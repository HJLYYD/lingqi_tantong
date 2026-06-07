#include "yolov8_pose_estimator.h"
#include "logger.h"
#include "utils.h"
#include "ort_inference_context.h"
#include "yolo_postprocess.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include "ort_common.h"
#else
#error "yolov8_pose_estimator requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

#define POSE_MAX_INPUT_BYTES (100UL * 1024UL * 1024UL)
#define POSE_MAX_PER_GROUP        200
#define POSE_SOFT_CAP_PER_GROUP   2000
#define POSE_MAX_PROPOSALS        6000
#define POSE_TOP_K_PROPOSALS      50

static const char* COCO_KEYPOINT_NAMES[] = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

/*
 * COCO per-keypoint sigmas for OKS computation.
 * These are the standard COCO keypoint evaluation constants,
 * used here for OKS-based NMS instead of IoU-based NMS.
 * Source: COCO dataset keypoint evaluation metric.
 */
static const float COCO_KPT_SIGMAS[17] = {
    0.026f,  /* nose */
    0.025f,  /* left_eye */
    0.025f,  /* right_eye */
    0.035f,  /* left_ear */
    0.035f,  /* right_ear */
    0.079f,  /* left_shoulder */
    0.079f,  /* right_shoulder */
    0.072f,  /* left_elbow */
    0.072f,  /* right_elbow */
    0.062f,  /* left_wrist */
    0.062f,  /* right_wrist */
    0.107f,  /* left_hip */
    0.107f,  /* right_hip */
    0.087f,  /* left_knee */
    0.087f,  /* right_knee */
    0.089f,  /* left_ankle */
    0.089f   /* right_ankle */
};

YOLOv8PoseEstimator* yolov8_pose_estimator_create(const char* model_path, int input_w, int input_h,
                                                   float conf_thresh, float iou_thresh) {
    YOLOv8PoseEstimator* est = (YOLOv8PoseEstimator*)calloc(1, sizeof(YOLOv8PoseEstimator));
    if (!est) return NULL;

    est->input_width = input_w > 0 ? input_w : YOLOV8_POSE_INPUT_SIZE;
    est->input_height = input_h > 0 ? input_h : YOLOV8_POSE_INPUT_SIZE;
    est->confidence_threshold = conf_thresh;
    est->iou_threshold = iou_thresh;

    /* Cache invalid until first frame probes output format */
    est->format_cached = false;
    est->cached_num_outputs = 0;
    est->is_xquant_split = false;
    est->num_pose_groups = 0;
    memset(est->pose_split_groups, -1, sizeof(est->pose_split_groups));

    if (!model_path || !yolov8_pose_estimator_load_model(est, model_path)) {
        log_error("YOLOv8PoseEstimator: failed to load model %s", model_path ? model_path : "(null)");
        free(est);
        return NULL;
    }

    return est;
}

void yolov8_pose_estimator_destroy(YOLOv8PoseEstimator* est) {
    if (!est) return;
    ort_ctx_destroy(est->ctx);
    const OrtApi* ort = ort_get_api();
    if (est->session && ort) {
        ort->ReleaseSession(est->session);
    }
    free(est);
}

bool yolov8_pose_estimator_load_model(YOLOv8PoseEstimator* est, const char* model_path) {
    if (!est || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    strncpy(est->input_name, "images", sizeof(est->input_name) - 1);

    if (!ort_global_init()) {
        log_error("YOLOv8Pose: ORT runtime not initialized");
        return false;
    }

    /* Use CPU EP to reserve SpacemiT EP slots for detection + face.
     * Pose estimation is optional and loaded for future use. */
    est->session = ort_create_session(model_path, 4, true);
    if (!est->session) {
        log_error("YOLOv8Pose: ONNX session creation failed for %s", model_path);
        return false;
    }

    int dims[8] = {0};
    int rank = ort_get_input_shape(est->session, dims, 8);
    if (rank == 4 && dims[2] > 0 && dims[3] > 0) {
        if (est->input_width != dims[3] || est->input_height != dims[2]) {
            log_info("YOLOv8Pose: overriding requested %dx%d with model's actual input %dx%d",
                     est->input_width, est->input_height, dims[3], dims[2]);
        }
        est->input_width = dims[3];
        est->input_height = dims[2];
    } else if (rank > 0) {
        log_warning("YOLOv8Pose: unexpected input rank=%d, keeping configured %dx%d",
                    rank, est->input_width, est->input_height);
    }

    log_info("YOLOv8-Pose model loaded: %s (%.2f MB) input=%dx%d",
             model_path, file_size / (1024.0f * 1024.0f), est->input_width, est->input_height);

    est->ctx = ort_ctx_create(est->session, est->input_width, est->input_height, 3);
    if (!est->ctx) {
        log_error("YOLOv8Pose: failed to create inference context");
        return false;
    }
    strncpy(est->ctx->input_name, est->input_name, sizeof(est->ctx->input_name) - 1);

    return true;
}

static int detect_pose_xquant_split(size_t num_outputs, OrtValue** all_output_vals,
                                     int group_indices[3][3]) {
    return yolo_detect_xquant_split(num_outputs, all_output_vals, group_indices, 1);
}

/*
 * Compute Object Keypoint Similarity (OKS) between two poses.
 * OKS is the standard COCO keypoint similarity metric — more robust
 * than bbox IoU for pose NMS because it considers keypoint distances
 * weighted by per-keypoint difficulty and object scale.
 *
 * OKS = Σ_i[exp(-d_i² / (2s²k_i²)) · δ(v_i>0)] / Σ_i[δ(v_i>0)]
 *
 * where d_i = distance between keypoint i in two poses,
 *       s   = sqrt(bbox area),
 *       k_i = per-keypoint constant (COCO sigmas),
 *       v_i = visibility flag (confidence > threshold).
 */
static float compute_oks(const PoseEstimation* a, const PoseEstimation* b) {
    if (!a->has_bbox || !b->has_bbox) return 0.0f;

    float area_a = bbox_area(&a->bbox);
    float area_b = bbox_area(&b->bbox);
    float s2 = sqrtf((area_a + area_b) * 0.5f);  /* mean scale squared approx */

    float oks_sum = 0.0f;
    int valid = 0;

    int n_kpts = UTILS_MIN(a->num_keypoints, b->num_keypoints);
    n_kpts = UTILS_MIN(n_kpts, 17);

    for (int k = 0; k < n_kpts; k++) {
        /* Only consider keypoints visible in BOTH poses */
        if (a->keypoints[k].confidence < 0.3f || b->keypoints[k].confidence < 0.3f)
            continue;

        float dx = a->keypoints[k].x - b->keypoints[k].x;
        float dy = a->keypoints[k].y - b->keypoints[k].y;
        float k2 = COCO_KPT_SIGMAS[k] * COCO_KPT_SIGMAS[k];
        float e = (dx * dx + dy * dy) / (2.0f * s2 * k2 + 1e-6f);
        oks_sum += expf(-e);
        valid++;
    }

    if (valid == 0) {
        /* Fall back to bbox IoU if no keypoints are mutually visible */
        return bbox_iou(&a->bbox, &b->bbox);
    }

    return oks_sum / (float)valid;
}

static float pose_similarity(const void* a, const void* b) {
    const PoseEstimation* pa = (const PoseEstimation*)a;
    const PoseEstimation* pb = (const PoseEstimation*)b;
    if (pa->has_bbox && pb->has_bbox) {
        return compute_oks(pa, pb);
    }
    return bbox_iou(&pa->bbox, &pb->bbox);
}

int yolov8_pose_estimator_estimate(YOLOv8PoseEstimator* est, const uint8_t* image_data, int width, int height,
                                    PoseEstimation* out_poses, int max_poses) {
    if (!est || !image_data || !out_poses || !est->session) return 0;
    if (width <= 0 || height <= 0 || max_poses <= 0) return 0;

    const OrtApi* ort = ort_get_api();
    if (!ort) return 0;

    int input_w = est->input_width > 0 ? est->input_width : YOLOV8_POSE_INPUT_SIZE;
    int input_h = est->input_height > 0 ? est->input_height : YOLOV8_POSE_INPUT_SIZE;
    size_t pixels = (size_t)input_w * (size_t)input_h;
    size_t input_size = pixels * 3 * sizeof(float);
    if (input_size == 0 || input_size > POSE_MAX_INPUT_BYTES) {
        log_error("YOLOv8Pose: refused unreasonable input tensor size %zu bytes", input_size);
        return 0;
    }

    float* input_tensor = (float*)malloc(input_size);
    if (!input_tensor) {
        log_error("YOLOv8Pose: input tensor malloc failed");
        return 0;
    }

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    int crop_x = 0, crop_y = 0;
    yolo_preprocess(image_data, width, height, input_tensor,
                    input_w, input_h, &scale, &pad_x, &pad_y, &crop_x, &crop_y);

    if (!ort_ctx_prepare_input(est->ctx, input_tensor, input_size)) {
        log_error("YOLOv8Pose: prepare input failed");
        free(input_tensor);
        return 0;
    }
    free(input_tensor);

    size_t num_outputs;
    bool is_split;
    int pose_split_groups[3][3];
    int num_pose_groups;

    if (est->format_cached) {
        num_outputs     = (size_t)est->cached_num_outputs;
        is_split        = est->is_xquant_split;
        num_pose_groups = est->num_pose_groups;
        memcpy(pose_split_groups, est->pose_split_groups, sizeof(pose_split_groups));
    } else {
        OrtStatus* st_oc = ort->SessionGetOutputCount(est->session, &num_outputs);
        if (st_oc) { ort->ReleaseStatus(st_oc); num_outputs = 1; }
        if (num_outputs == 0) num_outputs = 1;
    }

    OrtValue** all_output_vals = (OrtValue**)calloc(num_outputs, sizeof(OrtValue*));
    if (!all_output_vals) {
        return 0;
    }

    int64_t t_run_start = utils_get_time_ms();

    int run_rc = ort_ctx_run(est->ctx, all_output_vals);

    int64_t t_run_ms = utils_get_time_ms() - t_run_start;

    if (run_rc != 0) {
        log_error("YOLOv8Pose inference failed");
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);
        return 0;
    }

    /*
     * ── Format detection (first frame only) ──
     *
     * Detect xquant-split format. For pose models, the split is:
     *   {[1,64,H,W], [1,1,H,W], [1,51,H,W]} × 3 strides
     * 64ch = DFL regression (4 coords × 16 bins)
     * 1ch  = objectness (likely broken ~0.5)
     * 51ch = keypoints (17 kpts × 3: x, y, conf)
     *
     * The cls/obj heads are broken by INT8 quantization. We use DFL
     * peakiness as the confidence signal and decode keypoints from the
     * 51ch output.
     */
    if (!est->format_cached) {
        /* First frame: probe and cache */
        log_info("YOLOv8Pose: %zu outputs, probing shapes (frame 0)...", num_outputs);
        for (size_t oi = 0; oi < num_outputs && oi < 12; oi++) {
            if (!all_output_vals[oi]) continue;
            OrtTensorTypeAndShapeInfo* si = NULL;
            OrtStatus* st = ort->GetTensorTypeAndShape(all_output_vals[oi], &si);
            if (st || !si) { if (st) ort->ReleaseStatus(st); continue; }
            size_t nd = 0;
            { OrtStatus* _s = ort->GetDimensionsCount(si, &nd); if (_s) ort->ReleaseStatus(_s); }
            int64_t dims[4] = {0};
            { OrtStatus* _s = ort->GetDimensions(si, dims, nd < 4 ? nd : 4); if (_s) ort->ReleaseStatus(_s); }
            ONNXTensorElementDataType dt;
            { OrtStatus* _s = ort->GetTensorElementType(si, &dt); if (_s) ort->ReleaseStatus(_s); }
            ort->ReleaseTensorTypeAndShapeInfo(si);
            log_debug("YOLOv8Pose output[%zu]: nd=%zu dims=[%lld,%lld,%lld,%lld] type=%d",
                     oi, nd, (long long)dims[0], (long long)dims[1],
                     (long long)dims[2], (long long)dims[3], (int)dt);
        }

        memset(pose_split_groups, -1, sizeof(pose_split_groups));
        num_pose_groups = detect_pose_xquant_split(num_outputs, all_output_vals, pose_split_groups);

        /* Cache for all subsequent frames */
        est->cached_num_outputs = (int)num_outputs;
        est->is_xquant_split    = (num_pose_groups > 0);
        est->num_pose_groups    = num_pose_groups;
        memcpy(est->pose_split_groups, pose_split_groups, sizeof(pose_split_groups));
        est->format_cached      = true;

        if (num_pose_groups > 0) {
            log_info("YOLOv8Pose: detected xquant-split format with %d stride groups (cached)", num_pose_groups);
        } else {
            log_info("YOLOv8Pose: standard decode format (cached, %zu outputs)", num_outputs);
        }

        is_split = est->is_xquant_split;
    } else {
        is_split        = est->is_xquant_split;
        num_pose_groups = est->num_pose_groups;
        memcpy(pose_split_groups, est->pose_split_groups, sizeof(pose_split_groups));
    }

    if (is_split) {

        PoseEstimation* temp_poses = (PoseEstimation*)calloc(POSE_MAX_PROPOSALS, sizeof(PoseEstimation));
        if (!temp_poses) {
            for (size_t oi = 0; oi < num_outputs; oi++) {
                if (all_output_vals[oi]) ort->ReleaseValue(all_output_vals[oi]);
            }
            free(all_output_vals);
            return 0;
        }
        int num_temp = 0;

        for (int g = 0; g < num_pose_groups; g++) {
            int reg_idx = pose_split_groups[g][0];
            int kpt_idx = pose_split_groups[g][2];

            float* reg_data = NULL;
            OrtStatus* st_r = ort->GetTensorMutableData(all_output_vals[reg_idx], (void**)&reg_data);
            if (st_r || !reg_data) { if (st_r) ort->ReleaseStatus(st_r); continue; }

            OrtTensorTypeAndShapeInfo* si_r = NULL;
            if (ort->GetTensorTypeAndShape(all_output_vals[reg_idx], &si_r)) continue;
            int64_t rdims[4] = {0}; size_t rnd = 0;
            { OrtStatus* _s = ort->GetDimensionsCount(si_r, &rnd); if (_s) ort->ReleaseStatus(_s); }
            { OrtStatus* _s = ort->GetDimensions(si_r, rdims, rnd); if (_s) ort->ReleaseStatus(_s); }
            ort->ReleaseTensorTypeAndShapeInfo(si_r);

            int reg_c = (rnd >= 2) ? (int)rdims[1] : 0;
            int reg_h = (rnd >= 3) ? (int)rdims[2] : 0;
            int reg_w = (rnd >= 4) ? (int)rdims[3] : 1;
            int hw = reg_h * reg_w;
            if (reg_c < 60 || reg_c > 70 || hw <= 0) continue;
            int stride = input_h / reg_h;

            float* kpt_data = NULL;
            OrtStatus* st_k = ort->GetTensorMutableData(all_output_vals[kpt_idx], (void**)&kpt_data);
            if (st_k || !kpt_data) { if (st_k) ort->ReleaseStatus(st_k); continue; }

            OrtTensorTypeAndShapeInfo* si_k = NULL;
            if (ort->GetTensorTypeAndShape(all_output_vals[kpt_idx], &si_k)) continue;
            int64_t kdims[4] = {0}; size_t knd = 0;
            { OrtStatus* _s = ort->GetDimensionsCount(si_k, &knd); if (_s) ort->ReleaseStatus(_s); }
            { OrtStatus* _s = ort->GetDimensions(si_k, kdims, knd); if (_s) ort->ReleaseStatus(_s); }
            ort->ReleaseTensorTypeAndShapeInfo(si_k);

            int kpt_c = (knd >= 2) ? (int)kdims[1] : 51;
            int kpt_hw_val = ((knd >= 3) ? (int)kdims[2] : 0) * ((knd >= 4) ? (int)kdims[3] : 1);
            if (kpt_hw_val != hw) continue;

            if (g == 0) {
                log_info("YOLOv8Pose group[%d]: reg=[1,%d,%d,%d] kpt=[1,%d,%d,%d] stride=%d hw=%d",
                         g, reg_c, reg_h, reg_w, kpt_c, reg_h, reg_w, stride, hw);
            }

            int out_contrib = 0;
            for (int gy = 0; gy < reg_h; gy++) {
                for (int gx = 0; gx < reg_w; gx++) {
                    if (out_contrib >= POSE_SOFT_CAP_PER_GROUP) goto pose_group_done;
                    if (num_temp >= POSE_MAX_PROPOSALS) goto pose_decode_done;
                    int pix = gy * reg_w + gx;

                    float dists[4];
                    float dfl_conf = yolo_dfl_decode_position(reg_data, pix, hw, dists);

                    if (dfl_conf < est->confidence_threshold) continue;

                    float x1_grid = ((float)gx + 0.5f) - dists[0];
                    float y1_grid = ((float)gy + 0.5f) - dists[1];
                    float x2_grid = ((float)gx + 0.5f) + dists[2];
                    float y2_grid = ((float)gy + 0.5f) + dists[3];

                    float x1, y1, x2, y2;
                    yolo_map_to_original(x1_grid * (float)stride, y1_grid * (float)stride,
                                        scale, pad_x, pad_y, crop_x, crop_y, &x1, &y1);
                    yolo_map_to_original(x2_grid * (float)stride, y2_grid * (float)stride,
                                        scale, pad_x, pad_y, crop_x, crop_y, &x2, &y2);

                    x1 = UTILS_CLAMP(x1, 0.0f, (float)(width - 1));
                    y1 = UTILS_CLAMP(y1, 0.0f, (float)(height - 1));
                    x2 = UTILS_CLAMP(x2, 0.0f, (float)(width - 1));
                    y2 = UTILS_CLAMP(y2, 0.0f, (float)(height - 1));

                    if (x2 - x1 < 4.0f || y2 - y1 < 8.0f) continue;

                    PoseEstimation* pose = &temp_poses[num_temp++];
                    memset(pose, 0, sizeof(PoseEstimation));
                    pose->confidence = dfl_conf;
                    pose->has_bbox = true;
                    pose->bbox.x_min = x1;
                    pose->bbox.y_min = y1;
                    pose->bbox.x_max = x2;
                    pose->bbox.y_max = y2;

                    int num_kpt = (kpt_c >= 51) ? 17 : (kpt_c / 3);
                    if (num_kpt > 17) num_kpt = 17;
                    for (int k = 0; k < num_kpt; k++) {
                        float kx = kpt_data[(k * 3 + 0) * hw + pix];
                        float ky = kpt_data[(k * 3 + 1) * hw + pix];
                        float kc = kpt_data[(k * 3 + 2) * hw + pix];
                        kc = 1.0f / (1.0f + expf(-kc));

                        yolo_map_to_original(kx, ky, scale, pad_x, pad_y, crop_x, crop_y, &kx, &ky);
                        if (kc >= 0.3f) {
                            pose->keypoints[k].x = UTILS_CLAMP(kx, 0.0f, (float)(width - 1));
                            pose->keypoints[k].y = UTILS_CLAMP(ky, 0.0f, (float)(height - 1));
                            pose->keypoints[k].confidence = kc;
                        } else {
                            pose->keypoints[k].x = -1.0f;
                            pose->keypoints[k].y = -1.0f;
                            pose->keypoints[k].confidence = 0.0f;
                        }
                        strncpy(pose->keypoints[k].name, COCO_KEYPOINT_NAMES[k], MAX_STRING_LEN - 1);
                        pose->keypoints[k].name[MAX_STRING_LEN - 1] = '\0';
                    }
                    pose->num_keypoints = num_kpt;
                    out_contrib++;
                }
            }

        pose_group_done:
            if (g == 0) {
                log_info("YOLOv8Pose group[%d]: %d proposals above threshold (%.3f)",
                         g, out_contrib, est->confidence_threshold);
            }
        }

    pose_decode_done:
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);

        if (num_temp == 0) { free(temp_poses); return 0; }

        num_temp = yolo_nms_suppress(temp_poses, num_temp, sizeof(PoseEstimation),
                                      est->confidence_threshold, est->iou_threshold,
                                      max_poses, pose_similarity,
                                      offsetof(PoseEstimation, confidence));
        int num_poses = UTILS_MIN(num_temp, max_poses);
        memcpy(out_poses, temp_poses, (size_t)num_poses * sizeof(PoseEstimation));
        free(temp_poses);

        log_debug("YOLOv8Pose: %d poses after DFL+NMS (ORT Run: %lld ms)", num_poses, (long long)t_run_ms);
        return num_poses;
    }

    /* ── Standard decode path (non-xquant-split model) ── */
    log_debug("YOLOv8Pose: standard decode path (%zu outputs)", num_outputs);
    OrtValue* output_val = all_output_vals[0];
    for (size_t oi = 0; oi < num_outputs; oi++) {
        if (all_output_vals[oi]) { output_val = all_output_vals[oi]; break; }
    }
    if (!output_val) {
        free(all_output_vals);
        return 0;
    }

    OrtTensorTypeAndShapeInfo* shape_info = NULL;
    OrtStatus* st_shape = ort->GetTensorTypeAndShape(output_val, &shape_info);
    if (st_shape) ort->ReleaseStatus(st_shape);

    size_t num_dims = 0;
    int64_t dims[4] = {0, 0, 0, 0};
    int output_attr_count = 0;
    int output_proposal_count = 0;
    if (shape_info) {
        OrtStatus* st_dc = ort->GetDimensionsCount(shape_info, &num_dims);
        if (st_dc) ort->ReleaseStatus(st_dc);
        if (num_dims >= 2 && num_dims <= 4) {
            OrtStatus* st_d = ort->GetDimensions(shape_info, dims, num_dims);
            if (st_d) ort->ReleaseStatus(st_d);
            if (num_dims == 3 && dims[0] == 1) {
                output_attr_count = (int)dims[1];
                output_proposal_count = (int)dims[2];
            } else if (num_dims == 2) {
                output_attr_count = (int)dims[0];
                output_proposal_count = (int)dims[1];
            }
        }
        ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    }

    size_t num_elements = 0;
    shape_info = NULL;
    st_shape = ort->GetTensorTypeAndShape(output_val, &shape_info);
    if (st_shape) ort->ReleaseStatus(st_shape);
    if (shape_info) {
        OrtStatus* st_ec = ort->GetTensorShapeElementCount(shape_info, &num_elements);
        if (st_ec) ort->ReleaseStatus(st_ec);
        ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    }

    float* output_data = NULL;
    OrtStatus* st_data = ort->GetTensorMutableData(output_val, (void**)&output_data);
    if (st_data || !output_data) {
        if (st_data) ort->ReleaseStatus(st_data);
        ort->ReleaseValue(output_val);
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi] && all_output_vals[oi] != output_val) {
                ort->ReleaseValue(all_output_vals[oi]);
            }
        }
        free(all_output_vals);
        return 0;
    }

    int expected_stride = 5 + YOLOV8_POSE_NUM_KEYPOINTS * 3;
    bool transposed_layout = false;
    int num_proposals = 0;
    if (output_attr_count > 0 && output_proposal_count > 0) {
        if (output_attr_count == expected_stride) {
            transposed_layout = true;
            num_proposals = output_proposal_count;
        } else if (output_proposal_count == expected_stride) {
            transposed_layout = false;
            num_proposals = output_attr_count;
        } else {
            num_proposals = (int)(num_elements / expected_stride);
        }
    } else {
        num_proposals = (int)(num_elements / expected_stride);
    }
    if (num_proposals <= 0) {
        ort->ReleaseValue(output_val);
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi] && all_output_vals[oi] != output_val) {
                ort->ReleaseValue(all_output_vals[oi]);
            }
        }
        free(all_output_vals);
        return 0;
    }

    /*
     * ── Top-K pre-filter for standard decode ──
     *
     * Standard YOLO output has 8400 proposals (80×80+40×40+20×20).
     * Most are background.  We build a sorted index of confidences,
     * keep only the top K, then decode and NMS.  This reduces NMS
     * complexity from O(8400²) to O(K²) — a 70× speedup at K=100.
     */
    #define POSE_STANDARD_TOP_K 150
    typedef struct { float conf; int idx; } PoseCandidate;
    PoseCandidate* candidates = NULL;
    int* top_indices = NULL;
    int orig_num_proposals = num_proposals;  /* save before Top-K truncation for transposed index math */

    if (num_proposals > POSE_STANDARD_TOP_K) {
        candidates = (PoseCandidate*)calloc((size_t)num_proposals, sizeof(PoseCandidate));
        if (candidates) {
            for (int i = 0; i < num_proposals; i++) {
                float conf;
                if (transposed_layout) {
                    conf = output_data[4 * num_proposals + i];  /* 5th attr: objectness */
                } else {
                    conf = output_data[i * expected_stride + 4];
                }
                candidates[i].conf = conf;
                candidates[i].idx = i;
            }
            /* Partial sort: top K bubble to front */
            for (int i = 0; i < POSE_STANDARD_TOP_K && i < num_proposals; i++) {
                int best = i;
                for (int j = i + 1; j < num_proposals; j++) {
                    if (candidates[j].conf > candidates[best].conf) best = j;
                }
                if (best != i) {
                    PoseCandidate tmp = candidates[i];
                    candidates[i] = candidates[best];
                    candidates[best] = tmp;
                }
            }
            top_indices = (int*)calloc(POSE_STANDARD_TOP_K, sizeof(int));
            if (top_indices) {
                for (int i = 0; i < POSE_STANDARD_TOP_K; i++) {
                    top_indices[i] = candidates[i].idx;
                }
            }
            num_proposals = POSE_STANDARD_TOP_K;
            free(candidates);
            candidates = NULL;
        }
    }

    int num_poses = 0;
    for (int pi = 0; pi < num_proposals && num_poses < max_poses; pi++) {
        int idx = (top_indices) ? top_indices[pi] : pi;
        float cx, cy, bw_box, bh_box, conf;
        if (transposed_layout) {
            /* transposed: each attribute is a row of orig_num_proposals elements */
            cx = output_data[0 * orig_num_proposals + idx];
            cy = output_data[1 * orig_num_proposals + idx];
            bw_box = output_data[2 * orig_num_proposals + idx];
            bh_box = output_data[3 * orig_num_proposals + idx];
            conf = output_data[4 * orig_num_proposals + idx];
        } else {
            const float* row = output_data + idx * expected_stride;
            cx = row[0];
            cy = row[1];
            bw_box = row[2];
            bh_box = row[3];
            conf = row[4];
        }

        if (conf < est->confidence_threshold) continue;

        float x_center, y_center;
        yolo_map_to_original(cx, cy, scale, pad_x, pad_y, crop_x, crop_y, &x_center, &y_center);
        float box_w = bw_box / scale;
        float box_h = bh_box / scale;

        PoseEstimation* pose = &out_poses[num_poses++];
        memset(pose, 0, sizeof(PoseEstimation));
        pose->confidence = conf;
        pose->has_bbox = true;
        pose->bbox.x_min = UTILS_CLAMP(x_center - box_w / 2.0f, 0.0f, (float)width - 1);
        pose->bbox.y_min = UTILS_CLAMP(y_center - box_h / 2.0f, 0.0f, (float)height - 1);
        pose->bbox.x_max = UTILS_CLAMP(x_center + box_w / 2.0f, 0.0f, (float)width - 1);
        pose->bbox.y_max = UTILS_CLAMP(y_center + box_h / 2.0f, 0.0f, (float)height - 1);

        for (int k = 0; k < YOLOV8_POSE_NUM_KEYPOINTS; k++) {
            float kx, ky, kc;
            if (transposed_layout) {
                kx = output_data[(5 + k * 3 + 0) * orig_num_proposals + idx];
                ky = output_data[(5 + k * 3 + 1) * orig_num_proposals + idx];
                kc = output_data[(5 + k * 3 + 2) * orig_num_proposals + idx];
            } else {
                const float* row = output_data + idx * expected_stride;
                kx = row[5 + k * 3];
                ky = row[5 + k * 3 + 1];
                kc = row[5 + k * 3 + 2];
            }

            yolo_map_to_original(kx, ky, scale, pad_x, pad_y, crop_x, crop_y, &kx, &ky);

            if (kc >= 0.3f) {
                pose->keypoints[k].x = UTILS_CLAMP(kx, 0.0f, (float)width - 1);
                pose->keypoints[k].y = UTILS_CLAMP(ky, 0.0f, (float)height - 1);
                pose->keypoints[k].confidence = kc;
            } else {
                pose->keypoints[k].x = -1.0f;
                pose->keypoints[k].y = -1.0f;
                pose->keypoints[k].confidence = 0.0f;
            }
            strncpy(pose->keypoints[k].name, COCO_KEYPOINT_NAMES[k], MAX_STRING_LEN - 1);
            pose->keypoints[k].name[MAX_STRING_LEN - 1] = '\0';
        }
        pose->num_keypoints = YOLOV8_POSE_NUM_KEYPOINTS;
    }

    ort->ReleaseValue(output_val);
    for (size_t oi = 0; oi < num_outputs; oi++) {
        if (all_output_vals[oi] && all_output_vals[oi] != output_val) {
            ort->ReleaseValue(all_output_vals[oi]);
        }
    }
    free(all_output_vals);
    free(top_indices);

    if (num_poses > 1) {
        num_poses = yolo_nms_suppress(out_poses, num_poses, sizeof(PoseEstimation),
                                       0.0f, est->iou_threshold,
                                       max_poses, pose_similarity,
                                       offsetof(PoseEstimation, confidence));
    }

    #undef POSE_STANDARD_TOP_K

    log_debug("YOLOv8-Pose: %d poses (layout=%s, Top-K=%s, ORT Run: %lld ms)",
              num_poses, transposed_layout ? "CHW" : "NHW",
              top_indices ? "yes" : "no", (long long)t_run_ms);
    return num_poses;
}

const char* yolov8_pose_get_keypoint_name(int idx) {
    if (idx >= 0 && idx < YOLOV8_POSE_NUM_KEYPOINTS) {
        return COCO_KEYPOINT_NAMES[idx];
    }
    return "unknown";
}
