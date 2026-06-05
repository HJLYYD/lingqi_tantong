#include "yolov8_pose_estimator.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

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

    if (!model_path || !yolov8_pose_estimator_load_model(est, model_path)) {
        log_error("YOLOv8PoseEstimator: failed to load model %s", model_path ? model_path : "(null)");
        free(est);
        return NULL;
    }

    return est;
}

void yolov8_pose_estimator_destroy(YOLOv8PoseEstimator* est) {
    if (!est) return;
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
    return true;
}

static void softmax_stable_pose(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = (sum > 1e-12f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static int detect_pose_xquant_split(size_t num_outputs, OrtValue** all_output_vals,
                                     int group_indices[3][3]) {
    if (num_outputs < 3 || num_outputs % 3 != 0) return 0;
    const OrtApi* ort = ort_get_api();
    if (!ort) return 0;

    int out_hw[12] = {0}, out_c[12] = {0};
    int valid = 0;
    for (size_t oi = 0; oi < num_outputs && oi < 12; oi++) {
        OrtTensorTypeAndShapeInfo* si = NULL;
        if (ort->GetTensorTypeAndShape(all_output_vals[oi], &si)) continue;
        size_t nd = 0;
        { OrtStatus* _s = ort->GetDimensionsCount(si, &nd); if (_s) ort->ReleaseStatus(_s); }
        int64_t dims[4] = {0};
        { OrtStatus* _s = ort->GetDimensions(si, dims, nd); if (_s) ort->ReleaseStatus(_s); }
        ort->ReleaseTensorTypeAndShapeInfo(si);
        if (nd < 3 || dims[0] != 1) continue;
        out_c[oi] = (int)dims[1];
        int h = (int)dims[2], w = (nd >= 4) ? (int)dims[3] : 1;
        out_hw[oi] = h * w;
        valid++;
    }
    if (valid != (int)num_outputs) return 0;

    int num_groups = 0;
    bool used[12] = {false};
    for (size_t i = 0; i < num_outputs && num_groups < 3; i++) {
        if (used[i]) continue;
        int hw_i = out_hw[i];
        int group[3] = {(int)i, -1, -1};
        int found = 1;
        for (size_t j = i + 1; j < num_outputs && found < 3; j++) {
            if (used[j]) continue;
            if (out_hw[j] == hw_i) { group[found++] = (int)j; used[j] = true; }
        }
        if (found == 3) {
            used[i] = true;
            int reg_idx = -1, cls_idx = -1, kpt_idx = -1;
            for (int k = 0; k < 3; k++) {
                int ch = out_c[group[k]];
                if (ch >= 60 && ch <= 70) reg_idx = group[k];
                else if (ch >= 40 && ch <= 60) kpt_idx = group[k];
                else cls_idx = group[k];
            }
            if (reg_idx >= 0 && kpt_idx >= 0) {
                group_indices[num_groups][0] = reg_idx;
                group_indices[num_groups][1] = cls_idx >= 0 ? cls_idx : -1;
                group_indices[num_groups][2] = kpt_idx;
                num_groups++;
            }
        }
    }
    if (num_groups != (int)(num_outputs / 3)) return 0;
    return num_groups;
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

static int nms_pose(PoseEstimation* poses, int num_poses, float iou_threshold) {
    if (num_poses <= 0) return 0;

    /* Sort by confidence descending */
    for (int i = 0; i < num_poses - 1; i++) {
        for (int j = i + 1; j < num_poses; j++) {
            if (poses[i].confidence < poses[j].confidence) {
                PoseEstimation tmp = poses[i];
                poses[i] = poses[j];
                poses[j] = tmp;
            }
        }
    }

    bool* suppressed = (bool*)calloc((size_t)num_poses, sizeof(bool));
    if (!suppressed) return num_poses;

    int keep_count = 0;
    for (int i = 0; i < num_poses; i++) {
        if (suppressed[i]) continue;
        poses[keep_count++] = poses[i];

        for (int j = i + 1; j < num_poses; j++) {
            if (suppressed[j]) continue;
            /* Use OKS when both poses have valid bboxes, fall back to IoU */
            float similarity;
            if (poses[i].has_bbox && poses[j].has_bbox) {
                similarity = compute_oks(&poses[i], &poses[j]);
            } else {
                similarity = poses[i].has_bbox && poses[j].has_bbox ?
                             bbox_iou(&poses[i].bbox, &poses[j].bbox) : 0.0f;
            }
            /* OKS threshold: 0.65 is roughly equivalent to IoU 0.5 in density */
            if (similarity > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    free(suppressed);
    return keep_count;
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
    uint8_t* resized = (uint8_t*)malloc(pixels * 3);
    if (!resized) {
        free(input_tensor);
        log_error("YOLOv8Pose: resize buffer malloc failed");
        return 0;
    }
    utils_letterbox(image_data, width, height, resized, input_w, input_h, 3, &scale, &pad_x, &pad_y);

    for (int y = 0; y < input_h; y++) {
        for (int x = 0; x < input_w; x++) {
            int src_idx = (y * input_w + x) * 3;
            input_tensor[0 * (int)pixels + y * input_w + x] = resized[src_idx + 0] / 255.0f;
            input_tensor[1 * (int)pixels + y * input_w + x] = resized[src_idx + 1] / 255.0f;
            input_tensor[2 * (int)pixels + y * input_w + x] = resized[src_idx + 2] / 255.0f;
        }
    }
    free(resized);

    int64_t input_shape[4] = {1, 3, input_h, input_w};
    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) {
        const char* msg = ort->GetErrorMessage(status);
        log_error("YOLOv8Pose: CreateCpuMemoryInfo failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(status);
        free(input_tensor);
        return 0;
    }

    OrtValue* input_tensor_val = NULL;
    status = ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor, input_size,
                                                 input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_val);
    ort->ReleaseMemoryInfo(memory_info);
    if (status) {
        const char* msg = ort->GetErrorMessage(status);
        log_error("YOLOv8Pose: CreateTensor failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(status);
        free(input_tensor);
        return 0;
    }

    /* ── Multi-output query (handles truncated xquant models) ── */
    size_t num_outputs = 0;
    OrtStatus* st_oc = ort->SessionGetOutputCount(est->session, &num_outputs);
    if (st_oc) { ort->ReleaseStatus(st_oc); num_outputs = 1; }
    if (num_outputs == 0) num_outputs = 1;

    OrtAllocator* allocator = NULL;
    OrtStatus* st_alloc = ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_alloc) ort->ReleaseStatus(st_alloc);

    const char** all_output_names = (const char**)calloc(num_outputs, sizeof(char*));
    OrtValue**  all_output_vals   = (OrtValue**)calloc(num_outputs, sizeof(OrtValue*));
    if (!all_output_names || !all_output_vals) {
        free(all_output_names); free(all_output_vals);
        ort->ReleaseValue(input_tensor_val);
        free(input_tensor);
        return 0;
    }

    bool names_ok = true;
    for (size_t oi = 0; oi < num_outputs; oi++) {
        char* name_ptr = NULL;
        OrtStatus* s = ort->SessionGetOutputName(est->session, oi, allocator, &name_ptr);
        if (s || !name_ptr) { if (s) ort->ReleaseStatus(s); names_ok = false; break; }
        all_output_names[oi] = name_ptr;
    }
    if (!names_ok) {
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_names[oi]) { OrtStatus* sf = ort->AllocatorFree(allocator, (void*)all_output_names[oi]); if (sf) ort->ReleaseStatus(sf); }
        }
        free(all_output_names); free(all_output_vals);
        ort->ReleaseValue(input_tensor_val);
        free(input_tensor);
        return 0;
    }

    const char* input_names[] = {est->input_name};
    OrtValue*   input_vals[]  = {input_tensor_val};

    status = ort->Run(est->session, NULL,
                      input_names, (const OrtValue* const*)input_vals, 1,
                      (const char* const*)all_output_names, num_outputs, all_output_vals);
    ort->ReleaseValue(input_tensor_val);
    free(input_tensor);

    for (size_t oi = 0; oi < num_outputs; oi++) {
        if (all_output_names[oi]) { OrtStatus* sf = ort->AllocatorFree(allocator, (void*)all_output_names[oi]); if (sf) ort->ReleaseStatus(sf); }
    }
    free(all_output_names);

    if (status) {
        const char* msg = ort->GetErrorMessage(status);
        log_error("YOLOv8Pose inference failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(status);
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);
        return 0;
    }

    /*
     * Detect xquant-split format. For pose models, the split is:
     *   {[1,64,H,W], [1,1,H,W], [1,51,H,W]} × 3 strides
     * 64ch = DFL regression (4 coords × 16 bins)
     * 1ch  = objectness (likely broken ~0.5, same as YOLO11 detector)
     * 51ch = keypoints (17 kpts × 3: x, y, conf)
     *
     * As with the YOLO11 detector, the cls/obj heads are broken by INT8
     * quantization. We use DFL peakiness as the confidence signal and
     * decode keypoints from the 51ch output. The kpt head may also be
     * partially broken; if keypoint confidences are all ~0.5, we still
     * get valid bounding boxes from DFL.
     */

    /* Diagnostic: log output shapes to help debug format detection */
    log_info("YOLOv8Pose: %zu outputs, probing shapes...", num_outputs);
    for (size_t oi = 0; oi < num_outputs && oi < 12; oi++) {
        if (!all_output_vals[oi]) { log_info("YOLOv8Pose output[%zu]: NULL", oi); continue; }
        OrtTensorTypeAndShapeInfo* si = NULL;
        OrtStatus* st = ort->GetTensorTypeAndShape(all_output_vals[oi], &si);
        if (st || !si) {
            if (st) ort->ReleaseStatus(st);
            log_info("YOLOv8Pose output[%zu]: GetTensorTypeAndShape failed", oi);
            continue;
        }
        size_t nd = 0;
        { OrtStatus* _s = ort->GetDimensionsCount(si, &nd); if (_s) ort->ReleaseStatus(_s); }
        int64_t dims[4] = {0};
        { OrtStatus* _s = ort->GetDimensions(si, dims, nd < 4 ? nd : 4); if (_s) ort->ReleaseStatus(_s); }
        ONNXTensorElementDataType dt;
        { OrtStatus* _s = ort->GetTensorElementType(si, &dt); if (_s) ort->ReleaseStatus(_s); }
        ort->ReleaseTensorTypeAndShapeInfo(si);
        log_info("YOLOv8Pose output[%zu]: nd=%zu dims=[%lld,%lld,%lld,%lld] type=%d",
                 oi, nd, (long long)dims[0], (long long)dims[1], (long long)dims[2], (long long)dims[3], (int)dt);
    }

    int pose_split_groups[3][3] = {{-1,-1,-1},{-1,-1,-1},{-1,-1,-1}};
    int num_pose_groups = detect_pose_xquant_split(num_outputs, all_output_vals, pose_split_groups);

    if (num_pose_groups > 0) {
        log_info("YOLOv8Pose: detected xquant-split format with %d stride groups (DFL decode active)",
                 num_pose_groups);

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
                    float dfl_peaks[4];
                    for (int coord = 0; coord < 4; coord++) {
                        float bins[16];
                        int base = coord * 16;
                        for (int b = 0; b < 16; b++) {
                            bins[b] = reg_data[(base + b) * hw + pix];
                        }
                        softmax_stable_pose(bins, 16);
                        float max_bin = bins[0];
                        float val = 0.0f;
                        for (int b = 0; b < 16; b++) {
                            if (bins[b] > max_bin) max_bin = bins[b];
                            val += bins[b] * (float)b;
                        }
                        dfl_peaks[coord] = max_bin;
                        dists[coord] = val;
                    }
                    float dfl_conf = sqrtf(sqrtf(
                        dfl_peaks[0] * dfl_peaks[1] * dfl_peaks[2] * dfl_peaks[3]));

                    if (dfl_conf < est->confidence_threshold) continue;

                    float x1_grid = ((float)gx + 0.5f) - dists[0];
                    float y1_grid = ((float)gy + 0.5f) - dists[1];
                    float x2_grid = ((float)gx + 0.5f) + dists[2];
                    float y2_grid = ((float)gy + 0.5f) + dists[3];

                    float x1_model = x1_grid * (float)stride;
                    float y1_model = y1_grid * (float)stride;
                    float x2_model = x2_grid * (float)stride;
                    float y2_model = y2_grid * (float)stride;

                    float x1 = (x1_model - pad_x) / scale;
                    float y1 = (y1_model - pad_y) / scale;
                    float x2 = (x2_model - pad_x) / scale;
                    float y2 = (y2_model - pad_y) / scale;

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

                        kx = (kx - pad_x) / scale;
                        ky = (ky - pad_y) / scale;

                        /* ── Keypoint visibility gating ──
                         * Keypoints with confidence < 0.3 are typically
                         * occluded or outside the frame.  Mark them as
                         * invalid so downstream consumers skip them. */
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

        for (int i = 0; i < num_temp - 1; i++) {
            for (int j = i + 1; j < num_temp; j++) {
                if (temp_poses[i].confidence < temp_poses[j].confidence) {
                    PoseEstimation tmp = temp_poses[i];
                    temp_poses[i] = temp_poses[j];
                    temp_poses[j] = tmp;
                }
            }
        }

        if (num_temp > POSE_TOP_K_PROPOSALS) num_temp = POSE_TOP_K_PROPOSALS;

        bool* suppressed = (bool*)calloc((size_t)num_temp, sizeof(bool));
        int num_poses = 0;
        if (suppressed) {
            for (int i = 0; i < num_temp && num_poses < max_poses; i++) {
                if (suppressed[i]) continue;
                out_poses[num_poses++] = temp_poses[i];
                for (int j = i + 1; j < num_temp; j++) {
                    if (suppressed[j]) continue;
                    if (temp_poses[i].has_bbox && temp_poses[j].has_bbox) {
                        if (bbox_iou(&temp_poses[i].bbox, &temp_poses[j].bbox) > est->iou_threshold) {
                            suppressed[j] = true;
                        }
                    }
                }
            }
            free(suppressed);
        } else {
            num_poses = UTILS_MIN(num_temp, max_poses);
            memcpy(out_poses, temp_poses, (size_t)num_poses * sizeof(PoseEstimation));
        }
        free(temp_poses);

        log_debug("YOLOv8Pose: %d poses after DFL+NMS", num_poses);
        return num_poses;
    }

    /* ── Standard decode path (non-xquant-split model) ── */
    log_info("YOLOv8Pose: xquant-split not detected, falling back to standard decode (%zu outputs)", num_outputs);
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
    status = ort->GetTensorMutableData(output_val, (void**)&output_data);
    if (status || !output_data) {
        if (status) ort->ReleaseStatus(status);
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

    int num_poses = 0;
    for (int i = 0; i < num_proposals && num_poses < max_poses; i++) {
        float cx, cy, bw_box, bh_box, conf;
        if (transposed_layout) {
            cx = output_data[0 * num_proposals + i];
            cy = output_data[1 * num_proposals + i];
            bw_box = output_data[2 * num_proposals + i];
            bh_box = output_data[3 * num_proposals + i];
            conf = output_data[4 * num_proposals + i];
        } else {
            const float* row = output_data + i * expected_stride;
            cx = row[0];
            cy = row[1];
            bw_box = row[2];
            bh_box = row[3];
            conf = row[4];
        }

        if (conf < est->confidence_threshold) continue;

        float x_center = (cx - pad_x) / scale;
        float y_center = (cy - pad_y) / scale;
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
                kx = output_data[(5 + k * 3 + 0) * num_proposals + i];
                ky = output_data[(5 + k * 3 + 1) * num_proposals + i];
                kc = output_data[(5 + k * 3 + 2) * num_proposals + i];
            } else {
                const float* row = output_data + i * expected_stride;
                kx = row[5 + k * 3];
                ky = row[5 + k * 3 + 1];
                kc = row[5 + k * 3 + 2];
            }

            kx = (kx - pad_x) / scale;
            ky = (ky - pad_y) / scale;

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

    if (num_poses > 1) {
        num_poses = nms_pose(out_poses, num_poses, est->iou_threshold);
    }

    log_debug("YOLOv8-Pose: %d poses (layout=%s)", num_poses, transposed_layout ? "CHW" : "NHW");
    return num_poses;
}

const char* yolov8_pose_get_keypoint_name(int idx) {
    if (idx >= 0 && idx < YOLOV8_POSE_NUM_KEYPOINTS) {
        return COCO_KEYPOINT_NAMES[idx];
    }
    return "unknown";
}
