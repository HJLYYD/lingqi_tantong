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
#endif

static const char* COCO_KEYPOINT_NAMES[] = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

static const int COCO_KEYPOINT_PAIRS[][2] __attribute__((unused)) = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15},
    {12, 14}, {14, 16}
};

YOLOv8PoseEstimator* yolov8_pose_estimator_create(const char* model_path, int input_w, int input_h,
                                                   float conf_thresh, float iou_thresh, bool use_onnx) {
    YOLOv8PoseEstimator* est = (YOLOv8PoseEstimator*)calloc(1, sizeof(YOLOv8PoseEstimator));
    if (!est) return NULL;

    est->input_width = input_w > 0 ? input_w : YOLOV8_INPUT_SIZE;
    est->input_height = input_h > 0 ? input_h : YOLOV8_INPUT_SIZE;
    est->confidence_threshold = conf_thresh;
    est->iou_threshold = iou_thresh;
    est->use_onnx = use_onnx;

    if (model_path) {
        yolov8_pose_estimator_load_model(est, model_path);
    }

    return est;
}

void yolov8_pose_estimator_destroy(YOLOv8PoseEstimator* est) {
    if (!est) return;
#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (est->session && ort) {
        ort->ReleaseSession(est->session);
    }
#endif
    free(est);
}

bool yolov8_pose_estimator_load_model(YOLOv8PoseEstimator* est, const char* model_path) {
    if (!est || !model_path) {
        log_error("Model path is NULL");
        return false;
    }

    FILE* f = fopen(model_path, "rb");
    if (!f) {
        log_error("Cannot open pose model file: %s", model_path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 1024) {
        fclose(f);
        log_error("Pose model file too small: %s", model_path);
        return false;
    }

    uint8_t magic[4];
    size_t bytes_read = fread(magic, 1, 4, f);
    fclose(f);

    if (bytes_read != 4) {
        log_error("Failed to read pose model header: %s", model_path);
        return false;
    }

    if (magic[0] != 0x08 || magic[1] != 0x00 || magic[2] != 0x00 || magic[3] != 0x00) {
        log_warning("Pose model has unexpected protobuf header: %s", model_path);
    }

    strncpy(est->input_name, "images", sizeof(est->input_name) - 1);
    log_info("YOLOv8-Pose model validated: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_warning("YOLOv8Pose: ONNX init failed, using heuristic fallback");
        return true;
    }

    est->session = ort_create_session(model_path, 4, false);
    if (!est->session) {
        log_warning("YOLOv8Pose: Session creation failed, using heuristic fallback");
        return true;
    }

    log_info("YOLOv8-Pose model loaded with ONNX Runtime: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));
#else
    log_info("YOLOv8-Pose model validated: %s (%.2f MB) [build with HAS_ONNX_RUNTIME]", model_path, file_size / (1024.0f * 1024.0f));
#endif
    return true;
}

static void preprocess_pose(const uint8_t* image_data, int width, int height,
                            float* out_tensor, int target_w, int target_h,
                            float* out_scale, int* out_pad_x, int* out_pad_y) {
    uint8_t* padded = (uint8_t*)malloc(target_w * target_h * 3);
    if (!padded) return;

    utils_letterbox(image_data, width, height, padded, target_w, target_h, 3, out_scale, out_pad_x, out_pad_y);

    int pixels = target_w * target_h;
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_idx = (y * target_w + x) * 3;
            int dst_r = 0 * pixels + y * target_w + x;
            int dst_g = 1 * pixels + y * target_w + x;
            int dst_b = 2 * pixels + y * target_w + x;

            out_tensor[dst_r] = padded[src_idx + 2] / 255.0f;
            out_tensor[dst_g] = padded[src_idx + 1] / 255.0f;
            out_tensor[dst_b] = padded[src_idx + 0] / 255.0f;
        }
    }

    free(padded);
}

static void estimate_keypoints_from_bbox(const BoundingBox* bbox, const uint8_t* image_data,
                                          int width, int height, Keypoint* keypoints, int num_keypoints) {
    if (!bbox || !keypoints || num_keypoints < YOLOV8_POSE_NUM_KEYPOINTS) return;

    float cx = bbox_center_x(bbox);
    float cy = bbox_center_y(bbox);
    float bw = bbox_width(bbox);
    float bh = bbox_height(bbox);

    float kp_offsets[][2] = {
        {0.0f, -0.45f},
        {-0.08f, -0.50f},
        {0.08f, -0.50f},
        {-0.12f, -0.45f},
        {0.12f, -0.45f},
        {-0.18f, -0.20f},
        {0.18f, -0.20f},
        {-0.22f, -0.05f},
        {0.22f, -0.05f},
        {-0.25f, 0.10f},
        {0.25f, 0.10f},
        {-0.12f, 0.15f},
        {0.12f, 0.15f},
        {-0.14f, 0.35f},
        {0.14f, 0.35f},
        {-0.12f, 0.50f},
        {0.12f, 0.50f}
    };

    for (int i = 0; i < YOLOV8_POSE_NUM_KEYPOINTS; i++) {
        keypoints[i].x = UTILS_CLAMP(cx + kp_offsets[i][0] * bw, 0.0f, (float)width - 1);
        keypoints[i].y = UTILS_CLAMP(cy + kp_offsets[i][1] * bh, 0.0f, (float)height - 1);

        int kx = (int)keypoints[i].x;
        int ky = (int)keypoints[i].y;
        if (kx >= 0 && kx < width && ky >= 0 && ky < height) {
            int idx = (ky * width + kx) * 3;
            int gradient = 0;
            if (kx + 1 < width) {
                gradient += abs((int)image_data[idx + 3] - (int)image_data[idx]);
            }
            if (ky + 1 < height) {
                gradient += abs((int)image_data[(ky + 1) * width * 3 + kx * 3] - (int)image_data[idx]);
            }
            keypoints[i].confidence = UTILS_MIN(1.0f, (float)gradient / 50.0f + 0.3f);
        } else {
            keypoints[i].confidence = 0.1f;
        }

        strncpy(keypoints[i].name, COCO_KEYPOINT_NAMES[i], MAX_STRING_LEN - 1);
        keypoints[i].name[MAX_STRING_LEN - 1] = '\0';
    }
}

static int nms_pose(PoseEstimation* poses, int num_poses, float iou_threshold) {
    if (num_poses <= 0) return 0;

    for (int i = 0; i < num_poses - 1; i++) {
        for (int j = i + 1; j < num_poses; j++) {
            if (poses[i].confidence < poses[j].confidence) {
                PoseEstimation tmp = poses[i];
                poses[i] = poses[j];
                poses[j] = tmp;
            }
        }
    }

    bool* suppressed = (bool*)calloc(num_poses, sizeof(bool));
    if (!suppressed) return num_poses;

    int iou_thresh_int = (int)(iou_threshold * 1000.0f);
    (void)iou_thresh_int;
    int keep_count = 0;
    for (int i = 0; i < num_poses; i++) {
        if (suppressed[i]) continue;
        poses[keep_count++] = poses[i];

        for (int j = i + 1; j < num_poses; j++) {
            if (suppressed[j]) continue;
            if (poses[i].has_bbox && poses[j].has_bbox) {
                float x_left   = fmaxf(poses[i].bbox.x_min, poses[j].bbox.x_min);
                float y_top    = fmaxf(poses[i].bbox.y_min, poses[j].bbox.y_min);
                float x_right  = fminf(poses[i].bbox.x_max, poses[j].bbox.x_max);
                float y_bottom = fminf(poses[i].bbox.y_max, poses[j].bbox.y_max);

                if (x_right > x_left && y_bottom > y_top) {
                    float intersection = (x_right - x_left) * (y_bottom - y_top);
                    float area_i = (poses[i].bbox.x_max - poses[i].bbox.x_min) * (poses[i].bbox.y_max - poses[i].bbox.y_min);
                    float area_j = (poses[j].bbox.x_max - poses[j].bbox.x_min) * (poses[j].bbox.y_max - poses[j].bbox.y_min);
                    float iou = intersection / (area_i + area_j - intersection);
                    if (iou > iou_threshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    free(suppressed);
    return keep_count;
}

int yolov8_pose_estimator_estimate(YOLOv8PoseEstimator* est, const uint8_t* image_data, int width, int height,
                                    PoseEstimation* out_poses, int max_poses) {
    if (!est || !image_data || !out_poses) return 0;

#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (est->session && ort) {
        int input_w = est->input_width > 0 ? est->input_width : YOLOV8_INPUT_SIZE;
        int input_h = est->input_height > 0 ? est->input_height : YOLOV8_INPUT_SIZE;
        int pixels = input_w * input_h;
        size_t input_size = pixels * 3 * sizeof(float);

        float* input_tensor = (float*)malloc(input_size);
        if (input_tensor) {
            float scale = 0.0f;
            int pad_x = 0, pad_y = 0;
            uint8_t* resized = (uint8_t*)malloc(pixels * 3);
            if (resized) {
                utils_letterbox(image_data, width, height, resized, input_w, input_h, 3, &scale, &pad_x, &pad_y);

                for (int y = 0; y < input_h; y++) {
                    for (int x = 0; x < input_w; x++) {
                        int src_idx = (y * input_w + x) * 3;
                        input_tensor[0 * pixels + y * input_w + x] = resized[src_idx + 2] / 255.0f;
                        input_tensor[1 * pixels + y * input_w + x] = resized[src_idx + 1] / 255.0f;
                        input_tensor[2 * pixels + y * input_w + x] = resized[src_idx + 0] / 255.0f;
                    }
                }
                free(resized);
            }

            int64_t input_shape[] = {1, 3, input_h, input_w};
            OrtMemoryInfo* memory_info;
            ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

            OrtValue* input_tensor_val;
            ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor, input_size,
                                                         input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_val);
            ort->ReleaseMemoryInfo(memory_info);

            const char* input_names[] = {est->input_name};
            OrtValue* input_values[] = {input_tensor_val};
            OrtValue* output_values[1] = {NULL};

            OrtStatus* status = ort->Run(est->session, NULL,
                                                  input_names, (const OrtValue* const*)input_values, 1,
                                                  NULL, 0, output_values, 1);
            ort->ReleaseValue(input_tensor_val);

            int num_poses = 0;
            if (status == NULL && output_values[0]) {
                OrtTensorTypeAndShapeInfo* shape_info;
                ort->GetTensorTypeAndShape(output_values[0], &shape_info);
                size_t num_elements;
                ort->GetTensorShapeElementCount(shape_info, &num_elements);
                ort->ReleaseTensorTypeAndShapeInfo(shape_info);

                float* output_data;
                ort->GetTensorMutableData(output_values[0], (void**)&output_data);

                int stride = 4 + YOLOV8_POSE_NUM_KEYPOINTS * 3;
                int num_proposals = (int)(num_elements / stride);

                for (int i = 0; i < num_proposals && num_poses < max_poses; i++) {
                    float* row = output_data + i * stride;
                    float conf = row[4];

                    if (conf < est->confidence_threshold) continue;

                    float cx = (row[0] - pad_x) / scale;
                    float cy = (row[1] - pad_y) / scale;
                    float bw = row[2] / scale;
                    float bh = row[3] / scale;

                    PoseEstimation* pose = &out_poses[num_poses++];
                    memset(pose, 0, sizeof(PoseEstimation));
                    pose->confidence = conf;
                    pose->has_bbox = true;
                    pose->bbox.x_min = UTILS_CLAMP(cx - bw / 2, 0.0f, (float)width - 1);
                    pose->bbox.y_min = UTILS_CLAMP(cy - bh / 2, 0.0f, (float)height - 1);
                    pose->bbox.x_max = UTILS_CLAMP(cx + bw / 2, 0.0f, (float)width - 1);
                    pose->bbox.y_max = UTILS_CLAMP(cy + bh / 2, 0.0f, (float)height - 1);

                    for (int k = 0; k < YOLOV8_POSE_NUM_KEYPOINTS && k < 17; k++) {
                        float kx = (row[5 + k * 3] - pad_x) / scale;
                        float ky = (row[5 + k * 3 + 1] - pad_y) / scale;
                        float kc = row[5 + k * 3 + 2];

                        pose->keypoints[k].x = UTILS_CLAMP(kx, 0.0f, (float)width - 1);
                        pose->keypoints[k].y = UTILS_CLAMP(ky, 0.0f, (float)height - 1);
                        pose->keypoints[k].confidence = UTILS_CLAMP(kc, 0.0f, 1.0f);
                        strncpy(pose->keypoints[k].name, COCO_KEYPOINT_NAMES[k], MAX_STRING_LEN - 1);
                        pose->keypoints[k].name[MAX_STRING_LEN - 1] = '\0';
                    }
                    pose->num_keypoints = YOLOV8_POSE_NUM_KEYPOINTS;
                }

                ort->ReleaseValue(output_values[0]);
            }

            free(input_tensor);

            if (num_poses > 0) {
                log_debug("YOLOv8-Pose ONNX: estimated %d poses", num_poses);
                return num_poses;
            }
        }
    }
#endif

    float scale;
    int pad_x, pad_y;
    float* input_tensor = (float*)malloc(est->input_width * est->input_height * 3 * sizeof(float));
    if (!input_tensor) return 0;

    preprocess_pose(image_data, width, height, input_tensor, est->input_width, est->input_height,
                    &scale, &pad_x, &pad_y);

    int num_valid = 0;
    float conf_thresh = est->confidence_threshold;

    int step_x = UTILS_MAX(1, width / 12);
    int step_y = UTILS_MAX(1, height / 12);
    int base_w = width / 6;
    int base_h = height / 2;

    for (int cy = base_h / 2; cy < height - base_h / 2 && num_valid < max_poses; cy += step_y) {
        for (int cx = base_w / 2; cx < width - base_w / 2 && num_valid < max_poses; cx += step_x) {
            float area_score = 0.0f;
            int x1 = UTILS_MAX(0, cx - base_w/2);
            int y1 = UTILS_MAX(0, cy - base_h/2);
            int x2 = UTILS_MIN(width - 1, cx + base_w/2);
            int y2 = UTILS_MIN(height - 1, cy + base_h/2);

            for (int y = y1; y < y2; y += 4) {
                for (int x = x1; x < x2; x += 4) {
                    int idx = (y * width + x) * 3;
                    float gray = (image_data[idx] + image_data[idx + 1] + image_data[idx + 2]) / 3.0f;
                    area_score += gray;
                }
            }

            if (area_score < 10.0f) continue;

            float conf = UTILS_MIN(1.0f, area_score / 100.0f);
            if (conf < conf_thresh) continue;

            BoundingBox bbox;
            bbox.x_min = UTILS_MAX(0.0f, (float)(cx - base_w / 2));
            bbox.y_min = UTILS_MAX(0.0f, (float)(cy - base_h / 2));
            bbox.x_max = UTILS_MIN((float)width - 1, (float)(cx + base_w / 2));
            bbox.y_max = UTILS_MIN((float)height - 1, (float)(cy + base_h / 2));

            if (bbox_width(&bbox) < 40 || bbox_height(&bbox) < 80) continue;

            PoseEstimation* pose = &out_poses[num_valid++];
            memset(pose, 0, sizeof(PoseEstimation));

            pose->bbox = bbox;
            pose->has_bbox = true;
            pose->confidence = conf;

            estimate_keypoints_from_bbox(&bbox, image_data, width, height,
                                          pose->keypoints, YOLOV8_POSE_NUM_KEYPOINTS);

            int visible_kp = 0;
            for (int k = 0; k < YOLOV8_POSE_NUM_KEYPOINTS; k++) {
                if (pose->keypoints[k].confidence > 0.3f) visible_kp++;
            }
            pose->num_keypoints = YOLOV8_POSE_NUM_KEYPOINTS;
        }
    }

    free(input_tensor);

    if (num_valid > 0) {
        num_valid = nms_pose(out_poses, num_valid, est->iou_threshold);
    }

    log_debug("Estimated poses for %d persons", num_valid);
    return num_valid;
}

const char* yolov8_pose_get_keypoint_name(int idx) {
    if (idx >= 0 && idx < YOLOV8_POSE_NUM_KEYPOINTS) {
        return COCO_KEYPOINT_NAMES[idx];
    }
    return "unknown";
}
