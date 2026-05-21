#include "scrfd_detector.h"
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

#define SCRFD_MAX_FACES 20

typedef struct {
    BoundingBox bbox;
    float confidence;
    float keypoints[SCRFD_NUM_KEYPOINTS][2];
} RawFaceDetection;

SCRFDDetector* scrfd_detector_create(const char* model_path, int input_w, int input_h,
                                      float conf_thresh, float nms_thresh, bool use_onnx) {
    SCRFDDetector* det = (SCRFDDetector*)calloc(1, sizeof(SCRFDDetector));
    if (!det) return NULL;

    det->input_width = input_w > 0 ? input_w : 640;
    det->input_height = input_h > 0 ? input_h : 640;
    det->confidence_threshold = conf_thresh;
    det->nms_threshold = nms_thresh;
    det->use_onnx = use_onnx;

    if (model_path) {
        scrfd_detector_load_model(det, model_path);
    }

    return det;
}

void scrfd_detector_destroy(SCRFDDetector* det) {
    if (!det) return;
#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (det->session && ort) {
        ort->ReleaseSession(det->session);
    }
#endif
    free(det);
}

bool scrfd_detector_load_model(SCRFDDetector* det, const char* model_path) {
    if (!det || !model_path) {
        log_error("Model path is NULL");
        return false;
    }

    FILE* f = fopen(model_path, "rb");
    if (!f) {
        log_error("Cannot open SCRFD model file: %s", model_path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 1024) {
        fclose(f);
        log_error("SCRFD model file too small: %s", model_path);
        return false;
    }

    uint8_t magic[4];
    size_t bytes_read = fread(magic, 1, 4, f);
    fclose(f);

    if (bytes_read != 4) {
        log_error("Failed to read SCRFD model header: %s", model_path);
        return false;
    }

    if (magic[0] != 0x08 || magic[1] != 0x00 || magic[2] != 0x00 || magic[3] != 0x00) {
        log_warning("SCRFD model has unexpected protobuf header: %s", model_path);
    }

    strncpy(det->input_name, "input.1", sizeof(det->input_name) - 1);
    log_info("SCRFD model validated: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_warning("SCRFD: ONNX init failed, using heuristic fallback");
        return true;
    }

    det->session = ort_create_session(model_path, 4, false);
    if (!det->session) {
        log_warning("SCRFD: CreateSession failed, using heuristic fallback");
        return true;
    }

    log_info("SCRFD model loaded with ONNX Runtime: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));
#else
    log_info("SCRFD model validated: %s (%.2f MB) [build with HAS_ONNX_RUNTIME]", model_path, file_size / (1024.0f * 1024.0f));
#endif
    return true;
}

static float compute_skin_color_score(const uint8_t* image_data, int width, int height,
                                       int cx, int cy, int w, int h) {
    int x1 = UTILS_MAX(0, cx - w/2);
    int y1 = UTILS_MAX(0, cy - h/2);
    int x2 = UTILS_MIN(width - 1, cx + w/2);
    int y2 = UTILS_MIN(height - 1, cy + h/2);

    int skin_pixels = 0;
    int total_pixels = 0;

    for (int y = y1; y < y2; y += 2) {
        for (int x = x1; x < x2; x += 2) {
            int idx = (y * width + x) * 3;
            int r = image_data[idx + 2];
            int g = image_data[idx + 1];
            int b = image_data[idx + 0];

            if (r > 95 && g > 40 && b > 20 &&
                r > g && r > b &&
                (r - g) > 15 &&
                abs(r - g) > 15) {
                skin_pixels++;
            }
            total_pixels++;
        }
    }

    return total_pixels > 0 ? (float)skin_pixels / total_pixels : 0.0f;
}

static float compute_face_symmetry(const uint8_t* image_data, int width, int height,
                                    int cx, int cy, int w, int h) {
    int x1 = UTILS_MAX(0, cx - w/2);
    int y1 = UTILS_MAX(0, cy - h/2);
    int x2 = UTILS_MIN(width - 1, cx + w/2);
    int y2 = UTILS_MIN(height - 1, cy + h/2);

    float diff_sum = 0.0f;
    int count = 0;

    for (int y = y1; y < y2; y += 4) {
        for (int x = 0; x < w/2; x += 4) {
            int left_x = cx - x;
            int right_x = cx + x;

            if (left_x >= x1 && right_x < x2 && y >= y1 && y < y2) {
                int left_idx = (y * width + left_x) * 3;
                int right_idx = (y * width + right_x) * 3;

                for (int c = 0; c < 3; c++) {
                    float diff = (float)image_data[left_idx + c] - image_data[right_idx + c];
                    diff_sum += diff * diff;
                    count++;
                }
            }
        }
    }

    if (count == 0) return 0.0f;
    float mse = diff_sum / count;
    return UTILS_MAX(0.0f, 1.0f - mse / 1000.0f);
}

static int detect_face_candidates(const uint8_t* image_data, int width, int height,
                                   RawFaceDetection* faces, int max_faces, float conf_thresh) {
    int num_faces = 0;

    int step_x = UTILS_MAX(1, width / 15);
    int step_y = UTILS_MAX(1, height / 15);
    int face_w = UTILS_MIN(width, height) / 8;
    int face_h = face_w;

    for (int cy = face_h / 2; cy < height - face_h / 2 && num_faces < max_faces; cy += step_y) {
        for (int cx = face_w / 2; cx < width - face_w / 2 && num_faces < max_faces; cx += step_x) {
            float skin_score = compute_skin_color_score(image_data, width, height, cx, cy, face_w, face_h);
            if (skin_score < 0.2f) continue;

            float symmetry = compute_face_symmetry(image_data, width, height, cx, cy, face_w, face_h);
            if (symmetry < 0.3f) continue;

            float conf = (skin_score * 0.6f + symmetry * 0.4f);
            if (conf < conf_thresh) continue;

            RawFaceDetection* face = &faces[num_faces++];
            face->bbox.x_min = UTILS_MAX(0.0f, (float)(cx - face_w / 2));
            face->bbox.y_min = UTILS_MAX(0.0f, (float)(cy - face_h / 2));
            face->bbox.x_max = UTILS_MIN((float)width - 1, (float)(cx + face_w / 2));
            face->bbox.y_max = UTILS_MIN((float)height - 1, (float)(cy + face_h / 2));
            face->confidence = conf;

            float bw = bbox_width(&face->bbox);
            float bh = bbox_height(&face->bbox);
            float face_kp_offsets[SCRFD_NUM_KEYPOINTS][2] = {
                {-0.25f, -0.30f},
                {0.25f, -0.30f},
                {0.0f, -0.10f},
                {-0.20f, 0.15f},
                {0.20f, 0.15f}
            };

            for (int k = 0; k < SCRFD_NUM_KEYPOINTS; k++) {
                face->keypoints[k][0] = UTILS_CLAMP(bbox_center_x(&face->bbox) + face_kp_offsets[k][0] * bw, 0.0f, (float)width - 1);
                face->keypoints[k][1] = UTILS_CLAMP(bbox_center_y(&face->bbox) + face_kp_offsets[k][1] * bh, 0.0f, (float)height - 1);
            }
        }
    }

    return num_faces;
}

static int nms_faces(RawFaceDetection* faces, int num_faces, float iou_threshold) {
    if (num_faces <= 0) return 0;

    for (int i = 0; i < num_faces - 1; i++) {
        for (int j = i + 1; j < num_faces; j++) {
            if (faces[i].confidence < faces[j].confidence) {
                RawFaceDetection tmp = faces[i];
                faces[i] = faces[j];
                faces[j] = tmp;
            }
        }
    }

    bool* suppressed = (bool*)calloc(num_faces, sizeof(bool));
    if (!suppressed) return num_faces;

    int keep_count = 0;
    for (int i = 0; i < num_faces; i++) {
        if (suppressed[i]) continue;
        faces[keep_count++] = faces[i];

        for (int j = i + 1; j < num_faces; j++) {
            if (suppressed[j]) continue;
            float iou = bbox_iou(&faces[i].bbox, &faces[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    free(suppressed);
    return keep_count;
}

int scrfd_detector_detect_faces(SCRFDDetector* det, const uint8_t* image_data, int width, int height,
                                  FaceIdentity* out_faces, int max_faces) {
    if (!det || !image_data || !out_faces) return 0;

    RawFaceDetection raw_faces[SCRFD_MAX_FACES];
    int num_raw = 0;

#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (det->session && ort) {
        int input_w = det->input_width > 0 ? det->input_width : 640;
        int input_h = det->input_height > 0 ? det->input_height : 640;
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
                        input_tensor[0 * pixels + y * input_w + x] = resized[src_idx] / 255.0f;
                        input_tensor[1 * pixels + y * input_w + x] = resized[src_idx + 1] / 255.0f;
                        input_tensor[2 * pixels + y * input_w + x] = resized[src_idx + 2] / 255.0f;
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

            const char* input_names[] = {det->input_name};
            OrtValue* input_values[] = {input_tensor_val};
            OrtValue* output_values[1] = {NULL};

            OrtStatus* status = ort->Run(det->session, NULL,
                                           input_names, (const OrtValue* const*)input_values, 1,
                                           NULL, 0, output_values, 1);
            ort->ReleaseValue(input_tensor_val);

            if (status == NULL && output_values[0]) {
                OrtTensorTypeAndShapeInfo* shape_info;
                ort->GetTensorTypeAndShape(output_values[0], &shape_info);
                size_t num_elements;
                ort->GetTensorShapeElementCount(shape_info, &num_elements);
                ort->ReleaseTensorTypeAndShapeInfo(shape_info);

                float* output_data;
                ort->GetTensorMutableData(output_values[0], (void**)&output_data);

                int stride = 15;
                int num_proposals = (int)(num_elements / stride);

                for (int i = 0; i < num_proposals && num_raw < SCRFD_MAX_FACES; i++) {
                    float* row = output_data + i * stride;
                    float score = row[4];

                    if (score < det->confidence_threshold) continue;

                    float x1 = (row[0] - pad_x) / scale;
                    float y1 = (row[1] - pad_y) / scale;
                    float x2 = (row[2] - pad_x) / scale;
                    float y2 = (row[3] - pad_y) / scale;

                    raw_faces[num_raw].bbox.x_min = UTILS_CLAMP(x1, 0.0f, (float)width - 1);
                    raw_faces[num_raw].bbox.y_min = UTILS_CLAMP(y1, 0.0f, (float)height - 1);
                    raw_faces[num_raw].bbox.x_max = UTILS_CLAMP(x2, 0.0f, (float)width - 1);
                    raw_faces[num_raw].bbox.y_max = UTILS_CLAMP(y2, 0.0f, (float)height - 1);
                    raw_faces[num_raw].confidence = score;

                    for (int k = 0; k < SCRFD_NUM_KEYPOINTS && k < 5; k++) {
                        raw_faces[num_raw].keypoints[k][0] = UTILS_CLAMP((row[5 + k * 2] - pad_x) / scale, 0.0f, (float)width - 1);
                        raw_faces[num_raw].keypoints[k][1] = UTILS_CLAMP((row[5 + k * 2 + 1] - pad_y) / scale, 0.0f, (float)height - 1);
                    }

                    num_raw++;
                }

                ort->ReleaseValue(output_values[0]);
            }
            free(input_tensor);
        }

        if (num_raw > 0) {
            num_raw = nms_faces(raw_faces, num_raw, det->nms_threshold);
            goto output_results;
        }
    }
#endif

    num_raw = detect_face_candidates(image_data, width, height, raw_faces, SCRFD_MAX_FACES, det->confidence_threshold);

    if (num_raw <= 0) return 0;

    num_raw = nms_faces(raw_faces, num_raw, det->nms_threshold);

output_results:
    int num_faces = UTILS_MIN(num_raw, max_faces);
    for (int i = 0; i < num_faces; i++) {
        FaceIdentity* face = &out_faces[i];
        memset(face, 0, sizeof(FaceIdentity));

        face->bbox = raw_faces[i].bbox;
        face->confidence = raw_faces[i].confidence;
        strncpy(face->identity, "unknown", MAX_STRING_LEN - 1);
        face->identity[MAX_STRING_LEN - 1] = '\0';
        face->similarity = 0.0f;
        face->has_feature = false;
        face->has_keypoints = true;

        for (int k = 0; k < SCRFD_NUM_KEYPOINTS; k++) {
            face->keypoints[k][0] = raw_faces[i].keypoints[k][0];
            face->keypoints[k][1] = raw_faces[i].keypoints[k][1];
        }
    }

    log_debug("Detected %d faces", num_faces);
    return num_faces;
}

int scrfd_detector_crop_face(const SCRFDDetector* det, const uint8_t* image_data, int img_w, int img_h,
                              const FaceIdentity* face,
                              uint8_t* out_crop, int target_w, int target_h) {
    (void)det;
    if (!image_data || !face || !out_crop) return -1;

    int x1 = UTILS_MAX(0, (int)face->bbox.x_min);
    int y1 = UTILS_MAX(0, (int)face->bbox.y_min);
    int x2 = UTILS_MIN(img_w, (int)face->bbox.x_max);
    int y2 = UTILS_MIN(img_h, (int)face->bbox.y_max);

    int crop_w = x2 - x1;
    int crop_h = y2 - y1;

    if (crop_w <= 0 || crop_h <= 0) {
        memset(out_crop, 0, target_w * target_h * 3);
        return 0;
    }

    uint8_t* temp_crop = (uint8_t*)malloc(crop_w * crop_h * 3);
    if (!temp_crop) return -1;

    for (int y = 0; y < crop_h; y++) {
        for (int x = 0; x < crop_w; x++) {
            int src_idx = ((y1 + y) * img_w + (x1 + x)) * 3;
            int dst_idx = (y * crop_w + x) * 3;
            temp_crop[dst_idx + 0] = image_data[src_idx + 0];
            temp_crop[dst_idx + 1] = image_data[src_idx + 1];
            temp_crop[dst_idx + 2] = image_data[src_idx + 2];
        }
    }

    utils_resize_image(temp_crop, crop_w, crop_h, out_crop, target_w, target_h, 3);

    free(temp_crop);
    return 0;
}
