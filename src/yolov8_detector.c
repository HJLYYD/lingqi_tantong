#include "yolov8_detector.h"
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

static const char* COCO_CLASSES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

YOLOv8Detector* yolov8_detector_create(const char* model_path, int input_w, int input_h,
                                       float conf_thresh, float iou_thresh, bool use_onnx) {
    YOLOv8Detector* det = (YOLOv8Detector*)calloc(1, sizeof(YOLOv8Detector));
    if (!det) return NULL;

    det->input_width = input_w > 0 ? input_w : YOLOV8_INPUT_SIZE;
    det->input_height = input_h > 0 ? input_h : YOLOV8_INPUT_SIZE;
    det->confidence_threshold = conf_thresh;
    det->iou_threshold = iou_thresh;
    det->use_onnx = use_onnx;
    det->session = NULL;

    if (model_path) {
        yolov8_detector_load_model(det, model_path);
    }

    return det;
}

void yolov8_detector_destroy(YOLOv8Detector* det) {
    if (!det) return;
#ifdef HAS_ONNX_RUNTIME
    const OrtApi* g_ort = ort_get_api();
    if (det->session && g_ort) {
        g_ort->ReleaseSession(det->session);
    }
#endif
    free(det);
}

bool yolov8_detector_load_model(YOLOv8Detector* det, const char* model_path) {
    if (!det || !model_path) return false;

    FILE* f = fopen(model_path, "rb");
    if (!f) {
        log_error("Cannot open model file: %s", model_path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 1024) {
        fclose(f);
        log_error("Model file too small: %s (%zu bytes)", model_path, file_size);
        return false;
    }

    uint8_t magic[4];
    size_t bytes_read = fread(magic, 1, 4, f);
    fclose(f);

    if (bytes_read != 4) {
        log_error("Failed to read ONNX model header: %s", model_path);
        return false;
    }

    if (magic[0] != 0x08 || magic[1] != 0x00 || magic[2] != 0x00 || magic[3] != 0x00) {
        log_warning("YOLOv8 model has unexpected protobuf header: %s", model_path);
    }

    strncpy(det->input_name, "images", sizeof(det->input_name) - 1);

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_warning("ONNX Runtime init failed, using heuristic fallback for %s", model_path);
        log_info("YOLOv8 model validated: %s (%.2f MB) [heuristic mode]", model_path, file_size / (1024.0 * 1024.0));
        return true;
    }

    det->session = ort_create_session(model_path, 4, false);
    if (!det->session) {
        log_warning("Session creation failed, using heuristic fallback for %s", model_path);
        return true;
    }

    log_info("YOLOv8 model loaded with ONNX Runtime: %s (%.2f MB)", model_path, file_size / (1024.0 * 1024.0));
#else
    log_info("YOLOv8 model validated: %s (%.2f MB) [heuristic mode, build with HAS_ONNX_RUNTIME for real inference]", 
             model_path, file_size / (1024.0 * 1024.0));
#endif
    return true;
}

static void yolov8_preprocess(const uint8_t* image_data, int width, int height,
                       float* out_tensor, int target_w, int target_h,
                       float* out_scale, int* out_pad_x, int* out_pad_y) __attribute__((unused));
static void yolov8_preprocess(const uint8_t* image_data, int width, int height,
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

#ifdef HAS_ONNX_RUNTIME
static int onnx_postprocess(const float* output_data, size_t output_count,
                            int orig_w, int orig_h, float scale, int pad_x, int pad_y,
                            float conf_thresh, float iou_thresh,
                            Detection* out_detections, int max_detections) {
    int num_classes = YOLOV8_NUM_CLASSES;
    int num_proposals = (int)(output_count / (4 + num_classes));

    if (num_proposals <= 0 || num_proposals > YOLOV8_MAX_DETECTIONS) {
        num_proposals = YOLOV8_MAX_DETECTIONS;
    }

    Detection temp_dets[YOLOV8_MAX_DETECTIONS];
    int num_temp = 0;

    for (int i = 0; i < num_proposals && num_temp < YOLOV8_MAX_DETECTIONS; i++) {
        const float* row = output_data + i * (4 + num_classes);
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        const float* class_scores = row + 4;
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; c++) {
            if (class_scores[c] > max_score) {
                max_score = class_scores[c];
                best_class = c;
            }
        }

        if (max_score < conf_thresh) continue;

        float x1 = ((cx - w * 0.5f) - pad_x) / scale;
        float y1 = ((cy - h * 0.5f) - pad_y) / scale;
        float x2 = ((cx + w * 0.5f) - pad_x) / scale;
        float y2 = ((cy + h * 0.5f) - pad_y) / scale;

        x1 = UTILS_CLAMP(x1, 0.0f, (float)(orig_w - 1));
        y1 = UTILS_CLAMP(y1, 0.0f, (float)(orig_h - 1));
        x2 = UTILS_CLAMP(x2, 0.0f, (float)(orig_w - 1));
        y2 = UTILS_CLAMP(y2, 0.0f, (float)(orig_h - 1));

        if (x2 - x1 < 10.0f || y2 - y1 < 20.0f) continue;

        Detection* d = &temp_dets[num_temp++];
        d->bbox.x_min = x1;
        d->bbox.y_min = y1;
        d->bbox.x_max = x2;
        d->bbox.y_max = y2;
        d->confidence = max_score;
        d->class_id = best_class;
        const char* cls_name = yolov8_get_class_name(best_class);
        strncpy(d->class_name, cls_name, MAX_STRING_LEN - 1);
        d->class_name[MAX_STRING_LEN - 1] = '\0';
    }

    if (num_temp > 0) {
        utils_sort_detections_by_confidence(temp_dets, num_temp);

        bool* suppressed = (bool*)calloc(num_temp, sizeof(bool));
        if (suppressed) {
            int keep = 0;
            for (int i = 0; i < num_temp; i++) {
                if (suppressed[i]) continue;
                if (keep < max_detections) {
                    out_detections[keep++] = temp_dets[i];
                }
                for (int j = i + 1; j < num_temp; j++) {
                    if (suppressed[j]) continue;
                    if (bbox_iou(&temp_dets[i].bbox, &temp_dets[j].bbox) > iou_thresh) {
                        suppressed[j] = true;
                    }
                }
            }
            num_temp = keep;
            free(suppressed);
        }
    }

    return num_temp;
}
#endif

static float compute_iou(const BoundingBox* a, const BoundingBox* b) {
    float x_left   = fmaxf(a->x_min, b->x_min);
    float y_top    = fmaxf(a->y_min, b->y_min);
    float x_right  = fminf(a->x_max, b->x_max);
    float y_bottom = fminf(a->y_max, b->y_max);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection = (x_right - x_left) * (y_bottom - y_top);
    float area_a = (a->x_max - a->x_min) * (a->y_max - a->y_min);
    float area_b = (b->x_max - b->x_min) * (b->y_max - b->y_min);
    float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static int nms(Detection* dets, int num_dets, float iou_threshold) {
    if (num_dets <= 0) return 0;

    utils_sort_detections_by_confidence(dets, num_dets);

    bool* suppressed = (bool*)calloc(num_dets, sizeof(bool));
    if (!suppressed) return num_dets;

    int keep_count = 0;
    for (int i = 0; i < num_dets; i++) {
        if (suppressed[i]) continue;

        dets[keep_count++] = dets[i];

        for (int j = i + 1; j < num_dets; j++) {
            if (suppressed[j]) continue;
            if (compute_iou(&dets[i].bbox, &dets[j].bbox) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    free(suppressed);
    return keep_count;
}

static float compute_edge_strength(const uint8_t* image_data, int width, int height, int cx, int cy, int w, int h) {
    float strength = 0.0f;
    int count = 0;

    int x1 = UTILS_MAX(0, cx - w/2);
    int y1 = UTILS_MAX(0, cy - h/2);
    int x2 = UTILS_MIN(width - 1, cx + w/2);
    int y2 = UTILS_MIN(height - 1, cy + h/2);

    for (int y = y1; y < y2 - 1; y += 2) {
        for (int x = x1; x < x2 - 1; x += 2) {
            int idx = (y * width + x) * 3;
            int idx_right = (y * width + (x + 1)) * 3;
            int idx_down = ((y + 1) * width + x) * 3;

            float grad_x = 0.0f;
            float grad_y = 0.0f;
            for (int c = 0; c < 3; c++) {
                float dx = (float)image_data[idx_right + c] - image_data[idx + c];
                float dy = (float)image_data[idx_down + c] - image_data[idx + c];
                grad_x += dx * dx;
                grad_y += dy * dy;
            }

            strength += sqrtf(grad_x + grad_y);
            count++;
        }
    }

    return count > 0 ? strength / count : 0.0f;
}

static float compute_region_contrast(const uint8_t* image_data, int width, int height, int cx, int cy, int w, int h) {
    int x1 = UTILS_MAX(0, cx - w/2);
    int y1 = UTILS_MAX(0, cy - h/2);
    int x2 = UTILS_MIN(width - 1, cx + w/2);
    int y2 = UTILS_MIN(height - 1, cy + h/2);

    float min_val = 255.0f, max_val = 0.0f;
    int count = 0;

    for (int y = y1; y < y2; y += 4) {
        for (int x = x1; x < x2; x += 4) {
            int idx = (y * width + x) * 3;
            float gray = (image_data[idx + 0] + image_data[idx + 1] + image_data[idx + 2]) / 3.0f;
            if (gray < min_val) min_val = gray;
            if (gray > max_val) max_val = gray;
            count++;
        }
    }

    return count > 0 ? (max_val - min_val) / 255.0f : 0.0f;
}

static bool has_human_shape_ratio(float w, float h) {
    float ratio = w / UTILS_MAX(h, 1.0f);
    return ratio > 0.2f && ratio < 0.8f;
}

static int heuristic_detect(const uint8_t* image_data, int width, int height,
                            float conf_thresh, float iou_thresh,
                            Detection* out_detections, int max_detections) {
    int num_valid = 0;

    int step_x = UTILS_MAX(1, width / 20);
    int step_y = UTILS_MAX(1, height / 20);
    int base_w = width / 8;
    int base_h = height / 3;

    for (int cy = base_h / 2; cy < height - base_h / 2 && num_valid < max_detections; cy += step_y) {
        for (int cx = base_w / 2; cx < width - base_w / 2 && num_valid < max_detections; cx += step_x) {
            float edge_strength = compute_edge_strength(image_data, width, height, cx, cy, base_w, base_h);
            if (edge_strength < 5.0f) continue;

            float contrast = compute_region_contrast(image_data, width, height, cx, cy, base_w, base_h);
            if (contrast < 0.1f) continue;

            int w = base_w + (int)(edge_strength * 20.0f);
            int h = base_h + (int)(contrast * 100.0f);

            if (!has_human_shape_ratio((float)w, (float)h)) continue;

            float conf = (edge_strength / 50.0f + contrast) * 0.5f;
            conf = UTILS_CLAMP(conf, 0.0f, 1.0f);
            if (conf < conf_thresh) continue;

            float x1 = UTILS_MAX(0.0f, (float)(cx - w / 2));
            float y1 = UTILS_MAX(0.0f, (float)(cy - h / 2));
            float x2 = UTILS_MIN((float)width - 1, (float)(cx + w / 2));
            float y2 = UTILS_MIN((float)height - 1, (float)(cy + h / 2));

            if (x2 - x1 < 20 || y2 - y1 < 40) continue;

            Detection* d = &out_detections[num_valid++];
            d->bbox.x_min = x1;
            d->bbox.y_min = y1;
            d->bbox.x_max = x2;
            d->bbox.y_max = y2;
            d->confidence = conf;
            d->class_id = 0;
            strncpy(d->class_name, "person", MAX_STRING_LEN - 1);
            d->class_name[MAX_STRING_LEN - 1] = '\0';
        }
    }

    if (num_valid > 0) {
        num_valid = nms(out_detections, num_valid, iou_thresh);
    }

    return num_valid;
}

int yolov8_detector_detect(YOLOv8Detector* det, const uint8_t* image_data, int width, int height,
                            Detection* out_detections, int max_detections) {
    if (!det || !image_data || !out_detections) return 0;

#ifdef HAS_ONNX_RUNTIME
    const OrtApi* g_ort = ort_get_api();
    if (det->session && g_ort) {
        float scale;
        int pad_x, pad_y;
        size_t tensor_size = (size_t)det->input_width * det->input_height * 3;
        float* input_tensor = (float*)malloc(tensor_size * sizeof(float));
        if (!input_tensor) goto fallback;

        yolov8_preprocess(image_data, width, height, input_tensor, det->input_width, det->input_height,
                   &scale, &pad_x, &pad_y);

        int64_t input_shape[] = {1, 3, det->input_height, det->input_width};
        OrtMemoryInfo* mem_info;
        OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
        if (status) {
            g_ort->ReleaseStatus(status);
            free(input_tensor);
            goto fallback;
        }

        OrtValue* input_tensor_ort = NULL;
        status = g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, input_tensor, tensor_size * sizeof(float),
            input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_ort);
        g_ort->ReleaseMemoryInfo(mem_info);

        if (status) {
            g_ort->ReleaseStatus(status);
            free(input_tensor);
            goto fallback;
        }

        const char* input_names[] = {det->input_name};
        const char* output_names[] = {"output0"};
        OrtValue* output_tensor = NULL;

        status = g_ort->Run(det->session, NULL,
                           input_names, (const OrtValue* const*)&input_tensor_ort, 1,
                           output_names, 1, &output_tensor);

        g_ort->ReleaseValue(input_tensor_ort);
        free(input_tensor);

        if (status) {
            const char* msg = g_ort->GetErrorMessage(status);
            log_warning("ONNX inference failed: %s, falling back to heuristic", msg ? msg : "unknown");
            g_ort->ReleaseStatus(status);
            goto fallback;
        }

        float* output_data;
        status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status) {
            g_ort->ReleaseStatus(status);
            g_ort->ReleaseValue(output_tensor);
            goto fallback;
        }

        OrtTensorTypeAndShapeInfo* shape_info;
        g_ort->GetTensorTypeAndShape(output_tensor, &shape_info);
        size_t output_count;
        g_ort->GetTensorShapeElementCount(shape_info, &output_count);
        g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

        int result = onnx_postprocess(output_data, output_count,
                                      width, height, scale, pad_x, pad_y,
                                      det->confidence_threshold, det->iou_threshold,
                                      out_detections, max_detections);

        g_ort->ReleaseValue(output_tensor);

        log_debug("ONNX detected %d objects", result);
        return result;
    }

fallback:
#endif
    {
        int result = heuristic_detect(image_data, width, height,
                                      det->confidence_threshold, det->iou_threshold,
                                      out_detections, max_detections);
        log_debug("Heuristic detected %d objects", result);
        return result;
    }
}

int yolov8_detector_detect_persons(YOLOv8Detector* det, const uint8_t* image_data, int width, int height,
                                    Detection* out_detections, int max_detections) {
    return yolov8_detector_detect(det, image_data, width, height, out_detections, max_detections);
}

const char* yolov8_get_class_name(int class_id) {
    if (class_id >= 0 && class_id < YOLOV8_NUM_CLASSES) {
        return COCO_CLASSES[class_id];
    }
    return "unknown";
}