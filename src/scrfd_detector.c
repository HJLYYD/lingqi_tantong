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
#else
#error "scrfd_detector requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

#define SCRFD_MAX_FACES         20
#define SCRFD_MAX_INPUT_BYTES   (100UL * 1024UL * 1024UL)

typedef struct {
    BoundingBox bbox;
    float confidence;
    float keypoints[SCRFD_NUM_KEYPOINTS][2];
} RawFaceDetection;

static const int SCRFD_STRIDES[3] = {8, 16, 32};
static const int SCRFD_NUM_ANCHORS = 2;

SCRFDDetector* scrfd_detector_create(const char* model_path, int input_w, int input_h,
                                      float conf_thresh, float nms_thresh) {
    SCRFDDetector* det = (SCRFDDetector*)calloc(1, sizeof(SCRFDDetector));
    if (!det) return NULL;

    det->input_width = input_w > 0 ? input_w : 640;
    det->input_height = input_h > 0 ? input_h : 640;
    det->confidence_threshold = conf_thresh;
    det->nms_threshold = nms_thresh;

    if (!model_path || !scrfd_detector_load_model(det, model_path)) {
        log_error("SCRFDDetector: failed to load model %s", model_path ? model_path : "(null)");
        free(det);
        return NULL;
    }

    return det;
}

void scrfd_detector_destroy(SCRFDDetector* det) {
    if (!det) return;
    const OrtApi* ort = ort_get_api();
    if (det->session && ort) {
        ort->ReleaseSession(det->session);
    }
    free(det);
}

bool scrfd_detector_load_model(SCRFDDetector* det, const char* model_path) {
    if (!det || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    strncpy(det->input_name, "input.1", sizeof(det->input_name) - 1);

    if (!ort_global_init()) {
        log_error("SCRFD: ORT runtime not initialized");
        return false;
    }

    det->session = ort_create_session(model_path, 4, true);
    if (!det->session) {
        log_error("SCRFD: ONNX session creation failed for %s", model_path);
        return false;
    }

    log_info("SCRFD model loaded: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));
    return true;
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

static void distance2bbox(float cx, float cy, const float* distance,
                          BoundingBox* out) {
    out->x_min = cx - distance[0];
    out->y_min = cy - distance[1];
    out->x_max = cx + distance[2];
    out->y_max = cy + distance[3];
}

static int decode_scrfd_multi_output(const OrtApi* ort, OrtValue** outputs, size_t num_outputs,
                                     int input_w, int input_h, int img_w, int img_h,
                                     float scale, int pad_x, int pad_y,
                                     float conf_thresh,
                                     RawFaceDetection* raw_faces, int max_faces) {
    int num_raw = 0;
    if (num_outputs < 9) {
        log_warning("SCRFD: unexpected num_outputs=%zu (expected 9 for kps variant)", num_outputs);
    }

    for (int stride_idx = 0; stride_idx < 3 && num_raw < max_faces; stride_idx++) {
        int stride = SCRFD_STRIDES[stride_idx];
        int fm_w = input_w / stride;
        int fm_h = input_h / stride;
        int num_anchors_total = fm_w * fm_h * SCRFD_NUM_ANCHORS;

        size_t score_idx = (size_t)stride_idx;
        size_t bbox_idx  = (size_t)stride_idx + 3;
        size_t kps_idx   = (size_t)stride_idx + 6;
        if (score_idx >= num_outputs || bbox_idx >= num_outputs) continue;

        float* score_data = NULL;
        float* bbox_data = NULL;
        float* kps_data = NULL;
        OrtStatus* status = ort->GetTensorMutableData(outputs[score_idx], (void**)&score_data);
        if (status) { ort->ReleaseStatus(status); continue; }
        status = ort->GetTensorMutableData(outputs[bbox_idx], (void**)&bbox_data);
        if (status) { ort->ReleaseStatus(status); continue; }
        if (kps_idx < num_outputs && outputs[kps_idx]) {
            status = ort->GetTensorMutableData(outputs[kps_idx], (void**)&kps_data);
            if (status) { ort->ReleaseStatus(status); kps_data = NULL; }
        }

        for (int anchor_idx = 0; anchor_idx < num_anchors_total && num_raw < max_faces; anchor_idx++) {
            float score = score_data[anchor_idx];
            if (score < conf_thresh) continue;

            int spatial = anchor_idx / SCRFD_NUM_ANCHORS;
            int row = spatial / fm_w;
            int col = spatial % fm_w;
            float anchor_cx = (col + 0.5f) * stride;
            float anchor_cy = (row + 0.5f) * stride;

            float dist[4];
            for (int k = 0; k < 4; k++) {
                dist[k] = bbox_data[anchor_idx * 4 + k] * stride;
            }
            BoundingBox box_lb;
            distance2bbox(anchor_cx, anchor_cy, dist, &box_lb);

            box_lb.x_min = (box_lb.x_min - pad_x) / scale;
            box_lb.y_min = (box_lb.y_min - pad_y) / scale;
            box_lb.x_max = (box_lb.x_max - pad_x) / scale;
            box_lb.y_max = (box_lb.y_max - pad_y) / scale;

            box_lb.x_min = UTILS_CLAMP(box_lb.x_min, 0.0f, (float)img_w - 1);
            box_lb.y_min = UTILS_CLAMP(box_lb.y_min, 0.0f, (float)img_h - 1);
            box_lb.x_max = UTILS_CLAMP(box_lb.x_max, 0.0f, (float)img_w - 1);
            box_lb.y_max = UTILS_CLAMP(box_lb.y_max, 0.0f, (float)img_h - 1);

            if (box_lb.x_max - box_lb.x_min < 8.0f || box_lb.y_max - box_lb.y_min < 8.0f) continue;

            RawFaceDetection* face = &raw_faces[num_raw++];
            face->bbox = box_lb;
            face->confidence = score;

            if (kps_data) {
                for (int k = 0; k < SCRFD_NUM_KEYPOINTS; k++) {
                    float kx = anchor_cx + kps_data[anchor_idx * 10 + k * 2 + 0] * stride;
                    float ky = anchor_cy + kps_data[anchor_idx * 10 + k * 2 + 1] * stride;
                    kx = (kx - pad_x) / scale;
                    ky = (ky - pad_y) / scale;
                    face->keypoints[k][0] = UTILS_CLAMP(kx, 0.0f, (float)img_w - 1);
                    face->keypoints[k][1] = UTILS_CLAMP(ky, 0.0f, (float)img_h - 1);
                }
            } else {
                float cx = (face->bbox.x_min + face->bbox.x_max) * 0.5f;
                float cy = (face->bbox.y_min + face->bbox.y_max) * 0.5f;
                for (int k = 0; k < SCRFD_NUM_KEYPOINTS; k++) {
                    face->keypoints[k][0] = cx;
                    face->keypoints[k][1] = cy;
                }
            }
        }
    }

    return num_raw;
}

int scrfd_detector_detect_faces(SCRFDDetector* det, const uint8_t* image_data, int width, int height,
                                  FaceIdentity* out_faces, int max_faces) {
    if (!det || !image_data || !out_faces || !det->session) return 0;
    if (width <= 0 || height <= 0 || max_faces <= 0) return 0;

    const OrtApi* ort = ort_get_api();
    if (!ort) return 0;

    int input_w = det->input_width > 0 ? det->input_width : 640;
    int input_h = det->input_height > 0 ? det->input_height : 640;
    size_t pixels = (size_t)input_w * (size_t)input_h;
    size_t input_size = pixels * 3 * sizeof(float);
    if (input_size == 0 || input_size > SCRFD_MAX_INPUT_BYTES) {
        log_error("SCRFD: refused unreasonable input tensor size %zu bytes", input_size);
        return 0;
    }

    float* input_tensor = (float*)malloc(input_size);
    if (!input_tensor) return 0;

    uint8_t* resized = (uint8_t*)malloc(pixels * 3);
    if (!resized) { free(input_tensor); return 0; }

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    utils_letterbox(image_data, width, height, resized, input_w, input_h, 3, &scale, &pad_x, &pad_y);

    for (int y = 0; y < input_h; y++) {
        for (int x = 0; x < input_w; x++) {
            int src_idx = (y * input_w + x) * 3;
            input_tensor[0 * (int)pixels + y * input_w + x] = (resized[src_idx + 0] - 127.5f) / 128.0f;
            input_tensor[1 * (int)pixels + y * input_w + x] = (resized[src_idx + 1] - 127.5f) / 128.0f;
            input_tensor[2 * (int)pixels + y * input_w + x] = (resized[src_idx + 2] - 127.5f) / 128.0f;
        }
    }
    free(resized);

    int64_t input_shape[4] = {1, 3, input_h, input_w};
    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) {
        const char* msg = ort->GetErrorMessage(status);
        log_error("SCRFD: CreateCpuMemoryInfo failed: %s", msg ? msg : "unknown");
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
        log_error("SCRFD: CreateTensor failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(status);
        free(input_tensor);
        return 0;
    }

    size_t num_outputs_meta = 0;
    OrtStatus* st_oc = ort->SessionGetOutputCount(det->session, &num_outputs_meta);
    if (st_oc) ort->ReleaseStatus(st_oc);
    if (num_outputs_meta == 0) num_outputs_meta = 9;

    OrtAllocator* allocator = NULL;
    OrtStatus* st_alloc = ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_alloc) ort->ReleaseStatus(st_alloc);

    char** output_names_buf = (char**)calloc(num_outputs_meta, sizeof(char*));
    OrtValue** output_values = (OrtValue**)calloc(num_outputs_meta, sizeof(OrtValue*));
    if (!output_names_buf || !output_values) {
        free(output_names_buf); free(output_values);
        ort->ReleaseValue(input_tensor_val);
        free(input_tensor);
        return 0;
    }

    bool name_ok = true;
    for (size_t oi = 0; oi < num_outputs_meta; oi++) {
        char* name_ptr = NULL;
        OrtStatus* s = ort->SessionGetOutputName(det->session, oi, allocator, &name_ptr);
        if (s || !name_ptr) {
            if (s) ort->ReleaseStatus(s);
            name_ok = false;
            break;
        }
        output_names_buf[oi] = name_ptr;
    }

    if (!name_ok) {
        for (size_t oi = 0; oi < num_outputs_meta; oi++) {
            if (output_names_buf[oi]) {
                OrtStatus* st_f = ort->AllocatorFree(allocator, output_names_buf[oi]);
                if (st_f) ort->ReleaseStatus(st_f);
            }
        }
        free(output_names_buf);
        free(output_values);
        ort->ReleaseValue(input_tensor_val);
        free(input_tensor);
        log_error("SCRFD: SessionGetOutputName failed");
        return 0;
    }

    const char* input_names[] = {det->input_name};
    OrtValue* input_values[] = {input_tensor_val};

    status = ort->Run(det->session, NULL,
                      input_names, (const OrtValue* const*)input_values, 1,
                      (const char* const*)output_names_buf, num_outputs_meta, output_values);
    ort->ReleaseValue(input_tensor_val);
    free(input_tensor);

    for (size_t oi = 0; oi < num_outputs_meta; oi++) {
        if (output_names_buf[oi]) {
            OrtStatus* st_f = ort->AllocatorFree(allocator, output_names_buf[oi]);
            if (st_f) ort->ReleaseStatus(st_f);
        }
    }
    free(output_names_buf);

    if (status) {
        const char* msg = ort->GetErrorMessage(status);
        log_error("SCRFD inference failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(status);
        for (size_t oi = 0; oi < num_outputs_meta; oi++) {
            if (output_values[oi]) ort->ReleaseValue(output_values[oi]);
        }
        free(output_values);
        return 0;
    }

    RawFaceDetection raw_faces[SCRFD_MAX_FACES];
    int num_raw = decode_scrfd_multi_output(ort, output_values, num_outputs_meta,
                                            input_w, input_h, width, height,
                                            scale, pad_x, pad_y,
                                            det->confidence_threshold,
                                            raw_faces, SCRFD_MAX_FACES);

    for (size_t oi = 0; oi < num_outputs_meta; oi++) {
        if (output_values[oi]) ort->ReleaseValue(output_values[oi]);
    }
    free(output_values);

    if (num_raw <= 0) return 0;

    num_raw = nms_faces(raw_faces, num_raw, det->nms_threshold);
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

    log_debug("SCRFD: detected %d faces", num_faces);
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
        memset(out_crop, 0, (size_t)target_w * target_h * 3);
        return 0;
    }

    uint8_t* temp_crop = (uint8_t*)malloc((size_t)crop_w * crop_h * 3);
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
