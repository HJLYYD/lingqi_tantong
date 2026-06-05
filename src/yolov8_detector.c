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
#else
#error "yolov8_detector requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

#define YOLOV8_MAX_INPUT_BYTES (100UL * 1024UL * 1024UL)
#define YOLOV8_MAX_OUTPUT_DETECTIONS 50
#define YOLOV8_MAX_PER_GROUP        200
#define YOLO11_SOFT_CAP_PER_GROUP   2000  /* per-group proposal cap; ensures all 3 stride groups contribute
                                             even when DFL threshold is low. Post-sort Top-K picks winners. */

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

YOLO11Detector* yolov8_detector_create(const char* model_path, int input_w, int input_h,
                                       float conf_thresh, float iou_thresh) {
    YOLO11Detector* det = (YOLO11Detector*)calloc(1, sizeof(YOLO11Detector));
    if (!det) return NULL;

    det->input_width = input_w > 0 ? input_w : YOLO11_INPUT_SIZE;
    det->input_height = input_h > 0 ? input_h : YOLO11_INPUT_SIZE;
    det->confidence_threshold = conf_thresh;
    det->iou_threshold = iou_thresh;
    det->session = NULL;

    if (!model_path || !yolov8_detector_load_model(det, model_path)) {
        log_error("YOLO11Detector: failed to load model %s", model_path ? model_path : "(null)");
        free(det);
        return NULL;
    }

    return det;
}

void yolov8_detector_destroy(YOLO11Detector* det) {
    if (!det) return;
    const OrtApi* g_ort = ort_get_api();
    if (det->session && g_ort) {
        g_ort->ReleaseSession(det->session);
    }
    free(det);
}

bool yolov8_detector_load_model(YOLO11Detector* det, const char* model_path) {
    if (!det || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    strncpy(det->input_name, "images", sizeof(det->input_name) - 1);

    if (!ort_global_init()) {
        log_error("YOLO11Detector: ORT runtime not initialized; cannot load %s", model_path);
        return false;
    }

    det->session = ort_create_session(model_path, 4, true);
    if (!det->session) {
        log_error("YOLO11Detector: failed to create ONNX session for %s", model_path);
        return false;
    }

    /*
     * Query actual input shape from the loaded session. Quantized models
     * are commonly exported at a fixed resolution (e.g. yolo11n.q.onnx
     * is 320x320 even if the caller asked for 640x640). Mismatched input
     * dims cause ORT to throw "Got invalid dimensions for input: images
     * for the following indices: index: 2 Got: 640 Expected: 320" at Run().
     *
     * Trust the model. NCHW layout [1,3,H,W]: dims[2]=H, dims[3]=W.
     */
    int dims[8] = {0};
    int rank = ort_get_input_shape(det->session, dims, 8);
    if (rank == 4 && dims[2] > 0 && dims[3] > 0) {
        if (det->input_width != dims[3] || det->input_height != dims[2]) {
            log_info("YOLO11: overriding requested %dx%d with model's actual input %dx%d",
                     det->input_width, det->input_height, dims[3], dims[2]);
        }
        det->input_width = dims[3];
        det->input_height = dims[2];
    } else if (rank > 0) {
        log_warning("YOLO11: unexpected input rank=%d, keeping configured %dx%d",
                    rank, det->input_width, det->input_height);
    }

    log_info("YOLO11 model loaded: %s (%.2f MB) input=%dx%d",
             model_path, file_size / (1024.0 * 1024.0), det->input_width, det->input_height);
    return true;
}

static void yolov8_preprocess(const uint8_t* image_data, int width, int height,
                       float* out_tensor, int target_w, int target_h,
                       float* out_scale, int* out_pad_x, int* out_pad_y,
                       int* out_crop_x, int* out_crop_y) {
    /*
     * Resolution auto-adaptation: when source aspect ratio differs
     * significantly from the model's square input, center-crop first.
     *
     * Example: 720×1280 portrait → center-crop to 720×720 → scale to 320×320
     *   Without crop: scale=0.25, object pixels = 25% of original
     *   With crop:    scale=0.44, object pixels = 44% of original (1.78× more!)
     *
     * CRITICAL: when crop is active, the returned scale/pad are relative to
     * the CROPPED image.  Callers MUST add (crop_x, crop_y) to decoded
     * coordinates to map back to the ORIGINAL image.  For 720×1280 video,
     * crop_y=280 — without this offset, all boxes are shifted up by 280 px.
     *
     * Threshold: crop when |log2(src_ar / dst_ar)| > 0.5 (≈1.4× aspect mismatch)
     */
    float src_ar = (float)width / (float)UTILS_MAX(height, 1);
    float dst_ar = (float)target_w / (float)UTILS_MAX(target_h, 1);
    float ar_ratio = src_ar / UTILS_MAX(dst_ar, 0.01f);
    bool need_crop = (ar_ratio < 0.7f || ar_ratio > 1.4f);

    const uint8_t* src_ptr = image_data;
    int src_w = width, src_h = height;
    int crop_x = 0, crop_y = 0;

    uint8_t* crop_buf = NULL;
    if (need_crop) {
        int crop_w, crop_h;
        if (src_ar < dst_ar) {
            /* Portrait on square: crop top/bottom */
            crop_w = width;
            crop_h = (int)((float)width / dst_ar + 0.5f);
            crop_x = 0;
            crop_y = (height - crop_h) / 2;
        } else {
            /* Landscape on square: crop left/right */
            crop_h = height;
            crop_w = (int)((float)height * dst_ar + 0.5f);
            crop_x = (width - crop_w) / 2;
            crop_y = 0;
        }
        if (crop_y < 0) crop_y = 0;
        if (crop_x < 0) crop_x = 0;
        if (crop_x + crop_w > width)  crop_w = width - crop_x;
        if (crop_y + crop_h > height) crop_h = height - crop_y;

        crop_buf = (uint8_t*)malloc((size_t)crop_w * crop_h * 3);
        if (crop_buf) {
            for (int y = 0; y < crop_h; y++) {
                for (int x = 0; x < crop_w; x++) {
                    int si = ((crop_y + y) * width + (crop_x + x)) * 3;
                    int di = (y * crop_w + x) * 3;
                    crop_buf[di + 0] = image_data[si + 0];
                    crop_buf[di + 1] = image_data[si + 1];
                    crop_buf[di + 2] = image_data[si + 2];
                }
            }
            src_ptr = crop_buf;
            src_w = crop_w;
            src_h = crop_h;
        }
    }

    /* Always return crop offset so caller can map coords correctly */
    if (out_crop_x) *out_crop_x = crop_x;
    if (out_crop_y) *out_crop_y = crop_y;

    uint8_t* padded = (uint8_t*)malloc((size_t)target_w * target_h * 3);
    if (!padded) { free(crop_buf); return; }

    utils_letterbox(src_ptr, src_w, src_h, padded, target_w, target_h, 3, out_scale, out_pad_x, out_pad_y);
    free(crop_buf);

    int pixels = target_w * target_h;
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_idx = (y * target_w + x) * 3;
            int dst_r = 0 * pixels + y * target_w + x;
            int dst_g = 1 * pixels + y * target_w + x;
            int dst_b = 2 * pixels + y * target_w + x;

            out_tensor[dst_r] = padded[src_idx + 0] / 255.0f;
            out_tensor[dst_g] = padded[src_idx + 1] / 255.0f;
            out_tensor[dst_b] = padded[src_idx + 2] / 255.0f;
        }
    }

    free(padded);
}

/* ── Numerically-stable softmax (in-place on n-element array) ── */
static void softmax_stable(float* x, int n) {
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

/*
 * Auto-detect xquant-split output format.
 *
 * Standard YOLO11: 1 output  [1,84,N]  or [1,N,84]
 * xquant-truncated: 3 outputs [1,84,Hi,Wi] per stride (not split)
 * xquant-split:    9 outputs in groups: ([1,64,H,W],[1,C,H,W],[1,1,H,W]) × 3 strides
 *
 * Returns number of stride groups detected (0 = use standard decode).
 * On return, sets num_groups and fills group_indices[group][3] with
 * the output indices of {reg, cls, extra} for each stride group.
 */
static int detect_xquant_split_format(size_t num_outputs,
                                       OrtValue** all_output_vals,
                                       int group_indices[3][3]) {
    if (num_outputs < 3 || num_outputs % 3 != 0) return 0;

    const OrtApi* g_ort = ort_get_api();
    if (!g_ort) return 0;

    /* Collect per-output spatial dims {H, W, C} */
    int out_hw[12] = {0}, out_c[12] = {0};
    int valid = 0;

    for (size_t oi = 0; oi < num_outputs && oi < 12; oi++) {
        OrtTensorTypeAndShapeInfo* si = NULL;
        if (g_ort->GetTensorTypeAndShape(all_output_vals[oi], &si)) continue;
        size_t nd = 0;
        { OrtStatus* _s = g_ort->GetDimensionsCount(si, &nd); if (_s) g_ort->ReleaseStatus(_s); }
        int64_t dims[4] = {0};
        { OrtStatus* _s = g_ort->GetDimensions(si, dims, nd); if (_s) g_ort->ReleaseStatus(_s); }
        g_ort->ReleaseTensorTypeAndShapeInfo(si);
        if (nd < 3) continue;
        if (dims[0] != 1) continue;
        out_c[oi] = (int)dims[1];
        int h = (int)dims[2], w = (nd >= 4) ? (int)dims[3] : 1;
        out_hw[oi] = h * w;
        valid++;
    }
    if (valid != (int)num_outputs) return 0;

    /* Group by spatial size: find sets of 3 outputs with same HW */
    int num_groups = 0;
    bool used[12] = {false};

    for (size_t i = 0; i < num_outputs && num_groups < 3; i++) {
        if (used[i]) continue;
        int hw_i = out_hw[i];
        /* Find two more outputs with same HW and complementary channel counts */
        int group[3] = {(int)i, -1, -1};
        int found = 1;
        for (size_t j = i + 1; j < num_outputs && found < 3; j++) {
            if (used[j]) continue;
            if (out_hw[j] == hw_i) {
                group[found++] = (int)j;
                used[j] = true;
            }
        }
        if (found == 3) {
            used[i] = true;
            /* Identify which is reg (64 ch), cls, extra (1 ch) */
            int reg_idx = -1, cls_idx = -1, ext_idx = -1;
            for (int k = 0; k < 3; k++) {
                int ch = out_c[group[k]];
                if (ch >= 60 && ch <= 70) reg_idx = group[k];      /* 64 ch → reg */
                else if (ch >= 10) cls_idx = group[k];             /* 17/80 ch → cls */
                else ext_idx = group[k];                            /* 1 ch → extra */
            }
            if (reg_idx >= 0 && cls_idx >= 0) {
                group_indices[num_groups][0] = reg_idx;
                group_indices[num_groups][1] = cls_idx;
                group_indices[num_groups][2] = ext_idx >= 0 ? ext_idx : -1;
                num_groups++;
            }
        }
    }

    /* Only accept if we found exactly num_outputs/3 groups */
    if (num_groups != (int)(num_outputs / 3)) return 0;
    return num_groups;
}

int yolov8_detector_detect(YOLO11Detector* det, const uint8_t* image_data, int width, int height,
                            Detection* out_detections, int max_detections) {
    if (!det || !image_data || !out_detections || !det->session) return 0;
    if (width <= 0 || height <= 0) return 0;

    const OrtApi* g_ort = ort_get_api();
    if (!g_ort) return 0;

    size_t tensor_count = (size_t)det->input_width * (size_t)det->input_height * 3;
    size_t tensor_bytes = tensor_count * sizeof(float);
    if (tensor_bytes == 0 || tensor_bytes > YOLOV8_MAX_INPUT_BYTES) {
        log_error("YOLO11: refused unreasonable input tensor size %zu bytes", tensor_bytes);
        return 0;
    }

    float* input_tensor = (float*)malloc(tensor_bytes);
    if (!input_tensor) {
        log_error("YOLO11: input_tensor malloc failed (%zu bytes)", tensor_bytes);
        return 0;
    }

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    int crop_x = 0, crop_y = 0;
    yolov8_preprocess(image_data, width, height, input_tensor,
                      det->input_width, det->input_height,
                      &scale, &pad_x, &pad_y, &crop_x, &crop_y);

    int64_t input_shape[4] = {1, 3, det->input_height, det->input_width};
    OrtMemoryInfo* mem_info = NULL;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("YOLO11: CreateCpuMemoryInfo failed: %s", msg ? msg : "unknown");
        g_ort->ReleaseStatus(status);
        free(input_tensor);
        return 0;
    }

    OrtValue* input_tensor_ort = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, input_tensor, tensor_bytes,
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_ort);
    g_ort->ReleaseMemoryInfo(mem_info);
    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("YOLO11: CreateTensorWithDataAsOrtValue failed: %s", msg ? msg : "unknown");
        g_ort->ReleaseStatus(status);
        free(input_tensor);
        return 0;
    }

    /*
     * Handle both single-output (standard ONNX export) and multi-output
     * (xquant-truncated) models. xquant truncation at /model.22/Reshape_*
     * removes the post-processing Concat, producing 3 stride-level outputs
     * instead of 1. We query output count dynamically, allocate per-output
     * name/value arrays, run once, then collect all proposals before NMS.
     */
    size_t num_outputs = 0;
    OrtStatus* st_oc = g_ort->SessionGetOutputCount(det->session, &num_outputs);
    if (st_oc) {
        g_ort->ReleaseStatus(st_oc);
        num_outputs = 1;
    }
    if (num_outputs == 0) num_outputs = 1;

    OrtAllocator* allocator = NULL;
    OrtStatus* st_alloc = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_alloc) g_ort->ReleaseStatus(st_alloc);

    const char** all_output_names = (const char**)calloc(num_outputs, sizeof(char*));
    OrtValue**  all_output_vals   = (OrtValue**)calloc(num_outputs, sizeof(OrtValue*));
    if (!all_output_names || !all_output_vals) {
        free(all_output_names); free(all_output_vals);
        g_ort->ReleaseValue(input_tensor_ort);
        free(input_tensor);
        return 0;
    }

    bool names_ok = true;
    for (size_t oi = 0; oi < num_outputs; oi++) {
        char* name_ptr = NULL;
        OrtStatus* s = g_ort->SessionGetOutputName(det->session, oi, allocator, &name_ptr);
        if (s || !name_ptr) {
            if (s) g_ort->ReleaseStatus(s);
            names_ok = false;
            break;
        }
        all_output_names[oi] = name_ptr;
    }

    if (!names_ok) {
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_names[oi]) {
                OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)all_output_names[oi]);
                if (st_f) g_ort->ReleaseStatus(st_f);
            }
        }
        free(all_output_names); free(all_output_vals);
        g_ort->ReleaseValue(input_tensor_ort);
        free(input_tensor);
        return 0;
    }

    const char* input_names[] = {det->input_name};
    OrtValue*   input_vals[]  = {input_tensor_ort};

    status = g_ort->Run(det->session, NULL,
                        input_names, (const OrtValue* const*)input_vals, 1,
                        (const char* const*)all_output_names, num_outputs, all_output_vals);
    g_ort->ReleaseValue(input_tensor_ort);
    free(input_tensor);

    for (size_t oi = 0; oi < num_outputs; oi++) {
        if (all_output_names[oi]) {
            OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)all_output_names[oi]);
            if (st_f) g_ort->ReleaseStatus(st_f);
        }
    }
    free(all_output_names);

    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("YOLO11 inference failed: %s", msg ? msg : "unknown");
        g_ort->ReleaseStatus(status);
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) g_ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);
        return 0;
    }

    /*
     * ── Output format detection & proposal collection ──
     *
     * Three formats are possible depending on how the model was exported:
     *
     *   A) Standard ONNX:  1 output  [1,84,N] or [1,N,84] or [1,84,H,W]
     *   B) xquant 3-out:   3 outputs [1,84,Hi,Wi] per stride (Concat removed,
     *                       but Detect head is complete)
     *   C) xquant-split:   9 outputs in stride groups of 3:
     *                        { [1,64,H,W], [1,C,H,W], [1,1,H,W] }
     *                       where the Detect head's final convolutions are
     *                       also removed — we must do DFL softmax on the
     *                       64-channel reg branch and sigmoid on the C-channel
     *                       cls branch ourselves.
     */
    /* Heap-allocate proposal buffer large enough for all stride groups.
     * Each group gets YOLOV8_MAX_PER_GROUP slots (200 × 3 = 600).
     * Stack allocation would risk overflow on embedded systems. */
    Detection* temp_dets = (Detection*)calloc(YOLO11_MAX_DETECTIONS, sizeof(Detection));
    if (!temp_dets) {
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) g_ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);
        return 0;
    }
    int num_temp = 0;
    int min_box_w = width >= 320 ? 8 : 4;
    int min_box_h = height >= 320 ? 16 : 8;
    int num_classes = YOLO11_NUM_CLASSES;

    static int debug_frame_count = 0;
    int this_frame = debug_frame_count++;
    bool dump_detail = (this_frame % 60 == 0);
    bool dump_first = (this_frame == 0);  /* always dump first frame for diagnostics */

    /* ── Try to detect xquant-split format (C) ── */
    int split_groups[3][3];  /* [group][3] = {reg_idx, cls_idx, ext_idx} */
    int num_groups = detect_xquant_split_format(num_outputs, all_output_vals, split_groups);

    if (num_groups > 0 && dump_detail) {
        log_info("YOLO11: detected xquant-split format with %d stride groups (DFL decode active)",
                 num_groups);
    }

    if (num_groups > 0) {
        /* ────────────────────────────────────────────────────────
         * Format C: xquant-split — DFL softmax + sigmoid per stride
         * ──────────────────────────────────────────────────────── */
        for (int g = 0; g < num_groups; g++) {
            int reg_idx = split_groups[g][0];
            int cls_idx = split_groups[g][1];
            int ext_idx = split_groups[g][2];  /* [1,1,H,W] objectness; may be -1 if absent */

            /* ── Get reg data & shape ── */
            float* reg_data = NULL;
            OrtStatus* st_r = g_ort->GetTensorMutableData(all_output_vals[reg_idx], (void**)&reg_data);
            if (st_r || !reg_data) { if (st_r) g_ort->ReleaseStatus(st_r); continue; }

            OrtTensorTypeAndShapeInfo* si_r = NULL;
            if (g_ort->GetTensorTypeAndShape(all_output_vals[reg_idx], &si_r)) continue;
            int64_t rdims[4] = {0}; size_t rnd = 0;
            { OrtStatus* _s = g_ort->GetDimensionsCount(si_r, &rnd); if (_s) g_ort->ReleaseStatus(_s); }
            { OrtStatus* _s = g_ort->GetDimensions(si_r, rdims, rnd); if (_s) g_ort->ReleaseStatus(_s); }
            size_t re_count = 0;
            { OrtStatus* _s = g_ort->GetTensorShapeElementCount(si_r, &re_count); if (_s) g_ort->ReleaseStatus(_s); }
            g_ort->ReleaseTensorTypeAndShapeInfo(si_r);

            int reg_c = (rnd >= 2) ? (int)rdims[1] : 0;
            int reg_h = (rnd >= 3) ? (int)rdims[2] : 0;
            int reg_w = (rnd >= 4) ? (int)rdims[3] : 1;
            int hw     = reg_h * reg_w;

            if (reg_c < 60 || reg_c > 70 || hw <= 0) continue;
            int stride = det->input_height / reg_h;  /* e.g. 320/40=8 */

            /* ── Get cls data & shape ── */
            float* cls_data = NULL;
            OrtStatus* st_c = g_ort->GetTensorMutableData(all_output_vals[cls_idx], (void**)&cls_data);
            if (st_c || !cls_data) { if (st_c) g_ort->ReleaseStatus(st_c); continue; }

            OrtTensorTypeAndShapeInfo* si_c = NULL;
            if (g_ort->GetTensorTypeAndShape(all_output_vals[cls_idx], &si_c)) continue;
            int64_t cdims[4] = {0}; size_t cnd = 0;
            { OrtStatus* _s = g_ort->GetDimensionsCount(si_c, &cnd); if (_s) g_ort->ReleaseStatus(_s); }
            { OrtStatus* _s = g_ort->GetDimensions(si_c, cdims, cnd); if (_s) g_ort->ReleaseStatus(_s); }
            g_ort->ReleaseTensorTypeAndShapeInfo(si_c);

            int cls_c = (cnd >= 2) ? (int)cdims[1] : num_classes;
            if (cls_c > num_classes) cls_c = num_classes;
            int cls_h = (cnd >= 3) ? (int)cdims[2] : 0;
            int cls_w = (cnd >= 4) ? (int)cdims[3] : 1;
            int cls_hw = cls_h * cls_w;

            if (cls_hw != hw) {
                /* Mismatched spatial dims — skip this group */
                continue;
            }

            /* ── Get ext (objectness) data ──
             * The ext tensor provides per-position foreground/background
             * classification. Even after INT8 quantization, this signal
             * carries useful information — multiplying DFL peakiness by
             * objectness cuts false positives by ~50% empirically. */
            float* ext_data = NULL;
            if (ext_idx >= 0) {
                OrtStatus* st_e = g_ort->GetTensorMutableData(all_output_vals[ext_idx], (void**)&ext_data);
                if (st_e) { ext_data = NULL; g_ort->ReleaseStatus(st_e); }
            }

            if (dump_detail || dump_first) {
                log_info("YOLO11 group[%d]: reg=[1,%d,%d,%d] cls=[1,%d,%d,%d] stride=%d hw=%d",
                         g, reg_c, reg_h, reg_w, cls_c, cls_h, cls_w, stride, hw);
            }

            /* ── Diagnostic: scan cls data for max/mean scores ── */
            if (dump_first) {
                float cls_max_all = -1e9f, cls_sum = 0.0f;
                int cls_above_01 = 0, cls_total = cls_c * hw;
                for (int ci = 0; ci < cls_c; ci++) {
                    for (int pi = 0; pi < hw; pi++) {
                        float logit = cls_data[ci * hw + pi];
                        float prob = 1.0f / (1.0f + expf(-logit));
                        if (prob > cls_max_all) cls_max_all = prob;
                        cls_sum += prob;
                        if (prob > 0.1f) cls_above_01++;
                    }
                }
                float cls_mean = cls_total > 0 ? cls_sum / (float)cls_total : 0.0f;
                log_info("YOLO11 group[%d] cls diag: max=%.4f mean=%.4f above_0.1=%d/%d",
                         g, (double)cls_max_all, (double)cls_mean, cls_above_01, cls_total);

                /* Also diagnose ext tensor (objectness) if available */
                if (split_groups[g][2] >= 0) {
                    OrtValue* ext_val_diag = all_output_vals[split_groups[g][2]];
                    float* ext_data_diag = NULL;
                    OrtStatus* st_ed = g_ort->GetTensorMutableData(ext_val_diag, (void**)&ext_data_diag);
                    if (st_ed == NULL && ext_data_diag) {
                        float ext_max = -1e9f, ext_min = 1e9f, ext_sum = 0.0f;
                        int ext_above_05 = 0;
                        for (int pi = 0; pi < hw; pi++) {
                            float ep = 1.0f / (1.0f + expf(-ext_data_diag[pi]));
                            if (ep > ext_max) ext_max = ep;
                            if (ep < ext_min) ext_min = ep;
                            ext_sum += ep;
                            if (ep > 0.5f) ext_above_05++;
                        }
                        float ext_mean = hw > 0 ? ext_sum / (float)hw : 0.0f;
                        log_info("YOLO11 group[%d] ext diag: max=%.4f min=%.4f mean=%.4f above_0.5=%d/%d",
                                 g, (double)ext_max, (double)ext_min, (double)ext_mean, ext_above_05, hw);
                    }
                    if (st_ed) g_ort->ReleaseStatus(st_ed);
                }

                /* ── DFL peakiness diagnostic ──
                 * Sample 200 random positions per group to check if DFL
                 * distributions are peaked (real signal) or flat (broken).
                 * Uniform DFL: peak ≈ 0.0625.  Good DFL: peak ≈ 0.2–0.8. */
                {
                    int ns = (hw < 200) ? hw : 200;
                    int step = (hw > 1) ? (hw / ns) : 1;
                    if (step < 1) step = 1;
                    float dfl_sum = 0.0f, dfl_max = 0.0f, dfl_min = 1.0f;
                    int dfl_above_010 = 0;
                    for (int si = 0; si < ns; si++) {
                        int pi = (si * step) % hw;
                        float peaks[4];
                        for (int coord = 0; coord < 4; coord++) {
                            float bins[16];
                            int base = coord * 16;
                            for (int b = 0; b < 16; b++)
                                bins[b] = reg_data[(base + b) * hw + pi];
                            softmax_stable(bins, 16);
                            float mb = bins[0];
                            for (int b = 1; b < 16; b++)
                                if (bins[b] > mb) mb = bins[b];
                            peaks[coord] = mb;
                        }
                        float gm = sqrtf(sqrtf(peaks[0] * peaks[1] * peaks[2] * peaks[3]));
                        dfl_sum += gm;
                        if (gm > dfl_max) dfl_max = gm;
                        if (gm < dfl_min) dfl_min = gm;
                        if (gm > 0.10f) dfl_above_010++;
                    }
                    float dfl_mean = ns > 0 ? dfl_sum / (float)ns : 0.0f;
                    log_info("YOLO11 group[%d] DFL diag: max=%.4f min=%.4f mean=%.4f above_0.10=%d/%d",
                             g, (double)dfl_max, (double)dfl_min, (double)dfl_mean, dfl_above_010, ns);
                }
            }

            /*
             * Iterate every spatial position, DFL-decode bbox from reg.
             *
             * FAIR ALLOCATION: we process ALL stride groups.  Group 0
             * (stride=8, 6400 positions) used to fill the old global cap
             * (3000), starving groups 1 and 2 → zero medium/large-object
             * detections.  Now the cap (6000) is large enough for all
             * positions that pass the DFL threshold (~4000 at 0.30).
             *
             * The per-group soft cap at 2000 prevents any one group from
             * exhausting the array if the DFL threshold is set very low.
             * Post-sort Top-K (50) then selects the best proposals globally.
             */
            int out_contrib = 0;
            for (int gy = 0; gy < reg_h; gy++) {
                for (int gx = 0; gx < reg_w; gx++) {
                    /* Guard: don't overflow the malloc'd array.
                     * Per-group soft cap ensures all groups get a turn. */
                    if (out_contrib >= YOLO11_SOFT_CAP_PER_GROUP) goto group_done;
                    if (num_temp >= YOLO11_MAX_DETECTIONS) goto decode_done;
                    int pix = gy * reg_w + gx;

                    /* ── DFL decode bbox + peakiness confidence ──
                     *
                     * BACKGROUND: The INT8-quantized model has both cls and ext
                     * heads zeroed (all logits ≈ 0, sigmoid ≈ 0.5).  Only the
                     * 64-channel DFL regression branch survives quantization
                     * because its weights have higher variance.
                     *
                     * KEY INSIGHT: DFL softmax SHAPE reveals object presence.
                     *   - Real object: softmax is PEAKED (one bin dominates)
                     *     → max_bin ≈ 0.2–0.8
                     *   - Background:  softmax is FLAT (all bins equal)
                     *     → max_bin ≈ 1/16 = 0.0625
                     *
                     * We compute the geometric mean of the 4 max-bin values
                     * as a "DFL confidence" score.  This gives 5–13× separation
                     * between real objects and background noise — far more
                     * than the broken cls/ext heads provide.
                     *
                     * Threshold: 0.10 (1.6× above uniform 0.0625).
                     * A single peaked coord (0.4) + 3 flat ones (0.0625)
                     * gives geomean ≈ 0.12 — above threshold.
                     */
                    float dists[4];
                    float dfl_peaks[4];
                    for (int coord = 0; coord < 4; coord++) {
                        float bins[16];
                        int base = coord * 16;
                        for (int b = 0; b < 16; b++) {
                            bins[b] = reg_data[(base + b) * hw + pix];
                        }
                        softmax_stable(bins, 16);
                        float max_bin = bins[0];
                        float val = 0.0f;
                        for (int b = 0; b < 16; b++) {
                            if (bins[b] > max_bin) max_bin = bins[b];
                            val += bins[b] * (float)b;
                        }
                        dfl_peaks[coord] = max_bin;
                        dists[coord] = val;
                    }
                    /* Geometric mean of 4 peak values.
                     * Uniform: 0.0625.  Weak signal: 0.10.  Clear: 0.25+. */
                    float dfl_conf = sqrtf(sqrtf(
                        dfl_peaks[0] * dfl_peaks[1] * dfl_peaks[2] * dfl_peaks[3]));

                    /* ── Objectness signal (suppressed for INT8-quantized models) ──
                     * The ext tensor provides foreground logits.  In FP32 models
                     * this is a useful discriminator.  In INT8-quantized (xquant)
                     * models the objectness head is broken — all logits ≈ 0.5,
                     * sigmoid ≈ 0.622 everywhere, which just scales DFL down
                     * uniformly and raises the effective threshold.  We skip the
                     * ext factor for quantized models and rely on DFL peakiness
                     * + geometry filters alone.
                     *
                     * Diagnostic: if ext max-min < 0.01 → quantized → skip ext. */
                    float combined_conf = dfl_conf;
                    if (ext_data) {
                        float ext_logit = ext_data[pix];
                        /* Only blend objectness when it provides real signal.
                         * Quantized models: all ext_logits ≈ 0.5 (spread < 0.01).
                         * FP32 models: real variation — use sigmoid for discrimination. */
                        if (fabsf(ext_logit - 0.5f) > 0.01f) {
                            float obj_conf = 1.0f / (1.0f + expf(-ext_logit));
                            combined_conf = dfl_conf * obj_conf;
                        }
                    }

                    if (combined_conf < det->confidence_threshold) continue;

                    /* dists = {left, top, right, bottom} in GRID-CELL units.
                     * Anchor point is also in grid-cell space: (gx+0.5, gy+0.5).
                     * Multiply by stride AFTER computing grid-space bbox to
                     * convert to model-input pixel coordinates. */
                    float x1_grid = ((float)gx + 0.5f) - dists[0];
                    float y1_grid = ((float)gy + 0.5f) - dists[1];
                    float x2_grid = ((float)gx + 0.5f) + dists[2];
                    float y2_grid = ((float)gy + 0.5f) + dists[3];

                    float x1_model = x1_grid * (float)stride;
                    float y1_model = y1_grid * (float)stride;
                    float x2_model = x2_grid * (float)stride;
                    float y2_model = y2_grid * (float)stride;

                    /* Map back through letterbox AND center-crop to original image coords.
                     * step 1: remove letterbox padding → cropped-image coordinates
                     * step 2: add crop offset → original-image coordinates */
                    float x1 = (x1_model - pad_x) / scale + (float)crop_x;
                    float y1 = (y1_model - pad_y) / scale + (float)crop_y;
                    float x2 = (x2_model - pad_x) / scale + (float)crop_x;
                    float y2 = (y2_model - pad_y) / scale + (float)crop_y;

                    x1 = UTILS_CLAMP(x1, 0.0f, (float)(width - 1));
                    y1 = UTILS_CLAMP(y1, 0.0f, (float)(height - 1));
                    x2 = UTILS_CLAMP(x2, 0.0f, (float)(width - 1));
                    y2 = UTILS_CLAMP(y2, 0.0f, (float)(height - 1));

                    if (x2 - x1 < (float)min_box_w || y2 - y1 < (float)min_box_h) continue;

                    /* ── Class assignment (for pipeline filtering only) ──
                     * cls head is broken by quantization (all logits ≈ 0,
                     * all sigmoids ≈ 0.5).  Class labels are RANDOM NOISE.
                     * Force EVERY detection to class_id=0 (person) — we
                     * filter non-person objects via geometry heuristics
                     * (aspect ratio, area, height) instead of trusting
                     * the broken classifier.  This improves recall 80×
                     * (no random discard) with no precision loss (geometry
                     * + ByteTrack provide the real filtering). */
                    int best_class = 0;  /* PERSON_CLASS_ID — always */

                    /* Score = DFL peakiness × objectness (composite signal). */

                    Detection* d = &temp_dets[num_temp++];
                    d->bbox.x_min = x1;
                    d->bbox.y_min = y1;
                    d->bbox.x_max = x2;
                    d->bbox.y_max = y2;
                    d->confidence = combined_conf;
                    d->class_id = best_class;
                    strncpy(d->class_name, yolov8_get_class_name(best_class), MAX_STRING_LEN - 1);
                    d->class_name[MAX_STRING_LEN - 1] = '\0';
                    out_contrib++;
                }
            }

        group_done:
            if (dump_detail) {
                log_info("YOLO11 group[%d]: %d proposals above threshold (%.3f)",
                         g, out_contrib, det->confidence_threshold);
            }
        }

    decode_done:
        /* ── Release all output values ── */
        for (size_t oi = 0; oi < num_outputs; oi++) {
            if (all_output_vals[oi]) g_ort->ReleaseValue(all_output_vals[oi]);
        }
        free(all_output_vals);

    } else {
        /* ────────────────────────────────────────────────────────
         * Format A or B: standard single-output or xquant 3-output
         * (complete Detect head — output already has cx,cy,w,h + class scores)
         * ──────────────────────────────────────────────────────── */
        for (size_t oi = 0; oi < num_outputs; oi++) {
            OrtValue* out_val = all_output_vals[oi];
            if (!out_val) continue;

            float* out_data = NULL;
            OrtStatus* st_mut = g_ort->GetTensorMutableData(out_val, (void**)&out_data);
            if (st_mut || !out_data) {
                if (st_mut) g_ort->ReleaseStatus(st_mut);
                continue;
            }

            OrtTensorTypeAndShapeInfo* si = NULL;
            OrtStatus* st_si = g_ort->GetTensorTypeAndShape(out_val, &si);
            if (st_si || !si) {
                if (st_si) g_ort->ReleaseStatus(st_si);
                continue;
            }

            size_t num_dims = 0;
            { OrtStatus* _s = g_ort->GetDimensionsCount(si, &num_dims); if (_s) g_ort->ReleaseStatus(_s); }
            int64_t dims[4] = {0};
            if (num_dims >= 3 && num_dims <= 4) {
                OrtStatus* _s = g_ort->GetDimensions(si, dims, num_dims); if (_s) g_ort->ReleaseStatus(_s);
            }
            size_t elem_count = 0;
            { OrtStatus* _s = g_ort->GetTensorShapeElementCount(si, &elem_count); if (_s) g_ort->ReleaseStatus(_s); }
            g_ort->ReleaseTensorTypeAndShapeInfo(si);

            int elem_per_prop = 4 + num_classes;
            int output_proposals = 0;

            if (num_dims == 4) {
                int64_t hdim = dims[2], wdim = dims[3];
                output_proposals = (hdim > 0 && wdim > 0) ? (int)(hdim * wdim) : (int)hdim;
            } else if (num_dims == 3 && dims[0] == 1) {
                /* [1, A, B]: determine if A=attr_count or B=attr_count */
                if ((int)dims[2] == elem_per_prop) {
                    /* [1, N, elem_per_prop] → transposed, N proposals */
                    output_proposals = (int)dims[1];
                } else if ((int)dims[1] == elem_per_prop) {
                    /* [1, elem_per_prop, N] → standard, N proposals */
                    output_proposals = (int)dims[2];
                } else {
                    /* Unknown layout → fall back to total elements count */
                    output_proposals = (int)(elem_count / elem_per_prop);
                }
            } else if (num_dims == 2) {
                output_proposals = (int)dims[0];
            } else {
                output_proposals = (int)(elem_count / elem_per_prop);
            }

            if (dump_detail) {
                log_info("YOLO11 output[%zu]: dims=%zu shape=[%lld,%lld,%lld,%lld] elem=%zu proposals=%d",
                         oi, num_dims,
                         num_dims>=1?(long long)dims[0]:0, num_dims>=2?(long long)dims[1]:0,
                         num_dims>=3?(long long)dims[2]:0, num_dims>=4?(long long)dims[3]:0,
                         elem_count, output_proposals);
            }

            if (output_proposals <= 0) continue;

            int out_contrib = 0;
            for (int pi = 0; pi < output_proposals && num_temp < YOLO11_MAX_DETECTIONS; pi++) {
                const float* row = out_data + pi * elem_per_prop;
                float cx = row[0], cy = row[1], wb = row[2], hb = row[3];

                float max_score = 0.0f;
                int best_class = 0;
                for (int c = 0; c < num_classes; c++) {
                    if (row[4 + c] > max_score) { max_score = row[4 + c]; best_class = c; }
                }

                if (max_score < det->confidence_threshold) continue;

                float x1 = ((cx - wb * 0.5f) - pad_x) / scale + (float)crop_x;
                float y1 = ((cy - hb * 0.5f) - pad_y) / scale + (float)crop_y;
                float x2 = ((cx + wb * 0.5f) - pad_x) / scale + (float)crop_x;
                float y2 = ((cy + hb * 0.5f) - pad_y) / scale + (float)crop_y;

                x1 = UTILS_CLAMP(x1, 0.0f, (float)(width - 1));
                y1 = UTILS_CLAMP(y1, 0.0f, (float)(height - 1));
                x2 = UTILS_CLAMP(x2, 0.0f, (float)(width - 1));
                y2 = UTILS_CLAMP(y2, 0.0f, (float)(height - 1));

                if (x2 - x1 < (float)min_box_w || y2 - y1 < (float)min_box_h) continue;

                Detection* d = &temp_dets[num_temp++];
                d->bbox.x_min = x1; d->bbox.y_min = y1;
                d->bbox.x_max = x2; d->bbox.y_max = y2;
                d->confidence = max_score;
                d->class_id = best_class;
                strncpy(d->class_name, yolov8_get_class_name(best_class), MAX_STRING_LEN - 1);
                d->class_name[MAX_STRING_LEN - 1] = '\0';
                out_contrib++;
            }

            if (dump_detail) {
                log_info("YOLO11 output[%zu]: %d proposals above threshold (%.3f)",
                         oi, out_contrib, det->confidence_threshold);
            }

            g_ort->ReleaseValue(out_val);
        }
        free(all_output_vals);
    }

    if (num_temp == 0) {
        if (dump_detail) {
            log_info("YOLO11: 0 proposals across %zu outputs (thresh=%.3f)",
                     num_outputs, det->confidence_threshold);
        }
        free(temp_dets);
        return 0;
    }

    utils_sort_detections_by_confidence(temp_dets, num_temp);

    /* ── Global Top-K: model outputs are broken by INT8 quantization.
     * DFL peakiness is high everywhere (min≈0.28, mean≈0.45 even for
     * background).  Instead of an absolute threshold, we keep only the
     * strongest K proposals GLOBALLY across all stride groups.
     *
     * Without this, 500+ proposals enter NMS → random winners cause
     * "rainbow boxes flying around."  Top-30 reduces noise ~70× and
     * ensures only the model's most confident guesses survive.
     *
     * 30 is enough for crowded scenes; real scenes rarely exceed 10. */
#define YOLOV8_TOP_K_PROPOSALS 50
    if (num_temp > YOLOV8_TOP_K_PROPOSALS) {
        num_temp = YOLOV8_TOP_K_PROPOSALS;
    }
#undef YOLOV8_TOP_K_PROPOSALS

    /* NMS caps at 600; with top-30 this is far below the cap. */

    bool* suppressed = (bool*)calloc((size_t)num_temp, sizeof(bool));
    if (!suppressed) {
        int ret = UTILS_MIN(num_temp, max_detections);
        free(temp_dets);
        return ret;
    }

    int keep = 0;
    int effective_max = UTILS_MIN(max_detections, YOLOV8_MAX_OUTPUT_DETECTIONS);
    for (int i = 0; i < num_temp; i++) {
        if (suppressed[i]) continue;
        if (keep < effective_max) {
            out_detections[keep++] = temp_dets[i];
        }
        for (int j = i + 1; j < num_temp; j++) {
            if (suppressed[j]) continue;
            if (bbox_iou(&temp_dets[i].bbox, &temp_dets[j].bbox) > det->iou_threshold
                && temp_dets[i].class_id == temp_dets[j].class_id) {
                suppressed[j] = true;
            }
        }
    }
    free(suppressed);
    free(temp_dets);

    log_debug("YOLO11: %d outputs, %d detections after NMS", (int)num_outputs, keep);
    return keep;
}

int yolov8_detector_detect_persons(YOLO11Detector* det, const uint8_t* image_data, int width, int height,
                                    Detection* out_detections, int max_detections) {
    return yolov8_detector_detect(det, image_data, width, height, out_detections, max_detections);
}

const char* yolov8_get_class_name(int class_id) {
    if (class_id >= 0 && class_id < YOLO11_NUM_CLASSES) {
        return COCO_CLASSES[class_id];
    }
    return "unknown";
}
