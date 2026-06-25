#include "yolov5_face_detector.h"
#include "logger.h"
#include "utils.h"
#include "yolo_postprocess.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include "ort_common.h"
#else
#error "yolov5_face_detector requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

#define YOLOV5_FACE_MAX_FACES       50
#define YOLOV5_FACE_MAX_INPUT_BYTES (100UL * 1024UL * 1024UL)
#define YOLOV5_FACE_MAX_PROPOSALS   3000  /* per-stride temp buffer; 3 anchors × 80×80=19200 max, filtered down */

/*
 * ── YOLOv5-Face ONNX Output Format Auto-Detection ──
 *
 * Two formats are possible depending on how the model was exported:
 *
 * Format A — Standard 4D (ultralytics export / non-xquant):
 *   Output: [1, 16, Hi, Wi] × 3 strides  (NCHW layout)
 *   16 = 4(bbox) + 1(obj) + 1(cls) + 10(5 landmarks × 2 coords)
 *
 * Format B — xquant-split 5D (SpacemiT INT8 quantization):
 *   Output: [1, 3, Hi, Wi, 16] × 3 strides  (anchors, NHWC features-last)
 *   Anchors=3 (YOLOv5 uses 3 anchors per stride level)
 *   Features=16: same as Format A, but raw logits (xquant removes Detect head)
 *
 * CRITICAL: xquant truncates at /model.22/Reshape_*, removing the final
 * Detect layer. All outputs are RAW LOGITS — we must apply sigmoid + anchor
 * decode manually.  Standard export has the Detect layer baked in (sigmoid
 * already applied, anchor decode already done → cx,cy,w,h in grid units).
 *
 * The strides are derived from spatial dims:
 *   H=80  → stride = 640/80 = 8
 *   H=40  → stride = 640/40 = 16
 *   H=20  → stride = 640/20 = 32
 */

/*
 * Standard YOLOv5-Face anchors (from ultralytics YOLOv5-face training config).
 * These are pre-training constants embedded in the Detect layer for the
 * xquant variant (5D format). For standard 4D exports, the Detect layer
 * has already applied them — no need to re-apply.
 */
static const float YOLOV5_FACE_ANCHORS[3][3][2] = {
    /* stride=8  (P3 — small faces) */
    {{4.0f, 5.0f}, {8.0f, 10.0f}, {13.0f, 16.0f}},
    /* stride=16 (P4 — medium faces) */
    {{23.0f, 29.0f}, {43.0f, 55.0f}, {73.0f, 105.0f}},
    /* stride=32 (P5 — large faces) */
    {{146.0f, 217.0f}, {231.0f, 300.0f}, {335.0f, 433.0f}}
};

/* 5D format: 3 anchors × (4 bbox + 1 obj + 1 cls + 10 kpts) = 3 × 16  */
/* 4D format: [1, 16, H, W] — features are in dim[1]                   */
#define FACE_FEAT_PER_ANCHOR    16
#define FACE_ANCHORS_PER_STRIDE 3

typedef struct {
    BoundingBox bbox;
    float confidence;
    float keypoints[YOLOV5_FACE_NUM_KEYPOINTS][2];
} RawFaceDetection;

YOLOv5FaceDetector* yolov5_face_detector_create(const char* model_path, int input_w, int input_h,
                                                  float conf_thresh, float nms_thresh) {
    YOLOv5FaceDetector* det = (YOLOv5FaceDetector*)calloc(1, sizeof(YOLOv5FaceDetector));
    if (!det) return NULL;

    det->input_width = input_w > 0 ? input_w : 320;
    det->input_height = input_h > 0 ? input_h : 320;
    det->confidence_threshold = conf_thresh;
    det->nms_threshold = nms_thresh;
    det->session = NULL;

    if (!model_path || !yolov5_face_detector_load_model(det, model_path)) {
        log_error("YOLOv5FaceDetector: failed to load model %s", model_path ? model_path : "(null)");
        free(det);
        return NULL;
    }

    return det;
}

void yolov5_face_detector_destroy(YOLOv5FaceDetector* det) {
    if (!det) return;
    ort_ctx_destroy(det->ctx);
    const OrtApi* ort = ort_get_api();
    if (det->session && ort) {
        ort->ReleaseSession(det->session);
    }
    free(det);
}

bool yolov5_face_detector_load_model(YOLOv5FaceDetector* det, const char* model_path) {
    if (!det || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    strncpy(det->input_name, "images", sizeof(det->input_name) - 1);

    if (!ort_global_init()) {
        log_error("YOLOv5Face: ORT runtime not initialized");
        return false;
    }

    det->session = ort_create_session(model_path, 4, true);
    if (!det->session) {
        log_error("YOLOv5Face: ONNX session creation failed for %s", model_path);
        return false;
    }

    int dims[8] = {0};
    int rank = ort_get_input_shape(det->session, dims, 8);
    if (rank == 4 && dims[2] > 0 && dims[3] > 0) {
        if (det->input_width != dims[3] || det->input_height != dims[2]) {
            log_info("YOLOv5Face: overriding requested %dx%d with model's actual input %dx%d",
                     det->input_width, det->input_height, dims[3], dims[2]);
        }
        det->input_width = dims[3];
        det->input_height = dims[2];
    } else if (rank > 0) {
        log_warning("YOLOv5Face: unexpected input rank=%d, keeping configured %dx%d",
                    rank, det->input_width, det->input_height);
    }

    det->ctx = ort_ctx_create(det->session, det->input_width, det->input_height, 3);
    if (!det->ctx) {
        log_error("YOLOv5Face: failed to create inference context");
        const OrtApi* ort = ort_get_api();
        if (det->session && ort) ort->ReleaseSession(det->session);
        det->session = NULL;
        return false;
    }
    strncpy(det->ctx->input_name, det->input_name, sizeof(det->ctx->input_name) - 1);

    log_info("YOLOv5-face model loaded: %s (%.2f MB) input=%dx%d",
             model_path, file_size / (1024.0f * 1024.0f), det->input_width, det->input_height);
    return true;
}

/*
 * YOLOv5-style bbox + landmark decode for xquant-split (5D, raw logits).
 *
 * Standard YOLOv5 decode (Detect layer equivalent):
 *   cx = (sigmoid(tx) * 2.0 - 0.5 + grid_x) * stride
 *   cy = (sigmoid(ty) * 2.0 - 0.5 + grid_y) * stride
 *   w  = (sigmoid(tw) * 2.0)^2 * anchor_w * stride
 *   h  = (sigmoid(th) * 2.0)^2 * anchor_h * stride
 *
 * Landmark decode (same offset formula):
 *   kp_x = (sigmoid(lm_x) * 2.0 - 0.5 + grid_x) * stride
 *   kp_y = (sigmoid(lm_y) * 2.0 - 0.5 + grid_y) * stride
 */
/*
 * Decode a single 5D anchor proposal into model-input pixel coordinates.
 * Returns the decoded confidence (sigmoid(obj) * sigmoid(cls)).
 */
static float decode_5d_proposal(const float* feat,   /* [16] raw logits */
                                 int grid_x, int grid_y,
                                 int stride_idx,      /* 0..2 → P3/P4/P5 */
                                 int anchor_idx,      /* 0..2 */
                                 BoundingBox* out_bbox,
                                 float out_kpts[YOLOV5_FACE_NUM_KEYPOINTS][2]) {
    /* ── Objectness and classification ──
     * deepcam-cn/yolov5-face feature order (per official repo):
     *   [0..3]=bbox(tx,ty,tw,th)  [4]=obj_conf  [5..14]=landmarks(5×2)  [15]=cls_conf */
    float obj = feat[4];  /* raw logit */
    float obj_conf = utils_sigmoid(obj);

    float cls = feat[15];  /* raw logit — CLASS IS AT INDEX 15, NOT 5! */
    float cls_conf = utils_sigmoid(cls);

    float confidence = obj_conf * cls_conf;

    /* ── Bounding box decode ── */
    float tx = feat[0], ty = feat[1], tw = feat[2], th = feat[3];
    float sx = utils_sigmoid(tx);
    float sy = utils_sigmoid(ty);
    float sw = utils_sigmoid(tw);
    float sh = utils_sigmoid(th);

    int stride = (stride_idx == 0) ? 8 : ((stride_idx == 1) ? 16 : 32);
    const float* anchor = YOLOV5_FACE_ANCHORS[stride_idx][anchor_idx];
    float anchor_w = anchor[0];
    float anchor_h = anchor[1];

    float cx_model = (sx * 2.0f - 0.5f + (float)grid_x) * (float)stride;
    float cy_model = (sy * 2.0f - 0.5f + (float)grid_y) * (float)stride;
    float w_model  = (sw * 2.0f) * (sw * 2.0f) * anchor_w * (float)stride;
    float h_model  = (sh * 2.0f) * (sh * 2.0f) * anchor_h * (float)stride;

    out_bbox->x_min = cx_model - w_model * 0.5f;
    out_bbox->y_min = cy_model - h_model * 0.5f;
    out_bbox->x_max = cx_model + w_model * 0.5f;
    out_bbox->y_max = cy_model + h_model * 0.5f;

    /* ── Landmark decode (5 keypoints × 2 coords, at feat[5..14]) ──
     * Official formula (deepcam-cn/yolov5-face, community consensus):
     *   lm_x = raw_pred_x * anchor_w + grid_x * stride
     *   lm_y = raw_pred_y * anchor_h + grid_y * stride
     * Landmarks use LINEAR regression (no sigmoid) with anchor_w/h as the
     * scale reference — this makes predictions scale-invariant to face size.
     * See: docs/INFERENCE_OPTIMIZATION_ANALYSIS.md §1.1 */
    for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
        float lx = feat[5 + k * 2 + 0];  /* raw logit — NO sigmoid */
        float ly = feat[5 + k * 2 + 1];

        out_kpts[k][0] = lx * anchor_w + (float)grid_x * (float)stride;
        out_kpts[k][1] = ly * anchor_h + (float)grid_y * (float)stride;
    }

    return confidence;
}

/*
 * Decode 5D format output: [1, anchors, H, W, features] × 3 strides.
 *
 * This is the xquant-split output where the Detect head is removed.
 * Bbox + obj/cls: raw logits → must apply sigmoid + anchor decode.
 * Landmarks: raw logits → linear regression (NO sigmoid).
 *
 * 5D tensor layout (ONNX row-major):
 *   data[a * H * W * F + h * W * F + w * F + feat_idx]
 */
static int decode_5d_format(const uint8_t* image_data, int width, int height,
                             OrtValue** output_vals, size_t num_outputs,
                             const OrtApi* ort, float scale, int pad_x, int pad_y,
                             int crop_x, int crop_y, float conf_thresh,
                             RawFaceDetection* raw_faces, int max_faces) {
    int num_raw = 0;
    int model_h = -1;  /* derived from first output */

    (void)image_data;  /* kept for API consistency with standard decode path */

    for (size_t oi = 0; oi < num_outputs; oi++) {
        OrtValue* out_val = output_vals[oi];
        if (!out_val) continue;

        OrtTensorTypeAndShapeInfo* si = NULL;
        if (ort->GetTensorTypeAndShape(out_val, &si)) continue;

        size_t num_dims = 0;
        { OrtStatus* _s = ort->GetDimensionsCount(si, &num_dims); if (_s) ort->ReleaseStatus(_s); }
        int64_t dims[5] = {0};
        { OrtStatus* _s = ort->GetDimensions(si, dims, num_dims); if (_s) ort->ReleaseStatus(_s); }
        ort->ReleaseTensorTypeAndShapeInfo(si);

        /* Only handle 5D format */
        if (num_dims != 5) continue;
        if (dims[0] != 1) continue;

        int num_anchors = (int)dims[1];
        int feat_h = (int)dims[2];
        int feat_w = (int)dims[3];
        int feat_c = (int)dims[4];

        /* Validate: must be 3 anchors × 16 features */
        if (num_anchors != FACE_ANCHORS_PER_STRIDE) {
            log_warning("YOLOv5Face: 5D output[%zu] has %d anchors (expected %d), skipping",
                        oi, num_anchors, FACE_ANCHORS_PER_STRIDE);
            continue;
        }
        if (feat_c < FACE_FEAT_PER_ANCHOR) {
            log_warning("YOLOv5Face: 5D output[%zu] has %d features (expected %d), skipping",
                        oi, feat_c, FACE_FEAT_PER_ANCHOR);
            continue;
        }

        /* Determine stride from spatial dims */
        if (model_h < 0) model_h = feat_h;  /* use first output's H as reference */

        /* Model input_size / spatial_H = stride.
         * Input is square (e.g. 640×640), so H division gives stride.
         * For non-square: stride = scale_factor between input and feature map.
         * Typically: H=80→stride=8, H=40→stride=16, H=20→stride=32 */
        int stride_idx;
        if (feat_h >= 70) stride_idx = 0;      /* stride=8 */
        else if (feat_h >= 35) stride_idx = 1; /* stride=16 */
        else stride_idx = 2;                    /* stride=32 */

        float* out_data = NULL;
        OrtStatus* st = ort->GetTensorMutableData(out_val, (void**)&out_data);
        if (st || !out_data) { if (st) ort->ReleaseStatus(st); continue; }

        size_t H = (size_t)feat_h;
        size_t W = (size_t)feat_w;
        size_t F = (size_t)feat_c;
        size_t stride_HWF = H * W * F;

        static int frame_5d_log = 0;
        if (frame_5d_log++ == 0) {
            log_info("YOLOv5Face: 5D format detected! output[%zu]: [1,%d,%d,%d,%d] stride_idx=%d",
                     oi, num_anchors, feat_h, feat_w, feat_c, stride_idx);
        }

        for (int a = 0; a < num_anchors; a++) {
            for (int gy = 0; gy < feat_h; gy++) {
                for (int gx = 0; gx < feat_w; gx++) {
                    if (num_raw >= max_faces) goto decode_5d_done;

                    const float* feat = out_data + (size_t)a * stride_HWF
                                                + (size_t)gy * W * F
                                                + (size_t)gx * F;

                    BoundingBox model_bbox;
                    float out_kpts[YOLOV5_FACE_NUM_KEYPOINTS][2];
                    float confidence = decode_5d_proposal(feat, gx, gy, stride_idx, a,
                                                          &model_bbox, out_kpts);

                    if (confidence < conf_thresh) continue;

                    /* Clip to model input range */
                    int model_size = (stride_idx == 0) ? feat_h * 8 :
                                     (stride_idx == 1) ? feat_h * 16 : feat_h * 32;

                    model_bbox.x_min = UTILS_CLAMP(model_bbox.x_min, 0.0f, (float)(model_size - 1));
                    model_bbox.y_min = UTILS_CLAMP(model_bbox.y_min, 0.0f, (float)(model_size - 1));
                    model_bbox.x_max = UTILS_CLAMP(model_bbox.x_max, 0.0f, (float)(model_size - 1));
                    model_bbox.y_max = UTILS_CLAMP(model_bbox.y_max, 0.0f, (float)(model_size - 1));

                    /* Map back through letterbox + crop to original image coords */
                    float x1, y1, x2, y2;
                    yolo_map_to_original(model_bbox.x_min, model_bbox.y_min, scale, pad_x, pad_y, crop_x, crop_y, &x1, &y1);
                    yolo_map_to_original(model_bbox.x_max, model_bbox.y_max, scale, pad_x, pad_y, crop_x, crop_y, &x2, &y2);

                    x1 = UTILS_CLAMP(x1, 0.0f, (float)(width - 1));
                    y1 = UTILS_CLAMP(y1, 0.0f, (float)(height - 1));
                    x2 = UTILS_CLAMP(x2, 0.0f, (float)(width - 1));
                    y2 = UTILS_CLAMP(y2, 0.0f, (float)(height - 1));

                    if (x2 - x1 < 8.0f || y2 - y1 < 8.0f) continue;

                    RawFaceDetection* face = &raw_faces[num_raw++];
                    face->bbox.x_min = x1;
                    face->bbox.y_min = y1;
                    face->bbox.x_max = x2;
                    face->bbox.y_max = y2;
                    face->confidence = confidence;

                    for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
                        float kx = UTILS_CLAMP(out_kpts[k][0], 0.0f, (float)(model_size - 1));
                        float ky = UTILS_CLAMP(out_kpts[k][1], 0.0f, (float)(model_size - 1));
                        kx = (kx - pad_x) / scale + (float)crop_x;
                        ky = (ky - pad_y) / scale + (float)crop_y;
                        face->keypoints[k][0] = UTILS_CLAMP(kx, 0.0f, (float)(width - 1));
                        face->keypoints[k][1] = UTILS_CLAMP(ky, 0.0f, (float)(height - 1));
                    }
                }
            }
        }
    }
decode_5d_done:
    return num_raw;
}

int yolov5_face_detector_detect_faces(YOLOv5FaceDetector* det, const uint8_t* image_data, int width, int height,
                                       FaceIdentity* out_faces, int max_faces) {
    if (!det || !image_data || !out_faces || !det->session) return 0;
    if (width <= 0 || height <= 0 || max_faces <= 0) return 0;

    const OrtApi* ort = ort_get_api();
    if (!ort) return 0;

    int input_w = det->input_width > 0 ? det->input_width : 320;
    int input_h = det->input_height > 0 ? det->input_height : 320;
    size_t pixels = (size_t)input_w * (size_t)input_h;
    size_t input_size = pixels * 3 * sizeof(float);
    if (input_size == 0 || input_size > YOLOV5_FACE_MAX_INPUT_BYTES) {
        log_error("YOLOv5Face: refused unreasonable input tensor size %zu bytes", input_size);
        return 0;
    }

    float* input_tensor = (float*)malloc(input_size);
    if (!input_tensor) return 0;

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    int crop_x = 0, crop_y = 0;
    yolo_preprocess(image_data, width, height, input_tensor,
                            input_w, input_h, &scale, &pad_x, &pad_y, &crop_x, &crop_y);

    if (!ort_ctx_prepare_input(det->ctx, input_tensor, input_size)) {
        free(input_tensor);
        return 0;
    }
    free(input_tensor);

    size_t num_outputs = 0;
    OrtStatus* st_oc = ort->SessionGetOutputCount(det->session, &num_outputs);
    if (st_oc) { ort->ReleaseStatus(st_oc); num_outputs = 1; }
    if (num_outputs == 0) num_outputs = 1;

    OrtValue** output_vals = (OrtValue**)calloc(num_outputs, sizeof(OrtValue*));
    if (!output_vals) return 0;

    int run_rc = ort_ctx_run(det->ctx, output_vals);
    if (run_rc != 0) {
        log_error("YOLOv5Face inference failed");
        free(output_vals);
        return 0;
    }

    /* ── Auto-detect output format and decode ──
     *
     * Strategy: probe the first output's rank ONCE, then cache.
     *   num_dims == 5 → xquant-split 5D format (raw logits, anchor decode)
     *   num_dims == 4 → standard 4D format (Detect layer baked in)
     */

    /* Probe & cache output format on first frame */
    RawFaceDetection raw_faces[YOLOV5_FACE_MAX_FACES];
    int num_raw = 0;

    if (det->output_format_cached == 0) {
        if (num_outputs > 0 && output_vals[0]) {
            OrtTensorTypeAndShapeInfo* si = NULL;
            if (ort->GetTensorTypeAndShape(output_vals[0], &si) == NULL) {
                size_t nd = 0;
                { OrtStatus* _s = ort->GetDimensionsCount(si, &nd); if (_s) ort->ReleaseStatus(_s); }
                if (nd == 5) {
                    det->output_format_cached = 2;
                    log_info("YOLOv5Face: detected 5D output format (%zu outputs), using xquant anchor decode", num_outputs);
                } else {
                    det->output_format_cached = 1;
                    log_info("YOLOv5Face: detected 4D output format (%zu outputs), using standard decode", num_outputs);
                }
                ort->ReleaseTensorTypeAndShapeInfo(si);
            }
        }
        if (det->output_format_cached == 0) {
            det->output_format_cached = 1; /* default to 4D on probe failure */
        }
    }
    bool use_5d_decode = (det->output_format_cached == 2);

    if (use_5d_decode) {
        /* ── xquant-split 5D format path ── */
        num_raw = decode_5d_format(image_data, width, height,
                                    output_vals, num_outputs, ort,
                                    scale, pad_x, pad_y, crop_x, crop_y,
                                    det->confidence_threshold,
                                    raw_faces, YOLOV5_FACE_MAX_FACES);
    } else {
        /* ── Standard 4D format path ── */
        for (size_t oi = 0; oi < num_outputs; oi++) {
            OrtValue* out_val = output_vals[oi];
            if (!out_val) continue;

            float* out_data = NULL;
            OrtStatus* st_mut = ort->GetTensorMutableData(out_val, (void**)&out_data);
            if (st_mut || !out_data) { if (st_mut) ort->ReleaseStatus(st_mut); continue; }

            OrtTensorTypeAndShapeInfo* si = NULL;
            OrtStatus* st_si = ort->GetTensorTypeAndShape(out_val, &si);
            if (st_si || !si) { if (st_si) ort->ReleaseStatus(st_si); continue; }

            size_t num_dims_4d = 0;
            { OrtStatus* _s = ort->GetDimensionsCount(si, &num_dims_4d); if (_s) ort->ReleaseStatus(_s); }
            int64_t fdims[4] = {0};
            { OrtStatus* _s = ort->GetDimensions(si, fdims, num_dims_4d); if (_s) ort->ReleaseStatus(_s); }

            size_t elem_count = 0;
            OrtStatus* st_ec = ort->GetTensorShapeElementCount(si, &elem_count);
            if (st_ec) ort->ReleaseStatus(st_ec);
            ort->ReleaseTensorTypeAndShapeInfo(si);

            int nch = (num_dims_4d >= 2) ? (int)fdims[1] : 0;
            /* Skip outputs with too few channels (e.g. xquant partial: reg=64, cls=1, kpt=10) */
            if (nch < 16) continue;

            const int FACE_ELEM_PER_PROP = 16;
            int num_proposals = (int)(elem_count / FACE_ELEM_PER_PROP);
            if (num_proposals <= 0) continue;

            for (int pi = 0; pi < num_proposals && num_raw < YOLOV5_FACE_MAX_FACES; pi++) {
                const float* row = out_data + pi * FACE_ELEM_PER_PROP;

                /* Standard format (deepcam-cn official order):
                 * [0..3]=bbox(cx,cy,w,h) [4]=obj [5..14]=landmarks(5×2) [15]=cls
                 * NOTE: standard ONNX export has Detect layer baked in —
                 * values are already sigmoid'd. */
                float cx = row[0], cy = row[1];
                float wb = row[2], hb = row[3];
                float obj_conf = row[4];  /* already sigmoid'd */
                float cls_conf = row[15]; /* CLASS IS AT INDEX 15, NOT 5! */
                float conf_val = obj_conf * cls_conf;

                if (conf_val < det->confidence_threshold) continue;

                float x1, y1, x2, y2;
                yolo_map_to_original(cx - wb * 0.5f, cy - hb * 0.5f, scale, pad_x, pad_y, crop_x, crop_y, &x1, &y1);
                yolo_map_to_original(cx + wb * 0.5f, cy + hb * 0.5f, scale, pad_x, pad_y, crop_x, crop_y, &x2, &y2);

                x1 = UTILS_CLAMP(x1, 0.0f, (float)width - 1);
                y1 = UTILS_CLAMP(y1, 0.0f, (float)height - 1);
                x2 = UTILS_CLAMP(x2, 0.0f, (float)width - 1);
                y2 = UTILS_CLAMP(y2, 0.0f, (float)height - 1);

                if (x2 - x1 < 8.0f || y2 - y1 < 8.0f) continue;

                RawFaceDetection* face = &raw_faces[num_raw++];
                face->bbox.x_min = x1;
                face->bbox.y_min = y1;
                face->bbox.x_max = x2;
                face->bbox.y_max = y2;
                face->confidence = conf_val;

                for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
                    float kx = (row[5 + k * 2 + 0] - pad_x) / scale + (float)crop_x;  /* landmarks start at 5 */
                    float ky = (row[5 + k * 2 + 1] - pad_y) / scale + (float)crop_y;
                    face->keypoints[k][0] = UTILS_CLAMP(kx, 0.0f, (float)width - 1);
                    face->keypoints[k][1] = UTILS_CLAMP(ky, 0.0f, (float)height - 1);
                }
            }
        }
    }

    /* Release all remaining output values */
    ort_ctx_release_outputs(det->ctx, output_vals, num_outputs);
    free(output_vals);

    if (num_raw <= 0) return 0;

    num_raw = yolo_nms_suppress(raw_faces, num_raw, sizeof(RawFaceDetection),
                             0.0f, det->nms_threshold, YOLOV5_FACE_MAX_FACES,
                             NULL, (int)sizeof(BoundingBox));
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
        for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
            face->keypoints[k][0] = raw_faces[i].keypoints[k][0];
            face->keypoints[k][1] = raw_faces[i].keypoints[k][1];
        }
    }

    log_debug("YOLOv5Face: %zu outputs, %d faces after NMS (format=%s)",
              num_outputs, num_faces, use_5d_decode ? "5D" : "4D");
    return num_faces;
}

int yolov5_face_detector_crop_face(const YOLOv5FaceDetector* det, const uint8_t* image_data, int img_w, int img_h,
                                    const FaceIdentity* face,
                                    uint8_t* out_crop, int target_w, int target_h) {
    (void)det;
    if (!image_data || !face || !out_crop) return -1;

    int x1 = UTILS_MAX(0, (int)face->bbox.x_min);
    int y1 = UTILS_MAX(0, (int)face->bbox.y_min);
    int x2 = UTILS_MIN(img_w, (int)face->bbox.x_max);
    int y2 = UTILS_MIN(img_h, (int)face->bbox.y_max);

    /* Add margin: expand crop by 15% for ArcFace preprocessing */
    int crop_w = x2 - x1;
    int crop_h = y2 - y1;
    int margin_w = crop_w * 15 / 100;
    int margin_h = crop_h * 15 / 100;
    x1 = UTILS_MAX(0, x1 - margin_w);
    y1 = UTILS_MAX(0, y1 - margin_h);
    x2 = UTILS_MIN(img_w, x2 + margin_w);
    y2 = UTILS_MIN(img_h, y2 + margin_h);
    crop_w = x2 - x1;
    crop_h = y2 - y1;

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
