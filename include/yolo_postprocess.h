/*
 * yolo_postprocess.h — YOLO preprocessing + DFL decode + NMS for K1 edge
 */

#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include "ort_inference_context.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── NEW: Single-pass preprocess (zero intermediate buffer) ── */
void yolo_preprocess_sp(const uint8_t* src, int sw, int sh,
                        float* dst, int dw, int dh,
                        float* scale, int* padx, int* pady,
                        int* cropx, int* cropy);

/* ── Precomputed LUT for fixed input sizes ── */
typedef struct {
    int      sw, sh, dw, dh;
    float    scale;
    int      padx, pady, cropx, cropy;
    int16_t* mx;
    int16_t* my;
} YppLUT;

YppLUT* ypp_lut_build(int sw, int sh, int dw, int dh);
void    ypp_lut_free(YppLUT* lut);
void    ypp_lut_apply(const YppLUT* lut, const uint8_t* src, float* dst);

/* ── Legacy API (backward-compatible, kept for existing callers) ── */
void yolo_preprocess(const uint8_t* image_data, int width, int height,
                     float* out_tensor, int target_w, int target_h,
                     float* out_scale, int* out_pad_x, int* out_pad_y,
                     int* out_crop_x, int* out_crop_y);

int yolo_preprocess_pooled(OrtInferenceContext* ctx,
                           const uint8_t* image_data, int width, int height,
                           int target_w, int target_h,
                           float* out_scale, int* out_pad_x, int* out_pad_y,
                           int* out_crop_x, int* out_crop_y);

/* ── Stable softmax (in-place) ── */
void yolo_softmax_stable(float* restrict x, int n);

/* ── DFL decode ── */
float yolo_dfl_decode_position(const float* restrict reg, int pix, int hw, float d[4]);

/* ── Coordinate mapping ── */
void yolo_map_to_original(float mx, float my, float scale, int padx, int pady,
                          int cropx, int cropy, float* ox, float* oy);

/* ── NMS ── */
typedef float (*yolo_similarity_fn)(const void* a, const void* b);

int yolo_nms_suppress(void* items, int n, size_t item_size,
                      float conf_thresh, float iou_thresh, int max_out,
                      yolo_similarity_fn sim, int conf_offs);

/* ── xquant-split format detection ── */
#ifdef HAS_ONNX_RUNTIME
int yolo_detect_xquant_split(size_t nout, OrtValue** outs,
                              int groups[3][3], int mode);
#endif

#ifdef __cplusplus
}
#endif
#endif
