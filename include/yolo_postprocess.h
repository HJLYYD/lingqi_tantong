#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include "ort_inference_context.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void yolo_softmax_stable(float* restrict x, int n);

float yolo_dfl_decode_position(const float* restrict reg_data, int pix, int hw, float dists_out[4]);

void yolo_preprocess(const uint8_t* image_data, int width, int height,
                     float* out_tensor, int target_w, int target_h,
                     float* out_scale, int* out_pad_x, int* out_pad_y,
                     int* out_crop_x, int* out_crop_y);

/*
 * Pooled preprocess — writes directly into ctx->input_tensor (float* CHW),
 * using ctx->preproc_padded_buf as the intermediate letterbox buffer.
 * Zero malloc/free in the hot path.  Falls back to heap if ctx buffers
 * are NULL (backward compatible with bare OrtInferenceContext).
 *
 * Returns 0 on success, -1 on failure.
 */
int yolo_preprocess_pooled(OrtInferenceContext* ctx,
                           const uint8_t* image_data, int width, int height,
                           int target_w, int target_h,
                           float* out_scale, int* out_pad_x, int* out_pad_y,
                           int* out_crop_x, int* out_crop_y);

void yolo_map_to_original(float mx, float my, float scale, int pad_x, int pad_y,
                          int crop_x, int crop_y, float* ox, float* oy);

#ifdef HAS_ONNX_RUNTIME
int yolo_detect_xquant_split(size_t num_outputs, OrtValue** output_vals,
                             int group_indices[3][3], int mode);
#endif

typedef float (*yolo_similarity_fn)(const void* a, const void* b);

int yolo_nms_suppress(void* items, int num_items, size_t item_size,
                      float conf_threshold, float iou_threshold,
                      int max_output, yolo_similarity_fn sim_fn, int conf_offset);

#ifdef __cplusplus
}
#endif

#endif