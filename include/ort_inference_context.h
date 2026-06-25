#ifndef ORT_INFERENCE_CONTEXT_H
#define ORT_INFERENCE_CONTEXT_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>

typedef struct OrtInferenceContext {
    OrtSession* session;
    float* input_tensor;
    OrtMemoryInfo* memory_info;
    OrtAllocator* allocator;
    int input_width;
    int input_height;
    int input_channels;
    size_t input_tensor_bytes;
    char input_name[64];
    size_t num_outputs;
    char** output_names;
    bool names_cached;
    bool format_cached;
    int num_groups;
    int group_indices[3][3];

    /* ── Preprocessing buffer pool ──
     * Pre-allocated to avoid per-frame malloc/free in the hot path.
     * padded_buf holds the letterbox-resized RGB image (HWC layout).
     * crop_buf holds the cropped image when aspect ratio differs significantly. */
    uint8_t* preproc_padded_buf;
    size_t   preproc_padded_buf_size;
    uint8_t* preproc_crop_buf;
    size_t   preproc_crop_buf_size;

    /* ── Output value array pool ──
     * Pre-allocated OrtValue* array of size num_outputs.  Callers reuse
     * this instead of calloc(size, sizeof(OrtValue*)) per frame.
     * Zeroed before each Run() call via ort_ctx_reset_outputs(). */
    OrtValue** output_val_pool;
    size_t    output_val_pool_size;
} OrtInferenceContext;

OrtInferenceContext* ort_ctx_create(OrtSession* session, int input_w, int input_h, int input_c);
void ort_ctx_destroy(OrtInferenceContext* ctx);
bool ort_ctx_prepare_input(OrtInferenceContext* ctx, const float* data, size_t bytes);
/*
 * Mark input tensor as ready — the caller has already written preprocessed
 * data directly into ctx->input_tensor (e.g. via yolo_preprocess_pooled).
 * Skips the memcpy that ort_ctx_prepare_input would do.
 */
bool ort_ctx_input_ready(OrtInferenceContext* ctx, size_t bytes);
int ort_ctx_run(OrtInferenceContext* ctx, OrtValue** output_vals);
void ort_ctx_release_outputs(OrtInferenceContext* ctx, OrtValue** output_vals, size_t count);
/* ── Output value pool ──
 * Returns a pre-allocated OrtValue* array (size ctx->num_outputs, zeroed).
 * Callers pass this to ort_ctx_run() directly — no per-frame calloc needed.
 * After processing outputs, call ort_ctx_release_outputs() as usual, then
 * call ort_ctx_reset_outputs() to zero the pool for the next frame. */
OrtValue** ort_ctx_get_output_pool(OrtInferenceContext* ctx);
void ort_ctx_reset_outputs(OrtInferenceContext* ctx);
int ort_ctx_get_output_shape(OrtInferenceContext* ctx, OrtValue* val, int64_t* dims, int max_dims);
float* ort_ctx_get_output_data(OrtInferenceContext* ctx, OrtValue* val);

#endif

#ifdef __cplusplus
}
#endif

#endif