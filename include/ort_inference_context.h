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
} OrtInferenceContext;

OrtInferenceContext* ort_ctx_create(OrtSession* session, int input_w, int input_h, int input_c);
void ort_ctx_destroy(OrtInferenceContext* ctx);
bool ort_ctx_prepare_input(OrtInferenceContext* ctx, const float* data, size_t bytes);
int ort_ctx_run(OrtInferenceContext* ctx, OrtValue** output_vals);
void ort_ctx_release_outputs(OrtInferenceContext* ctx, OrtValue** output_vals, size_t count);
int ort_ctx_get_output_shape(OrtInferenceContext* ctx, OrtValue* val, int64_t* dims, int max_dims);
float* ort_ctx_get_output_data(OrtInferenceContext* ctx, OrtValue* val);

#endif

#ifdef __cplusplus
}
#endif

#endif