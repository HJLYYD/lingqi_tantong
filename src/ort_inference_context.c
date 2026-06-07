#include "ort_inference_context.h"
#include "ort_common.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>

#ifdef HAS_ONNX_RUNTIME

OrtInferenceContext* ort_ctx_create(OrtSession* session, int input_w, int input_h, int input_c) {
    if (!session || input_w <= 0 || input_h <= 0 || input_c <= 0) {
        log_error("ort_ctx_create: invalid parameters");
        return NULL;
    }

    OrtInferenceContext* ctx = (OrtInferenceContext*)calloc(1, sizeof(OrtInferenceContext));
    if (!ctx) return NULL;

    const OrtApi* ort = ort_get_api();
    if (!ort) {
        free(ctx);
        return NULL;
    }

    OrtStatus* st = ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
    if (st) {
        log_error("ort_ctx_create: GetAllocatorWithDefaultOptions failed");
        ort->ReleaseStatus(st);
        free(ctx);
        return NULL;
    }

    st = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    if (st) {
        const char* msg = ort->GetErrorMessage(st);
        log_error("ort_ctx_create: CreateCpuMemoryInfo failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(st);
        free(ctx);
        return NULL;
    }

    ctx->session = session;
    ctx->input_width = input_w;
    ctx->input_height = input_h;
    ctx->input_channels = input_c;
    ctx->input_tensor_bytes = (size_t)input_w * input_h * input_c * sizeof(float);
    ctx->input_tensor = (float*)calloc(1, ctx->input_tensor_bytes);
    if (!ctx->input_tensor) {
        log_error("ort_ctx_create: failed to allocate input tensor (%zu bytes)", ctx->input_tensor_bytes);
        ort->ReleaseMemoryInfo(ctx->memory_info);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void ort_ctx_destroy(OrtInferenceContext* ctx) {
    if (!ctx) return;
    const OrtApi* ort = ort_get_api();
    if (ort) {
        if (ctx->output_names && ctx->allocator) {
            for (size_t i = 0; i < ctx->num_outputs; i++) {
                if (ctx->output_names[i]) {
                    OrtStatus* st = ort->AllocatorFree(ctx->allocator, ctx->output_names[i]);
                    if (st) ort->ReleaseStatus(st);
                }
            }
            free(ctx->output_names);
        }
        if (ctx->memory_info) {
            ort->ReleaseMemoryInfo(ctx->memory_info);
        }
    }
    free(ctx->input_tensor);
    free(ctx);
}

bool ort_ctx_prepare_input(OrtInferenceContext* ctx, const float* data, size_t bytes) {
    if (!ctx || !data || bytes == 0) return false;

    if (bytes != ctx->input_tensor_bytes) {
        float* new_buf = (float*)realloc(ctx->input_tensor, bytes);
        if (!new_buf) {
            log_error("ort_ctx_prepare_input: realloc failed (%zu bytes)", bytes);
            return false;
        }
        ctx->input_tensor = new_buf;
        ctx->input_tensor_bytes = bytes;
    }

    memcpy(ctx->input_tensor, data, bytes);
    return true;
}

int ort_ctx_run(OrtInferenceContext* ctx, OrtValue** output_vals) {
    if (!ctx || !ctx->session || !output_vals) return -1;

    const OrtApi* ort = ort_get_api();
    if (!ort) return -1;

    if (!ctx->names_cached) {
        OrtStatus* st = ort->SessionGetOutputCount(ctx->session, &ctx->num_outputs);
        if (st) {
            ort->ReleaseStatus(st);
            ctx->num_outputs = 1;
        }
        if (ctx->num_outputs == 0) ctx->num_outputs = 1;

        ctx->output_names = (char**)calloc(ctx->num_outputs, sizeof(char*));
        if (!ctx->output_names) return -1;

        bool ok = true;
        for (size_t i = 0; i < ctx->num_outputs; i++) {
            char* name = NULL;
            OrtStatus* s = ort->SessionGetOutputName(ctx->session, i, ctx->allocator, &name);
            if (s || !name) {
                if (s) ort->ReleaseStatus(s);
                ok = false;
                break;
            }
            ctx->output_names[i] = name;
        }

        if (!ok) {
            for (size_t i = 0; i < ctx->num_outputs; i++) {
                if (ctx->output_names[i]) {
                    OrtStatus* sf = ort->AllocatorFree(ctx->allocator, ctx->output_names[i]);
                    if (sf) ort->ReleaseStatus(sf);
                }
            }
            free(ctx->output_names);
            ctx->output_names = NULL;
            return -1;
        }

        ctx->names_cached = true;
    }

    int64_t input_shape[4] = {1, ctx->input_channels, ctx->input_height, ctx->input_width};
    OrtValue* input_val = NULL;
    OrtStatus* st = ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info, ctx->input_tensor, ctx->input_tensor_bytes,
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_val);
    if (st) {
        const char* msg = ort->GetErrorMessage(st);
        log_error("ort_ctx_run: CreateTensorWithDataAsOrtValue failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(st);
        return -1;
    }

    const char* input_name = ctx->input_name[0] ? ctx->input_name : "images";
    const char* input_names[] = {input_name};
    const OrtValue* input_vals[] = {input_val};

    st = ort->Run(ctx->session, NULL,
                  input_names, input_vals, 1,
                  (const char* const*)ctx->output_names, ctx->num_outputs, output_vals);
    ort->ReleaseValue(input_val);

    if (st) {
        const char* msg = ort->GetErrorMessage(st);
        log_error("ort_ctx_run: inference failed: %s", msg ? msg : "unknown");
        ort->ReleaseStatus(st);
        return -1;
    }

    return 0;
}

void ort_ctx_release_outputs(OrtInferenceContext* ctx, OrtValue** output_vals, size_t count) {
    (void)ctx;
    if (!output_vals) return;
    const OrtApi* ort = ort_get_api();
    if (!ort) return;
    for (size_t i = 0; i < count; i++) {
        if (output_vals[i]) {
            ort->ReleaseValue(output_vals[i]);
            output_vals[i] = NULL;
        }
    }
}

int ort_ctx_get_output_shape(OrtInferenceContext* ctx, OrtValue* val, int64_t* dims, int max_dims) {
    (void)ctx;
    if (!val || !dims || max_dims <= 0) return -1;

    const OrtApi* ort = ort_get_api();
    if (!ort) return -1;

    OrtTensorTypeAndShapeInfo* si = NULL;
    OrtStatus* st = ort->GetTensorTypeAndShape(val, &si);
    if (st || !si) {
        if (st) ort->ReleaseStatus(st);
        return -1;
    }

    size_t nd = 0;
    {
        OrtStatus* _s = ort->GetDimensionsCount(si, &nd);
        if (_s) ort->ReleaseStatus(_s);
    }
    if (nd == 0 || (int)nd > max_dims) {
        ort->ReleaseTensorTypeAndShapeInfo(si);
        return -1;
    }

    {
        OrtStatus* _s = ort->GetDimensions(si, dims, nd);
        if (_s) ort->ReleaseStatus(_s);
    }
    ort->ReleaseTensorTypeAndShapeInfo(si);
    return (int)nd;
}

float* ort_ctx_get_output_data(OrtInferenceContext* ctx, OrtValue* val) {
    (void)ctx;
    if (!val) return NULL;

    const OrtApi* ort = ort_get_api();
    if (!ort) return NULL;

    float* data = NULL;
    OrtStatus* st = ort->GetTensorMutableData(val, (void**)&data);
    if (st) {
        ort->ReleaseStatus(st);
        return NULL;
    }
    return data;
}

#endif