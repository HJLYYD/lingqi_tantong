#define _DEFAULT_SOURCE

#include "ai_accel_adapter.h"
#include "logger.h"
#include "k1_platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef HAS_SPACENGINE_AI
#include <spacengine/spacengine.h>
#endif

struct AIAccelContext {
    bool initialized;
#ifdef HAS_SPACENGINE_AI
    spacengine_handle_t handles[AI_MODEL_COUNT];
    bool model_loaded[AI_MODEL_COUNT];
#endif
    void* tcm_buffer;
    size_t tcm_used;
};

AIAcclContext* ai_accel_context_create(void) {
    AIAcclContext* ctx = (AIAcclContext*)calloc(1, sizeof(AIAcclContext));
    if (!ctx) return NULL;

#ifdef HAS_SPACENGINE_AI
    spacengine_init_params_t params;
    memset(&params, 0, sizeof(params));
    params.device_id = 0;
    params.precision = SPACENGINE_PRECISION_INT8;
    params.power_mode = SPACENGINE_POWER_BALANCED;

    int ret = spacengine_init(&params);
    if (ret != 0) {
        log_error("Spacengine AI acceleration init failed: %d", ret);
        free(ctx);
        return NULL;
    }

    for (int i = 0; i < AI_MODEL_COUNT; i++) {
        ctx->handles[i] = NULL;
        ctx->model_loaded[i] = false;
    }

    ctx->initialized = true;
    log_info("Spacengine AI acceleration initialized (RISC-V X60, 2.0 TOPS INT8)");
#else
    ctx->initialized = false;
    log_warning("AI acceleration adapter running in stub mode (no Spacengine SDK)");
    log_warning("Set -DUSE_SPACENGINE_AI=ON -DSPACENGINE_DIR=<path> for RISC-V AI acceleration");
#endif

#ifdef HAS_K1_TCM
    if (k1_tcm_is_available()) {
        int tcm_size = k1_platform_get_tcm_size();
        ctx->tcm_buffer = k1_tcm_alloc(tcm_size);
        if (ctx->tcm_buffer) {
            ctx->tcm_used = 0;
            log_info("K1 TCM allocated: %dKB for AI model weights", tcm_size / 1024);
        }
    }
#endif

    return ctx;
}

void ai_accel_context_destroy(AIAcclContext* ctx) {
    if (!ctx) return;

#ifdef HAS_SPACENGINE_AI
    for (int i = 0; i < AI_MODEL_COUNT; i++) {
        if (ctx->handles[i]) {
            spacengine_unload(ctx->handles[i]);
            ctx->handles[i] = NULL;
        }
    }
    spacengine_deinit();
#endif

#ifdef HAS_K1_TCM
    if (ctx->tcm_buffer) {
        k1_tcm_free(ctx->tcm_buffer, k1_platform_get_tcm_size());
        ctx->tcm_buffer = NULL;
        ctx->tcm_used = 0;
    }
#endif

    free(ctx);
}

AIAcclStatus ai_accel_load_model(AIAcclContext* ctx, AIModelType type,
                                 const char* model_path,
                                 int input_w, int input_h) {
    if (!ctx || !model_path) return AI_ACCEL_ERROR_INIT;
    if (type >= AI_MODEL_COUNT) return AI_ACCEL_ERROR_LOAD;

#ifdef HAS_SPACENGINE_AI
    if (ctx->handles[type]) {
        spacengine_unload(ctx->handles[type]);
        ctx->handles[type] = NULL;
    }

    spacengine_model_params_t mp;
    memset(&mp, 0, sizeof(mp));
    mp.model_path = model_path;
    mp.input_width = input_w;
    mp.input_height = input_h;
    mp.precision = SPACENGINE_PRECISION_INT8;
    mp.cache_path = "/tmp/lingqi_ai_cache";

    int ret = spacengine_load(&mp, &ctx->handles[type]);
    if (ret != 0) {
        log_error("Failed to load AI model %d: %s (err=%d)", type, model_path, ret);
        return AI_ACCEL_ERROR_LOAD;
    }

    ctx->model_loaded[type] = true;
    log_info("AI model loaded: type=%d path=%s %dx%d", type, model_path, input_w, input_h);
    return AI_ACCEL_OK;
#else
    (void)input_w;
    (void)input_h;
    log_warning("AI accel stub: model %d would be loaded from %s", type, model_path);
    return AI_ACCEL_OK;
#endif
}

AIAcclStatus ai_accel_infer(AIAcclContext* ctx, AIModelType type,
                            const float* input_data,
                            AIAccelTensor** outputs, int* num_outputs) {
    (void)type;
    if (!ctx || !input_data || !outputs || !num_outputs) return AI_ACCEL_ERROR_INIT;

#ifdef HAS_SPACENGINE_AI
    if (!ctx->model_loaded[type] || !ctx->handles[type]) {
        return AI_ACCEL_ERROR_LOAD;
    }

    spacengine_run_params_t rp;
    memset(&rp, 0, sizeof(rp));
    rp.input_data = input_data;
    rp.input_format = SPACENGINE_FORMAT_NCHW_FLOAT;

    AIAccelTensor* out_tensors = NULL;
    int out_count = 0;

    int ret = spacengine_run(ctx->handles[type], &rp, (void**)&out_tensors, &out_count);
    if (ret != 0) {
        log_error("AI inference failed for model %d: %d", type, ret);
        return AI_ACCEL_ERROR_INFER;
    }

    *outputs = out_tensors;
    *num_outputs = out_count;
    return AI_ACCEL_OK;
#else
    *outputs = NULL;
    *num_outputs = 0;
    return AI_ACCEL_OK;
#endif
}

AIAcclStatus ai_accel_infer_uint8(AIAcclContext* ctx, AIModelType type,
                                  const uint8_t* input_data,
                                  AIAccelTensor** outputs, int* num_outputs) {
    (void)type;
    if (!ctx || !input_data || !outputs || !num_outputs) return AI_ACCEL_ERROR_INIT;

#ifdef HAS_SPACENGINE_AI
    if (!ctx->model_loaded[type] || !ctx->handles[type]) {
        return AI_ACCEL_ERROR_LOAD;
    }

    spacengine_run_params_t rp;
    memset(&rp, 0, sizeof(rp));
    rp.input_data = input_data;
    rp.input_format = SPACENGINE_FORMAT_NHWC_UINT8;

    AIAccelTensor* out_tensors = NULL;
    int out_count = 0;

    int ret = spacengine_run(ctx->handles[type], &rp, (void**)&out_tensors, &out_count);
    if (ret != 0) {
        log_error("AI inference (uint8) failed for model %d: %d", type, ret);
        return AI_ACCEL_ERROR_INFER;
    }

    *outputs = out_tensors;
    *num_outputs = out_count;
    return AI_ACCEL_OK;
#else
    *outputs = NULL;
    *num_outputs = 0;
    return AI_ACCEL_OK;
#endif
}

void ai_accel_release_outputs(AIAcclContext* ctx, AIAccelTensor* outputs, int num_outputs) {
    (void)ctx;
    (void)num_outputs;

#ifdef HAS_SPACENGINE_AI
    if (outputs) {
        spacengine_free_outputs(outputs, num_outputs);
    }
#else
    (void)outputs;
#endif
}

const char* ai_accel_status_string(AIAcclStatus status) {
    switch (status) {
        case AI_ACCEL_OK:           return "OK";
        case AI_ACCEL_ERROR_INIT:   return "INIT_ERROR";
        case AI_ACCEL_ERROR_LOAD:   return "LOAD_ERROR";
        case AI_ACCEL_ERROR_INFER:  return "INFER_ERROR";
        case AI_ACCEL_ERROR_MEMORY: return "MEMORY_ERROR";
        case AI_ACCEL_ERROR_TIMEOUT: return "TIMEOUT_ERROR";
        default:                    return "UNKNOWN";
    }
}

bool ai_accel_is_available(void) {
#ifdef HAS_SPACENGINE_AI
    return spacengine_probe() == 0;
#else
#ifdef HAS_K1_PIPELINE
    return k1_platform_is_k1();
#else
    return false;
#endif
#endif
}

void ai_accel_warmup(AIAcclContext* ctx, AIModelType type, int input_w, int input_h) {
    if (!ctx) return;

    size_t data_size = (size_t)input_w * input_h * 3 * sizeof(float);
    float* dummy = (float*)calloc(1, data_size);
    if (!dummy) return;

    AIAccelTensor* outputs = NULL;
    int num_outputs = 0;

    log_info("AI acceleration warmup: model=%d, %dx%d (3 runs)", type, input_w, input_h);
    for (int i = 0; i < 3; i++) {
        ai_accel_infer(ctx, type, dummy, &outputs, &num_outputs);
        ai_accel_release_outputs(ctx, outputs, num_outputs);
    }

    free(dummy);
    log_info("AI acceleration warmup complete for model %d", type);
}