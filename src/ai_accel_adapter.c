#define _DEFAULT_SOURCE

#include "ai_accel_adapter.h"
#include "logger.h"
#include "k1_platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#ifdef HAS_ONNX_RUNTIME
#include "ort_common.h"
#endif

struct AIAcclContext {
    bool initialized;
    bool spacemit_ep_available;
    bool tcm_active;
    void* tcm_buffer;
    size_t tcm_size;
    char description[256];
};

static void describe_status(AIAcclContext* ctx) {
    const char* ep = ctx->spacemit_ep_available ? "SpacemiT-EP (RVV+IME)" : "CPU";
    const char* tcm = ctx->tcm_active ? "TCM" : "no-TCM";
    snprintf(ctx->description, sizeof(ctx->description),
             "K1 AI Accel: backend=%s, %s (%zuKB)", ep, tcm, ctx->tcm_size / 1024);
}

AIAcclContext* ai_accel_context_create(void) {
    AIAcclContext* ctx = (AIAcclContext*)calloc(1, sizeof(AIAcclContext));
    if (!ctx) return NULL;

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_error("AI accel: ONNX Runtime init failed");
        free(ctx);
        return NULL;
    }
#ifdef HAS_SPACEMIT_EP
    ctx->spacemit_ep_available = (access("/dev/tcm", R_OK | W_OK) == 0);
    if (!ctx->spacemit_ep_available) {
        log_warning("AI accel: /dev/tcm not accessible (chmod 666 /dev/tcm), SpacemiT EP disabled");
    }
#endif
#endif

    if (k1_tcm_is_available()) {
        int tcm_size = k1_platform_get_tcm_size();
        ctx->tcm_buffer = k1_tcm_alloc(tcm_size);
        if (ctx->tcm_buffer) {
            ctx->tcm_size = (size_t)tcm_size;
            ctx->tcm_active = true;
            log_info("AI accel: K1 TCM mapped %dKB at %p", tcm_size / 1024, ctx->tcm_buffer);
        }
    }

    ctx->initialized = true;
    describe_status(ctx);
    log_info("%s", ctx->description);
    return ctx;
}

void ai_accel_context_destroy(AIAcclContext* ctx) {
    if (!ctx) return;

    if (ctx->tcm_buffer) {
        k1_tcm_free(ctx->tcm_buffer, ctx->tcm_size);
        ctx->tcm_buffer = NULL;
        ctx->tcm_active = false;
    }

    free(ctx);
}

bool ai_accel_is_available(void) {
#ifdef HAS_ONNX_RUNTIME
    return ort_get_api() != NULL;
#else
    return false;
#endif
}

bool ai_accel_is_hw_active(const AIAcclContext* ctx) {
    if (!ctx) return false;
#ifdef HAS_ONNX_RUNTIME
    return ctx->spacemit_ep_available && ort_spacemit_ep_active();
#else
    return false;
#endif
}

const char* ai_accel_describe(const AIAcclContext* ctx) {
    return ctx ? ctx->description : "(null AI accel context)";
}

AIAcclStatus ai_accel_load_model(AIAcclContext* ctx, AIModelType type,
                                 const char* model_path,
                                 int input_w, int input_h) {
    (void)ctx; (void)type; (void)input_w; (void)input_h;
    if (!model_path) return AI_ACCEL_ERROR_INIT;
    log_info("AI accel: model %d (%s) is loaded via ort_create_session() in the inference pipeline",
             type, model_path);
    return AI_ACCEL_OK;
}

void ai_accel_warmup(AIAcclContext* ctx, AIModelType type, int input_w, int input_h) {
    (void)ctx; (void)type; (void)input_w; (void)input_h;
    log_info("AI accel: warmup is performed implicitly by first ORT inference (skipped)");
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
