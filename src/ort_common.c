#include "ort_common.h"
#include "spacemit_ort_bridge.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#ifdef HAS_ONNX_RUNTIME

static const OrtApi* g_ort_api = NULL;
static OrtEnv* g_ort_env = NULL;
static bool g_ort_ready = false;
static pthread_mutex_t g_ort_mutex = PTHREAD_MUTEX_INITIALIZER;

bool ort_global_init(void) {
    pthread_mutex_lock(&g_ort_mutex);
    if (g_ort_ready) {
        pthread_mutex_unlock(&g_ort_mutex);
        return true;
    }

    g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort_api) {
        log_warning("ONNX Runtime API not available");
        pthread_mutex_unlock(&g_ort_mutex);
        return false;
    }

    OrtStatus* status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "LingQiTanTong", &g_ort_env);
    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_warning("Failed to create ONNX Runtime env: %s", msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        g_ort_api = NULL;
        pthread_mutex_unlock(&g_ort_mutex);
        return false;
    }

    g_ort_ready = true;
#ifdef HAS_SPACEMIT_EP
    log_info("ONNX Runtime initialized (SpacemiT EP: RISC-V Vector + IME acceleration)");
#else
    log_info("ONNX Runtime initialized (CPU only)");
#endif
    pthread_mutex_unlock(&g_ort_mutex);
    return true;
}

void ort_global_shutdown(void) {
    pthread_mutex_lock(&g_ort_mutex);
    if (g_ort_env && g_ort_api) {
        g_ort_api->ReleaseEnv(g_ort_env);
    }
    g_ort_env = NULL;
    g_ort_api = NULL;
    g_ort_ready = false;
    pthread_mutex_unlock(&g_ort_mutex);
}

const OrtApi* ort_get_api(void) {
    return g_ort_api;
}

OrtEnv* ort_get_env(void) {
    return g_ort_env;
}

int ort_validate_onnx_file(const char* model_path, size_t* out_file_size) {
    if (!model_path) return -1;

    FILE* f = fopen(model_path, "rb");
    if (!f) {
        log_error("Cannot open ONNX model: %s", model_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 256) {
        fclose(f);
        log_error("ONNX model file too small: %s (%zu bytes)", model_path, file_size);
        return -1;
    }

    uint8_t magic[8];
    size_t bytes_read = fread(magic, 1, 8, f);
    fclose(f);

    if (bytes_read != 8) {
        log_error("Failed to read ONNX model header: %s", model_path);
        return -1;
    }

    if (magic[0] != 0x08 || magic[1] != 0x00 || magic[2] != 0x00 || magic[3] != 0x00) {
        log_warning("ONNX model has unexpected IR version header: %s", model_path);
    }

    if (out_file_size) *out_file_size = file_size;
    return 0;
}

OrtSession* ort_create_session(const char* model_path, int num_threads, bool use_ep) {
    if (!g_ort_api || !g_ort_env || !model_path) return NULL;

    OrtSessionOptions* session_opts;
    OrtStatus* status = g_ort_api->CreateSessionOptions(&session_opts);
    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_error("CreateSessionOptions failed: %s", msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        return NULL;
    }

    g_ort_api->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL);
    g_ort_api->SetIntraOpNumThreads(session_opts, num_threads > 0 ? num_threads : 4);
    g_ort_api->SetInterOpNumThreads(session_opts, num_threads > 0 ? (num_threads > 2 ? 2 : num_threads) : 2);

#ifdef HAS_SPACEMIT_EP
    if (use_ep) {
        int ret = spacemit_ort_session_options_init(session_opts);
        if (ret == 0) {
            log_info("SpacemiT EP registered (RVV + IME, INT8 accelerated)");
        } else {
            log_warning("SpacemiT EP registration failed, falling back to CPU");
        }
    } else {
        log_info("SpacemiT EP skipped (FP32 model, CPU path faster)");
    }
#endif

    OrtSession* session = NULL;
    status = g_ort_api->CreateSession(g_ort_env, model_path, session_opts, &session);
    g_ort_api->ReleaseSessionOptions(session_opts);

    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_error("CreateSession failed for %s: %s", model_path, msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        return NULL;
    }

    log_info("Session created: %s (threads=%d)", model_path, num_threads);
    return session;
}

size_t ort_get_model_input_size(OrtSession* session) {
    if (!g_ort_api || !session) return 0;

    size_t num_inputs;
    OrtStatus* status = g_ort_api->SessionGetInputCount(session, &num_inputs);
    if (status) {
        g_ort_api->ReleaseStatus(status);
        return 0;
    }

    if (num_inputs == 0) return 0;

    OrtTypeInfo* type_info;
    status = g_ort_api->SessionGetInputTypeInfo(session, 0, &type_info);
    if (status) {
        g_ort_api->ReleaseStatus(status);
        return 0;
    }

    const OrtTensorTypeAndShapeInfo* tensor_info;
    status = g_ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status) {
        g_ort_api->ReleaseStatus(status);
        g_ort_api->ReleaseTypeInfo(type_info);
        return 0;
    }

    size_t num_dims;
    g_ort_api->GetDimensionsCount(tensor_info, &num_dims);

    int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
    if (!dims) {
        g_ort_api->ReleaseTypeInfo(type_info);
        return 0;
    }

    g_ort_api->GetDimensions(tensor_info, dims, num_dims);
    g_ort_api->ReleaseTypeInfo(type_info);

    size_t total = 1;
    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i] > 0) total *= (size_t)dims[i];
    }
    free(dims);

    return total;
}

#endif