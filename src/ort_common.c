#include "ort_common.h"
#include "spacemit_ort_bridge.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#ifdef HAS_ONNX_RUNTIME

static const OrtApi* g_ort_api = NULL;
static OrtEnv* g_ort_env = NULL;
static bool g_ort_ready = false;
static bool g_spacemit_ep_active = false;
static int g_ep_session_count = 0;
#define MAX_EP_SESSIONS 4  /* K1 cluster-0 has 4 AI cores; 4 EP sessions for detection+pose+face+arcface */
#ifdef HAS_SPACEMIT_EP
static bool g_ep_enabled_pref = true;
#else
static bool g_ep_enabled_pref = false;
#endif
static pthread_mutex_t g_ort_mutex = PTHREAD_MUTEX_INITIALIZER;

bool ort_spacemit_ep_active(void) {
    return g_spacemit_ep_active;
}

void ort_set_ep_enabled(bool enabled) {
    pthread_mutex_lock(&g_ort_mutex);
#ifdef HAS_SPACEMIT_EP
    g_ep_enabled_pref = enabled;
    log_info("SpacemiT EP preference set to %s", enabled ? "ENABLED" : "DISABLED");
#else
    (void)enabled;
    g_ep_enabled_pref = false;
    if (enabled) {
        log_warning("SpacemiT EP enable requested but binary was built without HAS_SPACEMIT_EP");
    }
#endif
    pthread_mutex_unlock(&g_ort_mutex);
}

bool ort_get_ep_enabled(void) {
    return g_ep_enabled_pref;
}

bool ort_global_init(void) {
    pthread_mutex_lock(&g_ort_mutex);
    if (g_ort_ready) {
        pthread_mutex_unlock(&g_ort_mutex);
        return true;
    }

    g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort_api) {
        log_error("ONNX Runtime API not available (incompatible libonnxruntime.so?)");
        pthread_mutex_unlock(&g_ort_mutex);
        return false;
    }

    OrtStatus* status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "LingQiTanTong", &g_ort_env);
    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_error("ONNX Runtime env creation failed: %s", msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        g_ort_api = NULL;
        pthread_mutex_unlock(&g_ort_mutex);
        return false;
    }

    g_ort_ready = true;
#ifdef HAS_SPACEMIT_EP
    /*
     * Do NOT pre-probe the EP here. Pre-probing forks the parent process,
     * but at this point the parent has already called CreateEnv (which
     * may spawn ORT threadpool threads). fork() in a multithreaded parent
     * leaves the child in an undefined state w.r.t. those pthreads ->
     * unreliable probe.
     *
     * Per-model probing is done inside ort_create_session() — it forks
     * BEFORE the parent's session is created and tests the specific
     * model+EP combo. The probe uses a fresh OrtEnv in the child.
     */
    if (g_ep_enabled_pref) {
        log_info("ONNX Runtime initialized (SpacemiT EP available, per-model probe will validate each session)");
    } else {
        log_info("ONNX Runtime initialized (SpacemiT EP disabled by config, using CPU EP)");
    }
#else
    log_info("ONNX Runtime initialized (CPU-only build, SpacemiT EP NOT compiled in)");
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
    g_spacemit_ep_active = false;
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

    if (magic[0] != 0x08) {
        log_warning("ONNX model has unexpected protobuf header byte 0x%02x (expected 0x08): %s",
                    magic[0], model_path);
    }

    if (out_file_size) *out_file_size = file_size;
    return 0;
}

OrtSession* ort_create_session(const char* model_path, int num_threads, bool use_ep) {
    if (!g_ort_api || !g_ort_env || !model_path) return NULL;

    int n_online = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_online <= 0) n_online = 4;

    /*
     * ── SpacemiT EP decision ──
     *
     * Multi-stage gate, ordered by cost (cheapest first):
     *   1. Config flag (use_ep && g_ep_enabled_pref)
     *   2. Model file accessible
     *   3. /dev/tcm present
     *   4. EP session count under limit
     *   5. Quantization check (scan model for INT8 ops)
     *   6. Direct EP session creation via safe wrapper
     *
     * NO fork-probe in the production path. The probe is a diagnostic
     * tool only. If a model passes the INT8 quantization check and TCM
     * is available, EP session creation should succeed. If it doesn't,
     * we fall back to CPU EP for THIS model only — no global disable.
     */
    bool want_ep = false;
#ifdef HAS_SPACEMIT_EP
    if (use_ep && g_ep_enabled_pref) {
        /* ── Gate 1: model file accessible ── */
        if (spacemit_ort_check_model_accessible(model_path) != 0) {
            log_error("Model file not accessible: %s (skipping EP)", model_path);
            goto use_cpu;
        }

        /* ── Gate 2: /dev/tcm present ── */
        if (spacemit_ort_check_tcm_available() != 1) {
            log_info("/dev/tcm not available, using CPU EP for %s", model_path);
            goto use_cpu;
        }

        /* ── Dump TCM diagnostics on first EP attempt ── */
        {
            static bool tcm_diag_dumped = false;
            if (!tcm_diag_dumped) {
                tcm_diag_dumped = true;
                spacemit_ort_dump_tcm_diagnostics(model_path);
            }
        }

        /* ── Gate 3: EP session limit ── */
        if (g_ep_session_count >= MAX_EP_SESSIONS) {
            log_info("EP session limit reached (%d/%d), using CPU EP for %s",
                     g_ep_session_count, MAX_EP_SESSIONS, model_path);
            goto use_cpu;
        }

        /* ── Gate 4: quantization check ── */
        char chk_err[512] = {0};
        int chk = spacemit_ort_check_model_supported(model_path, chk_err, sizeof(chk_err));
        if (chk != 0) {
            log_warning("SpacemiT EP: model check failed for %s: %s", model_path,
                        chk_err[0] ? chk_err : "unknown reason");
            log_info("  -> Using CPU EP (real ONNX, no RVV/IME acceleration)");
            goto use_cpu;
        }

        /* Gate 5 disabled — the fork-probe child's EP warmup Run()
         * corrupts K1 IME hardware state even though the child exits
         * cleanly.  This causes "Illegal instruction" in the parent
         * on the next floating-point operation.  Skip probe, try EP
         * directly (std::terminate handler catches unrecoverable crashes). */
        (void)0;

        /*
         * All gates passed. Try EP session creation directly.
         */
        want_ep = true;
    } else if (use_ep && !g_ep_enabled_pref) {
        log_info("SpacemiT EP disabled by config, using CPU EP for %s", model_path);
    }
#else
    if (use_ep) {
        log_warning("SpacemiT EP not compiled in (HAS_SPACEMIT_EP), using CPU EP");
    }
#endif

use_cpu:  /* fallthrough label for EP skip */

    int intra;
    int inter;
    if (want_ep) {
#ifdef HAS_SPACEMIT_EP
        intra = spacemit_ort_get_ep_intra_threads();
#else
        intra = 1;
#endif
        inter = 1;
    } else {
        /* CPU EP: use all available cores for maximum throughput.
         * K1 8-core: 6 intra-op (leaves 2 for system + pipeline threads) */
        intra = num_threads > 0 ? num_threads : (n_online >= 8 ? 6 : (n_online >= 4 ? 4 : n_online));
        /* If caller requested 4 (legacy default), bump to 6 on 8-core systems */
        if (intra == 4 && n_online >= 8) intra = 6;
        inter = intra > 2 ? 2 : 1;
    }

    OrtSessionOptions* session_opts = NULL;
    OrtStatus* status = g_ort_api->CreateSessionOptions(&session_opts);
    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_error("CreateSessionOptions failed: %s", msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        return NULL;
    }

    OrtStatus* st_opt;
    st_opt = g_ort_api->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL);
    if (st_opt) g_ort_api->ReleaseStatus(st_opt);
    st_opt = g_ort_api->SetIntraOpNumThreads(session_opts, intra);
    if (st_opt) g_ort_api->ReleaseStatus(st_opt);
    st_opt = g_ort_api->SetInterOpNumThreads(session_opts, inter);
    if (st_opt) g_ort_api->ReleaseStatus(st_opt);

    bool ep_registered = false;
#ifdef HAS_SPACEMIT_EP
    if (want_ep) {
        int ret = spacemit_ort_session_options_init(session_opts);
        if (ret == 0) {
            ep_registered = true;
        } else {
            log_warning("SpacemiT EP SessionOptions init failed (rc=%d) for %s, falling back to CPU EP",
                        ret, model_path);
            want_ep = false;
            /* Adjust thread counts for CPU fallback */
            intra = num_threads > 0 ? num_threads : (n_online >= 8 ? 6 : (n_online >= 4 ? 4 : n_online));
            inter = intra > 2 ? 2 : 1;
            st_opt = g_ort_api->SetIntraOpNumThreads(session_opts, intra);
            if (st_opt) g_ort_api->ReleaseStatus(st_opt);
            st_opt = g_ort_api->SetInterOpNumThreads(session_opts, inter);
            if (st_opt) g_ort_api->ReleaseStatus(st_opt);
        }
    }
#endif

    OrtSession* session = NULL;
    bool ep_succeeded = false;

#ifdef HAS_SPACEMIT_EP
    if (ep_registered) {
        char ep_err[512] = {0};
        int rc = spacemit_ort_create_session_safe(g_ort_env, model_path, session_opts, &session, ep_err, sizeof(ep_err));
        g_ort_api->ReleaseSessionOptions(session_opts);
        if (rc != 0) {
            /* EP failed cleanly (not a crash) — fall back to CPU for THIS model */
            log_error("SpacemiT EP CreateSession failed for %s (rc=%d): %s",
                      model_path, rc, ep_err[0] ? ep_err : "no error message");
            log_info("  -> Retrying with CPU EP (this model only, EP remains enabled for others)");

            /* Retry with CPU EP session options */
            OrtSessionOptions* cpu_opts = NULL;
            OrtStatus* st2 = g_ort_api->CreateSessionOptions(&cpu_opts);
            if (st2) { g_ort_api->ReleaseStatus(st2); return NULL; }

            int cpu_intra = num_threads > 0 ? num_threads : (n_online >= 8 ? 6 : (n_online >= 4 ? 4 : n_online));
            int cpu_inter = cpu_intra > 2 ? 2 : 1;
            OrtStatus* st_c;
            st_c = g_ort_api->SetSessionGraphOptimizationLevel(cpu_opts, ORT_ENABLE_ALL);
            if (st_c) g_ort_api->ReleaseStatus(st_c);
            st_c = g_ort_api->SetIntraOpNumThreads(cpu_opts, cpu_intra);
            if (st_c) g_ort_api->ReleaseStatus(st_c);
            st_c = g_ort_api->SetInterOpNumThreads(cpu_opts, cpu_inter);
            if (st_c) g_ort_api->ReleaseStatus(st_c);

            OrtStatus* st_cs = g_ort_api->CreateSession(g_ort_env, model_path, cpu_opts, &session);
            g_ort_api->ReleaseSessionOptions(cpu_opts);
            if (st_cs) {
                const char* msg2 = g_ort_api->GetErrorMessage(st_cs);
                log_error("CPU EP CreateSession also failed for %s: %s", model_path, msg2 ? msg2 : "unknown");
                g_ort_api->ReleaseStatus(st_cs);
                return NULL;
            }
            intra = cpu_intra;
            inter = cpu_inter;
            log_info("CPU EP session created for %s (intra=%d, inter=%d)", model_path, intra, inter);
        } else {
            ep_succeeded = true;
            g_ep_session_count++;
            g_spacemit_ep_active = true;
        }
    } else {
        status = g_ort_api->CreateSession(g_ort_env, model_path, session_opts, &session);
        g_ort_api->ReleaseSessionOptions(session_opts);
        if (status) {
            const char* msg = g_ort_api->GetErrorMessage(status);
            log_error("CreateSession (CPU EP) failed for %s: %s", model_path, msg ? msg : "unknown");
            g_ort_api->ReleaseStatus(status);
            return NULL;
        }
    }
#else
    status = g_ort_api->CreateSession(g_ort_env, model_path, session_opts, &session);
    g_ort_api->ReleaseSessionOptions(session_opts);
    if (status) {
        const char* msg = g_ort_api->GetErrorMessage(status);
        log_error("CreateSession failed for %s: %s", model_path, msg ? msg : "unknown");
        g_ort_api->ReleaseStatus(status);
        return NULL;
    }
#endif

    if (ep_succeeded) {
        log_info("Session: %s (intra=%d, inter=%d, ep=SpacemiT [%d/%d])",
                 model_path, intra, inter, g_ep_session_count, MAX_EP_SESSIONS);
    } else {
        log_info("Session: %s (intra=%d, inter=%d, ep=CPU)", model_path, intra, inter);
    }

    return session;
}

int ort_get_input_shape(OrtSession* session, int* out_dims, int max_dims) {
    if (!g_ort_api || !session || !out_dims || max_dims <= 0) return -1;

    size_t num_inputs = 0;
    OrtStatus* st = g_ort_api->SessionGetInputCount(session, &num_inputs);
    if (st) { g_ort_api->ReleaseStatus(st); return -1; }
    if (num_inputs == 0) return -1;

    OrtTypeInfo* type_info = NULL;
    st = g_ort_api->SessionGetInputTypeInfo(session, 0, &type_info);
    if (st) { g_ort_api->ReleaseStatus(st); return -1; }

    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
    st = g_ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (st) {
        g_ort_api->ReleaseStatus(st);
        g_ort_api->ReleaseTypeInfo(type_info);
        return -1;
    }

    size_t num_dims = 0;
    st = g_ort_api->GetDimensionsCount(tensor_info, &num_dims);
    if (st) {
        g_ort_api->ReleaseStatus(st);
        g_ort_api->ReleaseTypeInfo(type_info);
        return -1;
    }
    if (num_dims == 0 || (int)num_dims > max_dims) {
        g_ort_api->ReleaseTypeInfo(type_info);
        return -1;
    }

    int64_t dims[8] = {0};
    if (num_dims > 8) num_dims = 8;
    st = g_ort_api->GetDimensions(tensor_info, dims, num_dims);
    if (st) g_ort_api->ReleaseStatus(st);
    g_ort_api->ReleaseTypeInfo(type_info);

    for (size_t i = 0; i < num_dims; i++) {
        out_dims[i] = (int)dims[i];
    }
    return (int)num_dims;
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
    status = g_ort_api->GetDimensionsCount(tensor_info, &num_dims);
    if (status) {
        g_ort_api->ReleaseStatus(status);
        g_ort_api->ReleaseTypeInfo(type_info);
        return 0;
    }

    int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
    if (!dims) {
        g_ort_api->ReleaseTypeInfo(type_info);
        return 0;
    }

    status = g_ort_api->GetDimensions(tensor_info, dims, num_dims);
    if (status) g_ort_api->ReleaseStatus(status);
    g_ort_api->ReleaseTypeInfo(type_info);

    size_t total = 1;
    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i] > 0) total *= (size_t)dims[i];
    }
    free(dims);

    return total;
}

#endif