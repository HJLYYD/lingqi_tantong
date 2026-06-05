#include "spacemit_ort_bridge.h"

#ifdef HAS_SPACEMIT_EP
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <exception>
#include <atomic>

/*
 * SpacemiT EP can throw std::runtime_error from internal worker threads
 * during CreateSession / Run (commonly "tcm buffer alloc failed for core
 * id N"). Such throws cross the C ABI and the calling thread's try/catch
 * cannot see them — the C++ runtime calls std::terminate which aborts.
 *
 * Defense-in-depth layers (ordered by cost):
 *   1. Quantization check    — scan model file for INT8 op strings (fast, ~1ms)
 *   2. Model file check      — verify model exists and is readable (instant)
 *   3. /dev/tcm check        — verify TCM device node exists (instant)
 *   4. Safe CreateSession    — try/catch catches same-thread exceptions
 *   5. std::terminate handler — logs worker-thread crashes before _Exit(134)
 *   6. Fork-probe (optional) — can pre-test EP without risking parent process
 *
 * The fork-probe is a diagnostic tool, NOT the primary gate. The quant
 * check is the primary gate — if a model is INT8-quantized and /dev/tcm
 * is accessible, EP should succeed. The fork-probe is only used when
 * explicitly requested (debug builds or --ep-probe flag).
 */
static std::atomic<bool> g_terminate_installed{false};

static void spacemit_terminate_handler() noexcept {
    std::fputs("=== spacemit_ort_bridge: std::terminate called ===\n", stderr);
    try {
        std::exception_ptr ep = std::current_exception();
        if (ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  uncaught std::exception: %s\n", e.what() ? e.what() : "(null)");
            } catch (...) {
                std::fputs("  uncaught non-std::exception\n", stderr);
            }
        } else {
            std::fputs("  no active exception\n", stderr);
        }
    } catch (...) {
    }
    std::fputs("This is from a SpacemiT EP worker thread (uncatchable across C ABI).\n", stderr);
    std::fputs("Possible causes: FP32 model fed to EP, TCM exhausted, model weights too large.\n", stderr);
    std::fputs("Ensure model is INT8-quantized via xquant: https://bianbu.spacemit.com/en/brdk/\n", stderr);
    std::fflush(stderr);
    std::_Exit(134);
}

static void ensure_terminate_installed() {
    bool expected = false;
    if (g_terminate_installed.compare_exchange_strong(expected, true)) {
        std::set_terminate(spacemit_terminate_handler);
    }
}

/* ── Model file accessibility check ── */
extern "C" int spacemit_ort_check_model_accessible(const char* model_path) {
    if (!model_path || !model_path[0]) return -1;
    if (access(model_path, R_OK) != 0) return -2;
    return 0;
}

/* ── TCM device check (cached, board-level) ── */
static int g_tcm_available = -1;  /* -1=unchecked, 0=no, 1=yes */

extern "C" int spacemit_ort_check_tcm_available(void) {
    if (g_tcm_available >= 0) return g_tcm_available;
    if (access("/dev/tcm", F_OK) == 0) {
        g_tcm_available = 1;
    } else {
        g_tcm_available = 0;
    }
    return g_tcm_available;
}

/*
 * ── TCM diagnostic dump ──
 *
 * Logs detailed TCM state before EP session creation.  Helps diagnose
 * "tcm buffer alloc failed for core id 0" errors which can be caused by:
 *   1. Another process holding /dev/tcm (check with: sudo fuser /dev/tcm)
 *   2. Kernel TCM driver issue (check: dmesg | grep -i tcm)
 *   3. Model weights + intermediate buffers exceed per-core TCM (~512 KB)
 *   4. Previous EP session not fully released (TCM leak in driver)
 */
extern "C" void spacemit_ort_dump_tcm_diagnostics(const char* model_path) {
    std::fprintf(stderr, "── TCM Diagnostic for %s ──\n",
                 model_path ? model_path : "(unknown)");

    /* 1. Device node */
    if (access("/dev/tcm", F_OK) == 0) {
        std::fprintf(stderr, "  /dev/tcm: EXISTS\n");
        if (access("/dev/tcm", R_OK | W_OK) == 0) {
            std::fprintf(stderr, "  /dev/tcm: readable & writable\n");
        } else {
            std::fprintf(stderr, "  /dev/tcm: PERMISSION DENIED (try: sudo chmod 666 /dev/tcm)\n");
        }
    } else {
        std::fprintf(stderr, "  /dev/tcm: MISSING (TCM driver not loaded?)\n");
    }

    /* 2. Check for TCM holder processes via /proc */
    std::fprintf(stderr, "  Checking /proc for TCM holders...\n");
    /* Try opening /dev/tcm to test if another process holds it exclusively */
    int tcm_fd = open("/dev/tcm", O_RDONLY | O_NONBLOCK);
    if (tcm_fd >= 0) {
        std::fprintf(stderr, "  /dev/tcm: open() succeeded (fd=%d) — not locked\n", tcm_fd);
        close(tcm_fd);
    } else {
        std::fprintf(stderr, "  /dev/tcm: open() failed (errno=%d: %s) — may be held exclusively\n",
                     errno, strerror(errno));
    }

    /* 3. Kernel log hints */
    std::fprintf(stderr, "  To check kernel TCM state, run on board:\n");
    std::fprintf(stderr, "    sudo dmesg | grep -i tcm | tail -20\n");
    std::fprintf(stderr, "    sudo fuser /dev/tcm\n");
    std::fprintf(stderr, "    ls -la /dev/tcm\n");
    std::fprintf(stderr, "    cat /sys/kernel/debug/tcm/stats 2>/dev/null || echo 'debugfs not mounted'\n");
    std::fprintf(stderr, "── End TCM Diagnostic ──\n");
    std::fflush(stderr);
}

/*
 * ── Fork-based EP probe with stderr capture ──
 *
 * Forks a child process that attempts full EP session creation.
 * Captures child's stderr via pipe so we can log the actual ORT
 * error message when CreateSession fails.
 *
 * Returns: 0=EP functional, 1=EP not functional
 * On failure, child_stderr (if provided) is populated with the
 * child's error output.
 */
extern "C" int spacemit_ort_probe_ep_ex(
    const char* model_path,
    char* child_stderr_out,
    size_t child_stderr_size)
{
    if (child_stderr_out && child_stderr_size > 0) child_stderr_out[0] = '\0';

    if (spacemit_ort_check_tcm_available() != 1) {
        std::fprintf(stderr, "SpacemiT EP probe: /dev/tcm not found, EP unavailable\n");
        return 1;
    }

    if (model_path && model_path[0]) {
        if (spacemit_ort_check_model_accessible(model_path) != 0) {
            std::fprintf(stderr, "SpacemiT EP probe: model not accessible: %s\n", model_path);
            return 1;
        }
        char chk_err[512] = {0};
        if (spacemit_ort_check_model_supported(model_path, chk_err, sizeof(chk_err)) != 0) {
            std::fprintf(stderr, "SpacemiT EP probe: model not quantized: %s\n", chk_err);
            return 1;
        }
    }

    ensure_terminate_installed();

    int err_pipe[2];
    if (pipe(err_pipe) != 0) {
        std::fprintf(stderr, "SpacemiT EP probe: pipe() failed\n");
        return 1;
    }

    std::fprintf(stderr, "SpacemiT EP probe: forking child for %s...\n",
                 model_path ? model_path : "init-check");

    pid_t pid = fork();
    if (pid < 0) {
        close(err_pipe[0]); close(err_pipe[1]);
        std::fprintf(stderr, "SpacemiT EP probe: fork() failed\n");
        return 1;
    }

    if (pid == 0) {
        /* ── Child ── */
        close(err_pipe[0]);
        dup2(err_pipe[1], STDERR_FILENO);
        close(err_pipe[1]);

        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!api) {
            std::fprintf(stderr, "FATAL: OrtGetApiBase()->GetApi() returned NULL\n");
            _exit(10);
        }

        OrtEnv* child_env = nullptr;
        OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "EPProbe", &child_env);
        if (st) {
            std::fprintf(stderr, "CreateEnv: %s\n", api->GetErrorMessage(st));
            api->ReleaseStatus(st);
            _exit(11);
        }

        OrtSessionOptions* opts = nullptr;
        st = api->CreateSessionOptions(&opts);
        if (st) {
            std::fprintf(stderr, "CreateSessionOptions: %s\n", api->GetErrorMessage(st));
            api->ReleaseStatus(st);
            api->ReleaseEnv(child_env);
            _exit(12);
        }

        OrtStatus* s_gol = api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
        if (s_gol) api->ReleaseStatus(s_gol);
        OrtStatus* s_intra = api->SetIntraOpNumThreads(opts, 1);
        if (s_intra) api->ReleaseStatus(s_intra);
        OrtStatus* s_inter = api->SetInterOpNumThreads(opts, 1);
        if (s_inter) api->ReleaseStatus(s_inter);

        try {
            Ort::SessionOptions cpp_opts(opts);
            std::unordered_map<std::string, std::string> probe_provider_opts;
            probe_provider_opts["SPACEMIT_EP_INTRA_THREAD_NUM"] = "1";
            Ort::Status ep_status = Ort::SessionOptionsSpaceMITEnvInit(cpp_opts, probe_provider_opts);
            cpp_opts.release();
            if (!ep_status.IsOK()) {
                std::string emsg = ep_status.GetErrorMessage();
                std::fprintf(stderr, "SessionOptionsSpaceMITEnvInit: %s\n",
                             emsg.c_str());
                api->ReleaseSessionOptions(opts);
                api->ReleaseEnv(child_env);
                _exit(13);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "SessionOptionsSpaceMITEnvInit threw: %s\n", e.what());
            api->ReleaseSessionOptions(opts);
            api->ReleaseEnv(child_env);
            _exit(14);
        } catch (...) {
            std::fprintf(stderr, "SessionOptionsSpaceMITEnvInit threw unknown exception\n");
            api->ReleaseSessionOptions(opts);
            api->ReleaseEnv(child_env);
            _exit(14);
        }

        if (!model_path || !model_path[0]) {
            api->ReleaseSessionOptions(opts);
            api->ReleaseEnv(child_env);
            _exit(0);
        }

        OrtSession* session = nullptr;
        st = api->CreateSession(child_env, model_path, opts, &session);
        if (st) {
            std::fprintf(stderr, "CreateSession: %s\n", api->GetErrorMessage(st));
            api->ReleaseStatus(st);
            api->ReleaseSessionOptions(opts);
            api->ReleaseEnv(child_env);
            _exit(15);
        }

        /*
         * EP session created — now do a warmup Run() to fully exercise
         * the IME/TCM hardware.  CreateSession alone may leave the EP in
         * a state that crashes the next process to use it; a full Run()
         * pushes the hardware through a complete init→compute→teardown
         * cycle, leaving it clean for the parent.
         */
        {
            /* Query input count and shape */
            size_t n_inputs = 0;
            OrtStatus* s_cnt = api->SessionGetInputCount(session, &n_inputs);
            if (!s_cnt && n_inputs > 0) {
                OrtAllocator* alloc = nullptr;
                if (!api->GetAllocatorWithDefaultOptions(&alloc) && alloc) {
                    char* iname = nullptr;
                    if (!api->SessionGetInputName(session, 0, alloc, &iname) && iname) {
                        OrtTypeInfo* type_info = nullptr;
                        if (!api->SessionGetInputTypeInfo(session, 0, &type_info) && type_info) {
                            const OrtTensorTypeAndShapeInfo* tsi = nullptr;
                            { OrtStatus* _s = api->CastTypeInfoToTensorInfo(type_info, &tsi); if (_s) api->ReleaseStatus(_s); }
                            if (tsi) {
                                size_t nd = 0;
                                { OrtStatus* _s = api->GetDimensionsCount(tsi, &nd); if (_s) api->ReleaseStatus(_s); }
                                int64_t dims[4] = {1, 3, 320, 320};
                                { OrtStatus* _s = api->GetDimensions(tsi, dims, nd); if (_s) api->ReleaseStatus(_s); }
                                size_t n_elems = 1;
                                for (size_t di = 0; di < nd && di < 4; di++)
                                    if (dims[di] > 0) n_elems *= (size_t)dims[di];

                                /* Allocate zero input and run warmup inference */
                                size_t n_bytes = n_elems * sizeof(float);
                                void* warmup_data = std::calloc(1, n_bytes);
                                if (warmup_data) {
                                    OrtMemoryInfo* mi = nullptr;
                                    if (!api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi)) {
                                        OrtValue* in_val = nullptr;
                                        if (!api->CreateTensorWithDataAsOrtValue(
                                                mi, warmup_data, n_bytes, dims,
                                                (nd < 2) ? 2 : (int)nd,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                &in_val) && in_val) {
                                            const char* in_names[]  = {iname};
                                            const char* out_names[] = {nullptr};
                                            OrtValue*   in_vals[]   = {in_val};
                                            OrtValue*   out_val     = nullptr;

                                            char* oname = nullptr;
                                            if (!api->SessionGetOutputName(session, 0, alloc, &oname) && oname) {
                                                out_names[0] = oname;
                                            } else {
                                                out_names[0] = "output0";
                                            }

                                            OrtStatus* _s = api->Run(session, nullptr,
                                                     (const char* const*)in_names,
                                                     (const OrtValue* const*)in_vals, 1,
                                                     (const char* const*)out_names, 1,
                                                     &out_val);
                                            if (_s) api->ReleaseStatus(_s);
                                            if (out_val) api->ReleaseValue(out_val);
                                            api->ReleaseValue(in_val);
                                        }
                                        api->ReleaseMemoryInfo(mi);
                                    }
                                    std::free(warmup_data);
                                }
                            }
                            api->ReleaseTypeInfo(type_info);
                        }
                        { OrtStatus* _s = api->AllocatorFree(alloc, (void*)iname); if (_s) api->ReleaseStatus(_s); }
                    }
                    /* allocator not released — owned by ORT */
                }
            }
            if (s_cnt) api->ReleaseStatus(s_cnt);
        }

        api->ReleaseSession(session);
        api->ReleaseSessionOptions(opts);
        api->ReleaseEnv(child_env);
        _exit(0);
    }

    /* ── Parent ── */
    close(err_pipe[1]);

    /* Set pipe read end to non-blocking */
    int pipe_flags = fcntl(err_pipe[0], F_GETFL, 0);
    if (pipe_flags >= 0) fcntl(err_pipe[0], F_SETFL, pipe_flags | O_NONBLOCK);

    int wstatus;
    int waited = 0;
    const int PROBE_TIMEOUT_MS = 20000;
    bool child_done = false;

    while (waited < PROBE_TIMEOUT_MS) {
        pid_t ret = waitpid(pid, &wstatus, WNOHANG);
        if (ret > 0) { child_done = true; break; }
        if (ret < 0) {
            std::fprintf(stderr, "SpacemiT EP probe: waitpid error (errno=%d)\n", errno);
            close(err_pipe[0]);
            return 1;
        }
        usleep(100000);
        waited += 100;
    }

    if (!child_done) {
        std::fprintf(stderr, "SpacemiT EP probe: child timed out after %d ms, killing\n", PROBE_TIMEOUT_MS);
        kill(pid, SIGKILL);
        waitpid(pid, &wstatus, 0);
        close(err_pipe[0]);
        return 1;
    }

    /* Drain child's stderr from pipe */
    if (child_stderr_out && child_stderr_size > 0) {
        child_stderr_out[0] = '\0';
        /* Small delay to let pipe buffer flush */
        usleep(50000);
        ssize_t nr = read(err_pipe[0], child_stderr_out, (ssize_t)(child_stderr_size - 1));
        if (nr > 0) {
            child_stderr_out[nr] = '\0';
            /* Strip trailing newlines */
            char* end = child_stderr_out + nr - 1;
            while (end >= child_stderr_out && (*end == '\n' || *end == '\r'))
                *end-- = '\0';
        }
    } else {
        /* Still drain to prevent SIGPIPE issues */
        char drain[256];
        usleep(50000);
        while (read(err_pipe[0], drain, sizeof(drain)) > 0) {}
    }
    close(err_pipe[0]);

    if (WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0) {
        std::fprintf(stderr, "SpacemiT EP probe: SUCCESS — EP functional for %s\n",
                     model_path ? model_path : "init-check");
        /* Let kernel fully release child's /dev/tcm mmap */
        usleep(300000);
        return 0;
    }

    int exit_code = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1;
    int sig = WIFSIGNALED(wstatus) ? WTERMSIG(wstatus) : 0;
    std::fprintf(stderr, "SpacemiT EP probe: FAILED (exit=%d, sig=%d) for %s\n",
                 exit_code, sig, model_path ? model_path : "init-check");
    if (child_stderr_out && child_stderr_out[0]) {
        std::fprintf(stderr, "SpacemiT EP probe: child stderr: %s\n", child_stderr_out);
    }
    return 1;
}

/*
 * ── Simple probe (backward-compatible wrapper) ──
 * No longer caches result globally — each call probes fresh.
 */
extern "C" int spacemit_ort_probe_ep(const char* model_path) {
    char err_buf[4096];
    return spacemit_ort_probe_ep_ex(model_path, err_buf, sizeof(err_buf));
}

/* ── Per-EP-session intra-op thread count ── */
static int g_ep_intra_threads = 1;

extern "C" void spacemit_ort_set_ep_intra_threads(int n) {
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    g_ep_intra_threads = n;
}

extern "C" int spacemit_ort_get_ep_intra_threads(void) {
    return g_ep_intra_threads;
}

extern "C" int spacemit_ort_check_model_supported(
    const char* model_path,
    char* err_buf,
    size_t err_buf_size) {
    if (!model_path) return -1;

    auto write_err = [&](const char* msg) {
        if (err_buf && err_buf_size > 0) {
            std::snprintf(err_buf, err_buf_size, "%s", msg ? msg : "unknown");
        }
    };

    {
        const char* base = std::strrchr(model_path, '/');
        if (!base) base = std::strrchr(model_path, '\\');
        const char* name = base ? base + 1 : model_path;
        std::string lower(name);
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find(".q.onnx") != std::string::npos ||
            lower.find("_quant.onnx") != std::string::npos ||
            lower.find("_quantized.onnx") != std::string::npos ||
            lower.find(".int8.onnx") != std::string::npos) {
            return 0;
        }
    }

    std::FILE* f = std::fopen(model_path, "rb");
    if (!f) {
        write_err("cannot open model file");
        return -2;
    }
    constexpr size_t SCAN_BYTES = 4 * 1024 * 1024;
    std::vector<char> buf(SCAN_BYTES);
    size_t n = std::fread(buf.data(), 1, SCAN_BYTES, f);
    std::fclose(f);

    auto contains = [&](const char* needle) -> bool {
        size_t nlen = std::strlen(needle);
        if (n < nlen) return false;
        for (size_t i = 0; i + nlen <= n; i++) {
            if (std::memcmp(buf.data() + i, needle, nlen) == 0) return true;
        }
        return false;
    };

    if (contains("QuantizeLinear") || contains("DequantizeLinear") ||
        contains("QLinearConv") || contains("QLinearMatMul") || contains("QLinearAdd")) {
        return 0;
    }

    char msg[512];
    std::snprintf(msg, sizeof(msg),
        "model appears to be FP32 (no QuantizeLinear/QLinearConv ops in first %zu bytes). "
        "SpacemiT EP requires INT8-quantized models. Quantize with xquant: "
        "https://bianbu.spacemit.com/en/brdk/Advanced_development/7.1_Model_Quantization/ "
        "or use a prebuilt .q.onnx from https://gitee.com/bianbu/spacemit-demo",
        n);
    write_err(msg);
    return -3;
}

extern "C" int spacemit_ort_session_options_init(OrtSessionOptions* c_session_opts) {
    if (!c_session_opts) return -1;

    ensure_terminate_installed();

    if (access("/dev/tcm", F_OK) != 0) {
        return -3;
    }

    try {
        Ort::SessionOptions opts(c_session_opts);
        std::unordered_map<std::string, std::string> provider_options;
        provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(g_ep_intra_threads);
        Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(opts, provider_options);
        opts.release();
        if (status.IsOK()) {
            return 0;
        }
        return -4;
    } catch (const Ort::Exception&) {
        return -2;
    } catch (const std::exception&) {
        return -2;
    } catch (...) {
        return -2;
    }
}

extern "C" int spacemit_ort_create_session_safe(
    OrtEnv* env,
    const char* model_path,
    OrtSessionOptions* opts,
    OrtSession** out_session,
    char* err_buf,
    size_t err_buf_size) {
    if (!env || !model_path || !opts || !out_session) return -1;
    *out_session = nullptr;

    ensure_terminate_installed();

    auto write_err = [&](const char* msg) {
        if (err_buf && err_buf_size > 0) {
            std::snprintf(err_buf, err_buf_size, "%s", msg ? msg : "unknown");
        }
    };

    try {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!api) {
            write_err("OrtApi unavailable");
            return -2;
        }
        OrtStatus* status = api->CreateSession(env, model_path, opts, out_session);
        if (status) {
            write_err(api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            *out_session = nullptr;
            return -3;
        }
        return 0;
    } catch (const Ort::Exception& e) {
        write_err(e.what());
        *out_session = nullptr;
        return -4;
    } catch (const std::exception& e) {
        write_err(e.what());
        *out_session = nullptr;
        return -4;
    } catch (...) {
        write_err("unknown C++ exception in CreateSession");
        *out_session = nullptr;
        return -5;
    }
}

#else
/* ── Stubs when HAS_SPACEMIT_EP is not defined ── */
extern "C" int spacemit_ort_check_model_accessible(const char* model_path) {
    (void)model_path;
    return -1;
}

extern "C" int spacemit_ort_check_tcm_available(void) {
    return 0;
}

extern "C" int spacemit_ort_probe_ep_ex(
    const char* model_path,
    char* child_stderr_out,
    size_t child_stderr_size) {
    (void)model_path;
    if (child_stderr_out && child_stderr_size > 0) child_stderr_out[0] = '\0';
    return 1;
}

extern "C" int spacemit_ort_probe_ep(const char* model_path) {
    (void)model_path;
    return 1;
}

extern "C" int spacemit_ort_session_options_init(OrtSessionOptions* c_session_opts) {
    (void)c_session_opts;
    return -1;
}

extern "C" int spacemit_ort_check_model_supported(
    const char* model_path,
    char* err_buf,
    size_t err_buf_size) {
    (void)model_path; (void)err_buf; (void)err_buf_size;
    return -1;
}

extern "C" int spacemit_ort_create_session_safe(
    OrtEnv* env,
    const char* model_path,
    OrtSessionOptions* opts,
    OrtSession** out_session,
    char* err_buf,
    size_t err_buf_size) {
    (void)env; (void)model_path; (void)opts;
    (void)err_buf; (void)err_buf_size;
    if (out_session) *out_session = nullptr;
    return -2;
}

extern "C" void spacemit_ort_set_ep_intra_threads(int n) {
    (void)n;
}

extern "C" int spacemit_ort_get_ep_intra_threads(void) {
    return 1;
}

extern "C" void spacemit_ort_dump_tcm_diagnostics(const char* model_path) {
    (void)model_path;
}
#endif
