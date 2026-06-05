#ifndef SPACEMIT_ORT_BRIDGE_H
#define SPACEMIT_ORT_BRIDGE_H

#include <stddef.h>

struct OrtSessionOptions;
struct OrtSession;
struct OrtEnv;

#ifdef __cplusplus
extern "C" {
#endif

int spacemit_ort_session_options_init(struct OrtSessionOptions* session_options);

/*
 * Check whether a model file exists and is readable.
 * Returns: 0=ok, -1=null path, -2=not readable
 */
int spacemit_ort_check_model_accessible(const char* model_path);

/*
 * Check whether /dev/tcm is present on this board.
 * Result is cached after first call.
 * Returns: 1=available, 0=not available, -1=not yet checked
 */
int spacemit_ort_check_tcm_available(void);

/*
 * Scan the ONNX model file for quantization markers (QuantizeLinear /
 * QLinearConv / DequantizeLinear op type strings stored as protobuf
 * literals). SpacemiT EP requires INT8-quantized models — feeding it
 * an FP32 model causes std::runtime_error("tcm buffer alloc failed")
 * from worker threads during CreateSession (uncatchable).
 *
 * Returns:
 *    0  model has quantization ops — safe to register SpacemiT EP
 *   -1  bad arguments
 *   -2  file not readable
 *   -3  no quantization ops found (model is FP32) — err_buf populated
 *       with a guidance message pointing at xquant / spacemit-demo
 */
int spacemit_ort_check_model_supported(
    const char* model_path,
    char* err_buf,
    size_t err_buf_size);

/*
 * Fork-based EP probe with stderr capture.
 *
 * Forks a child that attempts full EP CreateSession. Captures child's
 * stderr via pipe so the actual ORT error message is available when
 * CreateSession fails (previously exit=15 was a black box).
 *
 * This is a DIAGNOSTIC tool. The primary EP gate is the quantization
 * check (spacemit_ort_check_model_supported). In production, we try EP
 * directly after quantization check passes. The fork-probe is useful
 * for debugging "why does EP fail for this model?" without risking
 * the main process.
 *
 * Returns:
 *   0  EP is functional (child created a session successfully)
 *   1  EP is NOT functional (child crashed or returned error)
 *
 * On failure, child_stderr_out (if provided) is populated with the
 * child process's stderr output (ORT error messages).
 */
int spacemit_ort_probe_ep_ex(
    const char* model_path,
    char* child_stderr_out,
    size_t child_stderr_size);

/*
 * Simple fork-probe (backward-compatible wrapper).
 * Each call probes fresh — no global caching.
 */
int spacemit_ort_probe_ep(const char* model_path);

/*
 * Safe CreateSession wrapper. Catches any C++ exceptions thrown from inside
 * libspacemit_ep.so (which can leak past the ORT C API in some failure modes,
 * e.g. corrupt model file or invalid weights). Returns:
 *   0  success
 *  -1  bad arguments
 *  -2  ORT API unavailable
 *  -3  CreateSession returned OrtStatus error (err_buf populated)
 *  -4  std::exception caught (err_buf populated with what())
 *  -5  unknown C++ exception caught
 */
int spacemit_ort_create_session_safe(
    struct OrtEnv* env,
    const char* model_path,
    struct OrtSessionOptions* opts,
    struct OrtSession** out_session,
    char* err_buf,
    size_t err_buf_size);

/*
 * Per-EP-session intra-op thread count.
 * K1 TCM is ~512KB per AI core. Multiple EP threads can exhaust TCM
 * (SpacemiT forum #1223). Default 1 is safe; raise to 2 only for small
 * single-model workloads.
 */
void spacemit_ort_set_ep_intra_threads(int n);
int  spacemit_ort_get_ep_intra_threads(void);

/*
 * Dump detailed TCM diagnostic info to stderr. Helps diagnose
 * "tcm buffer alloc failed for core id 0" errors.
 * Call before first EP CreateSession attempt.
 */
void spacemit_ort_dump_tcm_diagnostics(const char* model_path);

#ifdef __cplusplus
}
#endif

#endif