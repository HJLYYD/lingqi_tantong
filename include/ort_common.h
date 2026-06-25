#ifndef ORT_COMMON_H
#define ORT_COMMON_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>

const OrtApi* ort_get_api(void);
OrtEnv* ort_get_env(void);
bool ort_global_init(void);
void ort_global_shutdown(void);     /* Release global ORT environment (called once at process exit) */
bool ort_spacemit_ep_active(void);

/*
 * Enable/disable SpacemiT EP registration globally. Defaults to enabled
 * when HAS_SPACEMIT_EP is compiled in. If disabled, ort_create_session()
 * skips SpacemiT EP and falls back to the CPU EP (still real ONNX
 * inference — NOT a heuristic fallback).
 *
 * Why this exists: libspacemit_ep.so can throw std::runtime_error from
 * worker threads (e.g. "tcm buffer alloc failed for core id N") during
 * CreateSession. Those exceptions cannot be caught across the C ABI by
 * the calling thread and cause std::terminate -> abort. Until the EP is
 * stable on this board, users can toggle it off via config
 * `system.use_spacemit_ep`.
 */
void ort_set_ep_enabled(bool enabled);
bool ort_get_ep_enabled(void);

int ort_validate_onnx_file(const char* model_path, size_t* out_file_size);

OrtSession* ort_create_session(const char* model_path, int num_threads, bool use_ep);

/*
 * Query the model's first input tensor shape. Writes up to `max_dims`
 * dimension values into out_dims and returns the actual rank (number of
 * dims), or -1 on error. Useful for detectors that need to know whether
 * a quantized model was exported at 320x320 vs 640x640.
 *
 * Example for YOLOv8: shape [1,3,H,W] → rank=4, out_dims={1,3,H,W}.
 */
int ort_get_input_shape(OrtSession* session, int* out_dims, int max_dims);

#endif

#ifdef __cplusplus
}
#endif

#endif