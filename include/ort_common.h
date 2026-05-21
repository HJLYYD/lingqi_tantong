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
void ort_global_shutdown(void);

int ort_validate_onnx_file(const char* model_path, size_t* out_file_size);

OrtSession* ort_create_session(const char* model_path, int num_threads, bool use_ep);

size_t ort_get_model_input_size(OrtSession* session);

#endif

#ifdef __cplusplus
}
#endif

#endif