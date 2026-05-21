#ifndef AI_ACCEL_ADAPTER_H
#define AI_ACCEL_ADAPTER_H

#include "core_types.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI_ACCEL_MAX_INPUT_TENSORS   4
#define AI_ACCEL_MAX_OUTPUT_TENSORS  8
#define AI_ACCEL_MAX_MODEL_PATH      256

typedef enum {
    AI_MODEL_YOLOV8N,
    AI_MODEL_YOLOV8_POSE,
    AI_MODEL_SCRFD,
    AI_MODEL_ARCFACE,
    AI_MODEL_MIDAS,
    AI_MODEL_COUNT
} AIModelType;

typedef enum {
    AI_ACCEL_OK = 0,
    AI_ACCEL_ERROR_INIT = -1,
    AI_ACCEL_ERROR_LOAD = -2,
    AI_ACCEL_ERROR_INFER = -3,
    AI_ACCEL_ERROR_MEMORY = -4,
    AI_ACCEL_ERROR_TIMEOUT = -5,
} AIAccelStatus;

typedef struct {
    float* data;
    int dims[4];
    int ndim;
    size_t size_bytes;
    int data_type;
} AIAccelTensor;

typedef struct AIAccelContext AIAccelContext;

AIAcclContext* ai_accel_context_create(void);
void ai_accel_context_destroy(AIAcclContext* ctx);

AIAcclStatus ai_accel_load_model(AIAcclContext* ctx, AIModelType type,
                                 const char* model_path,
                                 int input_w, int input_h);

AIAcclStatus ai_accel_infer(AIAcclContext* ctx, AIModelType type,
                            const float* input_data,
                            AIAccelTensor** outputs, int* num_outputs);

AIAcclStatus ai_accel_infer_uint8(AIAcclContext* ctx, AIModelType type,
                                  const uint8_t* input_data,
                                  AIAccelTensor** outputs, int* num_outputs);

void ai_accel_release_outputs(AIAcclContext* ctx, AIAccelTensor* outputs, int num_outputs);

const char* ai_accel_status_string(AIAcclStatus status);

bool ai_accel_is_available(void);

void ai_accel_warmup(AIAcclContext* ctx, AIModelType type, int input_w, int input_h);

#ifdef __cplusplus
}
#endif

#endif