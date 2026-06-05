#ifndef AI_ACCEL_ADAPTER_H
#define AI_ACCEL_ADAPTER_H

#include "core_types.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AI_MODEL_YOLO11N,
    AI_MODEL_YOLOV8_POSE,
    AI_MODEL_STGCN,
    AI_MODEL_YOLOV5_FACE,
    AI_MODEL_ARCFACE,
    AI_MODEL_COUNT
} AIModelType;

typedef enum {
    AI_ACCEL_OK = 0,
    AI_ACCEL_ERROR_INIT = -1,
    AI_ACCEL_ERROR_LOAD = -2,
    AI_ACCEL_ERROR_INFER = -3,
    AI_ACCEL_ERROR_MEMORY = -4,
    AI_ACCEL_ERROR_TIMEOUT = -5,
} AIAcclStatus;

typedef struct AIAcclContext AIAcclContext;
typedef AIAcclContext AIAccelContext;

AIAcclContext* ai_accel_context_create(void);
void ai_accel_context_destroy(AIAcclContext* ctx);

bool ai_accel_is_available(void);
bool ai_accel_is_hw_active(const AIAcclContext* ctx);
const char* ai_accel_describe(const AIAcclContext* ctx);

AIAcclStatus ai_accel_load_model(AIAcclContext* ctx, AIModelType type,
                                 const char* model_path,
                                 int input_w, int input_h);

void ai_accel_warmup(AIAcclContext* ctx, AIModelType type, int input_w, int input_h);

const char* ai_accel_status_string(AIAcclStatus status);

#ifdef __cplusplus
}
#endif

#endif
