#ifndef MODEL_STORE_H
#define MODEL_STORE_H

#include "core_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MODEL_NAME_LEN      64
#define MAX_MODEL_PATH_LEN      256
#define MAX_MODELS              16

typedef struct OrtSession OrtSession;

typedef struct {
    char name[MAX_MODEL_NAME_LEN];
    char path[MAX_MODEL_PATH_LEN];
    bool exists;
    size_t size_mb;
    OrtSession* session;
    bool loaded;
} ModelInfo;

typedef struct {
    char base_path[MAX_PATH_LEN];
    ModelInfo models[MAX_MODELS];
    int num_models;
} ModelStore;

ModelStore* model_store_create(const char* base_path);
void model_store_destroy(ModelStore* store);

const char* model_store_resolve_path(const ModelStore* store, const char* model_name);
bool model_store_validate(const ModelStore* store, const char* model_name);

OrtSession* model_store_load_onnx(ModelStore* store, const char* model_name);
void model_store_clear_cache(ModelStore* store);

#ifdef __cplusplus
}
#endif

#endif
