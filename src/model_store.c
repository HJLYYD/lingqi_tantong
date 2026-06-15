#include "model_store.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>


static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

ModelStore* model_store_create(const char* base_path) {
    ModelStore* store = (ModelStore*)calloc(1, sizeof(ModelStore));
    if (!store) return NULL;

    if (base_path && base_path[0]) {
        strncpy(store->base_path, base_path, MAX_PATH_LEN - 1);
    } else {
        strncpy(store->base_path, "models", MAX_PATH_LEN - 1);
    }
    store->base_path[MAX_PATH_LEN - 1] = '\0';
    store->num_models = 0;

    if (!file_exists(store->base_path)) {
        log_warning("Model directory not found: %s", store->base_path);
    }

    return store;
}

void model_store_destroy(ModelStore* store) {
    if (!store) return;
    model_store_clear_cache(store);
    free(store);
}

const char* model_store_resolve_path(const ModelStore* store, const char* model_name) {
    for (int i = 0; i < store->num_models; i++) {
        if (strcmp(store->models[i].name, model_name) == 0) {
            if (store->models[i].exists) {
                return store->models[i].path;
            }
            log_warning("Model file not found: %s", store->models[i].path);
            return NULL;
        }
    }
    log_error("Unknown model: %s", model_name);
    return NULL;
}

bool model_store_validate(const ModelStore* store, const char* model_name) {
    const char* path = model_store_resolve_path(store, model_name);
    return path != NULL && file_exists(path);
}

#include "ort_common.h"

OrtSession* model_store_load_onnx(ModelStore* store, const char* model_name) {
    if (!store || !model_name) return NULL;

    const char* model_path = model_store_resolve_path(store, model_name);
    if (!model_path) return NULL;

    for (int i = 0; i < store->num_models; i++) {
        if (strcmp(store->models[i].name, model_name) == 0) {
            if (store->models[i].session != NULL) {
                log_debug("ModelStore: returning cached session for %s", model_name);
                return store->models[i].session;
            }
            break;
        }
    }

    if (!ort_global_init()) {
        log_warning("ModelStore: ONNX Runtime not available, cannot load %s", model_name);
        return NULL;
    }
    OrtSession* session = ort_create_session(model_path, 4, true);
    if (!session) {
        log_warning("ModelStore: ort_create_session failed for %s", model_name);
        return NULL;
    }
    for (int i = 0; i < store->num_models; i++) {
        if (strcmp(store->models[i].name, model_name) == 0) {
            store->models[i].session = session;
            store->models[i].loaded = true;
            break;
        }
    }
    log_info("ModelStore: loaded ONNX model %s from %s", model_name, model_path);
    return session;
}

void model_store_clear_cache(ModelStore* store) {
    const OrtApi* ort = ort_get_api();
    if (ort) {
        for (int i = 0; i < store->num_models; i++) {
            if (store->models[i].session != NULL) {
                ort->ReleaseSession(store->models[i].session);
                store->models[i].session = NULL;
            }
            store->models[i].loaded = false;
        }
    } else {
        for (int i = 0; i < store->num_models; i++) {
            store->models[i].session = NULL;
            store->models[i].loaded = false;
        }
    }
    log_info("Model cache cleared");
}

