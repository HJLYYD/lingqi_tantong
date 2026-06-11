#include "model_store.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static const struct {
    const char* name;
    const char* path;
} MODEL_REGISTRY[] = {
    {"yolo11n_person", "Human Recognition/yolo11n.q.onnx"},
    {"yolov8n_pose", "Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx"},
    {"stgcn_action", "Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx"},
    {"yolov5_face", "Face Recognition/yolov5n-face_cut.q.onnx"},
    {"arcface", "Face Recognition/arcface_mobilefacenet_cut.q.onnx"},
    {NULL, NULL}
};

static bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static size_t get_file_size_mb(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return st.st_size / (1024 * 1024);
    }
    return 0;
}

ModelStore* model_store_create(const char* base_path) {
    ModelStore* store = (ModelStore*)calloc(1, sizeof(ModelStore));
    if (!store) return NULL;

    if (base_path) {
        strncpy(store->base_path, base_path, MAX_PATH_LEN - 1);
    } else {
        strncpy(store->base_path, "models", MAX_PATH_LEN - 1);
    }
    store->base_path[MAX_PATH_LEN - 1] = '\0';

    for (int i = 0; MODEL_REGISTRY[i].name != NULL && store->num_models < MAX_MODELS; i++) {
        ModelInfo* info = &store->models[store->num_models++];
        strncpy(info->name, MODEL_REGISTRY[i].name, MAX_MODEL_NAME_LEN - 1);
        info->name[MAX_MODEL_NAME_LEN - 1] = '\0';

        char full_path[MAX_PATH_LEN * 4];
        int n = snprintf(full_path, sizeof(full_path), "%s/%s",
                         store->base_path, MODEL_REGISTRY[i].path);
        if (n < 0 || n >= (int)sizeof(full_path)) {
            /* Path too long — truncation would corrupt; skip this model */
            log_warning("Model path overflow for %s, skipping",
                        MODEL_REGISTRY[i].name);
            store->num_models--;
            continue;
        }
        strncpy(info->path, full_path, MAX_MODEL_PATH_LEN - 1);
        info->path[MAX_MODEL_PATH_LEN - 1] = '\0';

        info->exists = file_exists(full_path);
        info->size_mb = get_file_size_mb(full_path);
        info->session = NULL;
        info->loaded = false;
    }

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

const char* model_store_get_path(const ModelStore* store, const char* model_name) {
    return model_store_resolve_path(store, model_name);
}

bool model_store_validate(const ModelStore* store, const char* model_name) {
    const char* path = model_store_resolve_path(store, model_name);
    return path != NULL && file_exists(path);
}

#ifdef HAS_ONNX_RUNTIME
#include "ort_common.h"
#endif

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

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_warning("ModelStore: ONNX Runtime not available, cannot load %s", model_name);
        return NULL;
    }

    /*
     * Delegate to ort_create_session() so both load paths share the
     * same per-model fork-probe / EP-gating / CPU-fallback logic.
     * Avoids drift between two near-identical session creation flows.
     */
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
#else
    log_info("ModelStore: ONNX loading for %s (build with HAS_ONNX_RUNTIME)", model_name);
    return NULL;
#endif
}

void model_store_clear_cache(ModelStore* store) {
#ifdef HAS_ONNX_RUNTIME
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
#else
    for (int i = 0; i < store->num_models; i++) {
        store->models[i].session = NULL;
        store->models[i].loaded = false;
    }
#endif
    log_info("Model cache cleared");
}

void model_store_list_available(const ModelStore* store, char* out_names, int max_names, int name_len) {
    int count = 0;
    for (int i = 0; i < store->num_models && count < max_names; i++) {
        strncpy(out_names + count * name_len, store->models[i].name, name_len - 1);
        out_names[count * name_len + name_len - 1] = '\0';
        count++;
    }
}
