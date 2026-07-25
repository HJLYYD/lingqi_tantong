#include "arcface_recognizer.h"
#include "logger.h"
#include "utils.h"
#include "ort_inference_context.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include "ort_common.h"
#else
#error "arcface_recognizer requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

#define ARCFACE_MAX_INPUT_BYTES (16UL * 1024UL * 1024UL)

ArcFaceRecognizer* arcface_recognizer_create(const char* model_path, int input_w, int input_h,
                                              float sim_thresh) {
    ArcFaceRecognizer* rec = (ArcFaceRecognizer*)calloc(1, sizeof(ArcFaceRecognizer));
    if (!rec) return NULL;

    rec->input_width = input_w > 0 ? input_w : 112;
    rec->input_height = input_h > 0 ? input_h : 112;
    rec->similarity_threshold = sim_thresh;
    rec->num_entries = 0;

    if (!model_path || !arcface_recognizer_load_model(rec, model_path)) {
        log_error("ArcFaceRecognizer: failed to load model %s", model_path ? model_path : "(null)");
        free(rec);
        return NULL;
    }

    return rec;
}

void arcface_recognizer_destroy(ArcFaceRecognizer* rec) {
    if (!rec) return;
    if (rec->ctx) {
        ort_ctx_destroy(rec->ctx);
    }
    if (rec->session) {
        ort_release_session(rec->session);
    }
    free(rec);
}

bool arcface_recognizer_load_model(ArcFaceRecognizer* rec, const char* model_path) {
    if (!rec || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    strncpy(rec->input_name, "data", sizeof(rec->input_name) - 1);  /* default fallback */

    if (!ort_global_init()) {
        log_error("ArcFace: ORT runtime not initialized");
        return false;
    }

    rec->session = ort_create_session(model_path, 2, true);
    if (!rec->session) {
        log_error("ArcFace: ONNX session creation failed for %s", model_path);
        return false;
    }

    /* ── Query real input name from model ── */
    {
        const OrtApi* ort = ort_get_api();
        OrtAllocator* allocator = NULL;
        OrtStatus* st_a = ort->GetAllocatorWithDefaultOptions(&allocator);
        if (st_a) ort->ReleaseStatus(st_a);
        if (allocator) {
            char* real_name = NULL;
            OrtStatus* s = ort->SessionGetInputName(rec->session, 0, allocator, &real_name);
            if (s == NULL && real_name) {
                strncpy(rec->input_name, real_name, sizeof(rec->input_name) - 1);
                rec->input_name[sizeof(rec->input_name) - 1] = '\0';
                OrtStatus* sf = ort->AllocatorFree(allocator, real_name);
                if (sf) ort->ReleaseStatus(sf);
            } else {
                if (s) ort->ReleaseStatus(s);
            }
        }
    }

    int dims[8] = {0};
    int rank = ort_get_input_shape(rec->session, dims, 8);
    if (rank == 4 && dims[2] > 0 && dims[3] > 0) {
        if (rec->input_width != dims[3] || rec->input_height != dims[2]) {
            log_info("ArcFace: overriding requested %dx%d with model's actual input %dx%d",
                     rec->input_width, rec->input_height, dims[3], dims[2]);
        }
        rec->input_width = dims[3];
        rec->input_height = dims[2];
    }

    rec->ctx = ort_ctx_create(rec->session, rec->input_width, rec->input_height, 3);
    if (!rec->ctx) {
        log_error("ArcFace: failed to create inference context");
        ort_release_session(rec->session);
        rec->session = NULL;
        return false;
    }
    strncpy(rec->ctx->input_name, rec->input_name, sizeof(rec->ctx->input_name) - 1);

    log_info("ArcFace model loaded: %s (%.2f MB) input=%dx%d",
             model_path, file_size / (1024.0f * 1024.0f), rec->input_width, rec->input_height);
    return true;
}

static void preprocess_arcface(const uint8_t* face_image, int width, int height,
                               float* out_tensor, int target_w, int target_h) {
    uint8_t* resized = (uint8_t*)malloc((size_t)target_w * target_h * 3);
    if (!resized) return;

    utils_resize_image(face_image, width, height, resized, target_w, target_h, 3);

    int pixels = target_w * target_h;
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_idx = (y * target_w + x) * 3;
            int dst_r = 0 * pixels + y * target_w + x;
            int dst_g = 1 * pixels + y * target_w + x;
            int dst_b = 2 * pixels + y * target_w + x;

            out_tensor[dst_r] = (resized[src_idx + 0] - 127.5f) / 127.5f;
            out_tensor[dst_g] = (resized[src_idx + 1] - 127.5f) / 127.5f;
            out_tensor[dst_b] = (resized[src_idx + 2] - 127.5f) / 127.5f;
        }
    }

    free(resized);
}

int arcface_recognizer_extract_feature(ArcFaceRecognizer* rec, const uint8_t* face_image, int width, int height,
                                        float* out_feature) {
    if (!rec || !face_image || !out_feature || !rec->session || !rec->ctx) return -1;
    if (width <= 0 || height <= 0) return -1;

    int input_w = rec->input_width > 0 ? rec->input_width : 112;
    int input_h = rec->input_height > 0 ? rec->input_height : 112;
    size_t pixels = (size_t)input_w * (size_t)input_h;
    size_t input_size = pixels * 3 * sizeof(float);
    if (input_size == 0 || input_size > ARCFACE_MAX_INPUT_BYTES) {
        log_error("ArcFace: refused unreasonable input tensor size %zu bytes", input_size);
        return -1;
    }

    float* input_tensor = (float*)malloc(input_size);
    if (!input_tensor) return -1;

    preprocess_arcface(face_image, width, height, input_tensor, input_w, input_h);

    if (!ort_ctx_prepare_input(rec->ctx, input_tensor, input_size)) {
        free(input_tensor);
        return -1;
    }
    free(input_tensor);

    OrtValue* output_values[1] = {NULL};
    if (ort_ctx_run(rec->ctx, output_values) != 0) {
        return -1;
    }

    float* output_data = ort_ctx_get_output_data(rec->ctx, output_values[0]);
    if (!output_data) {
        ort_ctx_release_outputs(rec->ctx, output_values, 1);
        return -1;
    }

    int feat_dim = ARCFACE_FEATURE_DIM;

    float norm = 0.0f;
    for (int i = 0; i < feat_dim; i++) {
        out_feature[i] = output_data[i];
        norm += out_feature[i] * out_feature[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < feat_dim; i++) {
            out_feature[i] *= inv_norm;
        }
    }

    ort_ctx_release_outputs(rec->ctx, output_values, 1);
    return 0;
}

bool arcface_recognizer_register_face(ArcFaceRecognizer* rec, const char* identity,
                                       const uint8_t* face_image, int width, int height) {
    if (!rec || !identity || !face_image) return false;
    if (rec->num_entries >= ARCFACE_MAX_FACES) {
        log_error("Face database full");
        return false;
    }

    FaceDatabaseEntry* entry = &rec->database[rec->num_entries++];
    strncpy(entry->identity, identity, MAX_STRING_LEN - 1);
    entry->identity[MAX_STRING_LEN - 1] = '\0';
    entry->active = true;

    if (arcface_recognizer_extract_feature(rec, face_image, width, height, entry->feature) != 0) {
        rec->num_entries--;
        return false;
    }

    log_info("Registered face identity: %s", identity);
    return true;
}

FaceIdentity arcface_recognizer_recognize(ArcFaceRecognizer* rec, const uint8_t* face_image, int width, int height) {
    FaceIdentity result;
    memset(&result, 0, sizeof(FaceIdentity));
    strncpy(result.identity, "unknown", MAX_STRING_LEN - 1);
    result.identity[MAX_STRING_LEN - 1] = '\0';
    result.confidence = 0.0f;
    result.similarity = -1.0f;

    if (!rec || !face_image) return result;

    float feature[ARCFACE_FEATURE_DIM];
    if (arcface_recognizer_extract_feature(rec, face_image, width, height, feature) != 0) {
        return result;
    }

    float max_similarity = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < rec->num_entries; i++) {
        if (!rec->database[i].active) continue;

        float sim = arcface_calculate_similarity(feature, rec->database[i].feature, ARCFACE_FEATURE_DIM);
        if (sim > max_similarity) {
            max_similarity = sim;
            best_idx = i;
        }
    }

    if (best_idx >= 0 && max_similarity >= rec->similarity_threshold) {
        strncpy(result.identity, rec->database[best_idx].identity, MAX_STRING_LEN - 1);
        result.identity[MAX_STRING_LEN - 1] = '\0';
        result.confidence = max_similarity;
    } else {
        result.confidence = 0.0f;
    }

    result.similarity = max_similarity;
    memcpy(result.feature_vector, feature, sizeof(feature));
    result.has_feature = true;

    return result;
}

float arcface_calculate_similarity(const float* feature1, const float* feature2, int dim) {
    if (!feature1 || !feature2 || dim <= 0) return 0.0f;

    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += feature1[i] * feature2[i];
    }

    return dot;
}

void arcface_recognizer_clear_database(ArcFaceRecognizer* rec) {
    if (!rec) return;
    rec->num_entries = 0;
    log_info("Face database cleared");
}

bool arcface_recognizer_register_feature(ArcFaceRecognizer* rec, const char* identity,
                                          const float* feature) {
    if (!rec || !identity || !feature) return false;
    if (rec->num_entries >= ARCFACE_MAX_FACES) {
        log_error("Face database full");
        return false;
    }

    /* Check if identity already exists — update instead of duplicate */
    int existing = arcface_recognizer_find_entry(rec, identity);
    if (existing >= 0) {
        memcpy(rec->database[existing].feature, feature, ARCFACE_FEATURE_DIM * sizeof(float));
        rec->database[existing].active = true;
        log_info("Updated face identity: %s", identity);
        return true;
    }

    FaceDatabaseEntry* entry = &rec->database[rec->num_entries++];
    strncpy(entry->identity, identity, MAX_STRING_LEN - 1);
    entry->identity[MAX_STRING_LEN - 1] = '\0';
    entry->active = true;
    memcpy(entry->feature, feature, ARCFACE_FEATURE_DIM * sizeof(float));

    log_info("Registered face identity: %s", identity);
    return true;
}

int arcface_recognizer_find_entry(ArcFaceRecognizer* rec, const char* identity) {
    if (!rec || !identity) return -1;
    for (int i = 0; i < rec->num_entries; i++) {
        if (rec->database[i].active && strcmp(rec->database[i].identity, identity) == 0) {
            return i;
        }
    }
    return -1;
}

bool arcface_recognizer_delete_entry(ArcFaceRecognizer* rec, const char* identity) {
    if (!rec || !identity) return false;
    int idx = arcface_recognizer_find_entry(rec, identity);
    if (idx < 0) return false;
    rec->database[idx].active = false;
    log_info("Deleted face identity: %s", identity);
    return true;
}

int arcface_recognizer_get_entry_count(const ArcFaceRecognizer* rec) {
    if (!rec) return 0;
    int count = 0;
    for (int i = 0; i < rec->num_entries; i++) {
        if (rec->database[i].active) count++;
    }
    return count;
}

bool arcface_recognizer_save_database(ArcFaceRecognizer* rec, const char* filepath) {
    if (!rec || !filepath) return false;

    FILE* f = fopen(filepath, "w");
    if (!f) {
        log_error("ArcFace: cannot open database for writing: %s", filepath);
        return false;
    }

    fprintf(f, "{\n  \"version\": 1,\n  \"entries\": [\n");

    int written = 0;
    for (int i = 0; i < rec->num_entries; i++) {
        if (!rec->database[i].active) continue;
        if (written > 0) fprintf(f, ",\n");

        fprintf(f, "    {\n      \"identity\": \"");
        /* Escape backslash and double-quote in identity */
        for (const char* s = rec->database[i].identity; *s; s++) {
            if (*s == '\\' || *s == '"') fputc('\\', f);
            fputc(*s, f);
        }
        fprintf(f, "\",\n      \"feature\": [");

        for (int j = 0; j < ARCFACE_FEATURE_DIM; j++) {
            if (j > 0) fprintf(f, ", ");
            fprintf(f, "%.8f", (double)rec->database[i].feature[j]);
        }
        fprintf(f, "]\n    }");
        written++;
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);

    log_info("ArcFace: saved %d entries to %s", written, filepath);
    return true;
}

/* Minimal JSON parser — extracts identity and feature array from database file */
bool arcface_recognizer_load_database(ArcFaceRecognizer* rec, const char* filepath) {
    if (!rec || !filepath) return false;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        log_warn("ArcFace: database file not found: %s (starting with empty DB)", filepath);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0 || fsize > (4L * 1024 * 1024)) {  /* 4MB limit */
        fclose(f);
        return false;
    }
    fseek(f, 0, SEEK_SET);

    char* buf = (char*)malloc((size_t)fsize + 1);
    if (!buf) { fclose(f); return false; }
    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    if (nread <= 0) { free(buf); return false; }
    buf[nread] = '\0';

    int loaded = 0;
    const char* p = buf;

    while (*p) {
        /* Find "identity": "..." */
        const char* id_start = strstr(p, "\"identity\":");
        if (!id_start) break;

        /* Find opening quote of value */
        const char* id_val = strchr(id_start + 11, '"');
        if (!id_val) break;
        id_val++;  /* skip opening quote */

        char identity[MAX_STRING_LEN];
        int id_len = 0;
        while (*id_val && *id_val != '"' && id_len < (int)(sizeof(identity) - 1)) {
            if (*id_val == '\\' && *(id_val + 1)) {
                id_val++;
                identity[id_len++] = *id_val++;
            } else {
                identity[id_len++] = *id_val++;
            }
        }
        identity[id_len] = '\0';
        if (id_len == 0) { p = id_val + 1; continue; }
        p = id_val + 1;  /* advance past closing quote */

        /* Find "feature": [...] */
        const char* feat_start = strstr(p, "\"feature\":");
        if (!feat_start) break;

        const char* bracket = strchr(feat_start + 10, '[');
        if (!bracket) break;

        float feature[ARCFACE_FEATURE_DIM];
        int feat_count = 0;
        p = bracket + 1;

        while (feat_count < ARCFACE_FEATURE_DIM && *p) {
            /* Skip whitespace and commas */
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
            if (*p == ']' || *p == '\0') break;

            char* end = NULL;
            feature[feat_count++] = strtof(p, &end);
            if (end == p) break;
            p = end;
        }

        if (feat_count == ARCFACE_FEATURE_DIM) {
            if (arcface_recognizer_register_feature(rec, identity, feature)) {
                loaded++;
            }
        }

        /* Advance past closing ']' */
        p = strchr(p, ']');
        if (!p) break;
        p++;
    }

    free(buf);
    log_info("ArcFace: loaded %d entries from %s", loaded, filepath);
    return loaded > 0;
}
