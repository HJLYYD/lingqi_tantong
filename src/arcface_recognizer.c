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
    const OrtApi* ort = ort_get_api();
    if (rec->session && ort) {
        ort->ReleaseSession(rec->session);
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
        const OrtApi* o2 = ort_get_api();
        if (o2) o2->ReleaseSession(rec->session);
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
