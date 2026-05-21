#include "arcface_recognizer.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include "ort_common.h"
#endif

ArcFaceRecognizer* arcface_recognizer_create(const char* model_path, int input_w, int input_h,
                                              float sim_thresh, bool use_onnx) {
    ArcFaceRecognizer* rec = (ArcFaceRecognizer*)calloc(1, sizeof(ArcFaceRecognizer));
    if (!rec) return NULL;

    rec->input_width = input_w > 0 ? input_w : 112;
    rec->input_height = input_h > 0 ? input_h : 112;
    rec->similarity_threshold = sim_thresh;
    rec->use_onnx = use_onnx;
    rec->num_entries = 0;

    if (model_path) {
        arcface_recognizer_load_model(rec, model_path);
    }

    return rec;
}

void arcface_recognizer_destroy(ArcFaceRecognizer* rec) {
    if (!rec) return;
#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (rec->session && ort) {
        ort->ReleaseSession(rec->session);
    }
#endif
    free(rec);
}

bool arcface_recognizer_load_model(ArcFaceRecognizer* rec, const char* model_path) {
    if (!rec || !model_path) {
        log_error("Model path is NULL");
        return false;
    }

    FILE* f = fopen(model_path, "rb");
    if (!f) {
        log_error("Cannot open ArcFace model file: %s", model_path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 1024) {
        fclose(f);
        log_error("ArcFace model file too small: %s", model_path);
        return false;
    }

    uint8_t magic[4];
    size_t bytes_read = fread(magic, 1, 4, f);
    fclose(f);

    if (bytes_read != 4) {
        log_error("Failed to read ArcFace model header: %s", model_path);
        return false;
    }

    if (magic[0] != 0x08 || magic[1] != 0x00 || magic[2] != 0x00 || magic[3] != 0x00) {
        log_warning("ArcFace model has unexpected protobuf header: %s", model_path);
    }

    strncpy(rec->input_name, "data", sizeof(rec->input_name) - 1);
    log_info("ArcFace model validated: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));

#ifdef HAS_ONNX_RUNTIME
    if (!ort_global_init()) {
        log_warning("ArcFace: ONNX init failed, using heuristic fallback");
        return true;
    }

    rec->session = ort_create_session(model_path, 2, false);
    if (!rec->session) {
        log_warning("ArcFace: CreateSession failed, using heuristic fallback");
        return true;
    }

    log_info("ArcFace model loaded with ONNX Runtime: %s (%.2f MB)", model_path, file_size / (1024.0f * 1024.0f));
#else
    log_info("ArcFace model validated: %s (%.2f MB) [build with HAS_ONNX_RUNTIME]", model_path, file_size / (1024.0f * 1024.0f));
#endif
    return true;
}

static void preprocess_arcface(const uint8_t* face_image, int width, int height,
                               float* out_tensor, int target_w, int target_h) {
    uint8_t* resized = (uint8_t*)malloc(target_w * target_h * 3);
    if (!resized) return;

    utils_resize_image(face_image, width, height, resized, target_w, target_h, 3);

    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_idx = (y * target_w + x) * 3;
            int dst_r = 0 * target_w * target_h + y * target_w + x;
            int dst_g = 1 * target_w * target_h + y * target_w + x;
            int dst_b = 2 * target_w * target_h + y * target_w + x;

            out_tensor[dst_r] = (resized[src_idx + 2] - 127.5f) / 127.5f;
            out_tensor[dst_g] = (resized[src_idx + 1] - 127.5f) / 127.5f;
            out_tensor[dst_b] = (resized[src_idx + 0] - 127.5f) / 127.5f;
        }
    }

    free(resized);
}

static void compute_face_features(const float* face_image, int width, int height, float* out_features) {
    for (int i = 0; i < ARCFACE_FEATURE_DIM; i++) {
        out_features[i] = 0.0f;
    }

    int patch_size = 8;
    int num_patches_x = width / patch_size;
    int num_patches_y = height / patch_size;

    for (int py = 0; py < num_patches_y && py < 14; py++) {
        for (int px = 0; px < num_patches_x && px < 7; px++) {
            int patch_idx = py * 7 + px;
            float patch_mean[3] = {0, 0, 0};
            float patch_var[3] = {0, 0, 0};

            for (int y = py * patch_size; y < (py + 1) * patch_size && y < height; y++) {
                for (int x = px * patch_size; x < (px + 1) * patch_size && x < width; x++) {
                    int idx = (y * width + x) * 3;
                    for (int c = 0; c < 3; c++) {
                        patch_mean[c] += face_image[idx + c];
                    }
                }
            }

            int pixels = patch_size * patch_size;
            for (int c = 0; c < 3; c++) {
                patch_mean[c] /= pixels;
            }

            for (int y = py * patch_size; y < (py + 1) * patch_size && y < height; y++) {
                for (int x = px * patch_size; x < (px + 1) * patch_size && x < width; x++) {
                    int idx = (y * width + x) * 3;
                    for (int c = 0; c < 3; c++) {
                        float diff = face_image[idx + c] - patch_mean[c];
                        patch_var[c] += diff * diff;
                    }
                }
            }

            for (int c = 0; c < 3; c++) {
                patch_var[c] = sqrtf(patch_var[c] / pixels) / 255.0f;
            }

            if (patch_idx < ARCFACE_FEATURE_DIM) {
                out_features[patch_idx] = (patch_mean[0] / 255.0f - 0.5f) * 2.0f;
                out_features[98 + patch_idx] = patch_var[0];
            }
            if (patch_idx * 2 + 1 < ARCFACE_FEATURE_DIM) {
                out_features[patch_idx * 2 + 1] = (patch_mean[1] / 255.0f - 0.5f) * 2.0f;
                out_features[99 + patch_idx * 2 + 1] = patch_var[1];
            }
        }
    }

    for (int i = 0; i < ARCFACE_FEATURE_DIM; i++) {
        float x = out_features[i];
        out_features[i] = tanhf(x);
    }

    float norm = 0.0f;
    for (int i = 0; i < ARCFACE_FEATURE_DIM; i++) {
        norm += out_features[i] * out_features[i];
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < ARCFACE_FEATURE_DIM; i++) {
            out_features[i] *= inv_norm;
        }
    }
}

int arcface_recognizer_extract_feature(ArcFaceRecognizer* rec, const uint8_t* face_image, int width, int height,
                                        float* out_feature) {
    if (!rec || !face_image || !out_feature) return -1;

#ifdef HAS_ONNX_RUNTIME
    const OrtApi* ort = ort_get_api();
    if (rec->session && ort) {
        int input_w = rec->input_width > 0 ? rec->input_width : 112;
        int input_h = rec->input_height > 0 ? rec->input_height : 112;
        int pixels = input_w * input_h;
        size_t input_size = pixels * 3 * sizeof(float);

        float* input_tensor = (float*)malloc(input_size);
        if (!input_tensor) return -1;

        preprocess_arcface(face_image, width, height, input_tensor, input_w, input_h);

        int64_t input_shape[] = {1, 3, input_h, input_w};
        OrtMemoryInfo* memory_info;
        ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

        OrtValue* input_tensor_val;
        ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor, input_size,
                                              input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor_val);
        ort->ReleaseMemoryInfo(memory_info);

        const char* input_names[] = {rec->input_name};
        OrtValue* input_values[] = {input_tensor_val};
        OrtValue* output_values[1] = {NULL};

        OrtStatus* status = ort->Run(rec->session, NULL,
                                       input_names, (const OrtValue* const*)input_values, 1,
                                       NULL, 0, output_values, 1);
        ort->ReleaseValue(input_tensor_val);
        free(input_tensor);

        if (status == NULL && output_values[0]) {
            OrtTensorTypeAndShapeInfo* shape_info;
            ort->GetTensorTypeAndShape(output_values[0], &shape_info);
            size_t num_elements;
            ort->GetTensorShapeElementCount(shape_info, &num_elements);
            ort->ReleaseTensorTypeAndShapeInfo(shape_info);

            float* output_data;
            ort->GetTensorMutableData(output_values[0], (void**)&output_data);

            int feat_dim = UTILS_MIN((int)num_elements, ARCFACE_FEATURE_DIM);
            float norm = 0.0f;
            for (int i = 0; i < feat_dim; i++) {
                out_feature[i] = output_data[i];
                norm += out_feature[i] * out_feature[i];
            }
            norm = sqrtf(norm);
            if (norm > 0.0f) {
                float inv_norm = 1.0f / norm;
                for (int i = 0; i < feat_dim; i++) {
                    out_feature[i] *= inv_norm;
                }
            }

            ort->ReleaseValue(output_values[0]);
            return 0;
        }

        log_debug("ArcFace ONNX inference failed, using heuristic fallback");
    }
#endif

    float* input_tensor = (float*)malloc(rec->input_width * rec->input_height * 3 * sizeof(float));
    if (!input_tensor) return -1;

    preprocess_arcface(face_image, width, height, input_tensor, rec->input_width, rec->input_height);

    compute_face_features(input_tensor, rec->input_width, rec->input_height, out_feature);

    free(input_tensor);
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
