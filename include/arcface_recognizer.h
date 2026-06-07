#ifndef ARCFACE_RECOGNIZER_H
#define ARCFACE_RECOGNIZER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARCFACE_FEATURE_DIM     128  /* ArcFace MobileFaceNet-cuted: output is [1,128]; must match core_types.h FEATURE_VECTOR_DIM */
#define ARCFACE_MAX_FACES       256

typedef struct OrtSession OrtSession;
typedef struct OrtInferenceContext OrtInferenceContext;

typedef struct {
    char identity[MAX_STRING_LEN];
    float feature[ARCFACE_FEATURE_DIM];
    bool active;
} FaceDatabaseEntry;

typedef struct {
    OrtSession* session;
    OrtInferenceContext* ctx;
    char input_name[MAX_STRING_LEN];
    int input_width;
    int input_height;
    float similarity_threshold;
    FaceDatabaseEntry database[ARCFACE_MAX_FACES];
    int num_entries;
} ArcFaceRecognizer;

ArcFaceRecognizer* arcface_recognizer_create(const char* model_path, int input_w, int input_h,
                                              float sim_thresh);
void arcface_recognizer_destroy(ArcFaceRecognizer* rec);

bool arcface_recognizer_load_model(ArcFaceRecognizer* rec, const char* model_path);

int  arcface_recognizer_extract_feature(ArcFaceRecognizer* rec, const uint8_t* face_image, int width, int height,
                                        float* out_feature);
bool arcface_recognizer_register_face(ArcFaceRecognizer* rec, const char* identity,
                                       const uint8_t* face_image, int width, int height);
FaceIdentity arcface_recognizer_recognize(ArcFaceRecognizer* rec, const uint8_t* face_image, int width, int height);

float arcface_calculate_similarity(const float* feature1, const float* feature2, int dim);
void arcface_recognizer_clear_database(ArcFaceRecognizer* rec);

#ifdef __cplusplus
}
#endif

#endif
