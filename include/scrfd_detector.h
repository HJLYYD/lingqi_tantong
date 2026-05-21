#ifndef SCRFD_DETECTOR_H
#define SCRFD_DETECTOR_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCRFD_NUM_KEYPOINTS     5

typedef struct OrtSession OrtSession;

typedef struct {
    OrtSession* session;
    char input_name[MAX_STRING_LEN];
    int input_width;
    int input_height;
    float confidence_threshold;
    float nms_threshold;
    bool use_onnx;
} SCRFDDetector;

SCRFDDetector* scrfd_detector_create(const char* model_path, int input_w, int input_h,
                                      float conf_thresh, float nms_thresh, bool use_onnx);
void scrfd_detector_destroy(SCRFDDetector* det);

bool scrfd_detector_load_model(SCRFDDetector* det, const char* model_path);
int  scrfd_detector_detect_faces(SCRFDDetector* det, const uint8_t* image_data, int width, int height,
                                  FaceIdentity* out_faces, int max_faces);

int scrfd_detector_crop_face(const SCRFDDetector* det, const uint8_t* image_data, int img_w, int img_h,
                              const FaceIdentity* face,
                              uint8_t* out_crop, int target_w, int target_h);

#ifdef __cplusplus
}
#endif

#endif
