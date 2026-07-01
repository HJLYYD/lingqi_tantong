#ifndef YOLOV5_FACE_DETECTOR_H
#define YOLOV5_FACE_DETECTOR_H

#include "core_types.h"
#include "ort_inference_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV5_FACE_NUM_KEYPOINTS  5

typedef struct OrtSession OrtSession;

typedef struct {
    OrtSession* session;
    OrtInferenceContext* ctx;
    char input_name[MAX_STRING_LEN];
    int input_width;
    int input_height;
    float confidence_threshold;
    float nms_threshold;
    bool use_ep;                 /* enable SpacemiT EP (needs IO Binding for safe output reads) */
    /* Cached after first inference — avoids per-frame tensor shape probing */
    int  output_format_cached;   /* 0=unset, 1=4D, 2=5D */
} YOLOv5FaceDetector;

YOLOv5FaceDetector* yolov5_face_detector_create(const char* model_path, int input_w, int input_h,
                                                  float conf_thresh, float nms_thresh, bool use_ep);
void yolov5_face_detector_destroy(YOLOv5FaceDetector* det);

bool yolov5_face_detector_load_model(YOLOv5FaceDetector* det, const char* model_path);
int  yolov5_face_detector_detect_faces(YOLOv5FaceDetector* det, const uint8_t* image_data, int width, int height,
                                        FaceIdentity* out_faces, int max_faces);

int  yolov5_face_detector_crop_face(const YOLOv5FaceDetector* det, const uint8_t* image_data, int img_w, int img_h,
                                     const FaceIdentity* face,
                                     uint8_t* out_crop, int target_w, int target_h);

#ifdef __cplusplus
}
#endif

#endif
