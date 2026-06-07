#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLO11_NUM_CLASSES      80
#define YOLO11_MAX_DETECTIONS   6000  /* all 3 stride groups combined (80×80+40×40+20×20=8400 max);
                                         at DFL≥0.30 threshold ~4000 pass; 6000 with safety margin; heap-allocated */
#define YOLO11_INPUT_SIZE       320

typedef struct OrtSession OrtSession;
typedef struct OrtInferenceContext OrtInferenceContext;

typedef struct {
    OrtSession* session;
    OrtInferenceContext* ctx;
    char input_name[MAX_STRING_LEN];
    int input_width;
    int input_height;
    float confidence_threshold;
    float iou_threshold;
    float scale;
    int pad_x;
    int pad_y;
} YOLO11Detector;

YOLO11Detector* yolov8_detector_create(const char* model_path, int input_w, int input_h,
                                       float conf_thresh, float iou_thresh);
void yolov8_detector_destroy(YOLO11Detector* det);

bool yolov8_detector_load_model(YOLO11Detector* det, const char* model_path);
int  yolov8_detector_detect(YOLO11Detector* det, const uint8_t* image_data, int width, int height,
                            Detection* out_detections, int max_detections);
int  yolov8_detector_detect_persons(YOLO11Detector* det, const uint8_t* image_data, int width, int height,
                                    Detection* out_detections, int max_detections);

const char* yolov8_get_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif
