#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV8_NUM_CLASSES      80
#define YOLOV8_MAX_DETECTIONS   8400
#define YOLOV8_INPUT_SIZE       640

typedef struct OrtSession OrtSession;

typedef struct {
    OrtSession* session;
    char input_name[MAX_STRING_LEN];
    int input_width;
    int input_height;
    float confidence_threshold;
    float iou_threshold;
    bool use_onnx;
    float scale;
    int pad_x;
    int pad_y;
} YOLOv8Detector;

YOLOv8Detector* yolov8_detector_create(const char* model_path, int input_w, int input_h,
                                       float conf_thresh, float iou_thresh, bool use_onnx);
void yolov8_detector_destroy(YOLOv8Detector* det);

bool yolov8_detector_load_model(YOLOv8Detector* det, const char* model_path);
int  yolov8_detector_detect(YOLOv8Detector* det, const uint8_t* image_data, int width, int height,
                            Detection* out_detections, int max_detections);
int  yolov8_detector_detect_persons(YOLOv8Detector* det, const uint8_t* image_data, int width, int height,
                                    Detection* out_detections, int max_detections);

const char* yolov8_get_class_name(int class_id);

#ifdef __cplusplus
}
#endif

#endif
