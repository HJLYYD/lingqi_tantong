#ifndef INFERENCE_PIPELINE_H
#define INFERENCE_PIPELINE_H

#include "core_types.h"
#include "config_manager.h"
#include "yolov8_detector.h"
#include "yolov8_pose_estimator.h"
#include "yolov5_face_detector.h"
#include "arcface_recognizer.h"
#include "stgcn_action_recognizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIPELINE_ENABLE_DETECTION   0x01
#define PIPELINE_ENABLE_POSE        0x02
#define PIPELINE_ENABLE_FACE        0x04
#define PIPELINE_ENABLE_ACTION      0x08
#define PIPELINE_ENABLE_ALL         0x0F

typedef struct {
    YOLO11Detector* detector;
    YOLOv8PoseEstimator* pose_estimator;
    YOLOv5FaceDetector* face_detector;
    ArcFaceRecognizer* face_recognizer;
    STGCNActionRecognizer* action_recognizer;

    uint32_t enabled_stages;
    bool models_loaded[5];

    int frame_counter;
} AIInferencePipeline;

AIInferencePipeline* inference_pipeline_create(void);
void inference_pipeline_destroy(AIInferencePipeline* pipeline);

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config);
InferenceResult inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                                  const uint8_t* frame_data, int width, int height);

void inference_pipeline_configure(AIInferencePipeline* pipeline, uint32_t stages);
void inference_pipeline_reset(AIInferencePipeline* pipeline);

#ifdef __cplusplus
}
#endif

#endif
