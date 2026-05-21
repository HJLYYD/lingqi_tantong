#ifndef INFERENCE_PIPELINE_H
#define INFERENCE_PIPELINE_H

#include "core_types.h"
#include "yolov8_detector.h"
#include "yolov8_pose_estimator.h"
#include "scrfd_detector.h"
#include "arcface_recognizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIPELINE_ENABLE_DETECTION   0x01
#define PIPELINE_ENABLE_POSE        0x02
#define PIPELINE_ENABLE_FACE        0x04
#define PIPELINE_ENABLE_ALL         0x07

typedef struct {
    YOLOv8Detector* detector;
    YOLOv8PoseEstimator* pose_estimator;
    SCRFDDetector* face_detector;
    ArcFaceRecognizer* face_recognizer;

    uint32_t enabled_stages;
    bool models_loaded[4];

    int frame_counter;
    bool use_onnx;
} AIInferencePipeline;

AIInferencePipeline* inference_pipeline_create(bool use_onnx);
void inference_pipeline_destroy(AIInferencePipeline* pipeline);

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir);
InferenceResult inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                                  const uint8_t* frame_data, int width, int height);

void inference_pipeline_configure(AIInferencePipeline* pipeline, uint32_t stages);
void inference_pipeline_reset(AIInferencePipeline* pipeline);

#ifdef __cplusplus
}
#endif

#endif
