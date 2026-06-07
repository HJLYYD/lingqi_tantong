#ifndef YOLOV8_POSE_ESTIMATOR_H
#define YOLOV8_POSE_ESTIMATOR_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV8_POSE_NUM_KEYPOINTS   17
#define YOLOV8_POSE_OUTPUT_DIM      56
#define YOLOV8_POSE_INPUT_SIZE      640

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

    int  cached_num_outputs;
    bool format_cached;
    bool is_xquant_split;
    int  pose_split_groups[3][3];
    int  num_pose_groups;
} YOLOv8PoseEstimator;

YOLOv8PoseEstimator* yolov8_pose_estimator_create(const char* model_path, int input_w, int input_h,
                                                   float conf_thresh, float iou_thresh);
void yolov8_pose_estimator_destroy(YOLOv8PoseEstimator* est);

bool yolov8_pose_estimator_load_model(YOLOv8PoseEstimator* est, const char* model_path);
int  yolov8_pose_estimator_estimate(YOLOv8PoseEstimator* est, const uint8_t* image_data, int width, int height,
                                    PoseEstimation* out_poses, int max_poses);

const char* yolov8_pose_get_keypoint_name(int idx);

#ifdef __cplusplus
}
#endif

#endif
