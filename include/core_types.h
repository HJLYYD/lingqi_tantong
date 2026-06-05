#ifndef CORE_TYPES_H
#define CORE_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_KEYPOINTS           17
#define MAX_TRAJECTORY_LEN      300
#define MAX_DETECTIONS_PER_FRAME 100
#define MAX_TRACKED_OBJECTS     100
#define MAX_FACES_PER_FRAME     20
#define MAX_POSES_PER_FRAME     20
#define MAX_ACTIONS_PER_FRAME   5
#define FEATURE_VECTOR_DIM      128  /* ArcFace MobileFaceNet-cuted model output (validated via analyze_models.py) */
#define MAX_STRING_LEN          64
#define MAX_PATH_LEN            256
#define MAX_LABEL_LEN           128
#define MAX_LOG_MSG_LEN         1024
#define MAX_BUFFER_SIZE         4096
#define MAX_SESSION_RESULTS     1000
#define ARROW_RING_BUFFER_SIZE  16
#define ARROW_MAX_FRAME_LEN     65536

typedef enum {
    PIPELINE_MODE_OFFLINE = 0,
    PIPELINE_MODE_REALTIME = 1,
    PIPELINE_MODE_BENCHMARK = 2,
} PipelineMode;

typedef enum {
    ARROW_STATE_IDLE = 0,
    ARROW_STATE_SYNCING,
    ARROW_STATE_LOCKED,
    ARROW_STATE_ERROR,
} ArrowReceiveState;

typedef enum {
    DETECTION_CLASS_PERSON = 0,
    DETECTION_CLASS_BICYCLE = 1,
    DETECTION_CLASS_CAR = 2,
    DETECTION_CLASS_UNKNOWN = 99
} DetectionClass;

typedef struct {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} BoundingBox;

static inline float bbox_center_x(const BoundingBox* bbox) {
    return (bbox->x_min + bbox->x_max) * 0.5f;
}

static inline float bbox_center_y(const BoundingBox* bbox) {
    return (bbox->y_min + bbox->y_max) * 0.5f;
}

static inline float bbox_width(const BoundingBox* bbox) {
    return bbox->x_max - bbox->x_min;
}

static inline float bbox_height(const BoundingBox* bbox) {
    return bbox->y_max - bbox->y_min;
}

static inline float bbox_area(const BoundingBox* bbox) {
    return bbox_width(bbox) * bbox_height(bbox);
}

static inline float bbox_iou(const BoundingBox* a, const BoundingBox* b) {
    float x_left   = fmaxf(a->x_min, b->x_min);
    float y_top    = fmaxf(a->y_min, b->y_min);
    float x_right  = fminf(a->x_max, b->x_max);
    float y_bottom = fminf(a->y_max, b->y_max);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection = (x_right - x_left) * (y_bottom - y_top);
    float union_area   = bbox_area(a) + bbox_area(b) - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

typedef struct {
    BoundingBox bbox;
    float confidence;
    int class_id;
    char class_name[MAX_STRING_LEN];
} Detection;

typedef struct {
    float x;
    float y;
    float confidence;
    char name[MAX_STRING_LEN];
} Keypoint;

typedef struct {
    Keypoint keypoints[MAX_KEYPOINTS];
    int num_keypoints;
    BoundingBox bbox;
    bool has_bbox;
    float confidence;
} PoseEstimation;

typedef struct {
    float x;
    float y;
    float z;
    float depth_confidence;
    bool is_valid;
    float world_x;
    float world_z;
    float estimated_height;
    float confidence;
} SpatialPosition;

static inline float spatial_distance(const SpatialPosition* a, const SpatialPosition* b) {
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    float dz = a->z - b->z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

static inline float spatial_distance_from_origin(const SpatialPosition* pos) {
    return sqrtf(pos->x*pos->x + pos->y*pos->y + pos->z*pos->z);
}

static inline float spatial_world_distance(const SpatialPosition* a, const SpatialPosition* b) {
    float dx = a->world_x - b->world_x;
    float dz = a->world_z - b->world_z;
    return sqrtf(dx*dx + dz*dz);
}

typedef struct {
    BoundingBox bbox;
    float confidence;
    char identity[MAX_STRING_LEN];
    float similarity;
    float feature_vector[FEATURE_VECTOR_DIM];
    bool has_feature;
    float keypoints[5][2];
    bool has_keypoints;
} FaceIdentity;

typedef struct {
    int action_id;
    char action_name[MAX_STRING_LEN];
    float confidence;
} ActionPrediction;

typedef struct {
    ActionPrediction actions[MAX_ACTIONS_PER_FRAME];
    int num_actions;
    int predicted_action_id;
    float predicted_confidence;
} ActionResult;

typedef struct {
    SpatialPosition positions[MAX_TRAJECTORY_LEN];
    int count;
} TrajectoryBuffer;

typedef struct {
    int track_id;
    Detection detection;
    SpatialPosition spatial_pos;
    PoseEstimation pose;
    bool has_pose;
    FaceIdentity face;
    bool has_face;
    TrajectoryBuffer trajectory;
    float velocity[3];
    bool has_velocity;
    float acceleration[3];
    bool has_acceleration;
    bool is_active;
    bool is_occluded;
    int occluded_frames;
    int frames_seen;
    int last_seen_frame;
    int first_seen_frame;
    float height_meters;
    bool has_height;
} TrackedObject;

typedef struct {
    uint8_t* data;
    int width;
    int height;
    int channels;
    int frame_index;
    double timestamp;
} FrameData;

typedef struct {
    double timestamp;
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
} IMUData;

typedef struct {
    Detection detections[MAX_DETECTIONS_PER_FRAME];
    int num_detections;
    PoseEstimation poses[MAX_POSES_PER_FRAME];
    int num_poses;
    FaceIdentity faces[MAX_FACES_PER_FRAME];
    int num_faces;
    ActionResult action;
    bool has_action;
    float* depth_map;
    bool has_depth_map;
    int depth_width;
    int depth_height;
    float processing_time_ms;
} InferenceResult;

typedef struct {
    TrackedObject tracked_objects[MAX_TRACKED_OBJECTS];
    int num_tracked;
    int new_tracks;
    int lost_tracks;
    int id_switches;
    float processing_time_ms;
} TrackingResult;

typedef struct {
    SpatialPosition position;
    float depth;
    float confidence;
} SpatialResult;

typedef struct {
    bool is_running;
    int frame_count;
    float fps;
    float processing_time_ms;
    int active_tracks;
    int total_tracks;
    int error_count;
    char message[MAX_LOG_MSG_LEN];
} SystemStatus;

typedef struct {
    float qw;
    float qx;
    float qy;
    float qz;
    float pitch;
    float roll;
    float yaw;
    float altitude_m;
    float temperature_c;
    uint32_t timestamp_ms;
    bool is_valid;
} IMUExternalPose;

typedef struct {
    uint8_t* jpeg_buf;
    size_t jpeg_len;
    int frame_index;
    IMUExternalPose pose;
    bool has_pose;
} ARFramePair;

typedef struct {
    uint8_t jpeg_data[ARROW_MAX_FRAME_LEN];
    size_t jpeg_len;
    IMUExternalPose pose;
    bool has_pose;
    int frame_index;
    double timestamp;
    bool is_valid;
} ArrowSourceFrame;

void detection_init(Detection* det, float x1, float y1, float x2, float y2,
                    float conf, int class_id, const char* class_name);
void pose_estimation_init(PoseEstimation* pose);
void tracked_object_init(TrackedObject* obj, int track_id, const Detection* det,
                         const SpatialPosition* pos);
void trajectory_buffer_append(TrajectoryBuffer* buf, const SpatialPosition* pos);
void inference_result_init(InferenceResult* result);
void tracking_result_init(TrackingResult* result);
void frame_data_init(FrameData* frame, uint8_t* data, int w, int h, int c, int idx, double ts);
void imu_data_init(IMUData* imu, double ts,
                   float ax, float ay, float az,
                   float gx, float gy, float gz);

#ifdef __cplusplus
}
#endif

#endif
