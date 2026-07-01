/*
 * core_types.h — Core data structures for LingQi TanTong inference pipeline
 *
 * Rewritten: cleaner layout, PipelineTiming, error categories.
 * ALL struct field names kept backward-compatible with existing codebase.
 * New type aliases at bottom for both old→new and new→old code.
 */

#ifndef CORE_TYPES_H
#define CORE_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * ── Capacity constants (optimized for K1 embedded deployment) ──
 *
 * Size rationale:
 *   MAX_DETS=40:  cascade filter caps at 20; double that for safety.
 *   MAX_TRACKS=40: typical scene ≤10 people; 4× headroom.
 *   MAX_FACES=10:  face detection runs infrequently; 10 faces is generous.
 *   MAX_POSES=12:  one pose per person + margin.
 *   TRAJ_LEN=60:   velocity estimation needs 2 points; EMA smoothing ~10;
 *                   60 gives 2s history @30fps.
 *   STR_LEN=32:     class names max 16 chars; identity strings max 32.
 */
enum {
    MAX_KPTS        = 17,
    MAX_DETS        = 40,
    MAX_TRACKS      = 40,
    MAX_FACES       = 10,
    MAX_POSES       = 12,
    MAX_ACTIONS     = 5,
    FEAT_DIM        = 128,
    STR_LEN         = 32,
    PATH_LEN        = 256,
    LABEL_LEN       = 128,
    TRAJ_LEN        = 60,
    JPEG_MAX        = 131072,  /* must be >= CoAP FRAME_BUF_CAPACITY (128KB) */
};

/* ── Old names (keep for backward compat — updated to match enum) ── */
#define MAX_KEYPOINTS            17
#define MAX_DETECTIONS_PER_FRAME 40
#define MAX_TRACKED_OBJECTS      40
#define MAX_FACES_PER_FRAME      10
#define MAX_POSES_PER_FRAME      12
#define MAX_ACTIONS_PER_FRAME    5
#define FEATURE_VECTOR_DIM       128
#define MAX_STRING_LEN           32
#define MAX_PATH_LEN             256
#define MAX_LABEL_LEN            128
#define MAX_TRAJECTORY_LEN       60
#define FRAME_MAX_JPEG_LEN       131072  /* must be >= CoAP FRAME_BUF_CAPACITY (128KB) */
#define MAX_LOG_MSG_LEN          1024
#define MAX_BUFFER_SIZE          4096
#define MAX_SESSION_RESULTS      1000
#define RM_MAX_SESSIONS          128
#define RM_MAX_ERRORS            64
#define RM_MAX_TRACKED_OBJS      256

/* ═══════════════════════════════════════════════════════════════════════
 * BBox — axis-aligned bounding box
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x_min, y_min, x_max, y_max;
} BBox;

/* Short alias for new code */
typedef BBox BoundingBox;

static inline float bbox_center_x(const BBox* b) { return (b->x_min + b->x_max) * 0.5f; }
static inline float bbox_center_y(const BBox* b) { return (b->y_min + b->y_max) * 0.5f; }
static inline float bbox_width(const BBox* b)    { return b->x_max - b->x_min; }
static inline float bbox_height(const BBox* b)   { return b->y_max - b->y_min; }
static inline float bbox_area(const BBox* b)     { return bbox_width(b) * bbox_height(b); }

static inline float bbox_iou(const BBox* a, const BBox* b) {
    float l = fmaxf(a->x_min, b->x_min), r = fminf(a->x_max, b->x_max);
    float t = fmaxf(a->y_min, b->y_min), bo = fminf(a->y_max, b->y_max);
    if (r < l || bo < t) return 0.0f;
    float inter = (r - l) * (bo - t);
    float uni = bbox_area(a) + bbox_area(b) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* Short aliases for hot-path code */
#define bbox_cx(b)  bbox_center_x(b)
#define bbox_cy(b)  bbox_center_y(b)
#define bbox_w(b)   bbox_width(b)
#define bbox_h(b)   bbox_height(b)

/* ═══════════════════════════════════════════════════════════════════════
 * Detection
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    BBox   bbox;
    float  confidence;
    int    class_id;
    char   class_name[STR_LEN];
    int    track_id_hint;
    bool   is_partial_body;
    int    num_visible_keypoints;
} Detection;

/* ═══════════════════════════════════════════════════════════════════════
 * Keypoint / Pose
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x, y, confidence;
} Keypoint;

typedef struct {
    Keypoint keypoints[MAX_KPTS];
    int      num_keypoints;
    BBox     bbox;
    bool     has_bbox;
    float    confidence;
} Pose;

typedef Pose PoseEstimation;

/* ═══════════════════════════════════════════════════════════════════════
 * Spatial position
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float x, y, z;
    float depth_confidence;
    bool  is_valid;
    float world_x, world_z;
    float estimated_height;
    float confidence;
} SpatialPos;

typedef SpatialPos SpatialPosition;

static inline float spatial_distance(const SpatialPos* a, const SpatialPos* b) {
    float dx = a->x - b->x, dy = a->y - b->y, dz = a->z - b->z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SpatialResult (for spatial_engine return type)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    SpatialPos position;
    float      depth;
    float      confidence;
} SpatialResult;

/* ═══════════════════════════════════════════════════════════════════════
 * Face identity
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    BBox   bbox;
    float  confidence;
    char   identity[STR_LEN];
    float  similarity;
    float  feature_vector[FEAT_DIM];
    bool   has_feature;
    float  keypoints[5][2];
    bool   has_keypoints;
} Face;

typedef Face FaceIdentity;

/* ═══════════════════════════════════════════════════════════════════════
 * Action prediction
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int   action_id;
    char  action_name[LABEL_LEN];
    float confidence;
} ActionPrediction;

typedef struct {
    ActionPrediction actions[MAX_ACTIONS];
    int              num_actions;
    int              predicted_action_id;
    float            predicted_confidence;
} ActionResult;

/* ═══════════════════════════════════════════════════════════════════════
 * Per-stage pipeline timing
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int64_t t_preprocess_us;
    int64_t t_inference_us;
    int64_t t_decode_us;
    int64_t t_nms_us;
    int64_t t_filter_us;
    int64_t t_face_detect_us;
    int64_t t_face_recog_us;
    int64_t t_action_push_us;
    int64_t t_total_us;
} PipelineTiming;

/* ═══════════════════════════════════════════════════════════════════════
 * InferenceResult — per-frame output
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    Detection       detections[MAX_DETS];
    int             num_detections;
    Pose            poses[MAX_POSES];
    int             num_poses;
    Face            faces[MAX_FACES];
    int             num_faces;
    ActionResult    action;
    bool            has_action;
    float*          depth_map;
    bool            has_depth_map;
    int             depth_width;
    int             depth_height;
    float           processing_time_ms;
    PipelineTiming  pipeline_timing;
} InferResult;

typedef InferResult InferenceResult;

/* ═══════════════════════════════════════════════════════════════════════
 * Trajectory buffer
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    SpatialPos positions[TRAJ_LEN];
    int        count;
} TrajBuf;

typedef TrajBuf TrajectoryBuffer;

/* ═══════════════════════════════════════════════════════════════════════
 * Tracked object
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int         track_id;
    Detection   detection;
    SpatialPos  spatial_pos;
    Pose        pose;
    bool        has_pose;
    Face        face;
    bool        has_face;
    TrajBuf     trajectory;
    float       velocity[3];
    bool        has_velocity;
    float       acceleration[3];
    bool        has_acceleration;
    bool        is_active;
    bool        is_occluded;
    int         occluded_frames;
    int         frames_seen;
    int         last_seen_frame;
    int         first_seen_frame;
    float       height_meters;
    bool        has_height;
} TrackedObj;

typedef TrackedObj TrackedObject;

/* ═══════════════════════════════════════════════════════════════════════
 * TrackingResult
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    TrackedObj  tracked_objects[MAX_TRACKS];
    int         num_tracked;
    int         new_tracks;
    int         lost_tracks;
    int         id_switches;
    float       processing_time_ms;
} TrackResult;

typedef TrackResult TrackingResult;

/* ═══════════════════════════════════════════════════════════════════════
 * Frame / IMU / System status
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t* data;
    int      width, height, channels;
    int      frame_index;
    double   timestamp;
} Frame;

typedef Frame FrameData;

typedef struct {
    double timestamp;
    float  accel_x, accel_y, accel_z;
    float  gyro_x, gyro_y, gyro_z;
} ImuSample;

typedef ImuSample IMUData;

typedef struct {
    bool  is_running;
    int   frame_count;
    float fps;
    float processing_time_ms;
    int   active_tracks;
    int   total_tracks;
    int   error_count;
    char  message[MAX_LOG_MSG_LEN];
} SysStatus;

typedef SysStatus SystemStatus;

/* ═══════════════════════════════════════════════════════════════════════
 * Pipeline mode + error categories (NEW)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    PIPELINE_MODE_OFFLINE  = 0,
    PIPELINE_MODE_REALTIME = 1,
    PIPELINE_MODE_BENCHMARK = 2,
} PipelineMode;

typedef enum {
    DETECTION_CLASS_PERSON = 0,
    DETECTION_CLASS_BICYCLE = 1,
    DETECTION_CLASS_CAR = 2,
    DETECTION_CLASS_UNKNOWN = 99,
} DetectionClass;

typedef enum {
    ERR_NONE      = 0,
    ERR_MODEL     = 1,
    ERR_IO        = 2,
    ERR_MEMORY    = 3,
    ERR_TIMEOUT   = 4,
    ERR_INTERNAL  = 5,
} ErrCategory;

/* ═══════════════════════════════════════════════════════════════════════
 * IMU / Sensor types (kept for backward compat)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    float qw, qx, qy, qz;
    float pitch, roll, yaw;
    float altitude_m, temperature_c;
    uint32_t timestamp_ms;
    bool is_valid;
} IMUExternalPose;

typedef struct {
    float qw, qx, qy, qz;
    float beta, sample_freq;
    float integral_fb[3];
    bool  initialized;
    double last_timestamp;
} MadgwickFilter;

typedef struct {
    float k1_qw, k1_qx, k1_qy, k1_qz;
    float cam_qw, cam_qx, cam_qy, cam_qz;
    float align_qw, align_qx, align_qy, align_qz;
    float rel_pitch, rel_roll, rel_yaw;
    bool  k1_valid, cam_valid, align_valid;
    uint32_t timestamp_ms;
} DualImuPose;

typedef struct {
    int    window_size, samples_collected, cam_idx;
    float  k1_accels[3][256];
    float  cam_accels[3][256];
    bool   done;
} FrameAlignmentCtx;

typedef struct {
    uint8_t         jpeg_data[JPEG_MAX];
    size_t          jpeg_len;
    IMUExternalPose pose;
    bool            has_pose;
    int             frame_index;
    double          timestamp;
    bool            is_valid;
} ArrowSourceFrame;

/* ═══════════════════════════════════════════════════════════════════════
 * Init helper declarations
 * ═══════════════════════════════════════════════════════════════════════ */

void detection_init(Detection* d, float x1, float y1, float x2, float y2,
                    float conf, int cls, const char* name);
void pose_estimation_init(Pose* p);
void tracked_object_init(TrackedObj* o, int id, const Detection* d, const SpatialPos* pos);
void trajectory_buffer_append(TrajBuf* b, const SpatialPos* pos);
void inference_result_init(InferResult* r);
void tracking_result_init(TrackResult* r);
void frame_data_init(Frame* f, uint8_t* data, int w, int h, int c, int idx, double ts);
void imu_data_init(ImuSample* imu, double ts, float ax, float ay, float az,
                   float gx, float gy, float gz);

#ifdef __cplusplus
}
#endif

#endif /* CORE_TYPES_H */
