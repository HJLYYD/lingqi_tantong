#ifndef SPATIAL_ENGINE_H
#define SPATIAL_ENGINE_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPATIAL_MAX_TRAJECTORY      300
#define SPATIAL_MIN_DEPTH           0.3f
#define SPATIAL_MAX_DEPTH           120.0f
#define SPATIAL_MAX_PERSONS         256

/* ── Anatomical body ratios (from anthropometric research) ── */
#define ANTHRO_HEAD_TO_HEIGHT       0.130f   /* head top to chin ≈ 13% of total height */
#define ANTHRO_SHOULDER_TO_HEIGHT   0.818f   /* shoulder to floor */
#define ANTHRO_HIP_TO_HEIGHT        0.530f   /* hip to floor */
#define ANTHRO_KNEE_TO_HEIGHT       0.285f   /* knee to floor */
#define ANTHRO_ANKLE_TO_HEIGHT      0.039f   /* ankle to floor */
#define ANTHRO_SHOULDER_WIDTH_RATIO 0.259f   /* shoulder width / height */

/* ── Depth estimation tuning ── */
#define DEPTH_EMA_ALPHA             0.30f   /* exponential moving average for depth smoothing */
#define DEPTH_OUTLIER_MAD_MULT      2.5f    /* MAD multiplier for outlier rejection */
#define DEPTH_MIN_CONFIDENCE        0.25f   /* floor for confidence values */
#define DEPTH_MIN_KEYPOINT_CONF     0.25f   /* minimum per-keypoint confidence for depth sampling */

typedef struct {
    float camera_matrix[3][3];
    float dist_coeffs[4];
    float focal_length;
    float avg_person_height;

    bool world_initialized;
    SpatialPosition world_origin;
    float world_reference_height;

    float first_frame_center_x;
    float first_frame_center_y;
    float first_frame_ref_depth;

    float ref_frame_height_pixels;
    float height_scale_factor;

    TrajectoryBuffer trajectories[SPATIAL_MAX_PERSONS];
    bool trajectory_active[SPATIAL_MAX_PERSONS];

    SpatialPosition coord_smoothing_buffer[SPATIAL_MAX_PERSONS];
    bool smoothing_active[SPATIAL_MAX_PERSONS];

    float coord_smoothing_alpha;

    /* ── Per-track depth EMA (temporal smoothing) ── */
    float depth_ema[SPATIAL_MAX_PERSONS];
    float depth_ema_variance[SPATIAL_MAX_PERSONS];
    bool  depth_ema_active[SPATIAL_MAX_PERSONS];

    float camera_pitch;
    float camera_roll;
    float camera_yaw;
    bool has_camera_pose;

    /* ── Runtime tuning ── */
    float depth_ema_alpha;
    float depth_outlier_mad_mult;
} SpatialLocalizationEngine;

SpatialLocalizationEngine* spatial_engine_create(const float* camera_matrix, const float* dist_coeffs,
                                                  float focal_length, float avg_person_height);
void spatial_engine_destroy(SpatialLocalizationEngine* engine);

void spatial_engine_initialize_coordinate_system(SpatialLocalizationEngine* engine,
                                                  int frame_height, int frame_width,
                                                  const Detection* first_detection);

SpatialResult spatial_engine_calculate_position(SpatialLocalizationEngine* engine,
                                                 const Detection* detection,
                                                 int frame_width, int frame_height,
                                                 const float* depth_map, int depth_w, int depth_h,
                                                 const PoseEstimation* pose, int track_id);

float spatial_engine_calculate_height(SpatialLocalizationEngine* engine,
                                       const Detection* detection,
                                       const PoseEstimation* pose);

void spatial_engine_update_trajectory(SpatialLocalizationEngine* engine, int track_id, const SpatialPosition* position);
const SpatialPosition* spatial_engine_get_trajectory(SpatialLocalizationEngine* engine, int track_id, int* out_count);
bool spatial_engine_get_velocity(SpatialLocalizationEngine* engine, int track_id, float dt, float out_velocity[3]);

/*
 * Check if a new depth estimate is consistent with the tracked object's
 * recent depth history.  Returns false for implausible depth jumps
 * (e.g. z goes from 3m to 20m in one frame — likely a false detection
 * on background or a different person).
 */
bool spatial_engine_check_depth_consistency(SpatialLocalizationEngine* engine,
                                              int track_id, float new_depth,
                                              float max_jump_ratio);

void spatial_engine_clear_trajectories(SpatialLocalizationEngine* engine);
int  spatial_engine_get_active_tracks(const SpatialLocalizationEngine* engine, int* out_ids, int max_count);

void spatial_engine_set_camera_pose(SpatialLocalizationEngine* engine, float pitch, float roll, float yaw);

#ifdef __cplusplus
}
#endif

#endif
