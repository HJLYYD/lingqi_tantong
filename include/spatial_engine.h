#ifndef SPATIAL_ENGINE_H
#define SPATIAL_ENGINE_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPATIAL_MAX_TRAJECTORY      300
#define SPATIAL_MIN_DEPTH           0.5f
#define SPATIAL_MAX_DEPTH           50.0f
#define SPATIAL_MAX_PERSONS         256

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

    float camera_pitch;
    float camera_roll;
    float camera_yaw;
    bool has_camera_pose;
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
                                                 const float* depth_map, int depth_w, int depth_h);

float spatial_engine_calculate_height(SpatialLocalizationEngine* engine,
                                       const Detection* detection,
                                       const PoseEstimation* pose);

void spatial_engine_update_trajectory(SpatialLocalizationEngine* engine, int track_id, const SpatialPosition* position);
const SpatialPosition* spatial_engine_get_trajectory(SpatialLocalizationEngine* engine, int track_id, int* out_count);
bool spatial_engine_get_velocity(SpatialLocalizationEngine* engine, int track_id, float dt, float out_velocity[3]);

void spatial_engine_clear_trajectories(SpatialLocalizationEngine* engine);
int  spatial_engine_get_active_tracks(const SpatialLocalizationEngine* engine, int* out_ids, int max_count);

void spatial_engine_set_camera_pose(SpatialLocalizationEngine* engine, float pitch, float roll, float yaw);

#ifdef __cplusplus
}
#endif

#endif
