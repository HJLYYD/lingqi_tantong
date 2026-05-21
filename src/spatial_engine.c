#include "spatial_engine.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AVERAGE_EYE_HEIGHT          1.6f
#define AVERAGE_PERSON_HEIGHT       1.7f
#define SHOULDER_TO_HEIGHT_RATIO    0.7f
#define COORD_SMOOTHING_ALPHA       0.25f

SpatialLocalizationEngine* spatial_engine_create(const float* camera_matrix, const float* dist_coeffs,
                                                  float focal_length, float avg_person_height) {
    SpatialLocalizationEngine* engine = (SpatialLocalizationEngine*)calloc(1, sizeof(SpatialLocalizationEngine));
    if (!engine) return NULL;

    if (camera_matrix) {
        memcpy(engine->camera_matrix, camera_matrix, 9 * sizeof(float));
    } else {
        engine->camera_matrix[0][0] = focal_length;
        engine->camera_matrix[0][1] = 0.0f;
        engine->camera_matrix[0][2] = 320.0f;
        engine->camera_matrix[1][0] = 0.0f;
        engine->camera_matrix[1][1] = focal_length;
        engine->camera_matrix[1][2] = 240.0f;
        engine->camera_matrix[2][0] = 0.0f;
        engine->camera_matrix[2][1] = 0.0f;
        engine->camera_matrix[2][2] = 1.0f;
    }

    if (dist_coeffs) {
        memcpy(engine->dist_coeffs, dist_coeffs, 4 * sizeof(float));
    } else {
        memset(engine->dist_coeffs, 0, 4 * sizeof(float));
    }

    engine->focal_length = focal_length;
    engine->avg_person_height = avg_person_height;
    engine->world_initialized = false;
    engine->world_reference_height = AVERAGE_EYE_HEIGHT;
    engine->coord_smoothing_alpha = COORD_SMOOTHING_ALPHA;
    engine->camera_pitch = 0.0f;
    engine->camera_roll = 0.0f;
    engine->camera_yaw = 0.0f;
    engine->has_camera_pose = false;

    return engine;
}

void spatial_engine_destroy(SpatialLocalizationEngine* engine) {
    free(engine);
}

void spatial_engine_initialize_coordinate_system(SpatialLocalizationEngine* engine,
                                                  int frame_height, int frame_width,
                                                  const Detection* first_detection) {
    (void)frame_width;
    if (!engine || !first_detection) return;
    if (engine->world_initialized) return;

    float bbox_h = bbox_height(&first_detection->bbox);
    if (bbox_h <= 0.0f) {
        log_error("Invalid first detection: bbox height is 0");
        return;
    }

    engine->first_frame_center_x = engine->camera_matrix[0][2];
    engine->first_frame_center_y = engine->camera_matrix[1][2];

    float focal = engine->camera_matrix[0][0];
    engine->first_frame_ref_depth = (engine->avg_person_height * focal) / bbox_h;

    float ref_x = (bbox_center_x(&first_detection->bbox) - engine->first_frame_center_x) *
                   engine->first_frame_ref_depth / focal;
    float ref_y = (bbox_center_y(&first_detection->bbox) - engine->first_frame_center_y) *
                   engine->first_frame_ref_depth / engine->camera_matrix[1][1];
    float ref_z = engine->first_frame_ref_depth;

    engine->world_origin.x = ref_x;
    engine->world_origin.y = ref_y;
    engine->world_origin.z = ref_z;

    engine->ref_frame_height_pixels = (float)frame_height;
    engine->height_scale_factor = AVERAGE_PERSON_HEIGHT / bbox_h;

    engine->world_initialized = true;
    log_info("World coordinate system initialized: origin=(%.2f, %.2f, %.2f)m, height_scale=%.4f",
             ref_x, ref_y, ref_z, engine->height_scale_factor);
}

static float estimate_depth_from_bbox(const SpatialLocalizationEngine* engine, const Detection* detection) {
    float bbox_h = bbox_height(&detection->bbox);
    if (bbox_h <= 0.0f) return SPATIAL_MAX_DEPTH;

    float depth = (engine->avg_person_height * engine->camera_matrix[0][0]) / bbox_h;
    return UTILS_CLAMP(depth, SPATIAL_MIN_DEPTH, SPATIAL_MAX_DEPTH);
}

static float get_depth_confidence(float depth) {
    return UTILS_MAX(0.3f, 1.0f - (depth / SPATIAL_MAX_DEPTH) * 0.7f);
}

SpatialResult spatial_engine_calculate_position(SpatialLocalizationEngine* engine,
                                                 const Detection* detection,
                                                 int frame_width, int frame_height,
                                                 const float* depth_map, int depth_w, int depth_h) {
    SpatialResult result;
    memset(&result, 0, sizeof(SpatialResult));

    if (!engine || !detection) return result;

    if (!engine->world_initialized) {
        spatial_engine_initialize_coordinate_system(engine, frame_height, frame_width, detection);
        if (!engine->world_initialized) {
            result.position = (SpatialPosition){0.0f, 0.0f, 10.0f, 0.5f, false, 0.0f, 0.0f, 0.0f, 0.5f};
            result.depth = 10.0f;
            result.confidence = 0.5f;
            return result;
        }
    }

    float depth;
    float confidence;

    if (depth_map && depth_w > 0 && depth_h > 0) {
        int cx = UTILS_CLAMP((int)bbox_center_x(&detection->bbox), 0, depth_w - 1);
        int cy = UTILS_CLAMP((int)bbox_center_y(&detection->bbox), 0, depth_h - 1);

        int patch_size = 5;
        int x_start = UTILS_MAX(0, cx - patch_size / 2);
        int x_end = UTILS_MIN(depth_w, cx + patch_size / 2 + 1);
        int y_start = UTILS_MAX(0, cy - patch_size / 2);
        int y_end = UTILS_MIN(depth_h, cy + patch_size / 2 + 1);

        float patch[25];
        int patch_count = 0;
        for (int y = y_start; y < y_end && patch_count < 25; y++) {
            for (int x = x_start; x < x_end && patch_count < 25; x++) {
                patch[patch_count++] = depth_map[y * depth_w + x];
            }
        }
        depth = utils_median_float(patch, patch_count);
        confidence = 0.9f;
    } else {
        depth = estimate_depth_from_bbox(engine, detection);
        confidence = get_depth_confidence(depth);
    }

    if (engine->has_camera_pose && fabsf(engine->camera_pitch) > 0.001f) {
        float pitch_rad = engine->camera_pitch * ((float)M_PI / 180.0f);
        float corrected_depth = depth * cosf(pitch_rad);
        depth = corrected_depth;
    }

    float norm_x = bbox_center_x(&detection->bbox) / frame_width;
    float norm_y = bbox_center_y(&detection->bbox) / frame_height;

    float fx = engine->camera_matrix[0][0];
    float fy = engine->camera_matrix[1][1];
    float cx = engine->camera_matrix[0][2];
    float cy = engine->camera_matrix[1][2];

    float u = bbox_center_x(&detection->bbox);
    float v = bbox_center_y(&detection->bbox);
    float world_x = (u - cx) * depth / fx;
    float world_y = (v - cy) * depth / fy;
    float world_z = depth;

    result.position.x = world_x - engine->world_origin.x;
    result.position.y = world_y - engine->world_origin.y;
    result.position.z = world_z - engine->world_origin.z;
    result.position.world_x = world_x;
    result.position.world_z = world_z;
    result.position.depth_confidence = 1.0f;
    result.position.is_valid = true;
    result.position.confidence = confidence;
    result.position.estimated_height = engine->avg_person_height;
    result.depth = depth;
    result.confidence = confidence;

    return result;
}

float spatial_engine_calculate_height(SpatialLocalizationEngine* engine,
                                       const Detection* detection,
                                       const PoseEstimation* pose) {
    if (!engine) return AVERAGE_PERSON_HEIGHT;

    if (pose) {
        bool has_ankle = false;
        bool has_nose = false;
        float min_ankle_y = 1e9f;
        float nose_y = 0.0f;

        for (int i = 0; i < pose->num_keypoints; i++) {
            if (pose->keypoints[i].confidence > 0.3f) {
                if (strstr(pose->keypoints[i].name, "ankle") != NULL) {
                    has_ankle = true;
                    if (pose->keypoints[i].y < min_ankle_y) min_ankle_y = pose->keypoints[i].y;
                }
                if (strcmp(pose->keypoints[i].name, "nose") == 0) {
                    has_nose = true;
                    nose_y = pose->keypoints[i].y;
                }
            }
        }

        if (has_ankle && has_nose) {
            float pixel_height = nose_y - min_ankle_y;
            return UTILS_CLAMP(pixel_height * engine->height_scale_factor * 1.1f, 0.5f, 2.5f);
        }
    }

    float bbox_h = bbox_height(&detection->bbox);
    if (bbox_h > 0.0f && engine->world_initialized) {
        return bbox_h * engine->height_scale_factor;
    }

    return AVERAGE_PERSON_HEIGHT;
}

void spatial_engine_update_trajectory(SpatialLocalizationEngine* engine, int track_id, const SpatialPosition* position) {
    if (!engine || !position) return;
    if (track_id < 0 || track_id >= SPATIAL_MAX_PERSONS) return;

    TrajectoryBuffer* buf = &engine->trajectories[track_id];

    if (engine->smoothing_active[track_id]) {
        const SpatialPosition* prev = &engine->coord_smoothing_buffer[track_id];
        SpatialPosition smoothed;
        smoothed.x = engine->coord_smoothing_alpha * position->x + (1.0f - engine->coord_smoothing_alpha) * prev->x;
        smoothed.y = engine->coord_smoothing_alpha * position->y + (1.0f - engine->coord_smoothing_alpha) * prev->y;
        smoothed.z = engine->coord_smoothing_alpha * position->z + (1.0f - engine->coord_smoothing_alpha) * prev->z;
        smoothed.depth_confidence = position->depth_confidence;

        engine->coord_smoothing_buffer[track_id] = *position;
        trajectory_buffer_append(buf, &smoothed);
    } else {
        engine->coord_smoothing_buffer[track_id] = *position;
        engine->smoothing_active[track_id] = true;
        trajectory_buffer_append(buf, position);
    }

    engine->trajectory_active[track_id] = true;
}

const SpatialPosition* spatial_engine_get_trajectory(SpatialLocalizationEngine* engine, int track_id, int* out_count) {
    if (!engine || track_id < 0 || track_id >= SPATIAL_MAX_PERSONS) {
        if (out_count) *out_count = 0;
        return NULL;
    }

    if (out_count) *out_count = engine->trajectories[track_id].count;
    return engine->trajectories[track_id].positions;
}

bool spatial_engine_get_velocity(SpatialLocalizationEngine* engine, int track_id, float dt, float out_velocity[3]) {
    if (!engine || !out_velocity || track_id < 0 || track_id >= SPATIAL_MAX_PERSONS) return false;

    const TrajectoryBuffer* buf = &engine->trajectories[track_id];
    if (buf->count < 2) return false;

    const SpatialPosition* last = &buf->positions[buf->count - 1];
    const SpatialPosition* prev = &buf->positions[buf->count - 2];

    out_velocity[0] = (last->x - prev->x) / dt;
    out_velocity[1] = (last->y - prev->y) / dt;
    out_velocity[2] = (last->z - prev->z) / dt;

    float speed = sqrtf(out_velocity[0]*out_velocity[0] + out_velocity[1]*out_velocity[1] + out_velocity[2]*out_velocity[2]);
    if (speed > 50.0f) return false;

    return true;
}

void spatial_engine_clear_trajectories(SpatialLocalizationEngine* engine) {
    if (!engine) return;
    memset(engine->trajectories, 0, sizeof(engine->trajectories));
    memset(engine->trajectory_active, 0, sizeof(engine->trajectory_active));
    memset(engine->smoothing_active, 0, sizeof(engine->smoothing_active));
    log_info("All trajectories cleared");
}

int spatial_engine_get_active_tracks(const SpatialLocalizationEngine* engine, int* out_ids, int max_count) {
    if (!engine || !out_ids) return 0;

    int count = 0;
    for (int i = 0; i < SPATIAL_MAX_PERSONS && count < max_count; i++) {
        if (engine->trajectory_active[i]) {
            out_ids[count++] = i;
        }
    }
    return count;
}

void spatial_engine_set_camera_pose(SpatialLocalizationEngine* engine, float pitch, float roll, float yaw) {
    if (!engine) return;

    engine->camera_pitch = pitch;
    engine->camera_roll = roll;
    engine->camera_yaw = yaw;
    engine->has_camera_pose = true;
}
