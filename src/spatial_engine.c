#include "spatial_engine.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Anatomical constants ── */
#define AVERAGE_EYE_HEIGHT          1.6f
#define AVERAGE_PERSON_HEIGHT       1.7f
#define COORD_SMOOTHING_ALPHA       0.25f

/* ── COCO keypoint indices (for fast lookup without strcmp) ── */
enum {
    KPT_NOSE = 0, KPT_LEFT_EYE = 1, KPT_RIGHT_EYE = 2,
    KPT_LEFT_EAR = 3, KPT_RIGHT_EAR = 4,
    KPT_LEFT_SHOULDER = 5, KPT_RIGHT_SHOULDER = 6,
    KPT_LEFT_ELBOW = 7, KPT_RIGHT_ELBOW = 8,
    KPT_LEFT_WRIST = 9, KPT_RIGHT_WRIST = 10,
    KPT_LEFT_HIP = 11, KPT_RIGHT_HIP = 12,
    KPT_LEFT_KNEE = 13, KPT_RIGHT_KNEE = 14,
    KPT_LEFT_ANKLE = 15, KPT_RIGHT_ANKLE = 16
};

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
    engine->depth_ema_alpha = DEPTH_EMA_ALPHA;
    engine->depth_outlier_mad_mult = DEPTH_OUTLIER_MAD_MULT;
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

/*
 * ── Improved depth estimation (2024-2026 academic refinements) ──
 *
 * Replaces the simple Z = fy × Havg / h_bbox with:
 *   1. Perspective tilt-correction: accounts for camera pitch AND
 *      person's vertical offset from optical axis
 *   2. Multi-point anatomical sampling: uses body-part proportions
 *      (head, shoulder, hip, knee, ankle) for independent depth samples
 *   3. Robust fusion: median + MAD outlier rejection on the samples
 *   4. Confidence: based on inter-sample agreement, not just distance
 *
 * When pose keypoints are available, anatomical ratios use actual
 * keypoint-to-keypoint pixel distances (more accurate than bbox subdivision).
 * Otherwise, subdivides the bbox by anthropometric ratios.
 *
 * Refs:
 *   - FOV zone model: Measurement 236 (Elsevier, 2024)
 *   - DGC geometric constraints: J. Electronic Imaging (2024)
 *   - Adaptive Surface Normal: TPAMI (2024)
 */

/* ── Single depth estimate from a pixel height + body ratio ── */
static inline float depth_from_pixel_height(float fy, float pixel_h, float body_ratio,
                                              float avg_height, float cos_tilt) {
    if (pixel_h <= 0.5f) return -1.0f;  /* degenerate */
    /* Z = (Havg × ratio × fy × cos_tilt) / pixel_h */
    return (avg_height * body_ratio * fy * cos_tilt) / pixel_h;
}

/* ── Quick median of float array (in-place, destroys input) ── */
static float quick_median(float* arr, int n) {
    if (n <= 0) return 0.0f;
    if (n == 1) return arr[0];
    /* Partial sort to middle (simple bubble for small n, n ≤ 6 typical) */
    for (int i = 0; i <= n / 2; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) min_idx = j;
        }
        float tmp = arr[i]; arr[i] = arr[min_idx]; arr[min_idx] = tmp;
    }
    return arr[n / 2];
}

/*
 * Estimate depth using anatomical multi-point sampling from keypoints.
 * Returns depth in meters and confidence [0,1] via *out_conf.
 */
static float estimate_depth_from_pose(const SpatialLocalizationEngine* engine,
                                        const Detection* detection,
                                        const PoseEstimation* pose,
                                        float* out_conf) {
    float fy = engine->camera_matrix[1][1];
    float cy = engine->camera_matrix[1][2];
    float Havg = engine->avg_person_height;
    float pitch_rad = engine->has_camera_pose ? engine->camera_pitch * ((float)M_PI / 180.0f) : 0.0f;
    float cos_tilt = cosf(pitch_rad);

    /* Collect keypoint y-coordinates with per-kpt confidence */
    float kp_y[17];
    float kp_c[17];
    for (int i = 0; i < 17 && i < pose->num_keypoints; i++) {
        kp_y[i] = pose->keypoints[i].y;
        kp_c[i] = pose->keypoints[i].confidence;
    }

    float estimates[8];
    int n_est = 0;

    /* Define measurement pairs: {upper_kpt, lower_kpt, body_ratio, label} */
    struct { int upper; int lower; float ratio; } pairs[] = {
        {KPT_NOSE,          KPT_LEFT_ANKLE,  ANTHRO_ANKLE_TO_HEIGHT},  /* nose→ankle ≈ full height */
        {KPT_NOSE,          KPT_RIGHT_ANKLE, ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_LEFT_SHOULDER, KPT_LEFT_ANKLE,  ANTHRO_SHOULDER_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_RIGHT_SHOULDER,KPT_RIGHT_ANKLE, ANTHRO_SHOULDER_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_LEFT_HIP,      KPT_LEFT_ANKLE,  ANTHRO_HIP_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_RIGHT_HIP,     KPT_RIGHT_ANKLE, ANTHRO_HIP_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_LEFT_KNEE,     KPT_LEFT_ANKLE,  ANTHRO_KNEE_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
        {KPT_RIGHT_KNEE,    KPT_RIGHT_ANKLE, ANTHRO_KNEE_TO_HEIGHT - ANTHRO_ANKLE_TO_HEIGHT},
    };
    int npairs = sizeof(pairs) / sizeof(pairs[0]);

    for (int i = 0; i < npairs; i++) {
        int u = pairs[i].upper, l = pairs[i].lower;
        if (kp_c[u] < DEPTH_MIN_KEYPOINT_CONF || kp_c[l] < DEPTH_MIN_KEYPOINT_CONF)
            continue;  /* skip low-confidence kpts */

        float pixel_h = kp_y[l] - kp_y[u];  /* upper is ABOVE lower in image (smaller y) */
        if (pixel_h <= 1.0f) {
            /* Try reversed order — ankle might be above nose in some poses */
            pixel_h = kp_y[u] - kp_y[l];
        }
        if (pixel_h <= 1.0f) continue;

        /* Tilt correction: person's vertical angle from optical axis */
        float v_center = (kp_y[u] + kp_y[l]) * 0.5f;
        float alpha = atan2f(fabsf(v_center - cy), fy);
        float cos_alpha = cosf(alpha);

        float z = depth_from_pixel_height(fy, pixel_h, pairs[i].ratio, Havg,
                                           cos_tilt * cos_alpha);
        if (z > 0.0f) {
            estimates[n_est++] = z;
        }
    }

    /* Fallback: use bbox subdivision if keypoints insufficient */
    if (n_est < 2) {
        float bbox_h = bbox_height(&detection->bbox);
        if (bbox_h <= 0.0f) { *out_conf = 0.0f; return SPATIAL_MAX_DEPTH; }

        float v_center = bbox_center_y(&detection->bbox);
        float alpha = atan2f(fabsf(v_center - cy), fy);
        float cos_alpha = cosf(alpha);

        /* Sample at 3 anatomical heights within bbox */
        float ratios[] = {1.0f, ANTHRO_SHOULDER_TO_HEIGHT, ANTHRO_HIP_TO_HEIGHT};
        for (int i = 0; i < 3; i++) {
            float z = depth_from_pixel_height(fy, bbox_h, ratios[i], Havg,
                                               cos_tilt * cos_alpha);
            if (z > 0.0f) estimates[n_est++] = z;
        }
    }

    if (n_est == 0) { *out_conf = 0.0f; return SPATIAL_MAX_DEPTH; }

    /* ── Robust fusion: median + MAD outlier rejection ── */
    float sorted[8];
    memcpy(sorted, estimates, (size_t)n_est * sizeof(float));
    float med = quick_median(sorted, n_est);

    /* MAD (Median Absolute Deviation) */
    float abs_dev[8];
    for (int i = 0; i < n_est; i++) abs_dev[i] = fabsf(estimates[i] - med);
    float mad = quick_median(abs_dev, n_est);
    if (mad < 0.01f) mad = 0.01f;  /* floor to avoid division by zero */

    /* Remove outliers and average inliers */
    float sum = 0.0f;
    int n_inliers = 0;
    float threshold = engine->depth_outlier_mad_mult * mad;
    for (int i = 0; i < n_est; i++) {
        if (fabsf(estimates[i] - med) <= threshold) {
            sum += estimates[i];
            n_inliers++;
        }
    }

    float fused = (n_inliers > 0) ? (sum / (float)n_inliers) : med;
    fused = UTILS_CLAMP(fused, SPATIAL_MIN_DEPTH, SPATIAL_MAX_DEPTH);

    /* Confidence: based on estimate agreement (inverse of normalized MAD) */
    float conf = 1.0f - UTILS_MIN(mad / (med + 0.01f), 0.75f);
    conf = UTILS_MAX(conf, DEPTH_MIN_CONFIDENCE);

    *out_conf = conf;
    return fused;
}

/*
 * Estimate depth from bbox only (no pose data available).
 * Uses 3-point anatomical sampling within the bbox.
 */
static float estimate_depth_from_bbox(const SpatialLocalizationEngine* engine,
                                        const Detection* detection,
                                        float* out_conf) {
    float fy = engine->camera_matrix[1][1];
    float cy = engine->camera_matrix[1][2];
    float Havg = engine->avg_person_height;
    float pitch_rad = engine->has_camera_pose ? engine->camera_pitch * ((float)M_PI / 180.0f) : 0.0f;
    float cos_tilt = cosf(pitch_rad);

    float bbox_h = bbox_height(&detection->bbox);
    if (bbox_h <= 0.0f) { *out_conf = 0.0f; return SPATIAL_MAX_DEPTH; }

    float v_center = bbox_center_y(&detection->bbox);
    float alpha = atan2f(fabsf(v_center - cy), fy);
    float cos_alpha = cosf(alpha);

    /* 3-point sampling at head, shoulder, hip proportions */
    float ratios[] = {ANTHRO_ANKLE_TO_HEIGHT,           /* ≈ full height */
                      ANTHRO_SHOULDER_TO_HEIGHT,         /* shoulder→foot */
                      ANTHRO_HIP_TO_HEIGHT};             /* hip→foot */
    float estimates[3];
    int n_est = 0;

    for (int i = 0; i < 3; i++) {
        float z = depth_from_pixel_height(fy, bbox_h, ratios[i], Havg,
                                           cos_tilt * cos_alpha);
        if (z > 0.0f) estimates[n_est++] = z;
    }

    if (n_est == 0) { *out_conf = 0.0f; return SPATIAL_MAX_DEPTH; }

    /* Median fusion (n ≤ 3, no MAD needed) */
    float fused;
    if (n_est == 1) {
        fused = estimates[0];
    } else {
        float sorted[3];
        memcpy(sorted, estimates, (size_t)n_est * sizeof(float));
        fused = quick_median(sorted, n_est);
    }

    fused = UTILS_CLAMP(fused, SPATIAL_MIN_DEPTH, SPATIAL_MAX_DEPTH);

    /* Distance-based confidence (far = less reliable) */
    float dist_conf = UTILS_MAX(0.3f, 1.0f - (fused / SPATIAL_MAX_DEPTH) * 0.7f);
    /* Inter-sample agreement */
    float spread = (n_est >= 2) ? fabsf(estimates[0] - estimates[n_est-1]) / (fused + 0.01f) : 0.0f;
    float agree_conf = 1.0f - UTILS_MIN(spread, 0.5f);

    *out_conf = UTILS_MAX(UTILS_MIN(dist_conf, agree_conf), DEPTH_MIN_CONFIDENCE);
    return fused;
}

/*
 * Temporal depth smoothing via EMA (Exponential Moving Average) per track.
 * Reduces frame-to-frame depth jitter while responding to real changes.
 */
static float smooth_depth_ema(SpatialLocalizationEngine* engine, int track_id,
                                float raw_depth, float raw_conf) {
    if (track_id < 0 || track_id >= SPATIAL_MAX_PERSONS) return raw_depth;

    if (!engine->depth_ema_active[track_id]) {
        engine->depth_ema[track_id] = raw_depth;
        engine->depth_ema_variance[track_id] = 0.0f;
        engine->depth_ema_active[track_id] = true;
        return raw_depth;
    }

    /* Adaptive alpha: lower when confidence is high (trust new measurement more) */
    float alpha = engine->depth_ema_alpha * raw_conf;

    float prev = engine->depth_ema[track_id];
    float smoothed = alpha * raw_depth + (1.0f - alpha) * prev;

    /* Update variance estimate (for future use) */
    float delta = raw_depth - smoothed;
    engine->depth_ema_variance[track_id] =
        alpha * delta * delta + (1.0f - alpha) * engine->depth_ema_variance[track_id];
    engine->depth_ema[track_id] = smoothed;

    return smoothed;
}

SpatialResult spatial_engine_calculate_position(SpatialLocalizationEngine* engine,
                                                 const Detection* detection,
                                                 int frame_width, int frame_height,
                                                 const float* depth_map, int depth_w, int depth_h,
                                                 const PoseEstimation* pose, int track_id) {
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

    /* ── Source 1: External depth map (future: depth model output) ── */
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
    }
    /* ── Source 2: Anatomical multi-point estimation (with or without pose) ── */
    else if (pose && pose->num_keypoints >= 4) {
        depth = estimate_depth_from_pose(engine, detection, pose, &confidence);
    }
    /* ── Source 3: Bbox subdivision (no pose data) ── */
    else {
        depth = estimate_depth_from_bbox(engine, detection, &confidence);
    }

    /* ── Temporal EMA smoothing per track ── */
    if (track_id >= 0) {
        depth = smooth_depth_ema(engine, track_id, depth, confidence);
    }

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
    result.position.depth_confidence = confidence;
    result.position.is_valid = true;
    result.position.confidence = confidence;
    result.position.estimated_height = engine->avg_person_height;
    result.depth = depth;
    result.confidence = confidence;

    /* Per-30-detections spatial log */
    {
        static int spatial_cnt = 0;
        spatial_cnt++;
        if (spatial_cnt % 30 == 0) {
            const char* method = "unknown";
            if (depth_map && depth_w > 0 && depth_h > 0) method = "depth_map";
            else if (pose && pose->num_keypoints >= 4) method = "pose";
            else method = "bbox";
            log_info("[Spatial] depth#%d | track=%d method=%s | "
                     "depth=%.2fm conf=%.2f | bbox=(%.0f,%.0f) | "
                     "tilt: pitch=%.2f° roll=%.2f°",
                     spatial_cnt, track_id, method,
                     depth, confidence,
                     bbox_center_x(&detection->bbox), bbox_center_y(&detection->bbox),
                     engine->camera_pitch * 57.3f, engine->camera_roll * 57.3f);
        }
    }

    return result;
}

float spatial_engine_calculate_height(SpatialLocalizationEngine* engine,
                                       const Detection* detection,
                                       const PoseEstimation* pose) {
    if (!engine) return AVERAGE_PERSON_HEIGHT;

    /* ── Use pose keypoints for direct pixel height measurement ── */
    if (pose && pose->num_keypoints >= 4) {
        /* Find nose (highest reliable point) and ankle (lowest, on ground) */
        float nose_y = 0.0f, nose_c = 0.0f;
        float ankle_y = 0.0f, ankle_c = 0.0f;

        if (pose->keypoints[KPT_NOSE].confidence > DEPTH_MIN_KEYPOINT_CONF) {
            nose_y = pose->keypoints[KPT_NOSE].y;
            nose_c = pose->keypoints[KPT_NOSE].confidence;
        }
        /* Use the higher-confidence ankle */
        float la_c = pose->keypoints[KPT_LEFT_ANKLE].confidence;
        float ra_c = pose->keypoints[KPT_RIGHT_ANKLE].confidence;
        if (la_c > DEPTH_MIN_KEYPOINT_CONF || ra_c > DEPTH_MIN_KEYPOINT_CONF) {
            if (la_c >= ra_c) {
                ankle_y = pose->keypoints[KPT_LEFT_ANKLE].y;
                ankle_c = la_c;
            } else {
                ankle_y = pose->keypoints[KPT_RIGHT_ANKLE].y;
                ankle_c = ra_c;
            }
        }

        if (nose_c > DEPTH_MIN_KEYPOINT_CONF && ankle_c > DEPTH_MIN_KEYPOINT_CONF) {
            float pixel_h = ankle_y - nose_y;  /* nose is above ankle */
            if (pixel_h > 5.0f) {
                /* Pixel height → metric using world calibration */
                if (engine->world_initialized) {
                    return pixel_h * engine->height_scale_factor;
                }
                /* Without world calibration, use depth-based scaling */
                return UTILS_CLAMP(pixel_h * 0.004f, 0.5f, 2.5f);
            }
        }
    }

    /* ── Fallback: bbox-based height estimation ── */
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
    memset(engine->depth_ema, 0, sizeof(engine->depth_ema));
    memset(engine->depth_ema_variance, 0, sizeof(engine->depth_ema_variance));
    memset(engine->depth_ema_active, 0, sizeof(engine->depth_ema_active));
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

/*
 * Depth consistency check: reject detections whose depth estimate
 * changes implausibly compared to the tracked object's recent history.
 *
 * A depth ratio > 2.0× or < 0.4× compared to the rolling average of
 * the last 5 trajectory points suggests either:
 *   - A false detection on background clutter
 *   - An occlusion event (different person briefly occluding the tracked one)
 *   - A depth estimation failure (person height prior violated)
 *
 * Returns true if the depth is consistent with recent history.
 */
bool spatial_engine_check_depth_consistency(SpatialLocalizationEngine* engine,
                                              int track_id, float new_depth,
                                              float max_jump_ratio) {
    if (!engine || track_id < 0 || track_id >= SPATIAL_MAX_PERSONS) return true;

    const TrajectoryBuffer* buf = &engine->trajectories[track_id];
    if (buf->count < 3) return true;  /* Not enough history yet */

    /* Compute rolling average of last N depth values */
    int n = UTILS_MIN(5, buf->count);
    float avg_depth = 0.0f;
    int valid_pts = 0;
    for (int i = buf->count - n; i < buf->count; i++) {
        if (buf->positions[i].is_valid) {
            avg_depth += buf->positions[i].z;
            valid_pts++;
        }
    }

    if (valid_pts == 0) return true;
    avg_depth /= (float)valid_pts;

    float ratio = new_depth / (avg_depth + 0.01f);
    return (ratio >= (1.0f / max_jump_ratio) && ratio <= max_jump_ratio);
}
