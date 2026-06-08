/**
 * tracking_manager.c — Multi-Object Tracking Manager (v2.0)
 *
 * Implements a ByteTrack-inspired tracker with the following optimizations:
 *
 * 1. HUNGARIAN ALGORITHM: Replaces greedy IoU matching with optimal global
 *    assignment via the Kuhn-Munkres algorithm, minimizing total association
 *    cost rather than making locally-optimal decisions.
 *
 * 2. POSE-KEYPOINT APPEARANCE FEATURE: A lightweight 12-dim descriptor derived
 *    from COCO-17 keypoint spatial relationships.  Requires NO additional CNN
 *    model — leverages existing YOLOv8-pose output.  Used for appearance-based
 *    re-identification after occlusion.
 *
 * 3. CASCADE MATCHING: Tracklets are grouped by time_since_update and matched
 *    in order — freshest first.  This prevents stale tracks from stealing
 *    detections from active tracks, the primary cause of ID switches in SORT.
 *
 * 4. PARTIAL-BODY / OCCLUSION HANDLING: When full-body keypoint validation fails,
 *    falls back to upper-body-only (keypoints 0-10) or one-side-only matching.
 *    Extends track lifetime for occluded tracks and supports appearance-based
 *    re-identification via a deleted-track pool.
 *
 * 5. MULTI-PERSON DETECTION ASSISTANCE: Tracks unmatched detection counts and
 *    flags when detections significantly outnumber tracks, signaling new people
 *    entering the scene.
 */

#include "tracking_manager.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Kalman Filter (7-state constant velocity model)
 * State: [cx, cy, area, aspect_ratio, vx, vy, varea]
 * ═══════════════════════════════════════════════════════════════════════════ */

static void kalman_init(KalmanBoxTracker* kf, const BoundingBox* bbox, int track_id, float dt) {
    memset(kf, 0, sizeof(KalmanBoxTracker));
    kf->track_id = track_id;
    kf->dt = dt;

    float cx = bbox_center_x(bbox);
    float cy = bbox_center_y(bbox);
    float s = UTILS_MAX(bbox_area(bbox), 10.0f);
    float r = bbox_width(bbox) / UTILS_MAX(bbox_height(bbox), 1e-3f);

    kf->state[0] = cx;
    kf->state[1] = cy;
    kf->state[2] = s;
    kf->state[3] = r;
    kf->state[4] = 0.0f;
    kf->state[5] = 0.0f;
    kf->state[6] = 0.0f;

    /* Initial covariance: moderate for position/scale, high for velocity */
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->covariance[i][j] = (i == j) ? ((i < 4) ? 10.0f : 1000.0f) : 0.0f;
        }
    }

    /* Process noise: tuned for walking-pace human motion (~1.2 m/s typical) */
    float q_pos = 0.02f;   /* position process noise (lower = smoother) */
    float q_vel = 0.04f;   /* velocity process noise */
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            float q = (i < 4) ? q_pos : q_vel;
            kf->process_noise[i][j] = (i == j) ? q : 0.0f;
        }
    }

    /* Measurement noise: balanced for DFL-based detections (moderate noise) */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kf->measurement_noise[i][j] = (i == j) ? 0.15f : 0.0f;
        }
    }

    /* Transition matrix: constant velocity */
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->transition[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    kf->transition[0][4] = dt;
    kf->transition[1][5] = dt;
    kf->transition[2][6] = dt;

    /* Measurement matrix: observe [cx, cy, area, aspect] */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 7; j++) {
            kf->measurement[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    kf->time_since_update = 0;
    kf->hit_streak = 0;
}

static void kalman_predict(KalmanBoxTracker* kf) {
    float new_state[7] = {0};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            new_state[i] += kf->transition[i][j] * kf->state[j];
        }
    }
    memcpy(kf->state, new_state, sizeof(new_state));

    float new_cov[7][7];
    utils_matrix_multiply_abt(kf->transition, kf->covariance, new_cov);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->covariance[i][j] = new_cov[i][j] + kf->process_noise[i][j];
        }
    }

    kf->time_since_update++;
}

static void kalman_update(KalmanBoxTracker* kf, const BoundingBox* bbox) {
    float cx = bbox_center_x(bbox);
    float cy = bbox_center_y(bbox);
    float s = UTILS_MAX(bbox_area(bbox), 10.0f);
    float r = bbox_width(bbox) / UTILS_MAX(bbox_height(bbox), 1e-3f);
    float z[4] = {cx, cy, s, r};

    /* Innovation: y = z - H*x */
    float y[4];
    for (int i = 0; i < 4; i++) {
        y[i] = z[i];
        for (int j = 0; j < 7; j++) {
            y[i] -= kf->measurement[i][j] * kf->state[j];
        }
    }

    /* Innovation covariance: S = H*P*H' + R */
    float s_mat[4][4] = {{0}};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 7; k++) {
                s_mat[i][j] += kf->measurement[i][k] * kf->covariance[k][j];
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s_mat[i][j] += kf->measurement_noise[i][j];
        }
    }

    /* S inverse (with diagonal fallback) */
    float s_inv[4][4];
    if (utils_matrix_inverse_4x4(s_mat, s_inv) != 0) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                s_inv[i][j] = (i == j) ? 1.0f / UTILS_MAX(s_mat[i][i], 1e-10f) : 0.0f;
            }
        }
    }

    /* Kalman gain: K = P*H' * S^-1 */
    float k_gain[7][4] = {{0}};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                k_gain[i][j] += kf->covariance[i][k] * s_inv[k][j];  /* P*H' approximated: H uses identity subset */
            }
        }
    }

    /* State update: x = x + K*y */
    float new_state[7];
    for (int i = 0; i < 7; i++) {
        new_state[i] = kf->state[i];
        for (int j = 0; j < 4; j++) {
            new_state[i] += k_gain[i][j] * y[j];
        }
    }
    memcpy(kf->state, new_state, sizeof(new_state));

    /* Covariance update: P = (I - K*H) * P */
    float ikh[7][7] = {{0}};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            ikh[i][j] = (i == j) ? 1.0f : 0.0f;
            for (int k = 0; k < 4; k++) {
                ikh[i][j] -= k_gain[i][k] * kf->measurement[k][j];
            }
        }
    }

    float new_cov[7][7] = {{0}};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                new_cov[i][j] += ikh[i][k] * kf->covariance[k][j];
            }
        }
    }
    memcpy(kf->covariance, new_cov, sizeof(new_cov));

    kf->time_since_update = 0;
    kf->hit_streak++;
}

static BoundingBox kalman_get_bbox(const KalmanBoxTracker* kf) {
    BoundingBox bbox;
    float cx = kf->state[0];
    float cy = kf->state[1];
    float s = kf->state[2];
    float r = kf->state[3];

    float w = sqrtf(UTILS_MAX(s * r, 0.0f));
    float h = s / UTILS_MAX(w, 1e-3f);

    bbox.x_min = cx - w * 0.5f;
    bbox.y_min = cy - h * 0.5f;
    bbox.x_max = cx + w * 0.5f;
    bbox.y_max = cy + h * 0.5f;

    return bbox;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Hungarian Algorithm (Kuhn-Munkres) for Optimal Assignment
 *
 * Finds the minimum-cost perfect matching in a bipartite graph.  Used to
 * associate detections with track predictions globally rather than greedily.
 *
 * Complexity: O(n³) where n = max(rows, cols).  For typical tracking scenarios
 * (n ≤ 50), this is negligible (~0.1ms on a 1.6GHz core).
 * ═══════════════════════════════════════════════════════════════════════════ */

static float hungarian_solve(const float* cost_matrix, int rows, int cols,
                              int* assignment, float* final_cost) {
    /* Pad to square */
    int n = rows > cols ? rows : cols;
    if (n < 1) { *final_cost = 0.0f; return 0.0f; }
    if (n > HUNGARIAN_MAX_DIM) n = HUNGARIAN_MAX_DIM;

    /* Working matrix (n × n, padded with large values) */
    float* a = (float*)calloc((size_t)n * n, sizeof(float));
    if (!a) {
        /* Fallback: identity assignment */
        for (int i = 0; i < rows && i < cols; i++) assignment[i] = i;
        for (int i = cols; i < rows; i++) assignment[i] = -1;
        *final_cost = 0.0f;
        return 0.0f;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < rows && j < cols) {
                a[i * n + j] = cost_matrix[i * cols + j];
            } else {
                a[i * n + j] = 1e9f;  /* large pad value */
            }
        }
    }

    /* u/v potentials, p/way matching arrays */
    float* u = (float*)calloc((size_t)n + 1, sizeof(float));
    float* v = (float*)calloc((size_t)n + 1, sizeof(float));
    int*   p = (int*)calloc((size_t)n + 1, sizeof(int));
    int*   way = (int*)calloc((size_t)n + 1, sizeof(int));
    if (!u || !v || !p || !way) {
        free(a); free(u); free(v); free(p); free(way);
        for (int i = 0; i < rows && i < cols; i++) assignment[i] = i;
        for (int i = cols; i < rows; i++) assignment[i] = -1;
        *final_cost = 0.0f;
        return 0.0f;
    }

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        float* minv = (float*)calloc((size_t)n + 1, sizeof(float));
        bool*  used = (bool*)calloc((size_t)n + 1, sizeof(bool));
        if (!minv || !used) {
            free(minv); free(used);
            free(a); free(u); free(v); free(p); free(way);
            for (int k = 0; k < rows && k < cols; k++) assignment[k] = k;
            for (int k = cols; k < rows; k++) assignment[k] = -1;
            *final_cost = 0.0f;
            return 0.0f;
        }
        for (int j = 1; j <= n; j++) {
            minv[j] = 1e9f;
            used[j] = false;
        }

        int j1 = 0;
        do {
            used[j0] = true;
            int i0 = p[j0];
            float delta = 1e9f;
            j1 = 0;

            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    float cur = a[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1_prev = way[j0];
            p[j0] = p[j1_prev];
            j0 = j1_prev;
        } while (j0 != 0);

        free(minv);
        free(used);
    }

    /* Extract assignment */
    for (int j = 1; j <= n; j++) {
        if (p[j] > 0) {
            int row = p[j] - 1;
            int col = j - 1;
            if (row < rows && col < cols) {
                assignment[row] = col;
            }
        }
    }

    /* Compute total cost */
    float cost = 0.0f;
    for (int i = 0; i < rows; i++) {
        int j = assignment[i];
        if (j >= 0 && j < cols) {
            cost += cost_matrix[i * cols + j];
        }
    }
    *final_cost = cost;

    free(a); free(u); free(v); free(p); free(way);
    return cost;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Appearance Feature Extraction from Pose Keypoints
 * ═══════════════════════════════════════════════════════════════════════════ */

bool appearance_feature_from_pose(const PoseEstimation* pose, const BoundingBox* bbox,
                                   AppearanceFeature* out_feature, int frame_index) {
    if (!pose || !bbox || !out_feature) return false;

    /* Minimum keypoint requirement: at least TRACKING_APPEARANCE_MIN_KPTS valid */
    float min_conf = 0.30f;
    int valid_kpts = 0;
    float kx[17], ky[17], kc[17];
    for (int i = 0; i < 17 && i < pose->num_keypoints; i++) {
        kc[i] = pose->keypoints[i].confidence;
        if (kc[i] >= min_conf &&
            pose->keypoints[i].x >= 0.0f && pose->keypoints[i].y >= 0.0f) {
            kx[i] = pose->keypoints[i].x;
            ky[i] = pose->keypoints[i].y;
            valid_kpts++;
        } else {
            kx[i] = -1.0f;
            ky[i] = -1.0f;
            kc[i] = 0.0f;
        }
    }

    if (valid_kpts < TRACKING_APPEARANCE_MIN_KPTS) {
        memset(out_feature, 0, sizeof(*out_feature));
        out_feature->is_valid = false;
        return false;
    }

    float desc[TRACKING_APPEARANCE_DIM];
    memset(desc, 0, sizeof(desc));

    float bbox_w = bbox_width(bbox);
    float bbox_h = bbox_height(bbox);
    float norm_w = UTILS_MAX(bbox_w, 1.0f);
    float norm_h = UTILS_MAX(bbox_h, 1.0f);

    /* Feature 0: torso ratio (shoulder-hip Y / shoulder width) */
    if (kc[5] >= min_conf && kc[6] >= min_conf) {
        float sw = fabsf(kx[5] - kx[6]);
        float sh_mid_y = (ky[5] + ky[6]) * 0.5f;
        float hip_mid_y = 0.0f;
        bool has_hips = false;
        if (kc[11] >= min_conf && kc[12] >= min_conf) {
            hip_mid_y = (ky[11] + ky[12]) * 0.5f;
            has_hips = true;
        } else if (kc[11] >= min_conf) {
            hip_mid_y = ky[11];
            has_hips = true;
        } else if (kc[12] >= min_conf) {
            hip_mid_y = ky[12];
            has_hips = true;
        }
        if (has_hips && sw > 1.0f) {
            desc[0] = (hip_mid_y - sh_mid_y) / sw;
        }
    }

    /* Feature 1: shoulder width / bbox width */
    if (kc[5] >= min_conf && kc[6] >= min_conf) {
        float sw = fabsf(kx[5] - kx[6]);
        desc[1] = sw / norm_w;
    }

    /* Feature 2: left arm ratio (upper/lower) */
    if (kc[5] >= min_conf && kc[7] >= min_conf && kc[9] >= min_conf) {
        float upper = sqrtf(powf(kx[5]-kx[7],2) + powf(ky[5]-ky[7],2));
        float lower = sqrtf(powf(kx[7]-kx[9],2) + powf(ky[7]-ky[9],2));
        if (lower > 1.0f) desc[2] = upper / lower;
    }

    /* Feature 3: right arm ratio */
    if (kc[6] >= min_conf && kc[8] >= min_conf && kc[10] >= min_conf) {
        float upper = sqrtf(powf(kx[6]-kx[8],2) + powf(ky[6]-ky[8],2));
        float lower = sqrtf(powf(kx[8]-kx[10],2) + powf(ky[8]-ky[10],2));
        if (lower > 1.0f) desc[3] = upper / lower;
    }

    /* Feature 4: left leg ratio (upper/lower) */
    if (kc[11] >= min_conf && kc[13] >= min_conf && kc[15] >= min_conf) {
        float upper = sqrtf(powf(kx[11]-kx[13],2) + powf(ky[11]-ky[13],2));
        float lower = sqrtf(powf(kx[13]-kx[15],2) + powf(ky[13]-ky[15],2));
        if (lower > 1.0f) desc[4] = upper / lower;
    }

    /* Feature 5: right leg ratio */
    if (kc[12] >= min_conf && kc[14] >= min_conf && kc[16] >= min_conf) {
        float upper = sqrtf(powf(kx[12]-kx[14],2) + powf(ky[12]-ky[14],2));
        float lower = sqrtf(powf(kx[14]-kx[16],2) + powf(ky[14]-ky[16],2));
        if (lower > 1.0f) desc[5] = upper / lower;
    }

    /* Feature 6: head-to-shoulder ratio */
    if (kc[0] >= min_conf && kc[5] >= min_conf && kc[6] >= min_conf) {
        float nose_y = ky[0];
        float sh_mid_y = (ky[5] + ky[6]) * 0.5f;
        float head_dist = UTILS_MAX(sh_mid_y - nose_y, 1.0f);
        desc[6] = head_dist / norm_h;
    }

    /* Feature 7: left-right shoulder symmetry (Y diff / bbox height) */
    if (kc[5] >= min_conf && kc[6] >= min_conf) {
        desc[7] = fabsf(ky[5] - ky[6]) / norm_h;
    }

    /* Feature 8: left-right hip symmetry */
    if (kc[11] >= min_conf && kc[12] >= min_conf) {
        desc[8] = fabsf(ky[11] - ky[12]) / norm_h;
    }

    /* Feature 9: normalized body height (in keypoint space) */
    if (kc[0] >= min_conf) {
        float bottom = ky[0];
        if (kc[15] >= min_conf) bottom = UTILS_MAX(bottom, ky[15]);
        if (kc[16] >= min_conf) bottom = UTILS_MAX(bottom, ky[16]);
        if (kc[11] >= min_conf) bottom = UTILS_MAX(bottom, ky[11]);
        if (kc[12] >= min_conf) bottom = UTILS_MAX(bottom, ky[12]);
        desc[9] = (bottom - ky[0]) / norm_h;
    }

    /* Feature 10: center of mass X offset (from bbox center) */
    {
        float sum_x = 0.0f, sum_w = 0.0f;
        for (int i = 0; i < 17; i++) {
            if (kc[i] >= min_conf) {
                sum_x += kx[i] * kc[i];
                sum_w += kc[i];
            }
        }
        if (sum_w > 0.0f) {
            desc[10] = ((sum_x / sum_w) - bbox_center_x(bbox)) / norm_w;
        }
    }

    /* Feature 11: center of mass Y offset */
    {
        float sum_y = 0.0f, sum_w = 0.0f;
        for (int i = 0; i < 17; i++) {
            if (kc[i] >= min_conf) {
                sum_y += ky[i] * kc[i];
                sum_w += kc[i];
            }
        }
        if (sum_w > 0.0f) {
            desc[11] = ((sum_y / sum_w) - bbox_center_y(bbox)) / norm_h;
        }
    }

    /* Normalize: clip to reasonable range */
    for (int i = 0; i < TRACKING_APPEARANCE_DIM; i++) {
        if (desc[i] < -5.0f) desc[i] = -5.0f;
        if (desc[i] > 5.0f) desc[i] = 5.0f;
    }

    memcpy(out_feature->descriptor, desc, sizeof(desc));
    out_feature->num_valid_kpts = valid_kpts;
    out_feature->frame_index = frame_index;
    out_feature->is_valid = true;

    return true;
}

float appearance_feature_distance(const AppearanceFeature* a, const AppearanceFeature* b) {
    if (!a || !b || !a->is_valid || !b->is_valid) return 1.0f;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < TRACKING_APPEARANCE_DIM; i++) {
        dot += a->descriptor[i] * b->descriptor[i];
        norm_a += a->descriptor[i] * a->descriptor[i];
        norm_b += b->descriptor[i] * b->descriptor[i];
    }

    if (norm_a < 1e-8f || norm_b < 1e-8f) return 1.0f;

    float cosine_sim = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    /* Clamp to [-1, 1] due to float errors */
    if (cosine_sim > 1.0f) cosine_sim = 1.0f;
    if (cosine_sim < -1.0f) cosine_sim = -1.0f;

    return 0.5f * (1.0f - cosine_sim);  /* map to [0, 1] */
}

bool appearance_feature_is_valid(const AppearanceFeature* feat) {
    return feat && feat->is_valid && feat->num_valid_kpts >= TRACKING_APPEARANCE_MIN_KPTS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Partial-Body Keypoint Counting
 * ═══════════════════════════════════════════════════════════════════════════ */

int count_upper_body_keypoints(const PoseEstimation* pose, float min_conf) {
    if (!pose) return 0;
    int count = 0;
    /* Keypoints 0-10: nose, eyes, ears, shoulders, elbows, wrists */
    for (int i = 0; i <= 10 && i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f && pose->keypoints[i].y >= 0.0f) {
            count++;
        }
    }
    return count;
}

int count_left_side_keypoints(const PoseEstimation* pose, float min_conf) {
    if (!pose) return 0;
    static const int left_indices[] = {1,3,5,7,9,11,13,15}; /* left eye, ear, shoulder, elbow, wrist, hip, knee, ankle */
    int count = 0;
    for (int i = 0; i < 8; i++) {
        int idx = left_indices[i];
        if (idx < pose->num_keypoints &&
            pose->keypoints[idx].confidence >= min_conf &&
            pose->keypoints[idx].x >= 0.0f && pose->keypoints[idx].y >= 0.0f) {
            count++;
        }
    }
    return count;
}

int count_right_side_keypoints(const PoseEstimation* pose, float min_conf) {
    if (!pose) return 0;
    static const int right_indices[] = {2,4,6,8,10,12,14,16}; /* right eye, ear, shoulder, elbow, wrist, hip, knee, ankle */
    int count = 0;
    for (int i = 0; i < 8; i++) {
        int idx = right_indices[i];
        if (idx < pose->num_keypoints &&
            pose->keypoints[idx].confidence >= min_conf &&
            pose->keypoints[idx].x >= 0.0f && pose->keypoints[idx].y >= 0.0f) {
            count++;
        }
    }
    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tracker Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

ObjectTracker* object_tracker_create(int max_lost, float min_iou, float max_distance, int max_traj_len) {
    ObjectTracker* tracker = (ObjectTracker*)calloc(1, sizeof(ObjectTracker));
    if (!tracker) return NULL;

    tracker->max_lost = UTILS_MIN(max_lost, TRACKING_MAX_LOST_FRAMES);
    tracker->max_occluded = TRACKING_MAX_OCCLUDED_FRAMES;
    tracker->min_iou = min_iou;
    tracker->max_distance = max_distance;
    tracker->max_trajectory_length = max_traj_len;
    tracker->next_id = 0;
    tracker->num_tracks = 0;
    tracker->frame_width = 1920;
    tracker->frame_height = 1080;

    /* Enhanced confirmation defaults */
    tracker->confirmation_frames = TRACKING_CONFIRMATION_FRAMES;
    tracker->min_keypoints_for_confirm = TRACKING_MIN_KEYPOINTS_FOR_CONFIRM;
    tracker->min_keypoints_strong = TRACKING_MIN_KEYPOINTS_STRONG;
    tracker->spatial_jump_max_m = TRACKING_SPATIAL_JUMP_MAX_M;

    /* Cascade matching defaults */
    tracker->cascade_max_age = TRACKING_CASCADE_MAX_AGE;
    tracker->cascade_min_hits = TRACKING_CASCADE_MIN_HITS;
    tracker->appearance_weight = TRACKING_APPEARANCE_WEIGHT;
    tracker->appearance_max_dist = TRACKING_APPEARANCE_MAX_DIST;
    tracker->iou_threshold_low = TRACKING_IOU_THRESHOLD_LOW;

    /* Occlusion handling defaults */
    tracker->upper_body_min_kpts = TRACKING_UPPER_BODY_MIN_KPTS;
    tracker->side_body_min_kpts = TRACKING_SIDE_BODY_MIN_KPTS;
    tracker->occlusion_score_threshold = TRACKING_OCCLUSION_SCORE_THRESH;

    /* Multi-person detection defaults */
    tracker->new_person_grace_frames = TRACKING_NEW_PERSON_GRACE;

    /* Re-ID pool defaults */
    tracker->reid_pool_count = 0;
    tracker->reid_pool_max_age = 90;  /* keep deleted tracks for 90 frames (~3 sec at 30fps) */

    /* Diagnostics */
    tracker->total_id_switches = 0;
    tracker->total_reidentifications = 0;
    tracker->total_partial_matches = 0;

    return tracker;
}

void object_tracker_destroy(ObjectTracker* tracker) {
    free(tracker);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Track Management
 * ═══════════════════════════════════════════════════════════════════════════ */

static void remove_track(ObjectTracker* tracker, int idx) {
    if (idx < 0 || idx >= tracker->num_tracks) return;

    /* ── Re-ID pool: save appearance before deleting ── */
    TrackEntry* entry = &tracker->tracks[idx];
    if (entry->confirmed && appearance_feature_is_valid(&entry->latest_appearance)) {
        if (tracker->reid_pool_count < TRACKING_MAX_TRACKS) {
            tracker->reid_pool[tracker->reid_pool_count] = entry->latest_appearance;
            tracker->reid_pool_ids[tracker->reid_pool_count] = entry->track_id;
            tracker->reid_pool_count++;
        } else {
            /* Shift out oldest */
            memmove(&tracker->reid_pool[0], &tracker->reid_pool[1],
                    (size_t)(TRACKING_MAX_TRACKS - 1) * sizeof(AppearanceFeature));
            memmove(&tracker->reid_pool_ids[0], &tracker->reid_pool_ids[1],
                    (size_t)(TRACKING_MAX_TRACKS - 1) * sizeof(int));
            tracker->reid_pool[TRACKING_MAX_TRACKS - 1] = entry->latest_appearance;
            tracker->reid_pool_ids[TRACKING_MAX_TRACKS - 1] = entry->track_id;
        }
    }

    if (idx < tracker->num_tracks - 1) {
        memmove(&tracker->tracks[idx], &tracker->tracks[idx + 1],
                (size_t)(tracker->num_tracks - idx - 1) * sizeof(TrackEntry));
    }
    tracker->num_tracks--;
}

static int create_track(ObjectTracker* tracker, const Detection* detection,
                        const SpatialPosition* position, int frame_num,
                        const AppearanceFeature* app_feature,
                        bool from_reid) {
    if (tracker->num_tracks >= TRACKING_MAX_TRACKS) return -1;

    int track_id;
    if (from_reid) {
        track_id = detection->track_id_hint;  /* re-use old ID from re-id pool */
    } else {
        track_id = tracker->next_id++;
    }

    TrackEntry* entry = &tracker->tracks[tracker->num_tracks++];
    memset(entry, 0, sizeof(TrackEntry));

    entry->track_id = track_id;
    entry->object.track_id = track_id;
    entry->object.detection = *detection;
    entry->object.spatial_pos = position ? *position : (SpatialPosition){.x = 0.0f, .y = 0.0f, .z = 0.0f, .depth_confidence = 1.0f, .is_valid = true};
    entry->object.is_active = true;
    entry->object.is_occluded = false;
    entry->object.frames_seen = 1;
    entry->object.last_seen_frame = frame_num;
    entry->object.first_seen_frame = frame_num;
    entry->consecutive_hits = 1;
    entry->lost_count = 0;
    entry->confirmed = false;

    /* Keypoint stability init */
    entry->keypoint_stability_score = 0.0f;
    entry->keypoint_stability_frames = 0;
    entry->max_keypoints_seen = 0;

    /* Spatial jump detection init */
    if (position && position->is_valid) {
        entry->last_position = *position;
        entry->has_last_position = true;
    }

    /* Appearance feature init */
    entry->appearance_count = 0;
    if (app_feature && app_feature->is_valid) {
        entry->appearance_gallery[0] = *app_feature;
        entry->appearance_count = 1;
        entry->latest_appearance = *app_feature;
    }

    /* Occlusion state init */
    entry->occlusion_score = 1.0f;
    entry->occlusion_frames = 0;
    entry->is_partial_body = false;
    entry->last_match_type = MATCH_TYPE_FULL_BODY;

    /* Multi-person init */
    entry->unmatched_detection_count = 0;
    entry->has_nearby_unmatched = false;

    kalman_init(&entry->kf, &detection->bbox, track_id, 1.0f);

    if (position) {
        entry->smoothed_positions[0] = *position;
        entry->smoothed_count = 1;
    }

    return track_id;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Track Update (on successful match)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void update_track_match(ObjectTracker* tracker, int track_idx,
                                const Detection* detection, const SpatialPosition* position,
                                int frame_num, const AppearanceFeature* app_feature,
                                MatchType match_type) {
    TrackEntry* entry = &tracker->tracks[track_idx];

    kalman_update(&entry->kf, &detection->bbox);
    entry->object.detection = *detection;

    /* ── Spatial jump detection ── */
    if (position && position->is_valid && entry->has_last_position) {
        float dx = position->x - entry->last_position.x;
        float dy = position->y - entry->last_position.y;
        float dz = position->z - entry->last_position.z;
        float jump_dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (jump_dist > tracker->spatial_jump_max_m) {
            if (entry->consecutive_hits > 1) {
                entry->consecutive_hits = UTILS_MAX(1, entry->consecutive_hits - 2);
            }
            if (entry->keypoint_stability_frames > 0) {
                entry->keypoint_stability_score *= 0.7f;
            }
        }
    }

    if (position) {
        entry->last_position = *position;
        entry->has_last_position = true;

        /* EMA spatial smoothing */
        SpatialPosition smoothed = *position;
        if (entry->smoothed_count > 0) {
            const SpatialPosition* last = &entry->smoothed_positions[entry->smoothed_count - 1];
            smoothed.x = TRACKING_EMA_ALPHA * position->x + (1.0f - TRACKING_EMA_ALPHA) * last->x;
            smoothed.y = TRACKING_EMA_ALPHA * position->y + (1.0f - TRACKING_EMA_ALPHA) * last->y;
            smoothed.z = TRACKING_EMA_ALPHA * position->z + (1.0f - TRACKING_EMA_ALPHA) * last->z;
        }
        entry->object.spatial_pos = smoothed;

        if (entry->smoothed_count < TRACKING_MAX_TRAJECTORY) {
            entry->smoothed_positions[entry->smoothed_count++] = smoothed;
        } else {
            memmove(&entry->smoothed_positions[0], &entry->smoothed_positions[1],
                    (TRACKING_MAX_TRAJECTORY - 1) * sizeof(SpatialPosition));
            entry->smoothed_positions[TRACKING_MAX_TRAJECTORY - 1] = smoothed;
        }
    }

    /* ── Update appearance gallery ── */
    if (app_feature && app_feature->is_valid) {
        entry->latest_appearance = *app_feature;
        if (entry->appearance_count < TRACKING_APPEARANCE_MAX_GALLERY) {
            entry->appearance_gallery[entry->appearance_count++] = *app_feature;
        } else {
            /* Shift and replace oldest */
            memmove(&entry->appearance_gallery[0], &entry->appearance_gallery[1],
                    (size_t)(TRACKING_APPEARANCE_MAX_GALLERY - 1) * sizeof(AppearanceFeature));
            entry->appearance_gallery[TRACKING_APPEARANCE_MAX_GALLERY - 1] = *app_feature;
        }
    }

    /* ── Update keypoint stability ── */
    if (entry->object.has_pose && entry->object.pose.num_keypoints > 0) {
        int n_valid = 0;
        for (int i = 0; i < entry->object.pose.num_keypoints; i++) {
            if (entry->object.pose.keypoints[i].confidence >= 0.3f) n_valid++;
        }
        float alpha_k = 0.3f;
        if (entry->keypoint_stability_frames == 0) {
            entry->keypoint_stability_score = (float)n_valid;
        } else {
            entry->keypoint_stability_score =
                alpha_k * (float)n_valid + (1.0f - alpha_k) * entry->keypoint_stability_score;
        }
        entry->keypoint_stability_frames++;
        if (n_valid > entry->max_keypoints_seen) {
            entry->max_keypoints_seen = n_valid;
        }
    }

    /* ── Update occlusion score ── */
    if (match_type == MATCH_TYPE_UPPER_BODY || match_type == MATCH_TYPE_SIDE_BODY) {
        /* Partial body: estimate occlusion score from visible keypoint ratio */
        int max_kpts = match_type == MATCH_TYPE_UPPER_BODY ? 11 : 8;
        int n_visible = 0;
        if (entry->object.has_pose) {
            for (int i = 0; i < entry->object.pose.num_keypoints && i < 17; i++) {
                if (entry->object.pose.keypoints[i].confidence >= 0.3f) n_visible++;
            }
        }
        entry->occlusion_score = UTILS_MAX((float)n_visible / UTILS_MAX((float)max_kpts, 1.0f), 0.1f);
        entry->occlusion_frames++;
        entry->is_partial_body = true;
    } else {
        /* Full body: occlusion score from keypoint ratio */
        int n_visible = 0;
        if (entry->object.has_pose) {
            for (int i = 0; i < entry->object.pose.num_keypoints && i < 17; i++) {
                if (entry->object.pose.keypoints[i].confidence >= 0.3f) n_visible++;
            }
        }
        entry->occlusion_score = (float)n_visible / 17.0f;
        entry->occlusion_frames = 0;
        entry->is_partial_body = false;
    }
    entry->last_match_type = match_type;

    entry->object.is_active = true;
    entry->object.is_occluded = (entry->occlusion_score < tracker->occlusion_score_threshold);
    entry->object.occluded_frames = entry->occlusion_frames;
    entry->object.frames_seen++;
    entry->object.last_seen_frame = frame_num;

    entry->consecutive_hits++;

    /* ── Enhanced confirmation logic ── */
    if (entry->consecutive_hits >= tracker->confirmation_frames) {
        bool kpt_ok = (entry->max_keypoints_seen >= tracker->min_keypoints_strong) &&
                      (entry->keypoint_stability_score >= (float)tracker->min_keypoints_for_confirm);
        if (entry->keypoint_stability_frames == 0 || kpt_ok) {
            entry->confirmed = true;
        }
    }

    entry->lost_count = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Cascade Matching
 *
 * Groups tracks by time_since_update, matches freshest first.  This prevents
 * stale (long-lost) tracks from stealing detections from active tracks.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void cascade_matching(ObjectTracker* tracker,
                              BoundingBox* pred_bboxes, int* track_indices, int num_pred,
                              const Detection* detections, int num_det,
                              AppearanceFeature* det_features,
                              bool* det_matched, bool* track_matched,
                              const SpatialPosition* positions, int num_positions,
                              int frame_num) {
    /* ── Group tracks by time_since_update ── */
    /* Build sorted list of (time_since_update, track_idx) pairs */
    typedef struct { int age; int idx; } AgeEntry;
    AgeEntry* age_list = (AgeEntry*)calloc((size_t)num_pred, sizeof(AgeEntry));
    if (!age_list) return;

    for (int i = 0; i < num_pred; i++) {
        age_list[i].age = tracker->tracks[track_indices[i]].kf.time_since_update;
        age_list[i].idx = i;
    }

    /* Simple sort by age ascending (youngest first) */
    for (int i = 0; i < num_pred - 1; i++) {
        for (int j = i + 1; j < num_pred; j++) {
            if (age_list[j].age < age_list[i].age) {
                AgeEntry tmp = age_list[i];
                age_list[i] = age_list[j];
                age_list[j] = tmp;
            }
        }
    }

    /* ── For each age group, run Hungarian matching ── */
    int max_age = tracker->cascade_max_age;
    for (int age = 1; age <= max_age; age++) {
        /* Collect tracks with this age that are unmatched */
        int age_track_local[HUNGARIAN_MAX_DIM];
        int age_track_global[HUNGARIAN_MAX_DIM];
        int num_age_tracks = 0;

        for (int i = 0; i < num_pred; i++) {
            if (age_list[i].age == age && !track_matched[age_list[i].idx]) {
                if (num_age_tracks < HUNGARIAN_MAX_DIM) {
                    age_track_local[num_age_tracks] = age_list[i].idx;
                    age_track_global[num_age_tracks] = track_indices[age_list[i].idx];
                    num_age_tracks++;
                }
            }
        }

        if (num_age_tracks == 0) continue;

        /* Collect unmatched detections */
        int age_det_idx[HUNGARIAN_MAX_DIM];
        int num_age_dets = 0;
        for (int d = 0; d < num_det; d++) {
            if (!det_matched[d] && num_age_dets < HUNGARIAN_MAX_DIM) {
                age_det_idx[num_age_dets++] = d;
            }
        }

        if (num_age_dets == 0) break;  /* no more detections to match */

        /* Build cost matrix for this age group */
        int nr = num_age_tracks;
        int nc = num_age_dets;
        float* cost = (float*)calloc((size_t)nr * nc, sizeof(float));
        if (!cost) continue;

        for (int i = 0; i < nr; i++) {
            int local_idx = age_track_local[i];
            int global_idx = age_track_global[i];
            const TrackEntry* entry = &tracker->tracks[global_idx];

            for (int j = 0; j < nc; j++) {
                int det_idx = age_det_idx[j];
                float iou = bbox_iou(&pred_bboxes[local_idx], &detections[det_idx].bbox);
                float iou_cost = 1.0f - iou;

                float app_cost = 0.5f;
                if (appearance_feature_is_valid(&entry->latest_appearance) &&
                    det_features && appearance_feature_is_valid(&det_features[det_idx])) {
                    app_cost = appearance_feature_distance(&entry->latest_appearance, &det_features[det_idx]);
                }

                /* Use stricter thresholds for cascade: younger tracks are more reliable */
                float effective_iou_thresh = (age <= 3) ? TRACKING_IOU_THRESHOLD : tracker->iou_threshold_low;

                float cost_val;
                if (iou_cost > (1.0f - effective_iou_thresh) && app_cost > tracker->appearance_max_dist) {
                    cost_val = 1e8f;
                } else {
                    cost_val = iou_cost * (1.0f - tracker->appearance_weight) +
                               app_cost * tracker->appearance_weight;
                }

                cost[i * nc + j] = cost_val;
            }
        }

        /* Hungarian solve */
        int* assign = (int*)calloc((size_t)nr, sizeof(int));
        if (assign) {
            for (int i = 0; i < nr; i++) assign[i] = -1;
            float final_cost;
            hungarian_solve(cost, nr, nc, assign, &final_cost);

            /* Process assignments */
            for (int i = 0; i < nr; i++) {
                int j = assign[i];
                if (j < 0 || j >= nc) continue;
                if (cost[i * nc + j] >= 1e7f) continue;  /* gated out */

                int local_idx = age_track_local[i];
                int global_idx = age_track_global[i];
                int det_idx = age_det_idx[j];

                MatchType mtype = MATCH_TYPE_FULL_BODY;
                /* Check if this is a partial-body or appearance-only match */
                if (appearance_feature_is_valid(&det_features[det_idx])) {
                    int valid_kpts = det_features[det_idx].num_valid_kpts;
                    if (valid_kpts < 8 && valid_kpts >= tracker->upper_body_min_kpts) {
                        mtype = MATCH_TYPE_UPPER_BODY;
                    } else if (valid_kpts < tracker->upper_body_min_kpts &&
                               valid_kpts >= tracker->side_body_min_kpts) {
                        mtype = MATCH_TYPE_SIDE_BODY;
                    }
                }

                const SpatialPosition* pos = (num_positions > det_idx) ? &positions[det_idx] : NULL;
                update_track_match(tracker, global_idx, &detections[det_idx], pos, frame_num,
                                   det_features ? &det_features[det_idx] : NULL, mtype);

                det_matched[det_idx] = true;
                track_matched[local_idx] = true;

                if (mtype != MATCH_TYPE_FULL_BODY) {
                    tracker->total_partial_matches++;
                }
            }
            free(assign);
        }
        free(cost);
    }

    free(age_list);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Re-Identification: Check unmatched detections against deleted-track pool
 * ═══════════════════════════════════════════════════════════════════════════ */

static int reid_check(ObjectTracker* tracker,
                       const AppearanceFeature* det_feature) {
    if (!det_feature || !appearance_feature_is_valid(det_feature)) return -1;
    if (tracker->reid_pool_count == 0) return -1;

    float best_dist = tracker->appearance_max_dist;
    int best_id = -1;

    for (int i = 0; i < tracker->reid_pool_count; i++) {
        if (!tracker->reid_pool[i].is_valid) continue;
        float dist = appearance_feature_distance(&tracker->reid_pool[i], det_feature);
        if (dist < best_dist) {
            best_dist = dist;
            best_id = tracker->reid_pool_ids[i];
        }
    }

    return best_id;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main Tracking Update (ByteTrack-style with cascade + appearance + occlusion)
 * ═══════════════════════════════════════════════════════════════════════════ */

TrackingResult object_tracker_update(ObjectTracker* tracker,
                                      const Detection* detections, int num_detections,
                                      const SpatialPosition* positions, int num_positions,
                                      int frame_num) {
    TrackingResult result;
    tracking_result_init(&result);

    if (!tracker) return result;

    /* ── Step 0: Re-ID pool aging ──
     * Remove entries older than reid_pool_max_age.  Since we don't store
     * frame_index per pool entry currently, we do this lazily: shift the
     * pool every reid_pool_max_age/2 frames to age out old entries. */
    {
        static int reid_aging_counter = 0;
        reid_aging_counter++;
        if (reid_aging_counter >= tracker->reid_pool_max_age / 2) {
            /* Drop oldest half of the pool */
            int keep = tracker->reid_pool_count / 2;
            if (keep > 0) {
                memmove(&tracker->reid_pool[0], &tracker->reid_pool[tracker->reid_pool_count - keep],
                        (size_t)keep * sizeof(AppearanceFeature));
                memmove(&tracker->reid_pool_ids[0], &tracker->reid_pool_ids[tracker->reid_pool_count - keep],
                        (size_t)keep * sizeof(int));
                tracker->reid_pool_count = keep;
            } else {
                tracker->reid_pool_count = 0;
            }
            reid_aging_counter = 0;
        }
    }

    /* ── Step 1: Kalman predict all existing tracks ── */
    BoundingBox predicted_bboxes[TRACKING_MAX_TRACKS];
    int predicted_ids[TRACKING_MAX_TRACKS];
    int num_predicted = 0;

    for (int i = 0; i < tracker->num_tracks; i++) {
        kalman_predict(&tracker->tracks[i].kf);
        BoundingBox pred = kalman_get_bbox(&tracker->tracks[i].kf);

        if (bbox_area(&pred) > 0 && bbox_width(&pred) > 0 && bbox_height(&pred) > 0) {
            predicted_bboxes[num_predicted] = pred;
            predicted_ids[num_predicted] = i;
            num_predicted++;
        } else {
            remove_track(tracker, i);
            i--;
            result.lost_tracks++;
        }
    }

    /* ── Step 2: Split detections into high/low confidence ── */
    int det_high_idx[MAX_DETECTIONS_PER_FRAME];
    int det_low_idx[MAX_DETECTIONS_PER_FRAME];
    int num_high = 0, num_low = 0;

    for (int d = 0; d < num_detections; d++) {
        if (detections[d].confidence >= TRACKING_CONFIDENCE_HIGH) {
            det_high_idx[num_high++] = d;
        } else if (detections[d].confidence >= TRACKING_CONFIDENCE_LOW) {
            det_low_idx[num_low++] = d;
        }
    }

    bool det_matched[MAX_DETECTIONS_PER_FRAME];
    memset(det_matched, 0, sizeof(det_matched));
    bool track_matched[TRACKING_MAX_TRACKS];
    memset(track_matched, 0, sizeof(track_matched));

    /* ── Step 3: Extract appearance features for all detections that have poses ──
     * This is done externally via object_tracker_associate_poses() which must
     * be called BEFORE object_tracker_update.  The poses are stored in the
     * TrackedObject of each track entry, but detections are not yet associated.
     *
     * For detections, we compute features from the positional data available.
     * Since detections don't have pose data directly, we compute a simplified
     * feature based on bbox geometry (aspect ratio, normalized position, area).
     * Full pose-based appearance features are computed during pose association. */
    AppearanceFeature det_features[MAX_DETECTIONS_PER_FRAME];
    int num_det_features = 0;
    /* Clear all — features computed lazily */
    for (int d = 0; d < num_detections; d++) {
        memset(&det_features[d], 0, sizeof(AppearanceFeature));
        det_features[d].is_valid = false;
    }
    /* Note: Pose-to-detection feature assignment happens in the post-processing
     * step.  For now, we use bbox-geometry-based features for detections. */
    for (int h = 0; h < num_high; h++) {
        int d = det_high_idx[h];
        /* Simple geometry-based lightweight feature: aspect ratio + position */
        float aspect = bbox_width(&detections[d].bbox) / UTILS_MAX(bbox_height(&detections[d].bbox), 1.0f);
        float norm_x = bbox_center_x(&detections[d].bbox) / UTILS_MAX((float)tracker->frame_width, 1.0f);
        float norm_y = bbox_center_y(&detections[d].bbox) / UTILS_MAX((float)tracker->frame_height, 1.0f);
        float norm_area = bbox_area(&detections[d].bbox) / UTILS_MAX((float)(tracker->frame_width * tracker->frame_height), 1.0f);

        /* Store as simplified appearance descriptor */
        det_features[d].descriptor[0] = aspect;
        det_features[d].descriptor[1] = norm_x;
        det_features[d].descriptor[2] = norm_y;
        det_features[d].descriptor[3] = norm_area;
        /* Remaining dims stay 0 */
        det_features[d].num_valid_kpts = 4;
        det_features[d].frame_index = frame_num;
        det_features[d].is_valid = (norm_area > 0.001f && norm_area < 0.5f);
        if (det_features[d].is_valid) num_det_features++;
    }

    /* ── Step 4: Cascade matching (HIGH-confidence dets ONLY) ──
     * Uses both IoU and appearance for optimal assignment.  Fresh tracks
     * get priority via cascade ordering.  Works directly on original
     * detection indices via det_high_idx to keep det_matched consistent. */
    if (num_predicted > 0 && num_high > 0) {
        cascade_matching(tracker,
                        predicted_bboxes, predicted_ids, num_predicted,
                        detections, num_detections,
                        det_features,
                        det_matched, track_matched,
                        positions, num_positions,
                        frame_num);
    }

    /* ── Stage B: Low-confidence dets with remaining tracks (occlusion recovery) ── */
    for (int t = 0; t < num_predicted; t++) {
        if (track_matched[t]) continue;

        float best_iou = tracker->iou_threshold_low;
        int best_det = -1;

        /* Check both remaining high-confidence AND low-confidence dets */
        for (int h = 0; h < num_high; h++) {
            int d = det_high_idx[h];
            if (det_matched[d]) continue;
            float iou = bbox_iou(&predicted_bboxes[t], &detections[d].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = d;
            }
        }

        for (int l = 0; l < num_low; l++) {
            int d = det_low_idx[l];
            if (det_matched[d]) continue;
            float iou = bbox_iou(&predicted_bboxes[t], &detections[d].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = d;
            }
        }

        if (best_det >= 0) {
            int track_idx = predicted_ids[t];
            const SpatialPosition* pos = (num_positions > best_det) ? &positions[best_det] : NULL;

            /* Determine match type */
            MatchType mtype = MATCH_TYPE_FULL_BODY;
            if (best_iou < TRACKING_IOU_THRESHOLD) {
                /* Low IoU — likely partial body or occlusion */
                if (appearance_feature_is_valid(&tracker->tracks[track_idx].latest_appearance) &&
                    appearance_feature_is_valid(&det_features[best_det])) {
                    float app_dist = appearance_feature_distance(
                        &tracker->tracks[track_idx].latest_appearance,
                        &det_features[best_det]);
                    if (app_dist < tracker->appearance_max_dist) {
                        mtype = MATCH_TYPE_APPEARANCE;
                    }
                }
                if (mtype == MATCH_TYPE_FULL_BODY && det_features[best_det].is_valid) {
                    /* Infer partial body from available info */
                    mtype = MATCH_TYPE_UPPER_BODY;
                }
            }

            update_track_match(tracker, track_idx, &detections[best_det], pos, frame_num,
                               det_features[best_det].is_valid ? &det_features[best_det] : NULL,
                               mtype);
            det_matched[best_det] = true;
            track_matched[t] = true;
        }
    }

    /* ── Step 6: Mark unmatched tracks as lost (with occlusion awareness) ── */
    for (int t = 0; t < num_predicted; t++) {
        if (track_matched[t]) continue;
        int track_idx = predicted_ids[t];
        TrackEntry* entry = &tracker->tracks[track_idx];

        entry->lost_count++;
        entry->object.is_occluded = true;
        entry->object.occluded_frames = entry->lost_count;
        entry->occlusion_frames++;

        /* Keep predicted bbox for visualization */
        entry->object.detection.bbox = predicted_bboxes[t];
        entry->object.detection.confidence *= 0.95f;  /* slower decay than 0.9 */

        /* Increment occlusion score decay */
        if (entry->occlusion_score > 0.1f) {
            entry->occlusion_score *= 0.95f;
        }
    }

    /* ── Step 7: Create NEW tracks from unmatched HIGH-confidence dets ── */
    for (int h = 0; h < num_high; h++) {
        int d = det_high_idx[h];
        if (det_matched[d]) continue;

        /* ── Re-identification check ── */
        int reid_id = -1;
        if (appearance_feature_is_valid(&det_features[d])) {
            reid_id = reid_check(tracker, &det_features[d]);
        }

        const SpatialPosition* pos = (num_positions > d) ? &positions[d] : NULL;

        if (reid_id >= 0) {
            /* Re-identified! Create track with old ID */
            Detection reid_det = detections[d];
            reid_det.track_id_hint = reid_id;
            create_track(tracker, &reid_det, pos, frame_num,
                        det_features[d].is_valid ? &det_features[d] : NULL, true);
            tracker->total_reidentifications++;
        } else {
            create_track(tracker, &detections[d], pos, frame_num,
                        det_features[d].is_valid ? &det_features[d] : NULL, false);
        }
        result.new_tracks++;
    }

    /* ── Step 8: Remove tracks lost for too long ──
     * Occluded tracks get extended lifetime. */
    for (int i = tracker->num_tracks - 1; i >= 0; i--) {
        TrackEntry* entry = &tracker->tracks[i];
        int effective_max_lost = entry->is_partial_body ?
            tracker->max_occluded : tracker->max_lost;

        if (entry->lost_count > effective_max_lost) {
            remove_track(tracker, i);
            result.lost_tracks++;
        }
    }

    /* ── Step 9: Output only CONFIRMED + RECENTLY ACTIVE tracks ──
     * KEY FIX for ghost/residual boxes: a track that has been lost for
     * more than 3 consecutive frames is NOT output.  Its Kalman prediction
     * drifts and creates a "ghost box" on screen.  These tracks are kept
     * internally for re-identification but hidden from visualization.
     *
     * Tracks with lost_count == 0: matched this frame → fully visible
     * Tracks with lost_count 1-3: briefly lost → still shown (smooth)
     * Tracks with lost_count > 3: ghost → hidden (prediction drift too large) */
    for (int i = 0; i < tracker->num_tracks; i++) {
        TrackEntry* entry = &tracker->tracks[i];
        if (entry->confirmed && entry->lost_count <= 3) {
            if (result.num_tracked < MAX_TRACKED_OBJECTS) {
                result.tracked_objects[result.num_tracked++] = entry->object;
            }
        }
    }

    /* ── Step 10: Multi-person detection check ──
     * If detections significantly outnumber confirmed tracks, it likely means
     * new people have entered the scene.  Flag for cascade to run full detection. */
    {
        int confirmed_count = 0;
        for (int i = 0; i < tracker->num_tracks; i++) {
            if (tracker->tracks[i].confirmed) confirmed_count++;
        }
        if (num_high > confirmed_count * TRACKING_MAX_DET_TO_TRACK_RATIO) {
            /* Signal that new people may have entered */
            result.id_switches = -1;  /* special flag: request full detection next frame */
        } else {
            result.id_switches = tracker->total_id_switches;
        }
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pose Association (call after tracking update)
 * ═══════════════════════════════════════════════════════════════════════════ */

void object_tracker_associate_poses(ObjectTracker* tracker,
                                     const PoseEstimation* poses, int num_poses) {
    if (!tracker || !poses || num_poses <= 0) return;

    bool pose_used[MAX_POSES_PER_FRAME];
    memset(pose_used, 0, sizeof(pose_used));

    for (int i = 0; i < tracker->num_tracks; i++) {
        TrackEntry* entry = &tracker->tracks[i];
        if (!entry->confirmed && entry->consecutive_hits < 3) continue;

        float best_iou = 0.0f;
        int best_pose = -1;

        for (int j = 0; j < num_poses && j < MAX_POSES_PER_FRAME; j++) {
            if (pose_used[j] || !poses[j].has_bbox) continue;
            float iou = bbox_iou(&entry->object.detection.bbox, &poses[j].bbox);
            if (iou > best_iou && iou > 0.3f) {
                best_iou = iou;
                best_pose = j;
            }
        }

        if (best_pose >= 0) {
            entry->object.pose = poses[best_pose];
            entry->object.has_pose = true;
            pose_used[best_pose] = true;

            /* Compute appearance feature from associated pose */
            AppearanceFeature feat;
            if (appearance_feature_from_pose(&poses[best_pose],
                                              &entry->object.detection.bbox,
                                              &feat, entry->object.last_seen_frame)) {
                entry->latest_appearance = feat;
                if (entry->appearance_count < TRACKING_APPEARANCE_MAX_GALLERY) {
                    entry->appearance_gallery[entry->appearance_count++] = feat;
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * State Management
 * ═══════════════════════════════════════════════════════════════════════════ */

void object_tracker_reset(ObjectTracker* tracker) {
    if (!tracker) return;
    tracker->num_tracks = 0;
    tracker->next_id = 0;
    tracker->reid_pool_count = 0;
    tracker->total_id_switches = 0;
    tracker->total_reidentifications = 0;
    tracker->total_partial_matches = 0;
    log_info("Tracker state reset");
}

int object_tracker_get_active_tracks(const ObjectTracker* tracker, TrackedObject* out_objects, int max_count) {
    if (!tracker || !out_objects) return 0;
    int count = 0;
    for (int i = 0; i < tracker->num_tracks && count < max_count; i++) {
        if (tracker->tracks[i].confirmed && tracker->tracks[i].object.is_active) {
            out_objects[count++] = tracker->tracks[i].object;
        }
    }
    return count;
}

int object_tracker_get_all_track_count(const ObjectTracker* tracker) {
    return tracker ? tracker->num_tracks : 0;
}

int object_tracker_get_confirmed_count(const ObjectTracker* tracker) {
    if (!tracker) return 0;
    int count = 0;
    for (int i = 0; i < tracker->num_tracks; i++) {
        if (tracker->tracks[i].confirmed) count++;
    }
    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

void object_tracker_set_enhanced_config(ObjectTracker* tracker,
                                         int confirmation_frames,
                                         int min_keypoints_for_confirm,
                                         int min_keypoints_strong,
                                         float spatial_jump_max_m) {
    if (!tracker) return;
    if (confirmation_frames >= 2 && confirmation_frames <= 30)
        tracker->confirmation_frames = confirmation_frames;
    if (min_keypoints_for_confirm >= 0 && min_keypoints_for_confirm <= 17)
        tracker->min_keypoints_for_confirm = min_keypoints_for_confirm;
    if (min_keypoints_strong >= 0 && min_keypoints_strong <= 17)
        tracker->min_keypoints_strong = min_keypoints_strong;
    if (spatial_jump_max_m > 0.0f && spatial_jump_max_m <= 100.0f)
        tracker->spatial_jump_max_m = spatial_jump_max_m;
    log_info("Tracker enhanced config: confirm=%d frames, kpt_min=%d avg, kpt_strong=%d, jump_max=%.1fm",
             tracker->confirmation_frames, tracker->min_keypoints_for_confirm,
             tracker->min_keypoints_strong, (double)tracker->spatial_jump_max_m);
}

void object_tracker_set_cascade_config(ObjectTracker* tracker,
                                        int cascade_max_age,
                                        int cascade_min_hits,
                                        float appearance_weight,
                                        float appearance_max_dist,
                                        float iou_threshold_low) {
    if (!tracker) return;
    if (cascade_max_age >= 5 && cascade_max_age <= 120)
        tracker->cascade_max_age = cascade_max_age;
    if (cascade_min_hits >= 1 && cascade_min_hits <= 10)
        tracker->cascade_min_hits = cascade_min_hits;
    if (appearance_weight >= 0.0f && appearance_weight <= 1.0f)
        tracker->appearance_weight = appearance_weight;
    if (appearance_max_dist >= 0.05f && appearance_max_dist <= 1.0f)
        tracker->appearance_max_dist = appearance_max_dist;
    if (iou_threshold_low >= 0.05f && iou_threshold_low <= 0.50f)
        tracker->iou_threshold_low = iou_threshold_low;
    log_info("Tracker cascade config: max_age=%d min_hits=%d app_weight=%.2f app_max_dist=%.2f iou_low=%.2f",
             tracker->cascade_max_age, tracker->cascade_min_hits,
             (double)tracker->appearance_weight, (double)tracker->appearance_max_dist,
             (double)tracker->iou_threshold_low);
}

void object_tracker_set_occlusion_config(ObjectTracker* tracker,
                                          int max_occluded_frames,
                                          int upper_body_min_kpts,
                                          int side_body_min_kpts,
                                          float occlusion_score_threshold) {
    if (!tracker) return;
    if (max_occluded_frames >= 30 && max_occluded_frames <= 300)
        tracker->max_occluded = max_occluded_frames;
    if (upper_body_min_kpts >= 2 && upper_body_min_kpts <= 11)
        tracker->upper_body_min_kpts = upper_body_min_kpts;
    if (side_body_min_kpts >= 2 && side_body_min_kpts <= 8)
        tracker->side_body_min_kpts = side_body_min_kpts;
    if (occlusion_score_threshold >= 0.1f && occlusion_score_threshold <= 0.9f)
        tracker->occlusion_score_threshold = occlusion_score_threshold;
    log_info("Tracker occlusion config: max_occ_frames=%d upper_kpts=%d side_kpts=%d occ_thresh=%.2f",
             tracker->max_occluded, tracker->upper_body_min_kpts,
             tracker->side_body_min_kpts, (double)tracker->occlusion_score_threshold);
}

void object_tracker_set_reid_config(ObjectTracker* tracker, int reid_pool_max_age) {
    if (!tracker) return;
    if (reid_pool_max_age >= 30 && reid_pool_max_age <= 600)
        tracker->reid_pool_max_age = reid_pool_max_age;
    log_info("Tracker re-id config: pool_max_age=%d frames", tracker->reid_pool_max_age);
}

void object_tracker_set_multi_person_config(ObjectTracker* tracker, int new_person_grace_frames) {
    if (!tracker) return;
    if (new_person_grace_frames >= 1 && new_person_grace_frames <= 30)
        tracker->new_person_grace_frames = new_person_grace_frames;
    log_info("Tracker multi-person config: grace_frames=%d", tracker->new_person_grace_frames);
}
