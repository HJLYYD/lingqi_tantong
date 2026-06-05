#include "tracking_manager.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->covariance[i][j] = (i == j) ? ((i < 4) ? 10.0f : 1000.0f) : 0.0f;
        }
    }

    float q = 0.04f;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->process_noise[i][j] = (i == j) ? ((i < 4) ? q : q * 10.0f) : 0.0f;
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kf->measurement_noise[i][j] = (i == j) ? 0.1f : 0.0f;
        }
    }

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            kf->transition[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    kf->transition[0][4] = dt;
    kf->transition[1][5] = dt;
    kf->transition[2][6] = dt;

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

    float y[4];
    for (int i = 0; i < 4; i++) {
        y[i] = z[i];
        for (int j = 0; j < 7; j++) {
            y[i] -= kf->measurement[i][j] * kf->state[j];
        }
    }

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

    float s_inv[4][4];
    if (utils_matrix_inverse_4x4(s_mat, s_inv) != 0) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                s_inv[i][j] = (i == j) ? 1.0f / UTILS_MAX(s_mat[i][i], 1e-10f) : 0.0f;
            }
        }
    }

    float k_gain[7][4] = {{0}};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                k_gain[i][j] += kf->covariance[i][k] * s_inv[k][j];
            }
        }
    }

    float new_state[7];
    for (int i = 0; i < 7; i++) {
        new_state[i] = kf->state[i];
        for (int j = 0; j < 4; j++) {
            new_state[i] += k_gain[i][j] * y[j];
        }
    }
    memcpy(kf->state, new_state, sizeof(new_state));

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

ObjectTracker* object_tracker_create(int max_lost, float min_iou, float max_distance, int max_traj_len) {
    ObjectTracker* tracker = (ObjectTracker*)calloc(1, sizeof(ObjectTracker));
    if (!tracker) return NULL;

    tracker->max_lost = UTILS_MIN(max_lost, TRACKING_MAX_LOST_FRAMES);
    tracker->min_iou = min_iou;
    tracker->max_distance = max_distance;
    tracker->max_trajectory_length = max_traj_len;
    tracker->next_id = 0;
    tracker->num_tracks = 0;
    tracker->frame_width = 1920;
    tracker->frame_height = 1080;

    return tracker;
}

void object_tracker_destroy(ObjectTracker* tracker) {
    free(tracker);
}

static int find_track(ObjectTracker* tracker, int track_id) __attribute__((unused));
static int find_track(ObjectTracker* tracker, int track_id) {
    for (int i = 0; i < tracker->num_tracks; i++) {
        if (tracker->tracks[i].track_id == track_id) {
            return i;
        }
    }
    return -1;
}

static void remove_track(ObjectTracker* tracker, int idx) {
    if (idx < 0 || idx >= tracker->num_tracks) return;
    if (idx < tracker->num_tracks - 1) {
        memmove(&tracker->tracks[idx], &tracker->tracks[idx + 1],
                (tracker->num_tracks - idx - 1) * sizeof(TrackEntry));
    }
    tracker->num_tracks--;
}

static int create_track(ObjectTracker* tracker, const Detection* detection,
                        const SpatialPosition* position, int frame_num) {
    if (tracker->num_tracks >= TRACKING_MAX_TRACKS) return -1;

    int track_id = tracker->next_id++;
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

    kalman_init(&entry->kf, &detection->bbox, track_id, 1.0f);

    if (position) {
        entry->smoothed_positions[0] = *position;
        entry->smoothed_count = 1;
    }

    return track_id;
}

static void update_track_match(ObjectTracker* tracker, int track_idx,
                                const Detection* detection, const SpatialPosition* position, int frame_num) {
    TrackEntry* entry = &tracker->tracks[track_idx];

    kalman_update(&entry->kf, &detection->bbox);
    entry->object.detection = *detection;

    if (position) {
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

    entry->object.is_active = true;
    entry->object.is_occluded = false;
    entry->object.occluded_frames = 0;
    entry->object.frames_seen++;
    entry->object.last_seen_frame = frame_num;

    entry->consecutive_hits++;
    if (entry->consecutive_hits >= TRACKING_CONFIRMATION_FRAMES) {
        entry->confirmed = true;
    }
    entry->lost_count = 0;
}

TrackingResult object_tracker_update(ObjectTracker* tracker,
                                      const Detection* detections, int num_detections,
                                      const SpatialPosition* positions, int num_positions,
                                      int frame_num) {
    TrackingResult result;
    tracking_result_init(&result);

    if (!tracker) return result;

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

    /* ── Step 2: Split detections into high/low confidence ──
     * ByteTrack insight: HIGH-confidence detections create/update tracks;
     * LOW-confidence detections only RECOVER temporarily lost tracks
     * (e.g. occluded people). This prevents false positives from
     * spawning new tracks while still recovering real tracks. */
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

    /* ── Step 3: Stage 1 — match HIGH-confidence dets with ALL tracks ── */
    for (int t = 0; t < num_predicted; t++) {
        if (track_matched[t]) continue;

        float best_iou = TRACKING_IOU_THRESHOLD;
        int best_det = -1;

        for (int h = 0; h < num_high; h++) {
            int d = det_high_idx[h];
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
            update_track_match(tracker, track_idx, &detections[best_det], pos, frame_num);
            det_matched[best_det] = true;
            track_matched[t] = true;
        }
    }

    /* ── Step 4: Stage 2 — match LOW-confidence dets with REMAINING tracks ──
     * This recovers tracks that were temporarily lost (occlusion, motion blur).
     * Low-confidence dets do NOT create new tracks. */
    for (int t = 0; t < num_predicted; t++) {
        if (track_matched[t]) continue;

        float best_iou = TRACKING_IOU_THRESHOLD;
        int best_det = -1;

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
            update_track_match(tracker, track_idx, &detections[best_det], pos, frame_num);
            det_matched[best_det] = true;
            track_matched[t] = true;
        }
    }

    /* ── Step 5: Mark unmatched tracks as lost ── */
    for (int t = 0; t < num_predicted; t++) {
        if (track_matched[t]) continue;
        int track_idx = predicted_ids[t];
        tracker->tracks[track_idx].lost_count++;
        tracker->tracks[track_idx].object.is_occluded = true;
        tracker->tracks[track_idx].object.occluded_frames = tracker->tracks[track_idx].lost_count;
        /* Keep predicted bbox for visualization of lost tracks */
        tracker->tracks[track_idx].object.detection.bbox = predicted_bboxes[t];
        tracker->tracks[track_idx].object.detection.confidence *= 0.9f;  /* decay confidence */
    }

    /* ── Step 6: Create NEW tracks from unmatched HIGH-confidence dets ONLY ── */
    for (int h = 0; h < num_high; h++) {
        int d = det_high_idx[h];
        if (det_matched[d]) continue;

        const SpatialPosition* pos = (num_positions > d) ? &positions[d] : NULL;
        create_track(tracker, &detections[d], pos, frame_num);
        result.new_tracks++;
    }
    /* Low-confidence unmatched dets are DISCARDED — they don't create tracks. */

    /* ── Step 7: Remove tracks lost for too long ── */
    for (int i = tracker->num_tracks - 1; i >= 0; i--) {
        if (tracker->tracks[i].lost_count > tracker->max_lost) {
            remove_track(tracker, i);
            result.lost_tracks++;
        }
    }

    /* ── Step 8: Output only CONFIRMED tracks ── */
    for (int i = 0; i < tracker->num_tracks; i++) {
        if (tracker->tracks[i].confirmed) {
            if (result.num_tracked < MAX_TRACKED_OBJECTS) {
                result.tracked_objects[result.num_tracked++] = tracker->tracks[i].object;
            }
        }
    }

    return result;
}

void object_tracker_reset(ObjectTracker* tracker) {
    if (!tracker) return;
    tracker->num_tracks = 0;
    tracker->next_id = 0;
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
