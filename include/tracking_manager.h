#ifndef TRACKING_MANAGER_H
#define TRACKING_MANAGER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Track capacity ── */
#define TRACKING_MAX_TRACKS             256
#define TRACKING_MAX_TRAJECTORY         300

/* ── Confirmation ──
 * REDUCED from 5 to 3: 5 consecutive hits at 3.5 FPS = 1.4s delay before
 * a detected person appears on screen.  3 hits ≈ 0.85s — still enough to
 * filter transient noise, much faster for real human detection. */
#define TRACKING_CONFIRMATION_FRAMES    3
#define TRACKING_CONFIRMATION_FRAMES_OLD 3
#define TRACKING_MIN_KEYPOINTS_FOR_CONFIRM 3   /* lowered from 4: faster confirm for partial-body */
#define TRACKING_MIN_KEYPOINTS_STRONG   6       /* lowered from 8: INT8 keypoints rarely reach 8 */

/* ── Track lifecycle ── */
#define TRACKING_MAX_LOST_FRAMES        30    /* reduced from 60: 30 frames @ 3.5 FPS = 8.5s max prediction */
#define TRACKING_MAX_OCCLUDED_FRAMES    45    /* reduced from 90: extended but not infinite for occlusion */
#define TRACKING_OCCLUSION_WINDOW       15

/* ── Spatial jump ── */
#define TRACKING_SPATIAL_JUMP_MAX_M     3.0f

/* ── ByteTrack-style two-stage confidence thresholds ── */
#define TRACKING_IOU_THRESHOLD          0.30f  /* lowered from 0.35 for better occlusion matching */
#define TRACKING_IOU_THRESHOLD_LOW      0.15f  /* second-stage recovery for occluded tracks */
#define TRACKING_CONFIDENCE_HIGH        0.18f
#define TRACKING_CONFIDENCE_LOW         0.12f

/* ── EMA smoothing ── */
#define TRACKING_EMA_ALPHA              0.3f

/* ── Appearance feature matching ── */
#define TRACKING_APPEARANCE_DIM         12     /* pose-keypoint-derived compact descriptor */
#define TRACKING_APPEARANCE_MAX_GALLERY 50     /* max stored features per track */
#define TRACKING_APPEARANCE_WEIGHT      0.35f  /* weight of appearance vs IoU in combined cost */
#define TRACKING_APPEARANCE_MAX_DIST    0.50f  /* cosine distance gate for appearance matching */
#define TRACKING_APPEARANCE_MIN_KPTS    5      /* minimum valid keypoints to compute appearance */

/* ── Cascade matching ── */
#define TRACKING_CASCADE_MAX_AGE        30     /* max frames since last update for cascade levels */
#define TRACKING_CASCADE_MIN_HITS       3      /* min hits before track enters cascade matching */

/* ── Occlusion / partial-body handling ── */
#define TRACKING_UPPER_BODY_KPT_COUNT   11     /* keypoints 0-10: nose through wrists */
#define TRACKING_UPPER_BODY_MIN_KPTS    4      /* minimum upper-body keypoints for partial detection */
#define TRACKING_SIDE_BODY_MIN_KPTS     3      /* minimum one-side keypoints for lateral view */
#define TRACKING_PARTIAL_BODY_CONF_BOOST 0.05f /* boost detection confidence for partial-body matches */
#define TRACKING_OCCLUSION_SCORE_THRESH 0.40f  /* below this ratio of visible/total kpts → occluded */

/* ── Multi-person detection assistance ── */
#define TRACKING_NEW_PERSON_GRACE       3      /* frames before a new unmatched detection is promoted */
#define TRACKING_MAX_DET_TO_TRACK_RATIO 2.0f   /* if dets >> tracks, likely new people → run secondary */

/* ── Hungarian algorithm ── */
#define HUNGARIAN_MAX_DIM               256    /* max dimension of cost matrix */

/* ── Kalman Box Tracker (7-state: [cx, cy, area, aspect, vx, vy, varea]) ── */
typedef struct {
    float state[7];
    float covariance[7][7];
    float process_noise[7][7];
    float measurement_noise[4][4];
    float transition[7][7];
    float measurement[4][7];
    int track_id;
    int time_since_update;
    int hit_streak;
    float dt;
} KalmanBoxTracker;

/* ── Pose-keypoint appearance feature ──
 *
 * A compact 12-dim descriptor extracted from COCO-17 keypoint spatial
 * relationships.  Computationally free (no extra model) and robust to
 * viewpoint changes.  Cosine distance between two features is used as
 * the appearance cost in Hungarian matching. ── */
typedef struct {
    float descriptor[TRACKING_APPEARANCE_DIM];
    int num_valid_kpts;               /* keypoints used to compute this feature */
    int frame_index;                  /* when this feature was captured */
    bool is_valid;                    /* false if too few keypoints */
} AppearanceFeature;

/* ── Detection match candidate ── */
typedef enum {
    MATCH_TYPE_FULL_BODY  = 0,        /* standard full-body match */
    MATCH_TYPE_UPPER_BODY = 1,        /* upper-body only (occlusion below waist) */
    MATCH_TYPE_SIDE_BODY  = 2,        /* one side visible (lateral occlusion) */
    MATCH_TYPE_APPEARANCE = 3,        /* appearance-only re-identification */
} MatchType;

/* ── Track Entry ── */
typedef struct {
    int track_id;
    TrackedObject object;
    KalmanBoxTracker kf;
    int lost_count;
    int consecutive_hits;
    bool confirmed;

    /* trajectory */
    SpatialPosition smoothed_positions[TRACKING_MAX_TRAJECTORY];
    int smoothed_count;

    /* keypoint stability */
    float keypoint_stability_score;
    int keypoint_stability_frames;
    int max_keypoints_seen;

    /* spatial jump detection */
    SpatialPosition last_position;
    bool has_last_position;

    /* ── NEW: Appearance feature gallery ── */
    AppearanceFeature appearance_gallery[TRACKING_APPEARANCE_MAX_GALLERY];
    int appearance_count;
    AppearanceFeature latest_appearance;  /* most recent feature for fast matching */

    /* ── NEW: Occlusion state ── */
    float occlusion_score;             /* 1.0 = fully visible, 0.0 = fully occluded */
    int occlusion_frames;              /* consecutive frames with occlusion_score < threshold */
    bool is_partial_body;              /* track currently in partial-body mode (upper/lateral) */
    MatchType last_match_type;         /* how this track was last matched */

    /* ── NEW: Multi-person detection state ── */
    int unmatched_detection_count;     /* detections near this track that weren't matched */
    bool has_nearby_unmatched;         /* flag for cascade decision */
} TrackEntry;

/* ── Object Tracker ── */
typedef struct {
    TrackEntry tracks[TRACKING_MAX_TRACKS];
    int num_tracks;
    int next_id;
    int max_lost;
    int max_occluded;                  /* extended max for occluded tracks */
    float min_iou;
    float max_distance;
    int max_trajectory_length;
    int frame_width;
    int frame_height;

    /* enhanced confirmation config */
    int confirmation_frames;
    int min_keypoints_for_confirm;
    int min_keypoints_strong;
    float spatial_jump_max_m;

    /* ── NEW: Cascade matching config ── */
    int cascade_max_age;               /* max age to consider in cascade */
    int cascade_min_hits;              /* min hits for cascade entry */
    float appearance_weight;           /* weight of appearance in combined cost */
    float appearance_max_dist;         /* cosine distance gate */
    float iou_threshold_low;           /* second-stage IoU threshold for occlusion */

    /* ── NEW: Occlusion handling config ── */
    int upper_body_min_kpts;           /* min keypoints for upper-body match */
    int side_body_min_kpts;            /* min keypoints for side-body match */
    float occlusion_score_threshold;   /* below this → considered occluded */

    /* ── NEW: Multi-person detection ── */
    int new_person_grace_frames;       /* grace period for new person detection */

    /* ── NEW: Re-identification pool ──
     * Recently deleted tracks kept for re-identification.  When a new
     * detection matches a deleted track's appearance, the old track_id
     * is reused instead of creating a new one. */
    AppearanceFeature reid_pool[TRACKING_MAX_TRACKS];
    int reid_pool_ids[TRACKING_MAX_TRACKS];
    int reid_pool_count;
    int reid_pool_max_age;             /* frames to keep deleted track in re-id pool */

    /* ── Diagnostics ── */
    int total_id_switches;
    int total_reidentifications;
    int total_partial_matches;
} ObjectTracker;

/* ── Lifecycle ── */
ObjectTracker* object_tracker_create(int max_lost, float min_iou, float max_distance, int max_traj_len);
void object_tracker_destroy(ObjectTracker* tracker);

/* ── Core update (main entry point) ── */
TrackingResult object_tracker_update(ObjectTracker* tracker,
                                      const Detection* detections, int num_detections,
                                      const SpatialPosition* positions, int num_positions,
                                      int frame_num);

/* ── Post-update: associate poses with tracked objects (call externally) ── */
void object_tracker_associate_poses(ObjectTracker* tracker,
                                     const PoseEstimation* poses, int num_poses);

/* ── State management ── */
void object_tracker_reset(ObjectTracker* tracker);
int  object_tracker_get_active_tracks(const ObjectTracker* tracker, TrackedObject* out_objects, int max_count);
int  object_tracker_get_all_track_count(const ObjectTracker* tracker);
int  object_tracker_get_confirmed_count(const ObjectTracker* tracker);

/* ── Configuration ── */
void object_tracker_set_enhanced_config(ObjectTracker* tracker,
                                         int confirmation_frames,
                                         int min_keypoints_for_confirm,
                                         int min_keypoints_strong,
                                         float spatial_jump_max_m);

/* ── NEW: Extended configuration for cascade + occlusion + re-id ── */
void object_tracker_set_cascade_config(ObjectTracker* tracker,
                                        int cascade_max_age,
                                        int cascade_min_hits,
                                        float appearance_weight,
                                        float appearance_max_dist,
                                        float iou_threshold_low);

void object_tracker_set_occlusion_config(ObjectTracker* tracker,
                                          int max_occluded_frames,
                                          int upper_body_min_kpts,
                                          int side_body_min_kpts,
                                          float occlusion_score_threshold);

void object_tracker_set_reid_config(ObjectTracker* tracker,
                                     int reid_pool_max_age);

void object_tracker_set_multi_person_config(ObjectTracker* tracker,
                                             int new_person_grace_frames);

/* ── Appearance feature extraction ── */
bool appearance_feature_from_pose(const PoseEstimation* pose, const BoundingBox* bbox,
                                   AppearanceFeature* out_feature, int frame_index);
float appearance_feature_distance(const AppearanceFeature* a, const AppearanceFeature* b);
bool appearance_feature_is_valid(const AppearanceFeature* feat);

/* ── Partial-body keypoint counting ──
 * Counts valid keypoints in upper-body (indices 0-10) or one-side subsets. */
int count_upper_body_keypoints(const PoseEstimation* pose, float min_conf);
int count_left_side_keypoints(const PoseEstimation* pose, float min_conf);
int count_right_side_keypoints(const PoseEstimation* pose, float min_conf);

#ifdef __cplusplus
}
#endif

#endif
