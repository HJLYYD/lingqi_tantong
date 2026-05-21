#ifndef TRACKING_MANAGER_H
#define TRACKING_MANAGER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRACKING_MAX_TRACKS             256
#define TRACKING_MAX_TRAJECTORY         300
#define TRACKING_CONFIRMATION_FRAMES    3
#define TRACKING_MAX_LOST_FRAMES        30
#define TRACKING_OCCLUSION_WINDOW       15
#define TRACKING_IOU_THRESHOLD_HIGH     0.3f
#define TRACKING_IOU_THRESHOLD_LOW      0.1f
#define TRACKING_CONFIDENCE_HIGH        0.5f
#define TRACKING_EMA_ALPHA              0.3f

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

typedef struct {
    int track_id;
    TrackedObject object;
    KalmanBoxTracker kf;
    int lost_count;
    int consecutive_hits;
    bool confirmed;
    SpatialPosition smoothed_positions[TRACKING_MAX_TRAJECTORY];
    int smoothed_count;
} TrackEntry;

typedef struct {
    TrackEntry tracks[TRACKING_MAX_TRACKS];
    int num_tracks;
    int next_id;
    int max_lost;
    float min_iou;
    float max_distance;
    int max_trajectory_length;
    int frame_width;
    int frame_height;
} ObjectTracker;

ObjectTracker* object_tracker_create(int max_lost, float min_iou, float max_distance, int max_traj_len);
void object_tracker_destroy(ObjectTracker* tracker);

TrackingResult object_tracker_update(ObjectTracker* tracker,
                                      const Detection* detections, int num_detections,
                                      const SpatialPosition* positions, int num_positions,
                                      int frame_num);

void object_tracker_reset(ObjectTracker* tracker);
int  object_tracker_get_active_tracks(const ObjectTracker* tracker, TrackedObject* out_objects, int max_count);

#ifdef __cplusplus
}
#endif

#endif
