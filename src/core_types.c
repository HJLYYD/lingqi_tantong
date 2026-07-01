#include "core_types.h"
#include <string.h>

void detection_init(Detection* d, float x1, float y1, float x2, float y2,
                    float conf, int cls, const char* name) {
    d->bbox.x_min = x1; d->bbox.y_min = y1;
    d->bbox.x_max = x2; d->bbox.y_max = y2;
    d->confidence = conf;
    d->class_id = cls;
    strncpy(d->class_name, name ? name : "unknown", STR_LEN - 1);
    d->class_name[STR_LEN - 1] = '\0';
    d->track_id_hint = -1;
    d->is_partial_body = false;
    d->num_visible_keypoints = 0;
}

void pose_estimation_init(Pose* p) {
    memset(p, 0, sizeof(Pose));
    p->num_keypoints = MAX_KPTS;
    p->has_bbox = false;
    p->confidence = 0.0f;
}

void tracked_object_init(TrackedObj* o, int id, const Detection* d, const SpatialPos* sp) {
    memset(o, 0, sizeof(TrackedObj));
    o->track_id = id;
    if (d)  o->detection = *d;
    if (sp) o->spatial_pos = *sp;
    o->is_active   = true;
    o->is_occluded = false;
    o->frames_seen = 1;
}

void trajectory_buffer_append(TrajBuf* b, const SpatialPos* sp) {
    if (b->count < TRAJ_LEN) {
        b->positions[b->count++] = *sp;
    } else {
        memmove(&b->positions[0], &b->positions[1], (TRAJ_LEN - 1) * sizeof(SpatialPos));
        b->positions[TRAJ_LEN - 1] = *sp;
    }
}

void inference_result_init(InferResult* r) {
    /* Preserve depth_map if already allocated: memset would leak it.
     * depth_map is owned externally (future depth-estimation model output). */
    float* saved_depth = r->depth_map;
    memset(r, 0, sizeof(InferResult));
    r->depth_map = saved_depth;
}

void tracking_result_init(TrackResult* r) {
    memset(r, 0, sizeof(TrackResult));
}

void frame_data_init(Frame* f, uint8_t* data, int w, int h, int c, int idx, double ts) {
    f->data = data; f->width = w; f->height = h; f->channels = c;
    f->frame_index = idx; f->timestamp = ts;
}

void imu_data_init(ImuSample* imu, double ts, float ax, float ay, float az,
                   float gx, float gy, float gz) {
    imu->timestamp = ts;
    imu->accel_x = ax; imu->accel_y = ay; imu->accel_z = az;
    imu->gyro_x = gx; imu->gyro_y = gy; imu->gyro_z = gz;
}
