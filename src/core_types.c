#include "core_types.h"
#include <string.h>

void detection_init(Detection* det, float x1, float y1, float x2, float y2,
                    float conf, int class_id, const char* class_name) {
    det->bbox.x_min = x1;
    det->bbox.y_min = y1;
    det->bbox.x_max = x2;
    det->bbox.y_max = y2;
    det->confidence = conf;
    det->class_id = class_id;
    strncpy(det->class_name, class_name ? class_name : "unknown", MAX_STRING_LEN - 1);
    det->class_name[MAX_STRING_LEN - 1] = '\0';
}

void pose_estimation_init(PoseEstimation* pose) {
    memset(pose, 0, sizeof(PoseEstimation));
    pose->num_keypoints = MAX_KEYPOINTS;
    pose->has_bbox = false;
    pose->confidence = 0.0f;
}

void tracked_object_init(TrackedObject* obj, int track_id, const Detection* det,
                         const SpatialPosition* pos) {
    memset(obj, 0, sizeof(TrackedObject));
    obj->track_id = track_id;
    if (det) obj->detection = *det;
    if (pos) obj->spatial_pos = *pos;
    obj->is_active = true;
    obj->is_occluded = false;
    obj->frames_seen = 1;
}

void trajectory_buffer_append(TrajectoryBuffer* buf, const SpatialPosition* pos) {
    if (buf->count < MAX_TRAJECTORY_LEN) {
        buf->positions[buf->count++] = *pos;
    } else {
        memmove(&buf->positions[0], &buf->positions[1],
                (MAX_TRAJECTORY_LEN - 1) * sizeof(SpatialPosition));
        buf->positions[MAX_TRAJECTORY_LEN - 1] = *pos;
    }
}

void inference_result_init(InferenceResult* result) {
    memset(result, 0, sizeof(InferenceResult));
}

void tracking_result_init(TrackingResult* result) {
    memset(result, 0, sizeof(TrackingResult));
}

void frame_data_init(FrameData* frame, uint8_t* data, int w, int h, int c, int idx, double ts) {
    frame->data = data;
    frame->width = w;
    frame->height = h;
    frame->channels = c;
    frame->frame_index = idx;
    frame->timestamp = ts;
}

void imu_data_init(IMUData* imu, double ts,
                   float ax, float ay, float az,
                   float gx, float gy, float gz) {
    imu->timestamp = ts;
    imu->accel_x = ax;
    imu->accel_y = ay;
    imu->accel_z = az;
    imu->gyro_x = gx;
    imu->gyro_y = gy;
    imu->gyro_z = gz;
}
