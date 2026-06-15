#include "imu_handler.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMU_MAX_ACCEL_G  16.0f
#define IMU_MAX_GYRO_DPS 2000.0f

static bool is_valid_float(float v) {
    return !isnan(v) && !isinf(v);
}

IMUHandler* imu_handler_create(int window_size, float gyro_noise_std, float accel_noise_std) {
    IMUHandler* handler = (IMUHandler*)calloc(1, sizeof(IMUHandler));
    if (!handler) return NULL;

    handler->window_size = UTILS_CLAMP(window_size, 1, IMU_MAX_WINDOW_SIZE);
    handler->gyro_noise_std = gyro_noise_std;
    handler->accel_noise_std = accel_noise_std;
    handler->buffer_count = 0;
    handler->is_calibrated = false;
    pthread_mutex_init(&handler->pose_mutex, NULL);
    memset(&handler->latest_pose, 0, sizeof(handler->latest_pose));
    handler->has_external_pose = false;

    return handler;
}

void imu_handler_destroy(IMUHandler* handler) {
    if (!handler) return;
    pthread_mutex_destroy(&handler->pose_mutex);
    free(handler);
}

bool imu_handler_validate(const IMUHandler* handler, const float accel[3], const float gyro[3]) {
    (void)handler;
    if (!accel || !gyro) return false;

    for (int i = 0; i < 3; i++) {
        if (!is_valid_float(accel[i]) || fabsf(accel[i]) > IMU_MAX_ACCEL_G) return false;
        if (!is_valid_float(gyro[i]) || fabsf(gyro[i]) > IMU_MAX_GYRO_DPS) return false;
    }
    return true;
}

IMUData imu_handler_parse(IMUHandler* handler, const float accel[3], const float gyro[3], double timestamp) {
    IMUData data = {0};

    if (!imu_handler_validate(handler, accel, gyro)) {
        log_error("Invalid IMU data format");
        return data;
    }

    data.timestamp = timestamp;
    data.accel_x = accel[0];
    data.accel_y = accel[1];
    data.accel_z = accel[2];
    data.gyro_x = gyro[0];
    data.gyro_y = gyro[1];
    data.gyro_z = gyro[2];

    return data;
}

IMUData imu_handler_smooth(IMUHandler* handler, const IMUData* data) {
    if (!handler || !data) return (IMUData){0};

    if (handler->buffer_count < handler->window_size) {
        handler->buffer[handler->buffer_count++] = *data;
    } else {
        memmove(&handler->buffer[0], &handler->buffer[1],
                (handler->window_size - 1) * sizeof(IMUData));
        handler->buffer[handler->window_size - 1] = *data;
    }

    if (handler->buffer_count < 2) {
        return *data;
    }

    IMUData smoothed = {0};
    smoothed.timestamp = data->timestamp;

    for (int i = 0; i < handler->buffer_count; i++) {
        smoothed.accel_x += handler->buffer[i].accel_x;
        smoothed.accel_y += handler->buffer[i].accel_y;
        smoothed.accel_z += handler->buffer[i].accel_z;
        smoothed.gyro_x += handler->buffer[i].gyro_x;
        smoothed.gyro_y += handler->buffer[i].gyro_y;
        smoothed.gyro_z += handler->buffer[i].gyro_z;
    }

    float inv_count = 1.0f / handler->buffer_count;
    smoothed.accel_x *= inv_count;
    smoothed.accel_y *= inv_count;
    smoothed.accel_z *= inv_count;
    smoothed.gyro_x *= inv_count;
    smoothed.gyro_y *= inv_count;
    smoothed.gyro_z *= inv_count;

    return smoothed;
}

void imu_handler_reset(IMUHandler* handler) {
    if (!handler) return;
    handler->buffer_count = 0;
    handler->is_calibrated = false;
    log_debug("IMU handler buffer cleared");
}

void imu_handler_set_external_pose(IMUHandler* handler, float qw, float qx, float qy, float qz, float pitch, float roll, float yaw, float altitude, float temp, uint32_t ts) {
    if (!handler) return;

    pthread_mutex_lock(&handler->pose_mutex);
    handler->latest_pose.qw = qw;
    handler->latest_pose.qx = qx;
    handler->latest_pose.qy = qy;
    handler->latest_pose.qz = qz;
    handler->latest_pose.pitch = pitch;
    handler->latest_pose.roll = roll;
    handler->latest_pose.yaw = yaw;
    handler->latest_pose.altitude_m = altitude;
    handler->latest_pose.temperature_c = temp;
    handler->latest_pose.timestamp_ms = ts;
    handler->latest_pose.is_valid = true;
    handler->has_external_pose = true;
    pthread_mutex_unlock(&handler->pose_mutex);
}

bool imu_handler_get_latest_pose(const IMUHandler* handler, IMUExternalPose* out_pose) {
    if (!handler || !out_pose) return false;

    pthread_mutex_lock(&((IMUHandler*)handler)->pose_mutex);
    if (!handler->has_external_pose || !handler->latest_pose.is_valid) {
        pthread_mutex_unlock(&((IMUHandler*)handler)->pose_mutex);
        return false;
    }

    *out_pose = handler->latest_pose;
    pthread_mutex_unlock(&((IMUHandler*)handler)->pose_mutex);
    return true;
}
