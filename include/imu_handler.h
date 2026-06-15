#ifndef IMU_HANDLER_H
#define IMU_HANDLER_H

#include "core_types.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMU_DEFAULT_WINDOW_SIZE     10
#define IMU_MAX_WINDOW_SIZE         64

typedef struct {
    int window_size;
    float gyro_noise_std;
    float accel_noise_std;
    IMUData buffer[IMU_MAX_WINDOW_SIZE];
    int buffer_count;
    bool is_calibrated;

    /* ── Thread-safe external pose ──
     * Written by Capture thread (Arrow/MJPEG IMU), read by PostProcess thread.
     * Mutex protects against torn reads of the 11-field struct. */
    pthread_mutex_t pose_mutex;
    IMUExternalPose latest_pose;
    bool has_external_pose;
} IMUHandler;

IMUHandler* imu_handler_create(int window_size, float gyro_noise_std, float accel_noise_std);
void imu_handler_destroy(IMUHandler* handler);

bool imu_handler_validate(const IMUHandler* handler, const float accel[3], const float gyro[3]);
IMUData imu_handler_parse(IMUHandler* handler, const float accel[3], const float gyro[3], double timestamp);
IMUData imu_handler_smooth(IMUHandler* handler, const IMUData* data);

void imu_handler_reset(IMUHandler* handler);

void imu_handler_set_external_pose(IMUHandler* handler, float qw, float qx, float qy, float qz, float pitch, float roll, float yaw, float altitude, float temp, uint32_t ts);
bool imu_handler_get_latest_pose(const IMUHandler* handler, IMUExternalPose* out_pose);

#ifdef __cplusplus
}
#endif

#endif
