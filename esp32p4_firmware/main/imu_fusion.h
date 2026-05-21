#ifndef IMU_FUSION_H
#define IMU_FUSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float q0, q1, q2, q3;
    float beta;
    float inv_sample_freq;
    float pitch, roll, yaw;
} madgwick_filter_t;

typedef struct {
    float qw, qx, qy, qz;
    float pitch, roll, yaw;
    float altitude_m;
    float temperature_c;
    uint32_t timestamp_ms;
} imu_pose_t;

typedef struct {
    float bias_x, bias_y, bias_z;
} gyro_bias_t;

typedef struct {
    float offset_x, offset_y, offset_z;
    float scale_x, scale_y, scale_z;
} mag_calibration_t;

void madgwick_init(madgwick_filter_t *f, float beta, float sample_freq);

void madgwick_update_9dof(madgwick_filter_t *f,
                          float gx, float gy, float gz,
                          float ax, float ay, float az,
                          float mx, float my, float mz,
                          float dt);

void madgwick_get_euler(madgwick_filter_t *f);

imu_pose_t imu_fusion_get_pose(madgwick_filter_t *f, float altitude,
                               float temperature, uint32_t ts);

gyro_bias_t imu_calibrate_gyro(const float *gx_samples, const float *gy_samples,
                               const float *gz_samples, uint32_t n);

mag_calibration_t imu_calibrate_mag(const float *mx_samples, const float *my_samples,
                                    const float *mz_samples, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif