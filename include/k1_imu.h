#ifndef K1_IMU_H
#define K1_IMU_H

#include "core_types.h"
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

/*
 * K1 本地 I2C IMU 驱动 — 直接读取 GY85 (ADXL345 + ITG3205)
 *
 * 参照 gy85_json_output.py，使用 Linux I2C 用户空间 API。
 * 环形缓冲 + 互斥锁，线程安全。
 */

#define K1_IMU_RING_SIZE      256
#define K1_IMU_CALIB_SAMPLES  200    /* 200样本@100Hz = 2s静止校准 */

/* ── GY85 I2C 地址 ── */
#define K1_IMU_ADXL345_ADDR   0x53
#define K1_IMU_ITG3205_ADDR   0x68

typedef enum {
    K1_IMU_STATE_UNINIT = 0,
    K1_IMU_STATE_CALIBRATING,
    K1_IMU_STATE_RUNNING,
    K1_IMU_STATE_ERROR
} K1ImuState;

typedef struct {
    float gyro_bias[3];          /* rad/s — 陀螺仪零偏 */
    float accel_bias[3];         /* m/s² — 加速度计零偏 */
    int   samples_collected;
    bool  done;
} K1ImuCalibration;

typedef struct {
    /* ── I2C ── */
    int   i2c_bus;
    int   i2c_fd;
    float sample_rate_hz;

    /* ── 状态 ── */
    K1ImuState       state;
    K1ImuCalibration calib;

    /* ── 环形缓冲 ── */
    IMUData ring[K1_IMU_RING_SIZE];
    int     ring_head;
    int     ring_count;
    pthread_mutex_t ring_mutex;

    /* ── 统计 ── */
    int    total_samples;
    int    error_count;
    double last_sample_time;
    float  actual_rate;
} K1Imu;

/* ── 生命周期 ── */
K1Imu* k1_imu_create(int i2c_bus, float sample_rate_hz);
void   k1_imu_destroy(K1Imu* imu);

/* ── 数据读取 (单次 I2C 采样, 返回校准后的 IMUData) ── */
bool   k1_imu_read_sample(K1Imu* imu, IMUData* out);

/* ── 批量读取最近 N 个样本 ── */
int    k1_imu_read_recent(K1Imu* imu, IMUData* out, int max_count);

/* ── 校准 ── */
bool   k1_imu_start_calibration(K1Imu* imu);
bool   k1_imu_is_calibrated(const K1Imu* imu);

/* ── 状态查询 ── */
K1ImuState k1_imu_get_state(const K1Imu* imu);
float      k1_imu_get_actual_rate(const K1Imu* imu);

#endif /* K1_IMU_H */
