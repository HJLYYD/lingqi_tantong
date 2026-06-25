#ifndef IMU_HANDLER_H
#define IMU_HANDLER_H

#include "core_types.h"
#include <pthread.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMU_DEFAULT_WINDOW_SIZE     10
#define IMU_MAX_WINDOW_SIZE         64

/* ── IMUHandler ──
 * 管理 K1 本地 IMU + ESP32 远端 IMU 的融合处理:
 *   - 两个独立 Madgwick 滤波器 (K1 + 相机)
 *   - Wahba 重力向量对齐 (启动时自动完成)
 *   - 输出 DualImuPose (双姿态 + 对齐旋转)
 *   - 保留原有 set_external_pose / get_latest_pose API
 */
typedef struct {
    /* ── 原有: 滑动窗口平滑 + 外部姿态 ── */
    int window_size;
    float gyro_noise_std;
    float accel_noise_std;
    IMUData buffer[IMU_MAX_WINDOW_SIZE];
    int buffer_count;
    bool is_calibrated;

    pthread_mutex_t pose_mutex;
    IMUExternalPose latest_pose;
    bool has_external_pose;

    /* ── Phase 2: Madgwick 滤波器 ×2 ── */
    MadgwickFilter  k1_filter;        /* K1 本地 IMU */
    MadgwickFilter  cam_filter;       /* ESP32 远端 IMU (原始数据) */

    /* ── Phase 2: Wahba 重力对齐 ── */
    FrameAlignmentCtx align_ctx;
    bool   alignment_done;
    float  align_qw, align_qx, align_qy, align_qz;

    /* ── Phase 2: 融合输出 ── */
    DualImuPose dual_pose;
    pthread_mutex_t dual_mutex;

    /* ── K1 本地 IMU 引用 ── */
    void*  k1_imu;                    /* K1Imu* — 前向声明 */

    /* ── IMU 数据记录到 CSV ── */
    FILE*  k1_csv;                    /* K1 本地 IMU 采样 CSV */
    FILE*  esp32_csv;                 /* ESP32 远端 IMU 采样 CSV */
} IMUHandler;

IMUHandler* imu_handler_create(int window_size, float gyro_noise_std, float accel_noise_std);
void imu_handler_destroy(IMUHandler* handler);

/* ── 原有 API (保留不变) ── */
bool imu_handler_validate(const IMUHandler* handler, const float accel[3], const float gyro[3]);
IMUData imu_handler_parse(IMUHandler* handler, const float accel[3], const float gyro[3], double timestamp);
IMUData imu_handler_smooth(IMUHandler* handler, const IMUData* data);
void imu_handler_reset(IMUHandler* handler);
void imu_handler_set_external_pose(IMUHandler* handler, float qw, float qx, float qy, float qz, float pitch, float roll, float yaw, float altitude, float temp, uint32_t ts);
bool imu_handler_get_latest_pose(const IMUHandler* handler, IMUExternalPose* out_pose);

/* ── Phase 2 API: Madgwick + 双 IMU ── */
void madgwick_init(MadgwickFilter* mf, float beta, float sample_freq);
void madgwick_update(MadgwickFilter* mf, float gx, float gy, float gz, float ax, float ay, float az, float dt);
void madgwick_get_quat(const MadgwickFilter* mf, float* qw, float* qx, float* qy, float* qz);
void madgwick_get_euler(const MadgwickFilter* mf, float* pitch, float* roll, float* yaw);

/* 馈送原始 IMU 数据到滤波器 */
void imu_handler_feed_k1_imu(IMUHandler* h, const IMUData* data);
void imu_handler_feed_external_raw(IMUHandler* h, const float accel[3], const float gyro[3]);

/* 获取双 IMU 融合结果 */
bool imu_handler_get_dual_pose(IMUHandler* h, DualImuPose* out);
bool imu_handler_is_alignment_done(const IMUHandler* h);

/* 设置 K1 本地 IMU 实例引用 (用于采集线程读取) */
void imu_handler_set_k1_imu(IMUHandler* h, void* k1_imu);

/* ── IMU 数据记录到 CSV ── */
void imu_handler_start_recording(IMUHandler* h, const char* output_dir, const char* session_id);
void imu_handler_stop_recording(IMUHandler* h);

#ifdef __cplusplus
}
#endif

#endif
