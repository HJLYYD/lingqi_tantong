#ifndef K1_ODOMETRY_H
#define K1_ODOMETRY_H

#include "core_types.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── ZUPT / GLRT parameters ── */
#define K1_ODOM_ZUPT_WINDOW       5       /* GLRT 滑动窗口大小 (samples) */
#define K1_ODOM_ZUPT_ACCEL_THRESH 0.15f   /* 加速度模长偏离 g 的阈值 (m/s²) */
#define K1_ODOM_ZUPT_GYRO_THRESH  0.15f   /* 陀螺仪模长阈值 (rad/s).  0.15 covers
                                                 * typical MEMS gyro noise floor
                                                 * after calibration (GY85/MPU6050
                                                 * class: ~0.05 rad/s residual after
                                                 * static calibration). */
#define K1_ODOM_ZUPT_SIGMA_A      0.01f   /* 加速度计噪声标准差 (m/s²) */
#define K1_ODOM_ZUPT_SIGMA_W      0.015f  /* 陀螺仪噪声标准差 (rad/s) */
#define K1_ODOM_ZUPT_GLRT_THRESH  500.0f   /* GLRT statistic threshold.
                                                 * For a stationary device with sigma_a=0.01,
                                                 * sigma_w=0.015: per-sample GLRT ≈ 25 (accel)
                                                 * + 0.5 (gyro) ≈ 25.5.  5-sample window ≈ 128.
                                                 * 500 gives ~4× headroom above baseline while
                                                 * still catching slow walking (<1 m/s).
                                                 * Previous value 3.0e5 (300,000) was ~1000× too
                                                 * high and ZUPT *never* fired. */
#define K1_ODOM_VEL_CLAMP_MPS      5.0f     /* Safety clamp: velocities beyond 5 m/s are
                                                 * physically impossible for a camera rig or
                                                 * slow-moving robot.  Clamping prevents
                                                 * unbounded integration runaway from
                                                 * uncalibrated gyro drift.
                                                 * Previous value 50 m/s was 180 km/h — FAR
                                                 * beyond any pedestrian/mobile-robot use case
                                                 * and allowed massive position runaway. */
#define K1_ODOM_INIT_SAMPLES      200     /* 重力对齐静止采样数 (2s @100Hz) */

/* ── EKF state dimension (9: pos err ×3, vel err ×3, attitude err ×3) ── */
#define K1_ODOM_EKF_STATE_DIM     9
#define K1_ODOM_EKF_MEAS_DIM      3

/*
 * K1Odometry — INS 捷联解算 + ZUPT EKF 里程计
 *
 * 以 K1 启动位置为世界原点 (ENU: Z 向上).
 * 输入: 校准后的加速度计 (m/s²) + 陀螺仪 (rad/s) @ 100Hz
 * 输出: 世界坐标系中的位置、速度、姿态四元数
 *
 * 算法:
 *   1. TRIAD 重力对齐初始化 (静止 2 秒)
 *   2. 四元数姿态积分 (一阶)
 *   3. 加速度世界坐标系转换 + 重力补偿
 *   4. 速度/位置积分
 *   5. GLRT 零速检测 (5 样本滑动窗)
 *   6. ZUPT EKF 误差校正
 *
 * ═══════════════════════════════════════════════════════════
 *  ── 限制: 偏航角不可观 ──
 *
 * ZUPT EKF 仅观测速度误差 (零速伪测量). 这意味着:
 *   - Roll/Pitch: 从加速度计重力分量可观 → ZUPT 可校正
 *   - Yaw (偏航角): 与速度误差无关, 完全不可观 (Ilyas et al.,
 *     Sensors 2016 正式证明)
 *   - Yaw 漂移: 陀螺仪 Z 轴偏置 + 积分漂移导致 ≈1-5°/min
 *
 * 对短时运行 (<60s): 偏航漂移 ≈0.1-0.5°, WorldCoord 自适应锚点
 *   策略可缓解 (静止人体世界坐标不随 K1 偏航漂移)
 * 对长时运行: 偏航角发散, 投影世界坐标围绕 K1 垂直轴旋转
 *
 * 未来改进:
 *   - 磁力计融合 (地磁场 → 绝对偏航) + MAD 异常检测
 *     (Ilyas et al. 2016, Song & Park 2018)
 *   - 视觉罗盘 (视觉里程计 → 偏航观测)
 *   - 双天线 GPS 航向
 *
 * 参考:
 *   Savage, "Strapdown Analytics", 2000
 *   Skog et al., "Zero-Velocity Detection", IEEE TBME 2010
 *   Black, "TRIAD Algorithm", AIAA Journal 1964
 *   Ilyas et al., "Drift Reduction in Pedestrian Navigation",
 *     Sensors 2016 (yaw unobservability proof)
 */
typedef struct {
    /* ── 当前导航状态 (ENU, Z 向上) ── */
    float pos[3];               /* 世界坐标位置 (m) */
    float vel[3];               /* 世界坐标速度 (m/s) */
    float quat[4];              /* 姿态四元数 (w,x,y,z), Body→World */

    /* ── 零偏估计 ── */
    float gyro_bias[3];         /* 陀螺仪零偏 (rad/s) */
    float accel_bias[3];        /* 加速度计零偏 (m/s²) */

    /* ── EKF 协方差 (9×9 行优先) ── */
    float cov[K1_ODOM_EKF_STATE_DIM * K1_ODOM_EKF_STATE_DIM];

    /* ── 过程噪声 ── */
    float sigma_accel;          /* 加速度计噪声 (m/s²/√Hz) */
    float sigma_gyro;           /* 陀螺仪噪声 (rad/s/√Hz) */

    /* ── ZUPT 状态 ── */
    float accel_history[K1_ODOM_ZUPT_WINDOW][3];
    float gyro_history[K1_ODOM_ZUPT_WINDOW][3];
    int   zupt_history_idx;
    bool  zupt_detected;
    bool  last_zupt_valid;

    /* ── 初始化状态 ── */
    bool  initialized;          /* TRIAD 重力对齐完成? */
    bool  init_collecting;      /* 正在收集静止样本? */
    float init_accel_sum[3];    /* 静止样本累加 */
    int   init_sample_count;    /* 已收集样本数 */

    /* ── 时间 ── */
    double last_time;           /* 上次更新时间 (s) */
    uint64_t frame_count;       /* 累计采样次数 */
    int    ins_update_count;    /* INS update diagnostic counter */
    int    zupt_streak;          /* consecutive non-ZUPT frames counter */

    /* ── 线程安全 ──
     * update() 在 capture 线程中调用, get_pose()/get_velocity()
     * 在 postprocess/viz 线程中读取. 此锁保护 pos/vel/quat/cov. */
    pthread_mutex_t state_mutex;

    /* ── 输出 ── */
    FILE* csv_out;              /* 轨迹 CSV */

    /* ── 可调参数 ── */
    float zupt_accel_thresh;
    float zupt_gyro_thresh;
    float zupt_sigma_a;
    float zupt_sigma_w;
    float zupt_glrt_thresh;
    int   init_duration_samples;
} K1Odometry;

/* ── 生命周期 ── */
K1Odometry* k1_odometry_create(void);
void k1_odometry_destroy(K1Odometry* odom);

/* ── 配置 ── */
void k1_odometry_set_params(K1Odometry* odom,
                             float zupt_accel_thresh, float zupt_gyro_thresh,
                             float sigma_a, float sigma_w, float glrt_thresh,
                             int init_samples);

/* 设置 IMU 零偏 (从 K1 IMU 校准获得).
 * 调用时机: IMU 校准完成后.
 * gyro_bias:  rad/s,  accel_bias:  m/s² (通常为零, MEMS 加速度计零偏较小). */
void k1_odometry_set_biases(K1Odometry* odom,
                             const float gyro_bias[3],
                             const float accel_bias[3]);

/* ── CSV 记录 ── */
void k1_odometry_start_recording(K1Odometry* odom, const char* dir, const char* session_id);
void k1_odometry_stop_recording(K1Odometry* odom);

/* ── 初始化 ── */
bool k1_odometry_is_initialized(const K1Odometry* odom);
bool k1_odometry_is_collecting(const K1Odometry* odom);

/* ── 核心更新 (每收到一个 IMU 样本调用一次) ── */
void k1_odometry_update(K1Odometry* odom, const IMUData* imu);

/* ── 读取当前位姿 ── */
void k1_odometry_get_pose(const K1Odometry* odom, float pos[3], float quat[4]);
void k1_odometry_get_velocity(const K1Odometry* odom, float vel[3]);

/* ── 构造变换矩阵 R (3×3 行优先) + t (3×1) ── */
void k1_odometry_get_transform(const K1Odometry* odom, float R[9], float t[3]);

/* ── 四元数工具函数 ── */
void k1_quat_identity(float q[4]);
void k1_quat_normalize(float q[4]);
void k1_quat_conjugate(const float q[4], float q_conj[4]);
void k1_quat_multiply(const float q1[4], const float q2[4], float q_out[4]);
void k1_quat_rotate(const float q[4], const float v[3], float v_out[3]);
void k1_quat_to_rotation_matrix(const float q[4], float R[9]);
void k1_quat_from_rotation_matrix(const float R[9], float q[4]);

/* ── GLRT 零速检测 (独立函数, 可测试) ── */
bool k1_odom_glrt_zupt_detect(const float accel_samples[][3],
                               const float gyro_samples[][3],
                               int window_size,
                               float sigma_a, float sigma_w, float threshold);

#ifdef __cplusplus
}
#endif

#endif /* K1_ODOMETRY_H */
