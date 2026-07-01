/*
 * K1Odometry — INS 捷联解算 + ZUPT EKF 里程计
 *
 * 以 K1 启动位置为世界原点 (ENU: Z 向上).
 *
 * 算法管线:
 *   1. TRIAD 重力对齐初始化 (Black 1964)
 *   2. 四元数一阶姿态积分
 *   3. 重力补偿 + 世界坐标系加速度
 *   4. 速度/位置积分
 *   5. GLRT 零速检测 (Skog et al. 2010)
 *   6. ZUPT EKF 误差校正 (9 维状态)
 *
 * 参考:
 *   Savage, "Strapdown Analytics", 2000
 *   Davenport, NASA TN D-4691, 1968
 *   Skog et al., IEEE TBME 2010
 */

#include "k1_odometry.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Earth's gravity magnitude (m/s²) — positive scalar */
#define GRAVITY_MAG  9.80665f

/* ═══════════════════════════════════════════════════════════
 *  四元数工具函数
 * ═══════════════════════════════════════════════════════════ */

void k1_quat_identity(float q[4]) {
    q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
}

void k1_quat_normalize(float q[4]) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-8f) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    }
}

void k1_quat_conjugate(const float q[4], float q_conj[4]) {
    q_conj[0] =  q[0];
    q_conj[1] = -q[1];
    q_conj[2] = -q[2];
    q_conj[3] = -q[3];
}

void k1_quat_multiply(const float q1[4], const float q2[4], float q_out[4]) {
    /* Hamilton product: q_out = q1 ⊗ q2 */
    float w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    float x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    float y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    float z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
    q_out[0] = w; q_out[1] = x; q_out[2] = y; q_out[3] = z;
}

void k1_quat_rotate(const float q[4], const float v[3], float v_out[3]) {
    /* v_out = q ⊗ v ⊗ q*  (treating v as pure quaternion (0, vx, vy, vz)) */
    float q_conj[4];
    k1_quat_conjugate(q, q_conj);

    /* qv = q ⊗ (0, v) */
    float qv[4], result[4];
    float vq[4] = {0.0f, v[0], v[1], v[2]};
    k1_quat_multiply(q, vq, qv);
    k1_quat_multiply(qv, q_conj, result);

    v_out[0] = result[1];
    v_out[1] = result[2];
    v_out[2] = result[3];
}

void k1_quat_to_rotation_matrix(const float q[4], float R[9]) {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float qw2 = qw*qw, qx2 = qx*qx, qy2 = qy*qy, qz2 = qz*qz;
    float qwqx = qw*qx, qwqy = qw*qy, qwqz = qw*qz;
    float qxqy = qx*qy, qxqz = qx*qz, qyqz = qy*qz;

    /* R = Body→World rotation (R * v_body = v_world) */
    R[0] = qw2 + qx2 - qy2 - qz2;  R[1] = 2.0f*(qxqy - qwqz);      R[2] = 2.0f*(qxqz + qwqy);
    R[3] = 2.0f*(qxqy + qwqz);      R[4] = qw2 - qx2 + qy2 - qz2;  R[5] = 2.0f*(qyqz - qwqx);
    R[6] = 2.0f*(qxqz - qwqy);      R[7] = 2.0f*(qyqz + qwqx);      R[8] = qw2 - qx2 - qy2 + qz2;
}

void k1_quat_from_rotation_matrix(const float R[9], float q[4]) {
    /* Convert rotation matrix to quaternion (Hamilton convention).
     * Based on Markley & Crassidis, "Fundamentals of Spacecraft ADCS", 2014 §2.9 */
    float tr = R[0] + R[4] + R[8];
    float S;

    if (tr > 0.0f) {
        S = sqrtf(tr + 1.0f) * 2.0f;
        q[0] = 0.25f * S;
        q[1] = (R[7] - R[5]) / S;
        q[2] = (R[2] - R[6]) / S;
        q[3] = (R[3] - R[1]) / S;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        S = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
        q[0] = (R[7] - R[5]) / S;
        q[1] = 0.25f * S;
        q[2] = (R[1] + R[3]) / S;
        q[3] = (R[2] + R[6]) / S;
    } else if (R[4] > R[8]) {
        S = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
        q[0] = (R[2] - R[6]) / S;
        q[1] = (R[1] + R[3]) / S;
        q[2] = 0.25f * S;
        q[3] = (R[5] + R[7]) / S;
    } else {
        S = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
        q[0] = (R[3] - R[1]) / S;
        q[1] = (R[2] + R[6]) / S;
        q[2] = (R[5] + R[7]) / S;
        q[3] = 0.25f * S;
    }
    k1_quat_normalize(q);
}

/* ═══════════════════════════════════════════════════════════
 *  GLRT 零速检测器 (Skog et al. 2010)
 * ═══════════════════════════════════════════════════════════ */

bool k1_odom_glrt_zupt_detect(const float accel_samples[][3],
                               const float gyro_samples[][3],
                               int window_size,
                               float sigma_a, float sigma_w, float threshold) {
    if (window_size <= 0) return false;

    /* 1. 窗口内加速度均值方向 (重力方向估计) */
    float avg_accel[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < 3; j++) {
            avg_accel[j] += accel_samples[i][j];
        }
    }
    float norm = sqrtf(avg_accel[0]*avg_accel[0] + avg_accel[1]*avg_accel[1] + avg_accel[2]*avg_accel[2]);
    if (norm < 1e-6f) return false;
    for (int j = 0; j < 3; j++) avg_accel[j] /= norm;

    /* 2. 计算 GLRT 统计量 */
    float T = 0.0f;
    float inv_sigma_a2 = 1.0f / (sigma_a * sigma_a);
    float inv_sigma_w2 = 1.0f / (sigma_w * sigma_w);

    for (int i = 0; i < window_size; i++) {
        /* 加速度残差 (偏离重力方向的成分) */
        float accel_diff = 0.0f;
        for (int j = 0; j < 3; j++) {
            float diff = accel_samples[i][j] - GRAVITY_MAG * avg_accel[j];
            accel_diff += diff * diff;
        }
        /* 陀螺仪模长平方 */
        float gyro_mag2 = gyro_samples[i][0]*gyro_samples[i][0]
                        + gyro_samples[i][1]*gyro_samples[i][1]
                        + gyro_samples[i][2]*gyro_samples[i][2];

        T += accel_diff * inv_sigma_a2 + gyro_mag2 * inv_sigma_w2;
    }
    T /= (float)window_size;

    return (T < threshold);
}

/* ═══════════════════════════════════════════════════════════
 *  简单阈值 ZUPT 检测 (备用/互补)
 * ═══════════════════════════════════════════════════════════ */

static bool simple_zupt_check(K1Odometry* odom, float ax_c, float ay_c, float az_c,
                               float gx_c, float gy_c, float gz_c) {
    float accel_mag = sqrtf(ax_c*ax_c + ay_c*ay_c + az_c*az_c);
    float gyro_mag  = sqrtf(gx_c*gx_c + gy_c*gy_c + gz_c*gz_c);

    return (fabsf(accel_mag - GRAVITY_MAG) < odom->zupt_accel_thresh)
        && (gyro_mag < odom->zupt_gyro_thresh);
}

/* ═══════════════════════════════════════════════════════════
 *  ZUPT EKF 误差校正
 *
 *  状态 (9维): X = [δp_x, δp_y, δp_z, δv_x, δv_y, δv_z, δφ_x, δφ_y, δφ_z]
 *  观测 (3维): Z = v_estimated (应等于 0)
 *  观测矩阵:  H = [0_{3×3}, I_{3×3}, 0_{3×3}]
 * ═══════════════════════════════════════════════════════════ */

static void zupt_ekf_update(K1Odometry* odom) {
    /* Measurement: pseudo-observation of zero velocity */
    float z[3] = {0.0f, 0.0f, 0.0f};  /* we "measure" velocity = 0 */

    /* Innovation: y = z - H*x = z - v_estimated */
    float y[3];
    y[0] = z[0] - odom->vel[0];
    y[1] = z[1] - odom->vel[1];
    y[2] = z[2] - odom->vel[2];

    /* Measurement noise covariance R (3×3 diagonal).
     * Small values → strong zero-velocity enforcement.
     * Tuned for walking-speed scenarios (0.05 m/s typical residual) */
    float R[9] = {
        0.01f, 0.0f,  0.0f,
        0.0f,  0.01f, 0.0f,
        0.0f,  0.0f,  0.01f
    };

    /* H = [0_{3×3}, I_{3×3}, 0_{3×3}] → P*H' is columns 3-5 of P */
    int n = K1_ODOM_EKF_STATE_DIM;  /* 9 */
    int m = K1_ODOM_EKF_MEAS_DIM;   /* 3 */

    /* S = H*P*H' + R  (3×3 innovation covariance) */
    float S[9] = {0};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            S[i*3 + j] = odom->cov[(3+i)*n + (3+j)] + R[i*3 + j];
        }
    }

    /* Compute S inverse (3×3) using analytic formula */
    float det = S[0]*(S[4]*S[8] - S[5]*S[7])
              - S[1]*(S[3]*S[8] - S[5]*S[6])
              + S[2]*(S[3]*S[7] - S[4]*S[6]);
    if (fabsf(det) < 1e-12f) return;

    float inv_det = 1.0f / det;
    float S_inv[9];
    S_inv[0] = (S[4]*S[8] - S[5]*S[7]) * inv_det;
    S_inv[1] = (S[2]*S[7] - S[1]*S[8]) * inv_det;
    S_inv[2] = (S[1]*S[5] - S[2]*S[4]) * inv_det;
    S_inv[3] = (S[5]*S[6] - S[3]*S[8]) * inv_det;
    S_inv[4] = (S[0]*S[8] - S[2]*S[6]) * inv_det;
    S_inv[5] = (S[2]*S[3] - S[0]*S[5]) * inv_det;
    S_inv[6] = (S[3]*S[7] - S[4]*S[6]) * inv_det;
    S_inv[7] = (S[1]*S[6] - S[0]*S[7]) * inv_det;
    S_inv[8] = (S[0]*S[4] - S[1]*S[3]) * inv_det;

    /* K = P*H' * S^{-1}  (9×3 = columns 3-5 of P × S^{-1}) */
    float K[9*3];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += odom->cov[i*n + (3+k)] * S_inv[k*3 + j];
            }
            K[i*3 + j] = sum;
        }
    }

    /* State update: x += K*y */
    float dx[K1_ODOM_EKF_STATE_DIM];
    for (int i = 0; i < n; i++) {
        dx[i] = K[i*3 + 0]*y[0] + K[i*3 + 1]*y[1] + K[i*3 + 2]*y[2];
    }

    /* ── Apply corrections ── */
    /* Position correction */
    odom->pos[0] -= dx[0];
    odom->pos[1] -= dx[1];
    odom->pos[2] -= dx[2];

    /* Velocity correction */
    odom->vel[0] -= dx[3];
    odom->vel[1] -= dx[4];
    odom->vel[2] -= dx[5];

    /* Attitude correction (small-angle approx in body frame) */
    float dphi[3] = {dx[6], dx[7], dx[8]};
    float dphi_mag = sqrtf(dphi[0]*dphi[0] + dphi[1]*dphi[1] + dphi[2]*dphi[2]);
    if (dphi_mag > 1e-8f) {
        float half = dphi_mag * 0.5f;
        float s = sinf(half) / dphi_mag;
        float dq[4] = {cosf(half), dphi[0]*s, dphi[1]*s, dphi[2]*s};
        float q_corrected[4];
        k1_quat_multiply(odom->quat, dq, q_corrected);
        memcpy(odom->quat, q_corrected, 4 * sizeof(float));
        k1_quat_normalize(odom->quat);
    }

    /* ── Covariance update: P = (I - K*H) * P ── */
    /* K*H: 9×9 — only columns 3-5 of K are non-zero (H = [0 I 0]) */
    float KH[81];
    memset(KH, 0, sizeof(KH));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            KH[i*n + (3+j)] = K[i*3 + j];
        }
    }

    /* I - K*H */
    float I_KH[81];
    for (int i = 0; i < 81; i++) I_KH[i] = -KH[i];
    for (int i = 0; i < n; i++) I_KH[i*n + i] += 1.0f;

    /* P_new = (I-KH) * P */
    float P_new[81];
    memset(P_new, 0, sizeof(P_new));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += I_KH[i*n + k] * odom->cov[k*n + j];
            }
            P_new[i*n + j] = sum;
        }
    }
    memcpy(odom->cov, P_new, sizeof(P_new));
}

/* ═══════════════════════════════════════════════════════════
 *  TRIAD 重力对齐初始化 (Black 1964)
 * ═══════════════════════════════════════════════════════════ */

static bool triad_gravity_init(K1Odometry* odom, const float accel_avg[3]) {
    /* 在静止时, 加速度计测得的力 = 重力在体坐标系的投影的反作用力
     * a_body = -R_W_B^T * g_world
     * 在 ENU (Z 向上) 中, g_world = (0, 0, -9.80665)
     * 所以 a_body 指向上 → 归一化后的 a_body ≈ 体坐标系中的 "上" 方向
     *
     * TRIAD:
     *   r1 = normalize(accel_avg)  → 体重力方向 (= ENU 的 -Z 方向在体坐标系中)
     *   b1 = (0, 0, -1)            → 世界系的重力方向 (ENU 中指向下)
     *
     *   副向量用陀螺仪平均方向提供一个水平参考:
     *   r2 = normalize(gyro_avg × r1)  → 正交于重力
     *   b2 = (0, 1, 0)                 → 世界 Y 轴 (East)
     */

    /* r1: 归一化后的加速度计平均值 (体坐标系中的重力方向) */
    float a_norm = sqrtf(accel_avg[0]*accel_avg[0] + accel_avg[1]*accel_avg[1] + accel_avg[2]*accel_avg[2]);
    if (a_norm < 1.0f || a_norm > 20.0f) {
        log_warning("[Odom] TRIAD: accel norm=%.2f out of range [1,20], cannot init", a_norm);
        return false;
    }
    float r1[3] = {accel_avg[0] / a_norm, accel_avg[1] / a_norm, accel_avg[2] / a_norm};

    /* b1: world Z-down direction (= gravity direction in ENU) */
    float b1[3] = {0.0f, 0.0f, -1.0f};

    /* r2: cross product of arbitrary reference with r1 (approximate East in body frame) */
    /* Use a reference that isn't parallel to r1: (1, 0, 0) unless r1 is close to X-axis */
    float ref[3] = {1.0f, 0.0f, 0.0f};
    float dot = r1[0]*ref[0] + r1[1]*ref[1] + r1[2]*ref[2];
    if (fabsf(dot) > 0.99f) {
        /* r1 is nearly parallel to X-axis, use Y-axis instead */
        ref[0] = 0.0f; ref[1] = 1.0f; ref[2] = 0.0f;
    }

    float r2[3];
    r2[0] = ref[1]*r1[2] - ref[2]*r1[1];
    r2[1] = ref[2]*r1[0] - ref[0]*r1[2];
    r2[2] = ref[0]*r1[1] - ref[1]*r1[0];
    float r2_norm = sqrtf(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
    if (r2_norm < 1e-6f) { r2[0] = 0.0f; r2[1] = 1.0f; r2[2] = 0.0f; }
    else { r2[0] /= r2_norm; r2[1] /= r2_norm; r2[2] /= r2_norm; }
    float b2[3] = {0.0f, 1.0f, 0.0f};

    /* r3 = r1 × r2, b3 = b1 × b2 */
    float r3[3], b3[3];
    r3[0] = r1[1]*r2[2] - r1[2]*r2[1];
    r3[1] = r1[2]*r2[0] - r1[0]*r2[2];
    r3[2] = r1[0]*r2[1] - r1[1]*r2[0];

    b3[0] = b1[1]*b2[2] - b1[2]*b2[1];
    b3[1] = b1[2]*b2[0] - b1[0]*b2[2];
    b3[2] = b1[0]*b2[1] - b1[1]*b2[0];

    /* Construct DCM: M_body = [r1, r2, r3], M_world = [b1, b2, b3]
     * R_W_B = M_body * M_world^T */
    float M_body[9] = {r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]};
    float M_world_T[9] = {b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b3[0], b3[1], b3[2]};

    float R[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) sum += M_body[i*3 + k] * M_world_T[k*3 + j];
            R[i*3 + j] = sum;
        }
    }

    k1_quat_from_rotation_matrix(R, odom->quat);
    k1_quat_normalize(odom->quat);

    log_info("[Odom] TRIAD init: q=(%.4f,%.4f,%.4f,%.4f) from %d samples",
             odom->quat[0], odom->quat[1], odom->quat[2], odom->quat[3],
             odom->init_sample_count);
    return true;
}

/* ═══════════════════════════════════════════════════════════
 *  生命周期
 * ═══════════════════════════════════════════════════════════ */

K1Odometry* k1_odometry_create(void) {
    K1Odometry* odom = (K1Odometry*)calloc(1, sizeof(K1Odometry));
    if (!odom) return NULL;

    /* 初始姿态: 单位四元数 (无旋转) */
    k1_quat_identity(odom->quat);

    /* 默认参数 */
    odom->zupt_accel_thresh = K1_ODOM_ZUPT_ACCEL_THRESH;
    odom->zupt_gyro_thresh  = K1_ODOM_ZUPT_GYRO_THRESH;
    odom->zupt_sigma_a      = K1_ODOM_ZUPT_SIGMA_A;
    odom->zupt_sigma_w      = K1_ODOM_ZUPT_SIGMA_W;
    odom->zupt_glrt_thresh  = K1_ODOM_ZUPT_GLRT_THRESH;
    odom->init_duration_samples = K1_ODOM_INIT_SAMPLES;
    odom->sigma_accel       = 0.01f;
    odom->sigma_gyro        = 0.001f;

    /* 非初始化状态: 开始收集静止样本 */
    odom->init_collecting = true;
    odom->initialized = false;
    odom->init_sample_count = 0;
    memset(odom->init_accel_sum, 0, sizeof(odom->init_accel_sum));

    /* ZUPT 状态 */
    odom->zupt_detected = false;
    odom->last_zupt_valid = false;
    odom->zupt_history_idx = 0;
    memset(odom->accel_history, 0, sizeof(odom->accel_history));
    memset(odom->gyro_history, 0, sizeof(odom->gyro_history));

    /* EKF 协方差初始化 (中等不确定性) */
    memset(odom->cov, 0, sizeof(odom->cov));
    for (int i = 0; i < 9; i++) {
        odom->cov[i*9 + i] = 1.0f;  /* position: 1m² */
    }
    /* velocity: smaller initial uncertainty */
    odom->cov[3*9 + 3] = 0.01f;
    odom->cov[4*9 + 4] = 0.01f;
    odom->cov[5*9 + 5] = 0.01f;
    /* attitude: moderate initial uncertainty */
    odom->cov[6*9 + 6] = 0.001f;
    odom->cov[7*9 + 7] = 0.001f;
    odom->cov[8*9 + 8] = 0.001f;

    pthread_mutex_init(&odom->state_mutex, NULL);
    odom->csv_out = NULL;
    odom->last_time = 0.0;

    return odom;
}

void k1_odometry_destroy(K1Odometry* odom) {
    if (!odom) return;
    k1_odometry_stop_recording(odom);
    pthread_mutex_destroy(&odom->state_mutex);
    free(odom);
}

void k1_odometry_set_params(K1Odometry* odom,
                             float zupt_accel_thresh, float zupt_gyro_thresh,
                             float sigma_a, float sigma_w, float glrt_thresh,
                             int init_samples) {
    if (!odom) return;
    odom->zupt_accel_thresh = zupt_accel_thresh;
    odom->zupt_gyro_thresh  = zupt_gyro_thresh;
    odom->zupt_sigma_a      = sigma_a;
    odom->zupt_sigma_w      = sigma_w;
    odom->zupt_glrt_thresh  = glrt_thresh;
    if (init_samples > 0 && init_samples <= 500) {
        odom->init_duration_samples = init_samples;
    }
}

void k1_odometry_set_biases(K1Odometry* odom,
                             const float gyro_bias[3],
                             const float accel_bias[3]) {
    if (!odom) return;
    pthread_mutex_lock(&odom->state_mutex);
    if (gyro_bias) {
        memcpy(odom->gyro_bias, gyro_bias, 3 * sizeof(float));
    }
    if (accel_bias) {
        memcpy(odom->accel_bias, accel_bias, 3 * sizeof(float));
    }
    pthread_mutex_unlock(&odom->state_mutex);
    log_info("[Odom] Biases updated: gyro=(%.4f,%.4f,%.4f) rad/s, accel=(%.4f,%.4f,%.4f) m/s²",
             odom->gyro_bias[0], odom->gyro_bias[1], odom->gyro_bias[2],
             odom->accel_bias[0], odom->accel_bias[1], odom->accel_bias[2]);
}

void k1_odometry_start_recording(K1Odometry* odom, const char* dir, const char* session_id) {
    if (!odom || !dir || !session_id) return;

    char odom_dir[MAX_PATH_LEN * 2];
    snprintf(odom_dir, sizeof(odom_dir), "%s/%s/odometry", dir, session_id);
    mkdir(odom_dir, 0755);

    char path[MAX_PATH_LEN * 3];
    snprintf(path, sizeof(path), "%s/k1_path.csv", odom_dir);
    odom->csv_out = fopen(path, "w");
    if (odom->csv_out) {
        fprintf(odom->csv_out, "timestamp_s,pos_x_m,pos_y_m,pos_z_m,"
                "vel_x_mps,vel_y_mps,vel_z_mps,"
                "qw,qx,qy,qz,zupt\n");
        log_info("[Odom] Recording K1 path: %s", path);
    }
}

void k1_odometry_stop_recording(K1Odometry* odom) {
    if (!odom) return;
    if (odom->csv_out) {
        fclose(odom->csv_out);
        odom->csv_out = NULL;
        log_info("[Odom] K1 trajectory recording closed");
    }
}

bool k1_odometry_is_initialized(const K1Odometry* odom) {
    return odom && odom->initialized;
}

bool k1_odometry_is_collecting(const K1Odometry* odom) {
    return odom && odom->init_collecting && !odom->initialized;
}

/* ═══════════════════════════════════════════════════════════
 *  核心递推 (100Hz)
 * ═══════════════════════════════════════════════════════════ */

void k1_odometry_update(K1Odometry* odom, const IMUData* imu) {
    if (!odom || !imu) return;

    pthread_mutex_lock(&odom->state_mutex);

    /* ── Compute dt ── */
    double now = imu->timestamp;
    float dt;
    if (odom->last_time > 0.0) {
        dt = (float)(now - odom->last_time);
        if (dt < 0.0005f) dt = 0.0005f;   /* min 0.5ms (~2000Hz max) */
        if (dt > 0.1f)    dt = 0.01f;      /* max 100ms — long gaps → assume 100Hz */
    } else {
        dt = 0.01f;  /* first sample: assume 100Hz */
    }
    odom->last_time = now;

    /* ── Step 0a: 如果仍在初始化阶段, 收集静止样本 ── */
    if (!odom->initialized && odom->init_collecting) {
        odom->init_accel_sum[0] += imu->accel_x;
        odom->init_accel_sum[1] += imu->accel_y;
        odom->init_accel_sum[2] += imu->accel_z;
        odom->init_sample_count++;

        if (odom->init_sample_count >= odom->init_duration_samples) {
            /* 计算平均加速度 */
            float inv = 1.0f / (float)odom->init_sample_count;
            float accel_avg[3] = {
                odom->init_accel_sum[0] * inv,
                odom->init_accel_sum[1] * inv,
                odom->init_accel_sum[2] * inv
            };
            if (triad_gravity_init(odom, accel_avg)) {
                odom->initialized = true;
                odom->init_collecting = false;
                /* Reset position/velocity to zero at initialization */
                memset(odom->pos, 0, sizeof(odom->pos));
                memset(odom->vel, 0, sizeof(odom->vel));
            } else {
                /* Retry: reset collection */
                log_warning("[Odom] TRIAD init failed, retrying in 2s...");
                odom->init_sample_count = 0;
                memset(odom->init_accel_sum, 0, sizeof(odom->init_accel_sum));
            }
        }
        /* Don't run INS until initialized */
        pthread_mutex_unlock(&odom->state_mutex);
        return;
    }

    if (!odom->initialized) {
        pthread_mutex_unlock(&odom->state_mutex);
        return;
    }

    /* Diagnostic: count INS updates after init (per-instance, not static) */
    odom->ins_update_count++;
    if (odom->ins_update_count <= 3 || odom->ins_update_count % 500 == 0) {
        log_info("[Odom] INS update #%d: ax=%.3f ay=%.3f az=%.3f gx=%.4f gy=%.4f gz=%.4f dt=%.4f",
                 odom->ins_update_count, imu->accel_x, imu->accel_y, imu->accel_z,
                 imu->gyro_x, imu->gyro_y, imu->gyro_z, dt);
    }

    /* ── Step 1: 去除零偏 ── */
    float ax_c = imu->accel_x - odom->accel_bias[0];
    float ay_c = imu->accel_y - odom->accel_bias[1];
    float az_c = imu->accel_z - odom->accel_bias[2];
    float gx_c = imu->gyro_x  - odom->gyro_bias[0];
    float gy_c = imu->gyro_y  - odom->gyro_bias[1];
    float gz_c = imu->gyro_z  - odom->gyro_bias[2];

    /* ── Step 0b: Update ZUPT sliding window (bias-corrected values) ── */
    int widx = odom->zupt_history_idx % K1_ODOM_ZUPT_WINDOW;
    odom->accel_history[widx][0] = ax_c;
    odom->accel_history[widx][1] = ay_c;
    odom->accel_history[widx][2] = az_c;
    odom->gyro_history[widx][0]  = gx_c;
    odom->gyro_history[widx][1]  = gy_c;
    odom->gyro_history[widx][2]  = gz_c;
    odom->zupt_history_idx++;

    /* ── Step 2: 姿态更新 (四元数一阶积分) ──
     * q̇ = 0.5 × q ⊗ ω  where ω = (0, gx, gy, gz) */
    float qw = odom->quat[0], qx = odom->quat[1], qy = odom->quat[2], qz = odom->quat[3];
    float dqw = 0.5f * (-qx*gx_c - qy*gy_c - qz*gz_c);
    float dqx = 0.5f * ( qw*gx_c + qy*gz_c - qz*gy_c);
    float dqy = 0.5f * ( qw*gy_c - qx*gz_c + qz*gx_c);
    float dqz = 0.5f * ( qw*gz_c + qx*gy_c - qy*gx_c);

    odom->quat[0] += dqw * dt;
    odom->quat[1] += dqx * dt;
    odom->quat[2] += dqy * dt;
    odom->quat[3] += dqz * dt;
    k1_quat_normalize(odom->quat);

    /* ── Step 3: 加速度转换到世界坐标系 + 重力补偿 ── */
    float a_body[3] = {ax_c, ay_c, az_c};
    float a_world[3];
    k1_quat_rotate(odom->quat, a_body, a_world);

    /* 减去重力 (ENU: g 指向 -Z 方向, 所以加速度计读数中减去重力分量是 +Z) */
    a_world[2] -= GRAVITY_MAG;

    /* ── Step 4: 位置积分 (使用旧速度, 二阶泰勒展开) ──
     * 必须在速度更新之前: p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt² */
    odom->pos[0] += odom->vel[0] * dt + 0.5f * a_world[0] * dt * dt;
    odom->pos[1] += odom->vel[1] * dt + 0.5f * a_world[1] * dt * dt;
    odom->pos[2] += odom->vel[2] * dt + 0.5f * a_world[2] * dt * dt;

    /* ── Step 5: 速度积分 (欧拉法, 必须在位置之后) ── */
    odom->vel[0] += a_world[0] * dt;
    odom->vel[1] += a_world[1] * dt;
    odom->vel[2] += a_world[2] * dt;

    /* ── Safety clamp: velocity beyond 50 m/s is physically impossible
     * for pedestrian INS.  Uncalibrated gyro drift can cause gravity
     * leakage into horizontal acceleration → quadratic position runaway.
     * Clamping velocity caps the damage while preserving valid motion. */
    for (int i = 0; i < 3; i++) {
        if (odom->vel[i] >  K1_ODOM_VEL_CLAMP_MPS) odom->vel[i] =  K1_ODOM_VEL_CLAMP_MPS;
        if (odom->vel[i] < -K1_ODOM_VEL_CLAMP_MPS) odom->vel[i] = -K1_ODOM_VEL_CLAMP_MPS;
    }

    /* ── Step 6a: 简单阈值 ZUPT 检测 ── */
    bool is_static_simple = simple_zupt_check(odom, ax_c, ay_c, az_c, gx_c, gy_c, gz_c);

    /* ── Step 6b: GLRT ZUPT 检测 (有足够历史数据时) ── */
    bool is_static_glrt = false;
    if (odom->zupt_history_idx >= K1_ODOM_ZUPT_WINDOW) {
        is_static_glrt = k1_odom_glrt_zupt_detect(
            odom->accel_history, odom->gyro_history, K1_ODOM_ZUPT_WINDOW,
            odom->zupt_sigma_a, odom->zupt_sigma_w, odom->zupt_glrt_thresh);
    }

    /* Combine: both must agree for ZUPT to fire */
    bool is_static = is_static_simple && is_static_glrt;
    odom->zupt_detected = is_static;

    /* ── Step 7: ZUPT EKF 校正 ── */
    if (is_static && odom->last_zupt_valid) {
        zupt_ekf_update(odom);
    }
    odom->last_zupt_valid = is_static;

    /* ── Step 8: EKF 预测 (协方差传播) ──
     * P = F·P·F' + G·Σ·G'·dt
     *
     * 状态 (9): [δp_x, δp_y, δp_z, δv_x, δv_y, δv_z, δφ_x, δφ_y, δφ_z]
     *
     * F = | 0_{3×3}  I_{3×3}  0_{3×3}  |  位置误差 ← 速度误差
     *     | 0_{3×3}  0_{3×3}  -a_world× |  速度误差 ← 姿态误差 × 加速度
     *     | 0_{3×3}  0_{3×3}  0_{3×3}  |  姿态误差 ← 陀螺噪声
     *
     * G = | 0_{3×3}  0_{3×3} |    Σ = diag(σ_accel², σ_gyro²)
     *     | I_{3×3}  0_{3×3} |
     *     | 0_{3×3}  I_{3×3} |
     */
    {
        int n = K1_ODOM_EKF_STATE_DIM;

        /* Build F (9×9) = I + A·dt, where A = state dynamics matrix */
        float F[81];
        memset(F, 0, sizeof(F));
        for (int i = 0; i < n; i++) F[i*9 + i] = 1.0f;

        /* Position ← Velocity: F[0..2][3..5] = I·dt */
        F[0*9 + 3] = dt;  F[1*9 + 4] = dt;  F[2*9 + 5] = dt;

        /* Velocity ← Attitude error × world acceleration: F[3..5][6..8] = -a_world× · dt */
        F[3*9 + 7] =  a_world[2] * dt;   /* -a×_z → vx ← φy */
        F[3*9 + 8] = -a_world[1] * dt;   /* -a×_y → vx ← φz */
        F[4*9 + 6] = -a_world[2] * dt;   /* -a×_z → vy ← φx */
        F[4*9 + 8] =  a_world[0] * dt;   /* -a×_x → vy ← φz */
        F[5*9 + 6] =  a_world[1] * dt;   /* -a×_y → vz ← φx */
        F[5*9 + 7] = -a_world[0] * dt;   /* -a×_x → vz ← φy */

        /* F·P (temp storage in F_times_P) */
        float FP[81];
        memset(FP, 0, sizeof(FP));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) sum += F[i*9 + k] * odom->cov[k*9 + j];
                FP[i*9 + j] = sum;
            }
        }

        /* P_new = F·P·F' */
        float P_pred[81];
        memset(P_pred, 0, sizeof(P_pred));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) sum += FP[i*9 + k] * F[j*9 + k];  /* F'[k,j] = F[j,k] */
                P_pred[i*9 + j] = sum;
            }
        }

        /* Add process noise Q = G·Σ·G'·dt
         *   Q_pos   = 0  (position has no direct process noise)
         *   Q_vel   = σ_accel² · dt · I
         *   Q_att   = σ_gyro²  · dt · I
         */
        float sigma_a2 = odom->sigma_accel * odom->sigma_accel;
        float sigma_g2 = odom->sigma_gyro  * odom->sigma_gyro;

        for (int i = 3; i < 6; i++) P_pred[i*9 + i] += sigma_a2 * dt;
        for (int i = 6; i < 9; i++) P_pred[i*9 + i] += sigma_g2 * dt;

        memcpy(odom->cov, P_pred, sizeof(P_pred));
    }

    odom->frame_count++;

    /* ── CSV 输出 ── */
    if (odom->csv_out) {
        fprintf(odom->csv_out, "%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%d\n",
                now,
                odom->pos[0], odom->pos[1], odom->pos[2],
                odom->vel[0], odom->vel[1], odom->vel[2],
                odom->quat[0], odom->quat[1], odom->quat[2], odom->quat[3],
                is_static ? 1 : 0);
    }

    /* ── 调试日志 (每 500 帧) ── */
    if (odom->frame_count % 500 == 0) {
        /* Warn if ZUPT never fired over the last 1500+ frames (~15s).
         * Indicates the GLRT threshold may still be too high for this IMU
         * or the sensor noise is larger than configured sigma values. */
        if (is_static) odom->zupt_streak = 0;
        else odom->zupt_streak++;
        if (odom->zupt_streak > 1500) {
            log_warning("[Odom] ZUPT inactive for %d frames (~%.0fs) — "
                        "check GLRT threshold (%.0f) vs sensor noise (sigma_a=%.3f sigma_w=%.3f)",
                        odom->zupt_streak, odom->zupt_streak * 0.01,
                        (double)odom->zupt_glrt_thresh,
                        (double)odom->zupt_sigma_a, (double)odom->zupt_sigma_w);
            odom->zupt_streak = 0;  /* reset to avoid log spam */
        }
        log_info("[Odom] frame=%llu pos=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) zupt=%d",
                 (unsigned long long)odom->frame_count,
                 odom->pos[0], odom->pos[1], odom->pos[2],
                 odom->vel[0], odom->vel[1], odom->vel[2],
                 is_static ? 1 : 0);
    }

    pthread_mutex_unlock(&odom->state_mutex);
}

/* ═══════════════════════════════════════════════════════════
 *  读取接口
 * ═══════════════════════════════════════════════════════════ */

void k1_odometry_get_pose(const K1Odometry* odom, float pos[3], float quat[4]) {
    if (!odom) {
        if (pos)  { pos[0] = 0; pos[1] = 0; pos[2] = 0; }
        if (quat) { k1_quat_identity(quat); }
        return;
    }
    K1Odometry* m = (K1Odometry*)odom;  /* const-cast for internal locking */
    pthread_mutex_lock(&m->state_mutex);
    if (pos)  memcpy(pos, odom->pos, 3 * sizeof(float));
    if (quat) memcpy(quat, odom->quat, 4 * sizeof(float));
    pthread_mutex_unlock(&m->state_mutex);
}

void k1_odometry_get_velocity(const K1Odometry* odom, float vel[3]) {
    if (!odom || !vel) return;
    K1Odometry* m = (K1Odometry*)odom;
    pthread_mutex_lock(&m->state_mutex);
    memcpy(vel, odom->vel, 3 * sizeof(float));
    pthread_mutex_unlock(&m->state_mutex);
}

void k1_odometry_get_transform(const K1Odometry* odom, float R[9], float t[3]) {
    if (!odom) {
        if (R) {
            memset(R, 0, 9 * sizeof(float));
            R[0] = R[4] = R[8] = 1.0f;
        }
        if (t) { t[0] = 0; t[1] = 0; t[2] = 0; }
        return;
    }
    K1Odometry* m = (K1Odometry*)odom;
    pthread_mutex_lock(&m->state_mutex);
    if (R) k1_quat_to_rotation_matrix(odom->quat, R);
    if (t) memcpy(t, odom->pos, 3 * sizeof(float));
    pthread_mutex_unlock(&m->state_mutex);
}
