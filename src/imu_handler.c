/*
 * IMUHandler — IMU 数据处理 + Madgwick AHRS + Wahba 双 IMU 对齐
 *
 * [Madgwick 2011] "An efficient orientation filter..."
 * [Wahba 1965]  "A Least Squares Estimate of Spacecraft Attitude"
 */

#include "imu_handler.h"
#include "k1_imu.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#define IMU_MAX_ACCEL_G  16.0f
#define IMU_MAX_GYRO_DPS 2000.0f

/* ═══════════════════════════════════════════════════════════
 *  Madgwick AHRS 滤波器 [Madgwick 2011]
 * ═══════════════════════════════════════════════════════════ */

void madgwick_init(MadgwickFilter* mf, float beta, float sample_freq) {
    memset(mf, 0, sizeof(*mf));
    mf->qw = 1.0f;                    /* 初始朝向: 无旋转 */
    mf->beta = beta;
    mf->sample_freq = sample_freq;
    mf->initialized = true;
}

void madgwick_update(MadgwickFilter* mf, float gx, float gy, float gz,
                     float ax, float ay, float az, float dt) {
    if (!mf->initialized || dt <= 0.0f) return;

    float qw = mf->qw, qx = mf->qx, qy = mf->qy, qz = mf->qz;
    float recip_norm;

    /* 归一化加速度计 */
    float a_norm = sqrtf(ax*ax + ay*ay + az*az);
    /* ADXL345 at ±2g: max output 2*9.81 = 19.6 m/s².  Thresholds:
     *   < 0.5g (free-fall / microgravity) or > 2.0g (high acceleration)
     *   trigger gyro-only fallback (accelerometer unreliable as gravity). */
    if (a_norm < 0.5f * 9.81f || a_norm > 2.0f * 9.81f) {
        /* 高加速度状态: 仅陀螺仪积分, 跳过加速度校正 */
        float qdot_w = 0.5f * (-qx*gx - qy*gy - qz*gz);
        float qdot_x = 0.5f * ( qw*gx + qy*gz - qz*gy);
        float qdot_y = 0.5f * ( qw*gy - qx*gz + qz*gx);
        float qdot_z = 0.5f * ( qw*gz + qx*gy - qy*gx);
        qw += qdot_w * dt; qx += qdot_x * dt;
        qy += qdot_y * dt; qz += qdot_z * dt;
        recip_norm = 1.0f / sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        mf->qw = qw * recip_norm; mf->qx = qx * recip_norm;
        mf->qy = qy * recip_norm; mf->qz = qz * recip_norm;
        return;
    }
    recip_norm = 1.0f / a_norm;
    ax *= recip_norm; ay *= recip_norm; az *= recip_norm;

    /* 目标函数 f_g(q, a) = 2*[qx*qz - qw*qy - ax/2, qw*qx + qy*qz - ay/2, qw²-qx²-qy²+qz² - az] */
    float _2qw = 2.0f * qw, _2qx = 2.0f * qx, _2qy = 2.0f * qy, _2qz = 2.0f * qz;
    float _4qx = 4.0f * qx, _4qy = 4.0f * qy;

    /* f_g */
    float f0 = _2qx*qz - _2qw*qy - ax;
    float f1 = _2qw*qx + _2qy*qz - ay;
    float f2 = 1.0f - _2qx*_2qx*0.5f - _2qy*_2qy*0.5f - az;

    /* J_g^T · f (雅可比转置乘目标函数, 4×3 · 3×1 = 4×1) */
    float j0 = -_2qy*f0 + _2qx*f1;
    float j1 =  _2qz*f0 + _2qw*f1 - _4qx*f2;
    float j2 = -_2qw*f0 + _2qz*f1 - _4qy*f2;
    float j3 =  _2qx*f0 + _2qy*f1;

    /* 梯度归一化 */
    float grad_norm = sqrtf(j0*j0 + j1*j1 + j2*j2 + j3*j3);
    if (grad_norm > 0.0f) {
        /* BUGFIX: Remove extra dt — Madgwick formula (2010) Eq.(21):
         * q_new = q + (½ q⊗ω − β·∇f/‖∇f‖) · Δt
         * The gradient correction β·∇f/‖∇f‖ is multiplied by Δt ONCE
         * at lines 89-92.  The old code multiplied by dt², making the
         * correction 100× too weak at 100Hz (effective beta=0.0008). */
        float step = mf->beta / grad_norm;
        j0 *= step; j1 *= step; j2 *= step; j3 *= step;
    }

    /* 陀螺仪四元数导数: q̇ = 0.5 * q ⊗ ω */
    float qdot_w = 0.5f * (-qx*gx - qy*gy - qz*gz);
    float qdot_x = 0.5f * ( qw*gx + qy*gz - qz*gy);
    float qdot_y = 0.5f * ( qw*gy - qx*gz + qz*gx);
    float qdot_z = 0.5f * ( qw*gz + qx*gy - qy*gx);

    /* 融合: q_new = q + (q̇ - ∇f_normalized) · dt */
    qw += (qdot_w - j0) * dt;
    qx += (qdot_x - j1) * dt;
    qy += (qdot_y - j2) * dt;
    qz += (qdot_z - j3) * dt;

    /* 归一化 */
    recip_norm = 1.0f / sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    mf->qw = qw * recip_norm;
    mf->qx = qx * recip_norm;
    mf->qy = qy * recip_norm;
    mf->qz = qz * recip_norm;
}

void madgwick_get_quat(const MadgwickFilter* mf, float* qw, float* qx, float* qy, float* qz) {
    *qw = mf->qw; *qx = mf->qx; *qy = mf->qy; *qz = mf->qz;
}

void madgwick_get_euler(const MadgwickFilter* mf, float* pitch, float* roll, float* yaw) {
    float qw = mf->qw, qx = mf->qx, qy = mf->qy, qz = mf->qz;
    *pitch = asinf(-2.0f * (qx*qz - qw*qy));
    *roll  = atan2f(2.0f * (qw*qx + qy*qz), qw*qw - qx*qx - qy*qy + qz*qz);
    *yaw   = atan2f(2.0f * (qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz);
}

/* ═══════════════════════════════════════════════════════════
 *  Wahba 重力向量对齐
 * ═══════════════════════════════════════════════════════════ */

static void wahba_align(const float v_k1[3], const float v_cam[3],
                        float* qw, float* qx, float* qy, float* qz) {
    /* 计算 v_k1 和 v_cam 之间的旋转 (轴角法) */
    float cx = v_k1[1]*v_cam[2] - v_k1[2]*v_cam[1];
    float cy = v_k1[2]*v_cam[0] - v_k1[0]*v_cam[2];
    float cz = v_k1[0]*v_cam[1] - v_k1[1]*v_cam[0];
    float sin_theta = sqrtf(cx*cx + cy*cy + cz*cz);
    float cos_theta = v_k1[0]*v_cam[0] + v_k1[1]*v_cam[1] + v_k1[2]*v_cam[2];

    if (sin_theta < 1e-6f) {
        /* 向量已对齐 (或反向) */
        if (cos_theta > 0.0f) {
            *qw = 1.0f; *qx = 0.0f; *qy = 0.0f; *qz = 0.0f;
        } else {
            *qw = 0.0f; *qx = 0.0f; *qy = 0.0f; *qz = 1.0f;  /* 180° */
        }
        return;
    }

    float theta = atan2f(sin_theta, cos_theta);
    float half = theta * 0.5f;
    float s = sinf(half) / sin_theta;
    *qw = cosf(half);
    *qx = cx * s;
    *qy = cy * s;
    *qz = cz * s;
}

static bool try_align_frames(IMUHandler* h) {
    FrameAlignmentCtx* ctx = &h->align_ctx;
    int k1_n = ctx->samples_collected;
    int cam_n = ctx->cam_idx;
    int n = (k1_n < cam_n) ? k1_n : cam_n;  /* 取较小值对齐 */
    if (!ctx->done && n >= ctx->window_size) {
        /* 计算均值重力向量 (使用相同数量的样本) */
        float k1_sum[3] = {0}, cam_sum[3] = {0};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                k1_sum[j]  += ctx->k1_accels[j][i];
                cam_sum[j] += ctx->cam_accels[j][i];
            }
        }
        float inv = 1.0f / (float)n;
        float v_k1[3]  = { k1_sum[0]*inv,  k1_sum[1]*inv,  k1_sum[2]*inv  };
        float v_cam[3] = { cam_sum[0]*inv, cam_sum[1]*inv, cam_sum[2]*inv };

        /* 归一化 */
        float nk = sqrtf(v_k1[0]*v_k1[0] + v_k1[1]*v_k1[1] + v_k1[2]*v_k1[2]);
        float nc = sqrtf(v_cam[0]*v_cam[0] + v_cam[1]*v_cam[1] + v_cam[2]*v_cam[2]);
        if (nk < 1.0f || nc < 1.0f) return false;  /* 无效数据 */
        v_k1[0] /= nk; v_k1[1] /= nk; v_k1[2] /= nk;
        v_cam[0] /= nc; v_cam[1] /= nc; v_cam[2] /= nc;

        wahba_align(v_k1, v_cam, &h->align_qw, &h->align_qx, &h->align_qy, &h->align_qz);

        h->alignment_done = true;
        ctx->done = true;

        log_info("[IMU-Align] Frames aligned: q=(%.4f, %.4f, %.4f, %.4f) | "
                 "K1 grav=(%.3f,%.3f,%.3f) Cam grav=(%.3f,%.3f,%.3f)",
                 h->align_qw, h->align_qx, h->align_qy, h->align_qz,
                 v_k1[0], v_k1[1], v_k1[2], v_cam[0], v_cam[1], v_cam[2]);
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════
 *  IMUHandler 核心
 * ═══════════════════════════════════════════════════════════ */

/*
 * CRITICAL: With -ffast-math, isnan()/isinf() are UB on Clang — the compiler
 * optimizes them to always return false.  Use IEEE 754 bit-level inspection
 * via memcpy to uint32_t, which the compiler cannot constant-fold away.
 */
static inline bool is_valid_float(float v) {
    uint32_t raw;
    memcpy(&raw, &v, sizeof(raw));
    /* Check for NaN: exponent all-1s, mantissa non-zero */
    if ((raw & 0x7F800000u) == 0x7F800000u && (raw & 0x007FFFFFu) != 0) return false;
    /* Check for Inf: exponent all-1s, mantissa zero */
    if (raw == 0x7F800000u || raw == 0xFF800000u) return false;
    /* Check for subnormals (exponent all-0s, mantissa non-zero) */
    if ((raw & 0x7F800000u) == 0 && (raw & 0x007FFFFFu) != 0) return false;
    return true;
}

IMUHandler* imu_handler_create(int window_size, float gyro_noise_std, float accel_noise_std) {
    IMUHandler* h = (IMUHandler*)calloc(1, sizeof(IMUHandler));
    if (!h) return NULL;

    h->window_size = UTILS_CLAMP(window_size, 1, IMU_MAX_WINDOW_SIZE);
    h->gyro_noise_std = gyro_noise_std;
    h->accel_noise_std = accel_noise_std;
    pthread_mutex_init(&h->pose_mutex, NULL);
    pthread_mutex_init(&h->dual_mutex, NULL);
    pthread_mutex_init(&h->align_mutex, NULL);  /* protects align_ctx */

    /* 初始化两个 Madgwick 滤波器 */
    madgwick_init(&h->k1_filter,  0.08f, 100.0f);
    madgwick_init(&h->cam_filter, 0.08f, 100.0f);

    /* 初始化对齐上下文 */
    h->align_ctx.window_size = 200;
    h->align_ctx.samples_collected = 0;
    h->align_ctx.done = false;
    h->alignment_done = false;
    h->k1_imu = NULL;

    /* 初始化 K1 里程计 (INS + ZUPT) */
    h->odometry = k1_odometry_create();

    return h;
}

void imu_handler_destroy(IMUHandler* h) {
    if (!h) return;
    imu_handler_stop_recording(h);
    if (h->odometry) {
        k1_odometry_destroy(h->odometry);
        h->odometry = NULL;
    }
    if (h->k1_imu) {
        k1_imu_destroy((K1Imu*)h->k1_imu);
        h->k1_imu = NULL;
    }
    pthread_mutex_destroy(&h->pose_mutex);
    pthread_mutex_destroy(&h->dual_mutex);
    pthread_mutex_destroy(&h->align_mutex);
    free(h);
}

/* ═══════════════════════════════════════════════════════════
 *  原有 API (保持兼容)
 * ═══════════════════════════════════════════════════════════ */

bool imu_handler_validate(const IMUHandler* h, const float accel[3], const float gyro[3]) {
    (void)h;
    if (!accel || !gyro) return false;
    for (int i = 0; i < 3; i++) {
        if (!is_valid_float(accel[i]) || fabsf(accel[i]) > IMU_MAX_ACCEL_G) return false;
        if (!is_valid_float(gyro[i]) || fabsf(gyro[i]) > IMU_MAX_GYRO_DPS) return false;
    }
    return true;
}

IMUData imu_handler_parse(IMUHandler* h, const float accel[3], const float gyro[3], double ts) {
    IMUData d = {0};
    if (!imu_handler_validate(h, accel, gyro)) return d;
    d.timestamp = ts;
    d.accel_x = accel[0]; d.accel_y = accel[1]; d.accel_z = accel[2];
    d.gyro_x = gyro[0];   d.gyro_y = gyro[1];   d.gyro_z = gyro[2];
    return d;
}

IMUData imu_handler_smooth(IMUHandler* h, const IMUData* data) {
    if (!h || !data) return (IMUData){0};
    if (h->buffer_count < h->window_size) {
        h->buffer[h->buffer_count++] = *data;
    } else {
        memmove(&h->buffer[0], &h->buffer[1], (h->window_size - 1) * sizeof(IMUData));
        h->buffer[h->window_size - 1] = *data;
    }
    if (h->buffer_count < 2) return *data;
    IMUData s = {0};
    s.timestamp = data->timestamp;
    for (int i = 0; i < h->buffer_count; i++) {
        s.accel_x += h->buffer[i].accel_x; s.accel_y += h->buffer[i].accel_y;
        s.accel_z += h->buffer[i].accel_z; s.gyro_x  += h->buffer[i].gyro_x;
        s.gyro_y  += h->buffer[i].gyro_y;  s.gyro_z  += h->buffer[i].gyro_z;
    }
    float inv = 1.0f / h->buffer_count;
    s.accel_x *= inv; s.accel_y *= inv; s.accel_z *= inv;
    s.gyro_x  *= inv; s.gyro_y  *= inv; s.gyro_z  *= inv;
    return s;
}

void imu_handler_reset(IMUHandler* h) {
    if (!h) return;
    h->buffer_count = 0;
    h->is_calibrated = false;
}

void imu_handler_set_external_pose(IMUHandler* h, float qw, float qx, float qy, float qz,
                                    float pitch, float roll, float yaw,
                                    float altitude, float temp, uint32_t ts) {
    if (!h) return;
    pthread_mutex_lock(&h->pose_mutex);
    h->latest_pose.qw = qw; h->latest_pose.qx = qx;
    h->latest_pose.qy = qy; h->latest_pose.qz = qz;
    h->latest_pose.pitch = pitch; h->latest_pose.roll = roll; h->latest_pose.yaw = yaw;
    h->latest_pose.altitude_m = altitude; h->latest_pose.temperature_c = temp;
    h->latest_pose.timestamp_ms = ts;
    h->latest_pose.is_valid = true;
    h->has_external_pose = true;
    pthread_mutex_unlock(&h->pose_mutex);
}

bool imu_handler_get_latest_pose(const IMUHandler* h, IMUExternalPose* out) {
    if (!h || !out) return false;
    pthread_mutex_lock(&((IMUHandler*)h)->pose_mutex);
    if (!h->has_external_pose || !h->latest_pose.is_valid) {
        pthread_mutex_unlock(&((IMUHandler*)h)->pose_mutex);
        return false;
    }
    *out = h->latest_pose;
    pthread_mutex_unlock(&((IMUHandler*)h)->pose_mutex);
    return true;
}

/* ═══════════════════════════════════════════════════════════
 *  Phase 2 API: 双 IMU 融合
 * ═══════════════════════════════════════════════════════════ */

void imu_handler_set_k1_imu(IMUHandler* h, void* k1_imu) {
    h->k1_imu = k1_imu;
}

void imu_handler_start_recording(IMUHandler* h, const char* output_dir, const char* session_id) {
    if (!h || !output_dir || !session_id) return;

    char dir[MAX_PATH_LEN * 2];
    snprintf(dir, sizeof(dir), "%s/%s/imu", output_dir, session_id);
    mkdir(dir, 0755);

    /* ── K1 local IMU CSV: only open if K1 IMU is actually available ──
     * When GY85 init fails (common on boards without the sensor),
     * h->k1_imu is NULL and no data will ever arrive.  Opening an
     * empty CSV with just a header wastes I/O and confuses the user. */
    if (h->k1_imu) {
        char path[MAX_PATH_LEN * 3];
        snprintf(path, sizeof(path), "%s/k1_local.csv", dir);
        h->k1_csv = fopen(path, "w");
        if (h->k1_csv) {
            fprintf(h->k1_csv, "timestamp_s,accel_x_mps2,accel_y_mps2,accel_z_mps2,"
                    "gyro_x_radps,gyro_y_radps,gyro_z_radps,qw,qx,qy,qz\n");
            log_info("[IMU] K1 local IMU recording: %s", path);
        }
    } else {
        log_info("[IMU] K1 local IMU unavailable — skipping CSV (no sensor)");
        h->k1_csv = NULL;
    }

    /* ESP32 remote IMU: same format */
    {
        char path[MAX_PATH_LEN * 3];
        snprintf(path, sizeof(path), "%s/esp32_remote.csv", dir);
        h->esp32_csv = fopen(path, "w");
        if (h->esp32_csv) {
            fprintf(h->esp32_csv, "timestamp_s,accel_x_mps2,accel_y_mps2,accel_z_mps2,"
                    "gyro_x_radps,gyro_y_radps,gyro_z_radps,qw,qx,qy,qz\n");
            log_info("[IMU] ESP32 remote IMU recording: %s", path);
        }
    }

    /* Start K1 odometry trajectory recording */
    if (h->odometry) {
        k1_odometry_start_recording(h->odometry, output_dir, session_id);
    }
}

void imu_handler_stop_recording(IMUHandler* h) {
    if (!h) return;
    if (h->k1_csv) { fclose(h->k1_csv); h->k1_csv = NULL; log_info("[IMU] K1 IMU recording closed"); }
    if (h->esp32_csv) { fclose(h->esp32_csv); h->esp32_csv = NULL; log_info("[IMU] ESP32 IMU recording closed"); }
    if (h->odometry) { k1_odometry_stop_recording(h->odometry); }
}

void imu_handler_feed_k1_imu(IMUHandler* h, const IMUData* data) {
    if (!h || !data) return;

    /* Compute actual dt from data timestamp (CLOCK_MONOTONIC seconds).
     * K1 IMU typically runs at ~100Hz.  Clamp to guard against clock jumps. */
    float dt;
    if (h->k1_filter.last_timestamp > 0.0) {
        dt = (float)(data->timestamp - h->k1_filter.last_timestamp);
        if (dt < 0.001f) dt = 0.001f;
        if (dt > 10.0f)  dt = 0.01f;
    } else {
        dt = 0.01f;  /* first sample: assume 100Hz */
    }
    h->k1_filter.last_timestamp = data->timestamp;

    /* Madgwick 更新 K1 滤波器 */
    madgwick_update(&h->k1_filter, data->gyro_x, data->gyro_y, data->gyro_z,
                    data->accel_x, data->accel_y, data->accel_z, dt);

    /* 对齐收集: 收集 K1 静止样本 (加锁: 与 feed_external_raw 并发) */
    pthread_mutex_lock(&h->align_mutex);
    if (!h->alignment_done && h->align_ctx.samples_collected < h->align_ctx.window_size) {
        int n = h->align_ctx.samples_collected;
        h->align_ctx.k1_accels[0][n] = data->accel_x;
        h->align_ctx.k1_accels[1][n] = data->accel_y;
        h->align_ctx.k1_accels[2][n] = data->accel_z;
        h->align_ctx.samples_collected++;
    }

    /* 检查是否双方都准备好对齐 */
    if (!h->alignment_done && h->align_ctx.samples_collected >= h->align_ctx.window_size
        && h->align_ctx.cam_idx >= h->align_ctx.window_size) {
        try_align_frames(h);
    }
    pthread_mutex_unlock(&h->align_mutex);

    /* 更新融合输出 */
    if (h->alignment_done) {
        pthread_mutex_lock(&h->dual_mutex);
        madgwick_get_quat(&h->k1_filter, &h->dual_pose.k1_qw, &h->dual_pose.k1_qx,
                          &h->dual_pose.k1_qy, &h->dual_pose.k1_qz);
        h->dual_pose.k1_valid = true;
        pthread_mutex_unlock(&h->dual_mutex);
    }

    /* ── 记录 K1 IMU 采样到 CSV ── */
    if (h->k1_csv) {
        float qw, qx, qy, qz;
        madgwick_get_quat(&h->k1_filter, &qw, &qx, &qy, &qz);
        fprintf(h->k1_csv, "%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f\n",
                data->timestamp,
                data->accel_x, data->accel_y, data->accel_z,
                data->gyro_x, data->gyro_y, data->gyro_z,
                qw, qx, qy, qz);
    }

    /* ── Phase A: K1 里程计更新 (INS 捷联解算 + ZUPT) ── */
    if (h->odometry) {
        k1_odometry_update(h->odometry, data);
    }
}

void imu_handler_feed_external_raw(IMUHandler* h, const float accel[3], const float gyro[3]) {
    if (!h || !accel || !gyro) return;

    /* Compute actual dt from wall-clock time.
     * ESP32 IMU data arrives at ~1Hz via CoAP — hardcoding dt=0.01 would
     * under-integrate gyro by 100× and cause filter divergence. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;

    float dt;
    if (h->cam_filter.last_timestamp > 0.0) {
        dt = (float)(now - h->cam_filter.last_timestamp);
        if (dt < 0.001f) dt = 0.001f;
        if (dt > 10.0f)  dt = 0.01f;  /* first sample or long pause: default */
    } else {
        dt = 0.01f;  /* first sample: assume 100Hz */
    }
    h->cam_filter.last_timestamp = now;

    /* Madgwick 更新相机滤波器 */
    madgwick_update(&h->cam_filter, gyro[0], gyro[1], gyro[2],
                    accel[0], accel[1], accel[2], dt);

    /* 对齐收集: 收集相机静止样本 (加锁: 与 feed_k1_imu 并发) */
    pthread_mutex_lock(&h->align_mutex);
    if (!h->alignment_done && h->align_ctx.cam_idx < h->align_ctx.window_size) {
        int n = h->align_ctx.cam_idx;
        h->align_ctx.cam_accels[0][n] = accel[0];
        h->align_ctx.cam_accels[1][n] = accel[1];
        h->align_ctx.cam_accels[2][n] = accel[2];
        h->align_ctx.cam_idx++;
    }

    /* 检查是否双方都准备好对齐 */
    if (!h->alignment_done && h->align_ctx.samples_collected >= h->align_ctx.window_size
        && h->align_ctx.cam_idx >= h->align_ctx.window_size) {
        try_align_frames(h);
    }
    pthread_mutex_unlock(&h->align_mutex);

    /* 更新融合输出 */
    if (h->alignment_done) {
        pthread_mutex_lock(&h->dual_mutex);
        madgwick_get_quat(&h->cam_filter, &h->dual_pose.cam_qw, &h->dual_pose.cam_qx,
                          &h->dual_pose.cam_qy, &h->dual_pose.cam_qz);
        h->dual_pose.cam_valid = true;
        h->dual_pose.align_qw = h->align_qw;
        h->dual_pose.align_qx = h->align_qx;
        h->dual_pose.align_qy = h->align_qy;
        h->dual_pose.align_qz = h->align_qz;
        h->dual_pose.align_valid = true;
        pthread_mutex_unlock(&h->dual_mutex);
    }

    /* ── 记录 ESP32 IMU 采样到 CSV ── */
    if (h->esp32_csv) {
        float qw, qx, qy, qz;
        madgwick_get_quat(&h->cam_filter, &qw, &qx, &qy, &qz);
        double ts = h->cam_filter.last_timestamp;
        fprintf(h->esp32_csv, "%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f\n",
                ts, accel[0], accel[1], accel[2],
                gyro[0], gyro[1], gyro[2],
                qw, qx, qy, qz);
    }
}

bool imu_handler_get_dual_pose(IMUHandler* h, DualImuPose* out) {
    if (!h || !out || !h->alignment_done) return false;
    pthread_mutex_lock(&h->dual_mutex);
    *out = h->dual_pose;
    bool ok = out->k1_valid || out->cam_valid;
    pthread_mutex_unlock(&h->dual_mutex);
    return ok;
}

bool imu_handler_is_alignment_done(const IMUHandler* h) {
    return h && h->alignment_done;
}

/* ═══════════════════════════════════════════════════════════
 *  Phase A: K1 里程计接口
 * ═══════════════════════════════════════════════════════════ */

K1Odometry* imu_handler_get_odometry(IMUHandler* h) {
    return h ? h->odometry : NULL;
}

void imu_handler_set_odometry_params(IMUHandler* h,
                                      float zupt_accel_thresh, float zupt_gyro_thresh,
                                      float sigma_a, float sigma_w, float glrt_thresh,
                                      int init_samples) {
    if (!h || !h->odometry) return;
    k1_odometry_set_params(h->odometry, zupt_accel_thresh, zupt_gyro_thresh,
                            sigma_a, sigma_w, glrt_thresh, init_samples);
}
