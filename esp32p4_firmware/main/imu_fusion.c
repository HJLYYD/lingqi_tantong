#include "imu_fusion.h"
#include <math.h>

#define RAD2DEG (57.295779513082320876798154814105f)
#define DEG2RAD (0.01745329251994329576923690768489f)

#include <stdint.h>

static inline float inv_sqrt(float x) {
    union { float f; int32_t i; } u = { .f = x };
    float halfx = 0.5f * x;
    u.i = 0x5f3759df - (u.i >> 1);
    x = u.f;
    x = x * (1.5f - (halfx * x * x));
    x = x * (1.5f - (halfx * x * x));
    return x;
}

void madgwick_init(madgwick_filter_t *f, float beta, float sample_freq)
{
    f->q0 = 1.0f;
    f->q1 = 0.0f;
    f->q2 = 0.0f;
    f->q3 = 0.0f;
    f->beta = beta;
    f->inv_sample_freq = 1.0f / sample_freq;
    f->pitch = 0.0f;
    f->roll = 0.0f;
    f->yaw = 0.0f;
}

void madgwick_update_9dof(madgwick_filter_t *f,
                          float gx, float gy, float gz,
                          float ax, float ay, float az,
                          float mx, float my, float mz,
                          float dt)
{
    float q0 = f->q0, q1 = f->q1, q2 = f->q2, q3 = f->q3;
    float recip_norm;
    float s0, s1, s2, s3;
    float q_dot0, q_dot1, q_dot2, q_dot3;
    float _2q0, _2q1, _2q2, _2q3;
    float _2q0q2, _2q2q3;
    float _2q0mx, _2q0my, _2q0mz, _2q1mx;
    float q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3;
    float _2bx, _2bz, _4bx, _4bz;
    float hx, hy;

    if (dt <= 0.0f) {
        dt = f->inv_sample_freq;
    }

    gx *= DEG2RAD;
    gy *= DEG2RAD;
    gz *= DEG2RAD;

    recip_norm = inv_sqrt(ax * ax + ay * ay + az * az);
    ax *= recip_norm;
    ay *= recip_norm;
    az *= recip_norm;

    recip_norm = inv_sqrt(mx * mx + my * my + mz * mz);
    mx *= recip_norm;
    my *= recip_norm;
    mz *= recip_norm;

    _2q0 = 2.0f * q0;
    _2q1 = 2.0f * q1;
    _2q2 = 2.0f * q2;
    _2q3 = 2.0f * q3;

    _2q0mx = 2.0f * q0 * mx;
    _2q0my = 2.0f * q0 * my;
    _2q0mz = 2.0f * q0 * mz;
    _2q1mx = 2.0f * q1 * mx;

    q0q0 = q0 * q0;
    q0q1 = q0 * q1;
    q0q2 = q0 * q2;
    q0q3 = q0 * q3;
    q1q1 = q1 * q1;
    q1q2 = q1 * q2;
    q1q3 = q1 * q3;
    q2q2 = q2 * q2;
    q2q3 = q2 * q3;
    q3q3 = q3 * q3;

    _2q0q2 = _2q0 * q2;
    _2q2q3 = _2q2 * q3;

    hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2
         + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3;
    hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1
         + my * q2q2 + _2q2 * mz * q3 - my * q3q3;
    _2bx = sqrtf(hx * hx + hy * hy);
    _2bz = _2q0mx * q2 + _2q0my * q3 + mz * q0q0 + _2q1mx * q3 - mz * q1q1
           + _2q2 * my * q3 - mz * q2q2 + mz * q3q3;
    _4bx = 2.0f * _2bx;
    _4bz = 2.0f * _2bz;

    s0 = -_2q2 * (2.0f * q1q3 - _2q0q2 - ax)
         + _2q1 * (2.0f * q0q1 + _2q2q3 - ay)
         - _2bz * q2 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
         + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
         + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);

    s1 = _2q3 * (2.0f * q1q3 - _2q0q2 - ax)
         + _2q0 * (2.0f * q0q1 + _2q2q3 - ay)
         - 4.0f * q1 * (1.0f - 2.0f * q1q1 - 2.0f * q2q2 - az)
         + _2bz * q3 * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
         + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
         + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);

    s2 = -_2q0 * (2.0f * q1q3 - _2q0q2 - ax)
         + _2q3 * (2.0f * q0q1 + _2q2q3 - ay)
         - 4.0f * q2 * (1.0f - 2.0f * q1q1 - 2.0f * q2q2 - az)
         + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
         + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
         + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);

    s3 = _2q1 * (2.0f * q1q3 - _2q0q2 - ax)
         + _2q2 * (2.0f * q0q1 + _2q2q3 - ay)
         + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5f - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
         + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
         + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5f - q1q1 - q2q2) - mz);

    recip_norm = inv_sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
    s0 *= recip_norm;
    s1 *= recip_norm;
    s2 *= recip_norm;
    s3 *= recip_norm;

    q_dot0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz) - f->beta * s0;
    q_dot1 = 0.5f * (q0 * gx + q2 * gz - q3 * gy) - f->beta * s1;
    q_dot2 = 0.5f * (q0 * gy - q1 * gz + q3 * gx) - f->beta * s2;
    q_dot3 = 0.5f * (q0 * gz + q1 * gy - q2 * gx) - f->beta * s3;

    q0 += q_dot0 * dt;
    q1 += q_dot1 * dt;
    q2 += q_dot2 * dt;
    q3 += q_dot3 * dt;

    recip_norm = inv_sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    f->q0 = q0 * recip_norm;
    f->q1 = q1 * recip_norm;
    f->q2 = q2 * recip_norm;
    f->q3 = q3 * recip_norm;
}

void madgwick_get_euler(madgwick_filter_t *f)
{
    float q0 = f->q0, q1 = f->q1, q2 = f->q2, q3 = f->q3;

    f->pitch = asinf(-2.0f * (q1 * q3 - q0 * q2)) * RAD2DEG;
    f->roll = atan2f(2.0f * (q0 * q1 + q2 * q3), 1.0f - 2.0f * (q1 * q1 + q2 * q2)) * RAD2DEG;
    f->yaw = atan2f(2.0f * (q0 * q3 + q1 * q2), 1.0f - 2.0f * (q2 * q2 + q3 * q3)) * RAD2DEG;
}

imu_pose_t imu_fusion_get_pose(madgwick_filter_t *f, float altitude,
                               float temperature, uint32_t ts)
{
    imu_pose_t pose;
    madgwick_get_euler(f);

    pose.qw = f->q0;
    pose.qx = f->q1;
    pose.qy = f->q2;
    pose.qz = f->q3;
    pose.pitch = f->pitch;
    pose.roll = f->roll;
    pose.yaw = f->yaw;
    pose.altitude_m = altitude;
    pose.temperature_c = temperature;
    pose.timestamp_ms = ts;
    return pose;
}

gyro_bias_t imu_calibrate_gyro(const float *gx_samples, const float *gy_samples,
                               const float *gz_samples, uint32_t n)
{
    gyro_bias_t bias;
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        sum_x += gx_samples[i];
        sum_y += gy_samples[i];
        sum_z += gz_samples[i];
    }

    bias.bias_x = sum_x / (float)n;
    bias.bias_y = sum_y / (float)n;
    bias.bias_z = sum_z / (float)n;
    return bias;
}

mag_calibration_t imu_calibrate_mag(const float *mx_samples, const float *my_samples,
                                    const float *mz_samples, uint32_t n)
{
    mag_calibration_t cal;
    float min_x, max_x, min_y, max_y, min_z, max_z;
    float avg_range;

    min_x = max_x = mx_samples[0];
    min_y = max_y = my_samples[0];
    min_z = max_z = mz_samples[0];

    for (uint32_t i = 1; i < n; i++) {
        if (mx_samples[i] < min_x) min_x = mx_samples[i];
        if (mx_samples[i] > max_x) max_x = mx_samples[i];
        if (my_samples[i] < min_y) min_y = my_samples[i];
        if (my_samples[i] > max_y) max_y = my_samples[i];
        if (mz_samples[i] < min_z) min_z = mz_samples[i];
        if (mz_samples[i] > max_z) max_z = mz_samples[i];
    }

    cal.offset_x = (max_x + min_x) * 0.5f;
    cal.offset_y = (max_y + min_y) * 0.5f;
    cal.offset_z = (max_z + min_z) * 0.5f;

    avg_range = ((max_x - min_x) + (max_y - min_y) + (max_z - min_z)) / 3.0f;

    cal.scale_x = avg_range / (max_x - min_x);
    cal.scale_y = avg_range / (max_y - min_y);
    cal.scale_z = avg_range / (max_z - min_z);

    return cal;
}