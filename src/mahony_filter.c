#include "mahony_filter.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void mahony_init(mahony_filter_t* f, float kp, float ki)
{
    f->q0 = 1.0f;
    f->q1 = 0.0f;
    f->q2 = 0.0f;
    f->q3 = 0.0f;
    f->kp = kp;
    f->ki = ki;
    f->integral_fbx = 0.0f;
    f->integral_fby = 0.0f;
    f->integral_fbz = 0.0f;
    f->pitch = 0.0f;
    f->roll = 0.0f;
    f->yaw = 0.0f;
}

void mahony_update_9dof(mahony_filter_t* f, float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float dt)
{
    float recip_norm;
    float q0 = f->q0;
    float q1 = f->q1;
    float q2 = f->q2;
    float q3 = f->q3;
    float kp = f->kp;
    float ki = f->ki;

    if (dt <= 0.0f) {
        dt = 0.005f;
    }

    gx = gx * (M_PI / 180.0f);
    gy = gy * (M_PI / 180.0f);
    gz = gz * (M_PI / 180.0f);

    recip_norm = 1.0f / sqrtf(ax * ax + ay * ay + az * az);
    ax *= recip_norm;
    ay *= recip_norm;
    az *= recip_norm;

    recip_norm = 1.0f / sqrtf(mx * mx + my * my + mz * mz);
    mx *= recip_norm;
    my *= recip_norm;
    mz *= recip_norm;

    float vx = 2.0f * (q1 * q3 - q0 * q2);
    float vy = 2.0f * (q0 * q1 + q2 * q3);
    float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

    float ex = (ay * vz - az * vy);
    float ey = (az * vx - ax * vz);
    float ez = (ax * vy - ay * vx);

    float hx = 2.0f * mx * (0.5f - q2 * q2 - q3 * q3) + 2.0f * my * (q1 * q2 - q0 * q3) + 2.0f * mz * (q1 * q3 + q0 * q2);
    float hy = 2.0f * mx * (q1 * q2 + q0 * q3) + 2.0f * my * (0.5f - q1 * q1 - q3 * q3) + 2.0f * mz * (q2 * q3 - q0 * q1);
    float hz = 2.0f * mx * (q1 * q3 - q0 * q2) + 2.0f * my * (q2 * q3 + q0 * q1) + 2.0f * mz * (0.5f - q1 * q1 - q2 * q2);

    float bx = sqrtf(hx * hx + hy * hy);
    float bz = hz;

    float wx = 2.0f * bx * (0.5f - q2 * q2 - q3 * q3) + 2.0f * bz * (q1 * q3 - q0 * q2);
    float wy = 2.0f * bx * (q1 * q2 - q0 * q3) + 2.0f * bz * (q0 * q1 + q2 * q3);
    float wz = 2.0f * bx * (q0 * q2 + q1 * q3) + 2.0f * bz * (0.5f - q1 * q1 - q2 * q2);

    ex += (my * wz - mz * wy);
    ey += (mz * wx - mx * wz);
    ez += (mx * wy - my * wx);

    f->integral_fbx += ex * dt * 0.5f;
    f->integral_fby += ey * dt * 0.5f;
    f->integral_fbz += ez * dt * 0.5f;

    gx -= kp * ex + ki * f->integral_fbx;
    gy -= kp * ey + ki * f->integral_fby;
    gz -= kp * ez + ki * f->integral_fbz;

    q0 += 0.5f * (-q1 * gx - q2 * gy - q3 * gz) * dt;
    q1 += 0.5f * ( q0 * gx + q2 * gz - q3 * gy) * dt;
    q2 += 0.5f * ( q0 * gy - q1 * gz + q3 * gx) * dt;
    q3 += 0.5f * ( q0 * gz + q1 * gy - q2 * gx) * dt;

    recip_norm = 1.0f / sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    f->q0 = q0 * recip_norm;
    f->q1 = q1 * recip_norm;
    f->q2 = q2 * recip_norm;
    f->q3 = q3 * recip_norm;

    mahony_get_euler(f);
}

void mahony_get_euler(mahony_filter_t* f)
{
    float q0 = f->q0;
    float q1 = f->q1;
    float q2 = f->q2;
    float q3 = f->q3;

    f->pitch = asinf(-2.0f * (q1 * q3 - q0 * q2)) * (180.0f / M_PI);
    f->roll  = atan2f(2.0f * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * (180.0f / M_PI);
    f->yaw   = atan2f(2.0f * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * (180.0f / M_PI);
}