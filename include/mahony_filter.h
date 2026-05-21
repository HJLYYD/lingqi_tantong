#ifndef MAHONY_FILTER_H
#define MAHONY_FILTER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float q0;
    float q1;
    float q2;
    float q3;
    float kp;
    float ki;
    float integral_fbx;
    float integral_fby;
    float integral_fbz;
    float pitch;
    float roll;
    float yaw;
} mahony_filter_t;

void mahony_init(mahony_filter_t* f, float kp, float ki);
void mahony_update_9dof(mahony_filter_t* f, float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float dt);
void mahony_get_euler(mahony_filter_t* f);

#ifdef __cplusplus
}
#endif

#endif