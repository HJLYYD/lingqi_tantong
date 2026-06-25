/*
 * K1 本地 I2C IMU 驱动 — GY85 (ADXL345 + ITG3205)
 *
 * 完全参照 gy85_json_output.py 的 I2C 寄存器映射和缩放公式。
 * Linux I2C 用户空间 API: open("/dev/i2c-N") + ioctl(I2C_SLAVE) + read/write。
 */

#define _DEFAULT_SOURCE
#include "k1_imu.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

/* ── ADXL345 寄存器 ── */
#define ADXL_REG_DATA_FORMAT  0x31
#define ADXL_REG_POWER_CTL    0x2D
#define ADXL_REG_DATAX0       0x32

/* ── ITG3205 寄存器 ── */
#define ITG_REG_DLPF_FS       0x16
#define ITG_REG_PWR_MGMT      0x3E
#define ITG_REG_GYRO_XH       0x1D

/* ── 缩放因子 ── */
#define ACCEL_SCALE  ((2.0f * 9.80665f) / 32768.0f)   /* ±2g → m/s² */
#define GYRO_SCALE   (((float)M_PI / 180.0f) / 14.375f)  /* ITG3205 fixed ±2000dps, 14.375 LSB/°/s */

/* ═══════════════════════════════════════════════════════════
 *  I2C 底层
 * ═══════════════════════════════════════════════════════════ */

static int i2c_open(int bus) {
    char path[32];
    snprintf(path, sizeof(path), "/dev/i2c-%d", bus);
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        log_error("[K1-IMU] Cannot open %s: %s", path, strerror(errno));
        return -1;
    }
    return fd;
}

static int i2c_select(int fd, uint8_t addr) {
    if (ioctl(fd, I2C_SLAVE, addr) < 0) {
        log_error("[K1-IMU] I2C_SLAVE 0x%02X failed: %s", addr, strerror(errno));
        return -1;
    }
    return 0;
}

static int16_t i2c_read_s16_le(int fd, uint8_t reg) {
    /* ADXL345 / ITG3205: 小端序 16-bit */
    uint8_t buf[2];
    if (write(fd, &reg, 1) != 1) return 0;
    if (read(fd, buf, 2) != 2)  return 0;
    return (int16_t)((uint16_t)buf[0] | ((uint16_t)buf[1] << 8));
}

static bool i2c_write_reg(int fd, uint8_t reg, uint8_t val) {
    uint8_t buf[2] = { reg, val };
    return write(fd, buf, 2) == 2;
}

/* ═══════════════════════════════════════════════════════════
 *  传感器初始化 (完全参照 gy85_json_output.py)
 * ═══════════════════════════════════════════════════════════ */

static bool init_adxl345(int fd) {
    if (i2c_select(fd, K1_IMU_ADXL345_ADDR) < 0) return false;
    /* ±2g, 10-bit模式 */
    if (!i2c_write_reg(fd, ADXL_REG_DATA_FORMAT, 0x00)) return false;
    /* 测量模式 */
    if (!i2c_write_reg(fd, ADXL_REG_POWER_CTL, 0x08)) return false;
    return true;
}

static bool init_itg3205(int fd) {
    if (i2c_select(fd, K1_IMU_ITG3205_ADDR) < 0) return false;
    /* 复位 */
    if (!i2c_write_reg(fd, ITG_REG_PWR_MGMT, 0x80)) return false;
    usleep(100000);  /* 100ms 复位等待 */
    /* DLPF_CFG=3(42Hz) + internal OSC, ±2000dps */
    if (!i2c_write_reg(fd, ITG_REG_DLPF_FS, 0x1B)) return false;
    return true;
}

/* ═══════════════════════════════════════════════════════════
 *  数据读取
 * ═══════════════════════════════════════════════════════════ */

static bool read_adxl345(int fd, float* ax, float* ay, float* az) {
    if (i2c_select(fd, K1_IMU_ADXL345_ADDR) < 0) return false;

    int16_t rx = i2c_read_s16_le(fd, ADXL_REG_DATAX0);
    int16_t ry = i2c_read_s16_le(fd, ADXL_REG_DATAX0 + 2);
    int16_t rz = i2c_read_s16_le(fd, ADXL_REG_DATAX0 + 4);

    *ax = rx * ACCEL_SCALE;
    *ay = ry * ACCEL_SCALE;
    *az = rz * ACCEL_SCALE;
    return true;
}

static bool read_itg3205(int fd, float* gx, float* gy, float* gz) {
    if (i2c_select(fd, K1_IMU_ITG3205_ADDR) < 0) return false;

    int16_t rx = i2c_read_s16_le(fd, ITG_REG_GYRO_XH);
    int16_t ry = i2c_read_s16_le(fd, ITG_REG_GYRO_XH + 2);
    int16_t rz = i2c_read_s16_le(fd, ITG_REG_GYRO_XH + 4);

    *gx = rx * GYRO_SCALE;
    *gy = ry * GYRO_SCALE;
    *gz = rz * GYRO_SCALE;
    return true;
}

/* ═══════════════════════════════════════════════════════════
 *  公开 API
 * ═══════════════════════════════════════════════════════════ */

K1Imu* k1_imu_create(int i2c_bus, float sample_rate_hz) {
    K1Imu* imu = (K1Imu*)calloc(1, sizeof(K1Imu));
    if (!imu) return NULL;

    imu->i2c_bus = i2c_bus;
    imu->sample_rate_hz = (sample_rate_hz > 0.0f) ? sample_rate_hz : 100.0f;
    imu->state = K1_IMU_STATE_UNINIT;

    if (pthread_mutex_init(&imu->ring_mutex, NULL) != 0) {
        free(imu);
        return NULL;
    }

    /* 打开 I2C */
    imu->i2c_fd = i2c_open(i2c_bus);
    if (imu->i2c_fd < 0) {
        log_warning("[K1-IMU] I2C bus %d unavailable — K1 local IMU disabled", i2c_bus);
        imu->state = K1_IMU_STATE_ERROR;
        return imu;  /* 返回但标记错误, 调用者检查 state */
    }

    /* 初始化传感器 */
    if (!init_adxl345(imu->i2c_fd) || !init_itg3205(imu->i2c_fd)) {
        log_warning("[K1-IMU] GY85 sensor init failed — K1 local IMU disabled");
        close(imu->i2c_fd);
        imu->i2c_fd = -1;
        imu->state = K1_IMU_STATE_ERROR;
        return imu;
    }

    /* 自动开始校准 */
    imu->calib.done = false;
    imu->calib.samples_collected = 0;
    imu->state = K1_IMU_STATE_CALIBRATING;
    imu->actual_rate = 0.0f;

    log_info("[K1-IMU] GY85 initialized on /dev/i2c-%d @ %.0fHz (calibrating...)",
             i2c_bus, imu->sample_rate_hz);
    return imu;
}

void k1_imu_destroy(K1Imu* imu) {
    if (!imu) return;
    if (imu->i2c_fd >= 0) close(imu->i2c_fd);
    pthread_mutex_destroy(&imu->ring_mutex);
    free(imu);
}

bool k1_imu_read_sample(K1Imu* imu, IMUData* out) {
    if (!imu || !out || imu->state == K1_IMU_STATE_ERROR) return false;
    if (imu->i2c_fd < 0) return false;

    float ax, ay, az, gx, gy, gz;

    if (!read_adxl345(imu->i2c_fd, &ax, &ay, &az)) {
        imu->error_count++;
        return false;
    }
    if (!read_itg3205(imu->i2c_fd, &gx, &gy, &gz)) {
        imu->error_count++;
        return false;
    }

    /* 应用校准 */
    if (imu->calib.done) {
        gx -= imu->calib.gyro_bias[0];
        gy -= imu->calib.gyro_bias[1];
        gz -= imu->calib.gyro_bias[2];
    } else if (imu->state == K1_IMU_STATE_CALIBRATING) {
        /* 积累校准样本 */
        imu->calib.gyro_bias[0] += gx;
        imu->calib.gyro_bias[1] += gy;
        imu->calib.gyro_bias[2] += gz;
        imu->calib.samples_collected++;

        if (imu->calib.samples_collected >= K1_IMU_CALIB_SAMPLES) {
            float inv = 1.0f / (float)imu->calib.samples_collected;
            imu->calib.gyro_bias[0] *= inv;
            imu->calib.gyro_bias[1] *= inv;
            imu->calib.gyro_bias[2] *= inv;
            imu->calib.done = true;
            imu->state = K1_IMU_STATE_RUNNING;
            log_info("[K1-IMU] Calibration done: gyro_bias=(%.4f, %.4f, %.4f) rad/s",
                     imu->calib.gyro_bias[0], imu->calib.gyro_bias[1],
                     imu->calib.gyro_bias[2]);
            /* 应用校准到当前读数 */
            gx -= imu->calib.gyro_bias[0];
            gy -= imu->calib.gyro_bias[1];
            gz -= imu->calib.gyro_bias[2];
        }
    }

    /* 时间戳 */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now_s = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;

    /* 写入环形缓冲 */
    pthread_mutex_lock(&imu->ring_mutex);
    IMUData* slot = &imu->ring[imu->ring_head];
    slot->timestamp = now_s;
    slot->accel_x = ax;
    slot->accel_y = ay;
    slot->accel_z = az;
    slot->gyro_x = gx;
    slot->gyro_y = gy;
    slot->gyro_z = gz;
    imu->ring_head = (imu->ring_head + 1) % K1_IMU_RING_SIZE;
    if (imu->ring_count < K1_IMU_RING_SIZE) imu->ring_count++;

    /* 速率统计 */
    if (imu->last_sample_time > 0.0 && imu->total_samples % 100 == 0) {
        double elapsed = now_s - imu->last_sample_time;
        if (elapsed > 0.5) {
            imu->actual_rate = 100.0f / (float)elapsed;
        }
    }
    imu->last_sample_time = now_s;
    imu->total_samples++;
    pthread_mutex_unlock(&imu->ring_mutex);

    /* 复制输出 (只有 RUNNING 和校准完成后的数据) */
    if (out) {
        out->timestamp = now_s;
        out->accel_x = ax;
        out->accel_y = ay;
        out->accel_z = az;
        out->gyro_x = gx;
        out->gyro_y = gy;
        out->gyro_z = gz;
    }

    return true;
}

int k1_imu_read_recent(K1Imu* imu, IMUData* out, int max_count) {
    if (!imu || !out || max_count <= 0) return 0;

    pthread_mutex_lock(&imu->ring_mutex);
    int count = (imu->ring_count < max_count) ? imu->ring_count : max_count;
    int start = (imu->ring_head - count + K1_IMU_RING_SIZE) % K1_IMU_RING_SIZE;
    for (int i = 0; i < count; i++) {
        out[i] = imu->ring[(start + i) % K1_IMU_RING_SIZE];
    }
    pthread_mutex_unlock(&imu->ring_mutex);
    return count;
}

bool k1_imu_start_calibration(K1Imu* imu) {
    if (!imu || imu->state == K1_IMU_STATE_ERROR) return false;
    imu->calib.done = false;
    imu->calib.samples_collected = 0;
    memset(imu->calib.gyro_bias, 0, sizeof(imu->calib.gyro_bias));
    imu->state = K1_IMU_STATE_CALIBRATING;
    log_info("[K1-IMU] Re-calibration started");
    return true;
}

bool k1_imu_is_calibrated(const K1Imu* imu) {
    return imu && imu->calib.done;
}

K1ImuState k1_imu_get_state(const K1Imu* imu) {
    return imu ? imu->state : K1_IMU_STATE_ERROR;
}

float k1_imu_get_actual_rate(const K1Imu* imu) {
    return imu ? imu->actual_rate : 0.0f;
}
