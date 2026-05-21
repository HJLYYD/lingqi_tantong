#ifndef GY87_DRIVER_H
#define GY87_DRIVER_H

#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "driver/gpio.h"
#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GY87_MPU6050_ADDR      0x68
#define GY87_QMC5883L_ADDR     0x0D
#define GY87_BMP180_ADDR       0x77

#define MPU6050_REG_SMPLRT_DIV     0x19
#define MPU6050_REG_CONFIG         0x1A
#define MPU6050_REG_GYRO_CONFIG    0x1B
#define MPU6050_REG_ACCEL_CONFIG   0x1C
#define MPU6050_REG_INT_PIN_CFG    0x37
#define MPU6050_REG_USER_CTRL      0x6A
#define MPU6050_REG_PWR_MGMT_1     0x6B
#define MPU6050_REG_ACCEL_XOUT_H   0x3B

#define QMC5883L_REG_DATA_X_LSB    0x00
#define QMC5883L_REG_CTRL1         0x09
#define QMC5883L_REG_CTRL2         0x0A

#define BMP180_REG_CTRL_MEAS       0xF4
#define BMP180_REG_OUT_MSB         0xF6
#define BMP180_CAL_START           0xAA

#define GY87_DEFAULT_CLOCK_SPEED   400000

typedef struct {
    i2c_port_t i2c_port;
    gpio_num_t sda_pin;
    gpio_num_t scl_pin;
    uint32_t clk_speed;
} gy87_config_t;

typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    float mag_x;
    float mag_y;
    float mag_z;
    float temperature;
    float pressure;
    uint32_t timestamp_ms;
} gy87_data_t;

esp_err_t gy87_init(i2c_port_t i2c_port, gpio_num_t sda_pin, gpio_num_t scl_pin);
esp_err_t gy87_deinit(void);
esp_err_t gy87_read_all(gy87_data_t *data);

#ifdef __cplusplus
}
#endif

#endif