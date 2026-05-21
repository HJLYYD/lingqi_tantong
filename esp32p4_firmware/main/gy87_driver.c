#include "gy87_driver.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

static gy87_config_t g_config;
static bool g_initialized = false;

static int16_t bmp180_ac1, bmp180_ac2, bmp180_ac3;
static uint16_t bmp180_ac4, bmp180_ac5, bmp180_ac6;
static int16_t bmp180_b1, bmp180_b2;
static int16_t bmp180_mb, bmp180_mc, bmp180_md;
static bool bmp180_cal_loaded = false;

static esp_err_t i2c_write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t value)
{
    uint8_t buf[2] = {reg_addr, value};
    return i2c_master_write_to_device(g_config.i2c_port, dev_addr, buf, 2,
                                      pdMS_TO_TICKS(100));
}

static esp_err_t i2c_read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(g_config.i2c_port, dev_addr,
                                        &reg_addr, 1, data, len,
                                        pdMS_TO_TICKS(100));
}

static esp_err_t mpu6050_init(void)
{
    esp_err_t ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_PWR_MGMT_1, 0x00);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(100));

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_CONFIG, 0x03);
    if (ret != ESP_OK) return ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_GYRO_CONFIG, 0x00);
    if (ret != ESP_OK) return ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_ACCEL_CONFIG, 0x00);
    if (ret != ESP_OK) return ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_SMPLRT_DIV, 0x04);
    if (ret != ESP_OK) return ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_INT_PIN_CFG, 0x02);
    if (ret != ESP_OK) return ret;

    ret = i2c_write_reg(GY87_MPU6050_ADDR, MPU6050_REG_USER_CTRL, 0x00);
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

static esp_err_t qmc5883l_init(void)
{
    esp_err_t ret;

    vTaskDelay(pdMS_TO_TICKS(10));

    ret = i2c_write_reg(GY87_QMC5883L_ADDR, QMC5883L_REG_CTRL2, 0x80);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    ret = i2c_write_reg(GY87_QMC5883L_ADDR, QMC5883L_REG_CTRL1, 0x01);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(10));

    ret = i2c_write_reg(GY87_QMC5883L_ADDR, QMC5883L_REG_CTRL2, 0x00);
    if (ret != ESP_OK) return ret;

    return ESP_OK;
}

static esp_err_t bmp180_load_calibration(void)
{
    uint8_t cal_data[22];
    esp_err_t ret;

    ret = i2c_read_reg(GY87_BMP180_ADDR, BMP180_CAL_START, cal_data, 22);
    if (ret != ESP_OK) return ret;

    bmp180_ac1 = (int16_t)((cal_data[0] << 8) | cal_data[1]);
    bmp180_ac2 = (int16_t)((cal_data[2] << 8) | cal_data[3]);
    bmp180_ac3 = (int16_t)((cal_data[4] << 8) | cal_data[5]);
    bmp180_ac4 = (uint16_t)((cal_data[6] << 8) | cal_data[7]);
    bmp180_ac5 = (uint16_t)((cal_data[8] << 8) | cal_data[9]);
    bmp180_ac6 = (uint16_t)((cal_data[10] << 8) | cal_data[11]);
    bmp180_b1  = (int16_t)((cal_data[12] << 8) | cal_data[13]);
    bmp180_b2  = (int16_t)((cal_data[14] << 8) | cal_data[15]);
    bmp180_mb  = (int16_t)((cal_data[16] << 8) | cal_data[17]);
    bmp180_mc  = (int16_t)((cal_data[18] << 8) | cal_data[19]);
    bmp180_md  = (int16_t)((cal_data[20] << 8) | cal_data[21]);

    bmp180_cal_loaded = true;
    return ESP_OK;
}

static esp_err_t bmp180_read_ut(int32_t *ut)
{
    esp_err_t ret;
    uint8_t data[2];

    ret = i2c_write_reg(GY87_BMP180_ADDR, BMP180_REG_CTRL_MEAS, 0x2E);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(5));

    ret = i2c_read_reg(GY87_BMP180_ADDR, BMP180_REG_OUT_MSB, data, 2);
    if (ret != ESP_OK) return ret;

    *ut = (int32_t)((data[0] << 8) | data[1]);
    return ESP_OK;
}

static esp_err_t bmp180_read_up(int32_t *up)
{
    esp_err_t ret;
    uint8_t data[3];

    ret = i2c_write_reg(GY87_BMP180_ADDR, BMP180_REG_CTRL_MEAS, 0x34);
    if (ret != ESP_OK) return ret;
    vTaskDelay(pdMS_TO_TICKS(5));

    ret = i2c_read_reg(GY87_BMP180_ADDR, BMP180_REG_OUT_MSB, data, 3);
    if (ret != ESP_OK) return ret;

    *up = (int32_t)(((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | data[2]) >> 8;
    return ESP_OK;
}

static float bmp180_calc_temperature(int32_t ut)
{
    int32_t x1 = (((int32_t)ut - (int32_t)bmp180_ac6) * (int32_t)bmp180_ac5) >> 15;
    int32_t x2 = ((int32_t)bmp180_mc << 11) / (x1 + bmp180_md);
    int32_t b5 = x1 + x2;
    return ((b5 + 8) >> 4) / 10.0f;
}

static float bmp180_calc_pressure(int32_t up, int32_t ut)
{
    int32_t x1, x2, x3, b3, b5, b6;
    uint32_t b4, b7;
    int32_t p;

    x1 = (((int32_t)ut - (int32_t)bmp180_ac6) * (int32_t)bmp180_ac5) >> 15;
    x2 = ((int32_t)bmp180_mc << 11) / (x1 + bmp180_md);
    b5 = x1 + x2;
    b6 = b5 - 4000;

    x1 = (bmp180_b2 * ((b6 * b6) >> 12)) >> 11;
    x2 = ((int32_t)bmp180_ac2 * b6) >> 11;
    x3 = x1 + x2;
    b3 = ((((int32_t)bmp180_ac1 * 4 + x3)) + 2) >> 2;

    x1 = ((int32_t)bmp180_ac3 * b6) >> 13;
    x2 = (bmp180_b1 * ((b6 * b6) >> 12)) >> 16;
    x3 = ((x1 + x2) + 2) >> 2;
    b4 = (bmp180_ac4 * (uint32_t)(x3 + 32768)) >> 15;

    b7 = ((uint32_t)(up - b3) * (uint32_t)50000);
    if (b7 < 0x80000000) {
        p = (int32_t)((b7 << 1) / b4);
    } else {
        p = (int32_t)((b7 / b4) << 1);
    }

    x1 = ((p >> 8) * (p >> 8) * 3038) >> 16;
    x2 = (-7357 * p) >> 16;
    p = p + ((x1 + x2 + 3791) >> 4);

    return (float)p;
}

esp_err_t gy87_init(i2c_port_t i2c_port, gpio_num_t sda_pin, gpio_num_t scl_pin)
{
    esp_err_t ret;

    g_config.i2c_port = i2c_port;
    g_config.sda_pin = sda_pin;
    g_config.scl_pin = scl_pin;
    g_config.clk_speed = GY87_DEFAULT_CLOCK_SPEED;

    i2c_config_t i2c_conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = sda_pin,
        .scl_io_num = scl_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = GY87_DEFAULT_CLOCK_SPEED,
    };

    ret = i2c_param_config(i2c_port, &i2c_conf);
    if (ret != ESP_OK) return ret;

    ret = i2c_driver_install(i2c_port, I2C_MODE_MASTER, 0, 0, 0);
    if (ret != ESP_OK) return ret;

    ret = mpu6050_init();
    if (ret != ESP_OK) return ret;

    ret = qmc5883l_init();
    if (ret != ESP_OK) return ret;

    ret = bmp180_load_calibration();
    if (ret != ESP_OK) return ret;

    g_initialized = true;
    return ESP_OK;
}

esp_err_t gy87_deinit(void)
{
    if (!g_initialized) return ESP_OK;

    esp_err_t ret = i2c_driver_delete(g_config.i2c_port);
    g_initialized = false;
    bmp180_cal_loaded = false;
    memset(&g_config, 0, sizeof(g_config));
    return ret;
}

esp_err_t gy87_read_all(gy87_data_t *data)
{
    if (!g_initialized || !data) return ESP_ERR_INVALID_STATE;

    esp_err_t ret;
    uint8_t buf[14];
    int32_t ut, up;

    memset(data, 0, sizeof(gy87_data_t));
    data->timestamp_ms = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);

    ret = i2c_read_reg(GY87_MPU6050_ADDR, MPU6050_REG_ACCEL_XOUT_H, buf, 14);
    if (ret != ESP_OK) return ret;

    int16_t raw_ax = (int16_t)((buf[0] << 8) | buf[1]);
    int16_t raw_ay = (int16_t)((buf[2] << 8) | buf[3]);
    int16_t raw_az = (int16_t)((buf[4] << 8) | buf[5]);
    int16_t raw_tp = (int16_t)((buf[6] << 8) | buf[7]);
    int16_t raw_gx = (int16_t)((buf[8] << 8) | buf[9]);
    int16_t raw_gy = (int16_t)((buf[10] << 8) | buf[11]);
    int16_t raw_gz = (int16_t)((buf[12] << 8) | buf[13]);

    data->accel_x = (float)raw_ax / 16384.0f * 9.81f;
    data->accel_y = (float)raw_ay / 16384.0f * 9.81f;
    data->accel_z = (float)raw_az / 16384.0f * 9.81f;

    data->gyro_x = (float)raw_gx / 131.0f;
    data->gyro_y = (float)raw_gy / 131.0f;
    data->gyro_z = (float)raw_gz / 131.0f;

    ret = i2c_read_reg(GY87_QMC5883L_ADDR, QMC5883L_REG_DATA_X_LSB, buf, 6);
    if (ret != ESP_OK) return ret;

    int16_t raw_mx = (int16_t)((uint16_t)buf[1] << 8 | buf[0]);
    int16_t raw_my = (int16_t)((uint16_t)buf[3] << 8 | buf[2]);
    int16_t raw_mz = (int16_t)((uint16_t)buf[5] << 8 | buf[4]);

    data->mag_x = (float)raw_mx / 12000.0f * 100.0f;
    data->mag_y = (float)raw_my / 12000.0f * 100.0f;
    data->mag_z = (float)raw_mz / 12000.0f * 100.0f;

    if (bmp180_cal_loaded) {
        ret = bmp180_read_ut(&ut);
        if (ret == ESP_OK) {
            data->temperature = bmp180_calc_temperature(ut);

            ret = bmp180_read_up(&up);
            if (ret == ESP_OK) {
                data->pressure = bmp180_calc_pressure(up, ut);
            }
        }
    }

    (void)raw_tp;
    return ESP_OK;
}