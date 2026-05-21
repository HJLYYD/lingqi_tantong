#include "camera_capture.h"
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "camera_capture";

/*
 * 摄像头引脚配置说明 (TODO: 根据实际硬件接线配置):
 *
 * 当前所有 pin_* 初始化为 -1 (无效), esp_camera_init() 将使用 ESP-IDF 默认值。
 * 必须在部署前根据实际硬件接线设置正确的 GPIO 引脚映射。
 *
 * 参考接线 (OV2640/OV5640 → ESP32P4, 以 ESP32-S3-EYE 为参考):
 *   - pin_xclk:    GPIO 15 (XCLK,  10-24MHz)
 *   - pin_sccb_sda: GPIO 12 (SIOD,  I2C 数据)
 *   - pin_sccb_scl: GPIO 13 (SIOC,  I2C 时钟)
 *   - pin_d0-d7:   GPIO 4,5,6,7,8,9,10,11 (Y0-Y7 数据线)
 *   - pin_vsync:   GPIO 16 (VSYNC, 垂直同步)
 *   - pin_href:    GPIO 18 (HREF,  水平参考)
 *   - pin_pclk:    GPIO 14 (PCLK,  像素时钟)
 *   - pin_pwdn:    GPIO -1 (PWDN,  通常接地, -1 表示不使用)
 *   - pin_reset:   GPIO -1 (RESET, 通常接 3.3V, -1 表示不使用)
 *
 * 实际引脚请参考:
 *   - ESP32P4 数据手册和硬件设计指南
 *   - 项目原理图和 PCB 布局
 *   - esp32-camera 组件文档: https://components.espressif.com/components/espressif/esp32-camera
 *
 * 建议: 通过 Kconfig (menuconfig) 配置引脚, 而非硬编码。
 *       在 sdkconfig.defaults 或 main/Kconfig.projbuild 中添加:
 *         CONFIG_CAMERA_PIN_XCLK=15
 *         CONFIG_CAMERA_PIN_D0=4
 *         ...
 */

static camera_config_t camera_config = {
    .pin_pwdn = -1,
    .pin_reset = -1,
    .pin_xclk = -1,
    .pin_sccb_sda = -1,
    .pin_sccb_scl = -1,

    .pin_d7 = -1,
    .pin_d6 = -1,
    .pin_d5 = -1,
    .pin_d4 = -1,
    .pin_d3 = -1,
    .pin_d2 = -1,
    .pin_d1 = -1,
    .pin_d0 = -1,

    .pin_vsync = -1,
    .pin_href = -1,
    .pin_pclk = -1,

    .xclk_freq_hz = 24000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,

    .jpeg_quality = 12,
    .fb_count = 2,
    .grab_mode = CAMERA_GRAB_LATEST,
};

static bool s_camera_initialized = false;
static camera_fb_t *s_last_fb = NULL;

esp_err_t camera_capture_init(void)
{
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: 0x%x", err);
        return err;
    }

    s_camera_initialized = true;
    ESP_LOGI(TAG, "camera init success");
    return ESP_OK;
}

esp_err_t camera_capture_start(void)
{
    if (!s_camera_initialized) {
        ESP_LOGE(TAG, "camera not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "camera capture started");
    return ESP_OK;
}

esp_err_t camera_capture_get_frame(jpeg_frame_t *frame)
{
    if (!frame) {
        return ESP_ERR_INVALID_ARG;
    }

    if (!s_camera_initialized) {
        ESP_LOGE(TAG, "camera not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "camera framebuffer get failed");
        return ESP_FAIL;
    }

    if (s_last_fb) {
        esp_camera_fb_return(s_last_fb);
    }
    s_last_fb = fb;

    frame->jpeg_buf = fb->buf;
    frame->jpeg_len = fb->len;
    frame->timestamp_ms = (uint32_t)(fb->timestamp.tv_sec * 1000ULL + fb->timestamp.tv_usec / 1000);

    return ESP_OK;
}

esp_err_t camera_capture_return_frame(void)
{
    if (!s_camera_initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (s_last_fb) {
        esp_camera_fb_return(s_last_fb);
        s_last_fb = NULL;
    }
    return ESP_OK;
}

esp_err_t camera_capture_deinit(void)
{
    if (!s_camera_initialized) {
        return ESP_OK;
    }

    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera deinit failed: 0x%x", err);
        return err;
    }

    s_camera_initialized = false;
    ESP_LOGI(TAG, "camera deinit success");
    return ESP_OK;
}