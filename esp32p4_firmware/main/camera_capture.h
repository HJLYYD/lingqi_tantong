#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <stdint.h>
#include "freertos/FreeRTOS.h"

typedef struct {
    uint8_t *jpeg_buf;
    size_t jpeg_len;
    uint32_t timestamp_ms;
} jpeg_frame_t;

esp_err_t camera_capture_init(void);
esp_err_t camera_capture_start(void);
esp_err_t camera_capture_get_frame(jpeg_frame_t *frame);
esp_err_t camera_capture_return_frame(void);
esp_err_t camera_capture_deinit(void);

#endif