idf_component_register(SRCS
    "main.c"
    "camera_capture.c"
    "gy87_driver.c"
    "imu_fusion.c"
    "protocol.c"
    INCLUDE_DIRS "."
    REQUIRES
    esp_driver_cam
    esp_driver_i2c
    esp_driver_uart
    esp_driver_gpio
    nvs_flash
)