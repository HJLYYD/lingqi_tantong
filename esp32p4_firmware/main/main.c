#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"

#include "camera_capture.h"
#include "gy87_driver.h"
#include "imu_fusion.h"
#include "protocol.h"

#define CAM_XCLK_GPIO  GPIO_NUM_15
#define CAM_SDA_GPIO   GPIO_NUM_12
#define CAM_SCL_GPIO   GPIO_NUM_13
#define I2C_SDA_GPIO   GPIO_NUM_45
#define I2C_SCL_GPIO   GPIO_NUM_46
#define UART_TX_GPIO   GPIO_NUM_43
#define UART_RX_GPIO   GPIO_NUM_44

#define UART_PORT       UART_NUM_1
#define UART_BAUDRATE   3000000
#define UART_BUF_SIZE   (16 * 1024)

#define FRAME_Q_LEN     4
#define IMU_Q_LEN       8

#define CAM_TASK_PRIO   10
#define IMU_TASK_PRIO   12
#define TX_TASK_PRIO    8

#define CAM_STACK       8192
#define IMU_STACK       4096
#define TX_STACK        8192

#define IMU_HZ          200
#define TX_HZ           100
#define IMU_BETA        0.1f

#define CALIB_SAMPLES   400
#define CALIB_DURATION_MS 2000

static const char *TAG = "esp32p4";

static QueueHandle_t frame_queue;
static QueueHandle_t imu_queue;
static madgwick_filter_t madgwick;

static void camera_task(void *arg)
{
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(66);

    camera_capture_init();
    vTaskDelay(pdMS_TO_TICKS(500));

    for (;;) {
        jpeg_frame_t *frame = malloc(sizeof(jpeg_frame_t));
        if (frame == NULL) {
            vTaskDelayUntil(&xLastWakeTime, xPeriod);
            continue;
        }
        if (camera_capture_get_frame(frame) == ESP_OK) {
            if (xQueueSend(frame_queue, &frame, 0) != pdPASS) {
                camera_capture_return_frame();
                free(frame);
            }
        } else {
            free(frame);
        }
        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

static void imu_task(void *arg)
{
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1000 / IMU_HZ);
    gy87_data_t imu_data;
    imu_pose_t pose;
    float gx_bias = 0.0f, gy_bias = 0.0f, gz_bias = 0.0f;
    int calib_count = 0;

    gy87_init(I2C_NUM_0, I2C_SDA_GPIO, I2C_SCL_GPIO);
    vTaskDelay(pdMS_TO_TICKS(100));

    madgwick_init(&madgwick, IMU_BETA, (float)IMU_HZ);

    TickType_t calib_start = xTaskGetTickCount();
    while ((xTaskGetTickCount() - calib_start) * portTICK_PERIOD_MS < CALIB_DURATION_MS) {
        if (gy87_read_all(&imu_data) != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(1000 / IMU_HZ));
            continue;
        }
        gx_bias += imu_data.gyro_x;
        gy_bias += imu_data.gyro_y;
        gz_bias += imu_data.gyro_z;
        calib_count++;
        vTaskDelay(pdMS_TO_TICKS(1000 / IMU_HZ));
    }
    gx_bias /= (float)calib_count;
    gy_bias /= (float)calib_count;
    gz_bias /= (float)calib_count;
    ESP_LOGI(TAG, "Gyro bias calibrated: %.3f, %.3f, %.3f", gx_bias, gy_bias, gz_bias);

    for (;;) {
        gy87_read_all(&imu_data);

        madgwick_update_9dof(&madgwick,
                             imu_data.gyro_x - gx_bias,
                             imu_data.gyro_y - gy_bias,
                             imu_data.gyro_z - gz_bias,
                             imu_data.accel_x, imu_data.accel_y, imu_data.accel_z,
                             imu_data.mag_x, imu_data.mag_y, imu_data.mag_z,
                             1.0f / (float)IMU_HZ);

        float altitude = 0.0f;
        if (imu_data.pressure > 10000.0f) {
            altitude = 44330.0f * (1.0f - powf(imu_data.pressure / 101325.0f, 0.1903f));
        }

        pose = imu_fusion_get_pose(&madgwick, altitude, imu_data.temperature,
                                   xTaskGetTickCount() * portTICK_PERIOD_MS);

        xQueueSend(imu_queue, &pose, 0);

        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

static void tx_task(void *arg)
{
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1000 / TX_HZ);
    uint8_t tx_buf[UART_BUF_SIZE];
    uint16_t pkt_len;
    imu_pose_t pose;
    bool has_pose = false;
    uint16_t seq = 0;

    uart_config_t uart_config = {
        .baud_rate = UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    uart_driver_install(UART_PORT, UART_BUF_SIZE, UART_BUF_SIZE, 0, NULL, 0);
    uart_param_config(UART_PORT, &uart_config);
    uart_set_pin(UART_PORT, UART_TX_GPIO, UART_RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);

    for (;;) {
        jpeg_frame_t *frame = NULL;
        if (xQueueReceive(frame_queue, &frame, pdMS_TO_TICKS(5)) == pdPASS) {
            imu_pose_t latest;
            while (xQueueReceive(imu_queue, &latest, 0) == pdPASS) {
                pose = latest;
                has_pose = true;
            }

            uint32_t ts = frame->timestamp_ms;

            pkt_len = protocol_pack_jpeg(tx_buf, seq++, ts,
                                         frame->jpeg_buf, (uint16_t)frame->jpeg_len);
            uart_write_bytes(UART_PORT, tx_buf, pkt_len);

            if (has_pose) {
                pkt_len = protocol_pack_imu_pose(tx_buf, seq++, pose.timestamp_ms,
                                                  pose.qw, pose.qx, pose.qy, pose.qz,
                                                  pose.altitude_m, pose.temperature_c);
                uart_write_bytes(UART_PORT, tx_buf, pkt_len);
            }

            camera_capture_return_frame();
            free(frame);
        }

        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

void app_main(void)
{
    frame_queue = xQueueCreate(FRAME_Q_LEN, sizeof(jpeg_frame_t *));
    imu_queue = xQueueCreate(IMU_Q_LEN, sizeof(imu_pose_t));

    xTaskCreatePinnedToCore(camera_task, "camera", CAM_STACK, NULL, CAM_TASK_PRIO, NULL, 0);
    xTaskCreatePinnedToCore(imu_task, "imu", IMU_STACK, NULL, IMU_TASK_PRIO, NULL, 0);
    xTaskCreatePinnedToCore(tx_task, "tx", TX_STACK, NULL, TX_TASK_PRIO, NULL, 1);

    vTaskStartScheduler();

    for (;;) {
        vTaskDelay(portMAX_DELAY);
    }
}