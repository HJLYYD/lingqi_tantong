#ifndef COAP_RECEIVER_H
#define COAP_RECEIVER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CoapReceiver — K1 端 CoAP/UDP 接收器
 *
 * 对应 ESP32 端 coap_server.h + ov-imu-pwm.ino 的新协议:
 *   - CoAP/UDP 端口 5683
 *   - /stream 端点: Block2 分块传输纯 JPEG 帧 (CoAP NON)
 *   - /imu 端点:   独立 IMU JSON 轮询 (1Hz)
 *   - /servo 端点: 舵机控制 PUT (预留)
 *
 * 输出统一为 ArrowSourceFrame，由 SystemController 采集线程消费。
 */

#define COAP_RECV_BUF_SIZE      65536   /* UDP 接收缓冲 */
#define COAP_MAX_PACKET         1400
#define COAP_DEFAULT_SZX        6
#define COAP_BLOCK_SIZE(szx)    ((size_t)16 << (szx))
#define COAP_RECONNECT_INITIAL_S 1
#define COAP_RECONNECT_MAX_S     16

/* ── CoAP 协议常量 ── */
#define COAP_TYPE_CON   0
#define COAP_TYPE_NON   1
#define COAP_TYPE_ACK   2

#define COAP_CODE_GET       0x01
#define COAP_CODE_CONTENT   0x45

#define COAP_OPT_ETAG       4
#define COAP_OPT_URI_PATH   11
#define COAP_OPT_CONTENT_FMT 12
#define COAP_OPT_BLOCK2     23
#define COAP_OPT_SIZE2      28

#define COAP_FMT_JSON   50
#define COAP_FMT_JPEG   22

/* ── Stream token (与 Python 客户端 ov-imu-pwm.py 一致) ── */
#define COAP_STREAM_TOKEN  0xCAFEBABE
#define COAP_IMU_TOKEN     0xDEADBEEF

typedef struct CoapReceiver CoapReceiver;

/* 原始 IMU 数据回调: 当收到 /imu 的原始 accel/gyro JSON 时调用 */
typedef void (*CoapImuRawCallback)(void* user, const float accel[3], const float gyro[3]);

CoapReceiver* coap_receiver_create(const char* esp_ip, int esp_port,
                                   const char* wifi_ssid,
                                   const char* wifi_password);
void coap_receiver_destroy(CoapReceiver* receiver);
void coap_receiver_update(CoapReceiver* receiver);
bool coap_receiver_get_latest_frame(CoapReceiver* receiver,
                                    ArrowSourceFrame* out_frame);
bool coap_receiver_is_connected(CoapReceiver* receiver);
void coap_receiver_set_imu_raw_callback(CoapReceiver* receiver,
                                         CoapImuRawCallback cb, void* user);

#ifdef __cplusplus
}
#endif

#endif /* COAP_RECEIVER_H */
