#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include "camera_capture.h"
#include "imu_fusion.h"

#define PROTO_MAGIC_START_0   0xA5
#define PROTO_MAGIC_START_1   0x5A
#define PROTO_MAGIC_END_0     0x5A
#define PROTO_MAGIC_END_1     0xA5

#define PROTO_TYPE_JPEG       0x01
#define PROTO_TYPE_IMU_POSE   0x02
#define PROTO_TYPE_IMU_RAW    0x03
#define PROTO_TYPE_HEARTBEAT  0x04
#define PROTO_TYPE_CTRL_ACK   0x05

#define PROTO_HEADER_SIZE     11
#define PROTO_MAGIC_SIZE      2
#define PROTO_CRC_SIZE        2
#define PROTO_OVERHEAD        (PROTO_HEADER_SIZE + PROTO_CRC_SIZE + PROTO_MAGIC_SIZE)

typedef struct {
    uint8_t  magic_start[2];
    uint8_t  type;
    uint16_t seq;
    uint32_t timestamp;
    uint16_t length;
} __attribute__((packed)) packet_header_t;

uint16_t crc16_ccitt(const uint8_t* data, uint16_t len);

uint16_t protocol_pack_jpeg(uint8_t* buf, uint16_t seq, uint32_t ts, const uint8_t* jpeg, uint16_t len);
uint16_t protocol_pack_imu_pose(uint8_t* buf, uint16_t seq, uint32_t ts, float qw, float qx, float qy, float qz, float altitude, float temp);
uint16_t protocol_pack_imu_raw(uint8_t* buf, uint16_t seq, uint32_t ts, float* accel, float* gyro, float* mag);
uint16_t protocol_pack_heartbeat(uint8_t* buf, uint16_t seq, uint32_t ts, float battery_v, float temp, uint8_t fps, uint8_t errors);

#endif