#include "protocol.h"
#include <string.h>

// Arrow Protocol: Mixed endian — Big-endian for headers/CRC, Little-endian for float payloads.
static void write_u16_be(uint8_t* buf, uint16_t val)
{
    buf[0] = (val >> 8) & 0xFF;
    buf[1] = val & 0xFF;
}

static void write_u32_be(uint8_t* buf, uint32_t val)
{
    buf[0] = (val >> 24) & 0xFF;
    buf[1] = (val >> 16) & 0xFF;
    buf[2] = (val >> 8) & 0xFF;
    buf[3] = val & 0xFF;
}

static void write_float_le(uint8_t* buf, float val)
{
    uint32_t raw;
    memcpy(&raw, &val, 4);
    buf[0] = raw & 0xFF;
    buf[1] = (raw >> 8) & 0xFF;
    buf[2] = (raw >> 16) & 0xFF;
    buf[3] = (raw >> 24) & 0xFF;
}

uint16_t crc16_ccitt(const uint8_t* data, uint16_t len)
{
    uint16_t crc = 0xFFFF;
    uint16_t i;
    uint8_t j;

    for (i = 0; i < len; i++) {
        crc ^= (uint16_t)(data[i] << 8);
        for (j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}

static uint16_t write_header(uint8_t* buf, uint8_t type, uint16_t seq, uint32_t timestamp, uint16_t data_len)
{
    buf[0] = PROTO_MAGIC_START_0;
    buf[1] = PROTO_MAGIC_START_1;
    buf[2] = type;
    write_u16_be(buf + 3, seq);
    write_u32_be(buf + 5, timestamp);
    write_u16_be(buf + 9, data_len);

    return PROTO_HEADER_SIZE;
}

static void write_crc_and_tail(uint8_t* buf, uint16_t total_before_crc, uint16_t* out_total)
{
    uint16_t crc = crc16_ccitt(buf, total_before_crc);
    write_u16_be(buf + total_before_crc, crc);
    buf[total_before_crc + 2] = PROTO_MAGIC_END_0;
    buf[total_before_crc + 3] = PROTO_MAGIC_END_1;
    *out_total = total_before_crc + PROTO_CRC_SIZE + PROTO_MAGIC_SIZE;
}

uint16_t protocol_pack_jpeg(uint8_t* buf, uint16_t seq, uint32_t ts, const uint8_t* jpeg, uint16_t len)
{
    uint16_t offset;
    uint16_t total;

    offset = write_header(buf, PROTO_TYPE_JPEG, seq, ts, len);
    memcpy(buf + offset, jpeg, len);
    offset += len;

    write_crc_and_tail(buf, offset, &total);
    return total;
}

uint16_t protocol_pack_imu_pose(uint8_t* buf, uint16_t seq, uint32_t ts, float qw, float qx, float qy, float qz, float altitude, float temp)
{
    uint16_t offset;
    uint16_t total;
    uint16_t data_len = 24;

    offset = write_header(buf, PROTO_TYPE_IMU_POSE, seq, ts, data_len);
    write_float_le(buf + offset, qw);
    write_float_le(buf + offset + 4, qx);
    write_float_le(buf + offset + 8, qy);
    write_float_le(buf + offset + 12, qz);
    write_float_le(buf + offset + 16, altitude);
    write_float_le(buf + offset + 20, temp);
    offset += data_len;

    write_crc_and_tail(buf, offset, &total);
    return total;
}

uint16_t protocol_pack_imu_raw(uint8_t* buf, uint16_t seq, uint32_t ts, float* accel, float* gyro, float* mag)
{
    uint16_t offset;
    uint16_t total;
    uint16_t data_len = 36;

    offset = write_header(buf, PROTO_TYPE_IMU_RAW, seq, ts, data_len);
    write_float_le(buf + offset, accel[0]);
    write_float_le(buf + offset + 4, accel[1]);
    write_float_le(buf + offset + 8, accel[2]);
    write_float_le(buf + offset + 12, gyro[0]);
    write_float_le(buf + offset + 16, gyro[1]);
    write_float_le(buf + offset + 20, gyro[2]);
    write_float_le(buf + offset + 24, mag[0]);
    write_float_le(buf + offset + 28, mag[1]);
    write_float_le(buf + offset + 32, mag[2]);
    offset += data_len;

    write_crc_and_tail(buf, offset, &total);
    return total;
}

uint16_t protocol_pack_heartbeat(uint8_t* buf, uint16_t seq, uint32_t ts, float battery_v, float temp, uint8_t fps, uint8_t errors)
{
    uint16_t offset;
    uint16_t total;
    uint16_t data_len = 10;

    offset = write_header(buf, PROTO_TYPE_HEARTBEAT, seq, ts, data_len);
    write_float_le(buf + offset, battery_v);
    write_float_le(buf + offset + 4, temp);
    buf[offset + 8] = fps;
    buf[offset + 9] = errors;
    offset += data_len;

    write_crc_and_tail(buf, offset, &total);
    return total;
}