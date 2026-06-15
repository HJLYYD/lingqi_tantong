#ifndef MJPEG_RECEIVER_H
#define MJPEG_RECEIVER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MjpegReceiver — EXACT 1:1 translation of Python test-ov-imu.py
 *
 * Python reference logic:
 *   buf = b''
 *   for chunk in stream.iter_content(4096):
 *       buf += chunk
 *       s = buf.find(b'--123456789000000000000987654321')
 *       if s == -1: continue
 *       frame = buf[:s]
 *       buf = buf[s + 67:]
 *       # extract IMU: frame.find(b'{"ax":') ... frame.find(b'}')
 *       # extract JPEG: frame.find(b'\xff\xd8') ... frame.find(b'\xff\xd9')
 *
 * API mirrors ArrowReceiver for drop-in compatibility with SystemController.
 */

#define MJPEG_RECV_BUF_SIZE      131072   /* 128 KB linear buffer */
#define MJPEG_MAX_FRAME_LEN      ARROW_MAX_FRAME_LEN  /* 65536 */
#define MJPEG_BOUNDARY_MAX       128
#define MJPEG_RECONNECT_INITIAL_S  1
#define MJPEG_RECONNECT_MAX_S      16

typedef struct MjpegReceiver MjpegReceiver;

MjpegReceiver* mjpeg_receiver_create(const char* esp_ip, int esp_port,
                                     const char* wifi_ssid,
                                     const char* wifi_password);
void mjpeg_receiver_destroy(MjpegReceiver* receiver);
void mjpeg_receiver_update(MjpegReceiver* receiver);
bool mjpeg_receiver_get_latest_frame(MjpegReceiver* receiver,
                                     ArrowSourceFrame* out_frame);
bool mjpeg_receiver_is_connected(MjpegReceiver* receiver);

#ifdef __cplusplus
}
#endif

#endif /* MJPEG_RECEIVER_H */
