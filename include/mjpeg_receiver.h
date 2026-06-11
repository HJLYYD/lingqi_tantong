#ifndef MJPEG_RECEIVER_H
#define MJPEG_RECEIVER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MjpegReceiver — WiFi HTTP MJPEG stream receiver for ESP32-Camera.
 *
 * Connects to an ESP32 running test-ov-imu.ino which serves:
 *   GET /     → multipart/x-mixed-replace MJPEG stream (JPEG frames)
 *   GET /imu  → JSON {"accel":[...], "gyro":[...]} (optional IMU data)
 *
 * The API mirrors ArrowReceiver so it can be used as a drop-in
 * alternative frame source in SystemController.
 *
 *   MjpegReceiver* r = mjpeg_receiver_create("192.168.4.1", 80, NULL, NULL);
 *   while (running) {
 *       mjpeg_receiver_update(r);
 *       ArrowSourceFrame frame;
 *       if (mjpeg_receiver_get_latest_frame(r, &frame)) {
 *           soft_jpeg_decode_to_rgb(frame.jpeg_data, frame.jpeg_len, ...);
 *       }
 *   }
 *   mjpeg_receiver_destroy(r);
 */

#define MJPEG_RECV_BUF_SIZE      131072   /* 128 KB TCP receive buffer */
#define MJPEG_MAX_FRAME_LEN      ARROW_MAX_FRAME_LEN  /* reuse: 65536 */
#define MJPEG_BOUNDARY_MAX       128
#define MJPEG_IMU_POLL_INTERVAL_MS 100
#define MJPEG_RECONNECT_INITIAL_S  1
#define MJPEG_RECONNECT_MAX_S      16

typedef struct MjpegReceiver MjpegReceiver;

/*
 * Create an MJPEG receiver.
 *
 *   esp_ip         — ESP32 IP address (e.g. "192.168.4.1")
 *   esp_port       — HTTP port (usually 80)
 *   wifi_ssid      — optional: WiFi SSID to auto-connect (may be NULL)
 *   wifi_password  — optional: WiFi password (may be NULL)
 *
 * WiFi auto-connect uses `nmcli` (NetworkManager). If NULL, assumes
 * WiFi is already configured externally.
 *
 * Returns NULL on failure (logs reason).
 */
MjpegReceiver* mjpeg_receiver_create(const char* esp_ip, int esp_port,
                                     const char* wifi_ssid,
                                     const char* wifi_password);

void mjpeg_receiver_destroy(MjpegReceiver* receiver);

/*
 * Non-blocking update: read available data from the TCP socket and
 * process any complete MJPEG parts.  Call this once per frame loop
 * iteration.  Safe to call even when disconnected (handles reconnect).
 */
void mjpeg_receiver_update(MjpegReceiver* receiver);

/*
 * Get the latest fully-received JPEG frame.
 *
 * Copies JPEG data into out_frame->jpeg_data (up to MJPEG_MAX_FRAME_LEN),
 * sets out_frame->jpeg_len, frame_index, timestamp.
 *
 * Also copies the latest IMU pose if available (out_frame->pose, has_pose).
 *
 * Returns true if a new frame was available, false otherwise.
 * The frame data is consumed — calling again returns false until
 * the next frame arrives.
 */
bool mjpeg_receiver_get_latest_frame(MjpegReceiver* receiver,
                                     ArrowSourceFrame* out_frame);

/* Returns true if TCP connection to ESP32 is currently established. */
bool mjpeg_receiver_is_connected(MjpegReceiver* receiver);

#ifdef __cplusplus
}
#endif

#endif /* MJPEG_RECEIVER_H */
