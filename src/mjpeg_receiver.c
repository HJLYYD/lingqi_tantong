#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "mjpeg_receiver.h"
#include "logger.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

/*
 * MJPEG receiver — Python test-ov-imu.py logic + HTTP chunked decode
 *
 * ESP32's httpd_resp_send_chunk() uses Transfer-Encoding: chunked, so the
 * raw TCP stream contains hex chunk-size headers that must be stripped
 * before MJPEG parsing.  Python's requests library does this transparently;
 * we replicate it here with a two-buffer pipeline:
 *
 *   socket → chunk_buf →[chunked_decode]→ recv_buf →[mjpeg_parse_data]→ frame
 *
 * Python reference (after requests transparently de-chunks):
 *   buf = b''
 *   for chunk in stream.iter_content(4096):
 *       buf += chunk
 *       s = buf.find(b'--123456789000000000000987654321')
 *       if s == -1: continue
 *       frame = buf[:s]
 *       buf = buf[s + 67:]
 *       imu_start = frame.find(b'{"ax":')
 *       ...
 */

/* ── Fixed MJPEG boundary (hardcoded, same as Python) ── */
#define BOUNDARY_STR    "123456789000000000000987654321"
#define BOUNDARY_MARKER "--" BOUNDARY_STR
#define BOUNDARY_SKIP   67   /* Python: buf[s + 67:] */

/* ── HTTP chunked decoding buffer ── */
#define CHUNK_BUF_SIZE  65536  /* large enough for one max JPEG chunk + overhead */

struct MjpegReceiver {
    /* ── Connection ── */
    char   esp_ip[64];
    int    esp_port;
    char   wifi_ssid[64];
    char   wifi_password[64];
    int    sock_fd;
    bool   connected;

    int    reconnect_delay_s;
    time_t last_reconnect_attempt;

    /* ── Chunked-decoding staging buffer (raw TCP data with HTTP chunk framing) ── */
    uint8_t chunk_buf[CHUNK_BUF_SIZE];
    int     chunk_len;   /* bytes of raw chunked data waiting in chunk_buf */

    /* ── De-chunked MJPEG linear buffer (Python: buf = b'') ── */
    uint8_t recv_buf[MJPEG_RECV_BUF_SIZE];
    int     recv_head;   /* write position (= len(buf) in Python) */
    bool    http_done;   /* HTTP response headers stripped */

    /* ── Latest complete JPEG frame ── */
    uint8_t* jpeg_buf;
    size_t   jpeg_len;
    int      frame_index;
    double   frame_timestamp;

    /* ── Latest IMU data ── */
    IMUExternalPose latest_pose;
    bool   has_pose;
    float  yaw_accum;       /* integrated yaw from gyro Z (rad) */
    double last_imu_time;   /* CLOCK_MONOTONIC seconds of last IMU sample */
    int    imu_log_cnt;     /* per-instance IMU log counter */

    /* ── Threading ── */
    pthread_t       reader_thread;
    volatile bool   running;
    volatile bool   shutdown;
    pthread_mutex_t mutex;
};

/* ═══════════════════════════════════════════════════════════════════════
 *  WiFi auto-connect
 * ═══════════════════════════════════════════════════════════════════════ */

static bool wifi_connect(const char* ssid, const char* password) {
    if (!ssid || ssid[0] == '\0') {
        log_info("[MjpegRX] No WiFi SSID given, assuming user pre-connected");
        return true;
    }
    log_info("[MjpegRX] Auto-connecting to WiFi: %s", ssid);
    char cmd[512];
    if (password && password[0] != '\0') {
        snprintf(cmd, sizeof(cmd),
            "nmcli dev wifi connect \"%s\" password \"%s\" 2>/dev/null", ssid, password);
    } else {
        snprintf(cmd, sizeof(cmd),
            "nmcli dev wifi connect \"%s\" 2>/dev/null", ssid);
    }
    int rc = system(cmd);
    if (rc != 0) {
        log_warning("[MjpegRX] nmcli wifi connect returned %d", rc);
    }
    usleep(500000);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  TCP socket helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static int tcp_connect(const char* ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { log_error("[MjpegRX] socket() failed: %s", strerror(errno)); return -1; }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        log_error("[MjpegRX] inet_pton(%s) failed", ip);
        close(fd); return -1;
    }

    int rc = connect(fd, (struct sockaddr*)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
        log_error("[MjpegRX] connect(%s:%d) failed: %s", ip, port, strerror(errno));
        close(fd); return -1;
    }

    if (rc < 0 && errno == EINPROGRESS) {
        struct timeval tv = {3, 0};
        fd_set wfds; FD_ZERO(&wfds); FD_SET(fd, &wfds);
        rc = select(fd + 1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) {
            log_error("[MjpegRX] connect(%s:%d) timeout", ip, port);
            close(fd); return -1;
        }
        int err = 0; socklen_t len = sizeof(err);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) {
            log_error("[MjpegRX] connect async error: %s", err ? strerror(err) : "unknown");
            close(fd); return -1;
        }
    }

    fcntl(fd, F_SETFL, flags);

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    struct timeval tv = {1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    log_info("[MjpegRX] Connected to %s:%d", ip, port);
    return fd;
}

static void tcp_close(int* fd) {
    if (*fd >= 0) { shutdown(*fd, SHUT_RDWR); close(*fd); *fd = -1; }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  HTTP request
 * ═══════════════════════════════════════════════════════════════════════ */

static int http_send_request(int fd, const char* host, const char* path) {
    char req[512];
    snprintf(req, sizeof(req),
        "GET %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n", path, host);
    size_t len = strlen(req);
    ssize_t n = send(fd, req, len, MSG_NOSIGNAL);
    if (n != (ssize_t)len) {
        log_error("[MjpegRX] HTTP GET %s failed: %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Strip HTTP response headers from chunk_buf — find \r\n\r\n, keep
 *  everything after (i.e. the raw chunked-encoded body).
 *  Equivalent to Python's requests library consuming the HTTP headers.
 * ═══════════════════════════════════════════════════════════════════════ */

static void strip_http_headers(MjpegReceiver* r) {
    if (r->http_done) return;
    for (int i = 0; i < r->chunk_len - 3; i++) {
        if (r->chunk_buf[i] == '\r' && r->chunk_buf[i+1] == '\n' &&
            r->chunk_buf[i+2] == '\r' && r->chunk_buf[i+3] == '\n') {
            int body_start = i + 4;
            int body_len   = r->chunk_len - body_start;
            if (body_len > 0) {
                memmove(r->chunk_buf, r->chunk_buf + body_start, body_len);
            }
            r->chunk_len = body_len;
            r->http_done = true;
            log_info("[MjpegRX] HTTP headers stripped, chunked body starts at offset 0");
            return;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  HTTP chunked transfer-encoding decoder
 *
 *  Reads raw chunked data from chunk_buf, writes de-chunked payload to
 *  recv_buf.  Incomplete chunks stay in chunk_buf for the next recv().
 *
 *  Chunked format (RFC 7230 §4.1):
 *    <hex-size>\r\n<data>\r\n  ...  0\r\n\r\n
 * ═══════════════════════════════════════════════════════════════════════ */

static int chunked_decode(MjpegReceiver* r) {
    int pos = 0;
    int decoded_any = 0;

    while (pos < r->chunk_len) {
        int chunk_start = pos;

        /* ── Find \r\n ending the hex-size line ── */
        int nl = -1;
        for (int i = pos; i < r->chunk_len - 1; i++) {
            if (r->chunk_buf[i] == '\r' && r->chunk_buf[i+1] == '\n') {
                nl = i;
                break;
            }
        }
        if (nl < 0) {
            /* Incomplete hex-size line — keep in chunk_buf */
            pos = chunk_start;
            break;
        }

        /* ── Parse hex chunk size ── */
        int csize = 0;
        for (int i = pos; i < nl; i++) {
            char c = (char)r->chunk_buf[i];
            if      (c >= '0' && c <= '9') csize = (csize << 4) | (c - '0');
            else if (c >= 'a' && c <= 'f') csize = (csize << 4) | (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') csize = (csize << 4) | (c - 'A' + 10);
            else {
                log_warning("[MjpegRX] Bad hex in chunk size at offset %d: 0x%02x",
                           i, r->chunk_buf[i]);
                return -1;
            }
        }
        pos = nl + 2;  /* skip hex-size + \r\n */

        /* ── Last chunk (size == 0) ── */
        if (csize == 0) {
            /* Skip optional trailer \r\n */
            if (pos + 1 < r->chunk_len &&
                r->chunk_buf[pos] == '\r' && r->chunk_buf[pos+1] == '\n')
                pos += 2;
            /* Consume everything up to pos */
            if (pos > 0) {
                int remain = r->chunk_len - pos;
                if (remain > 0) memmove(r->chunk_buf, r->chunk_buf + pos, remain);
                r->chunk_len = remain;
            } else {
                r->chunk_len = 0;
            }
            log_debug("[MjpegRX] HTTP chunked stream ended (0-size chunk)");
            return decoded_any;
        }

        /* ── Need complete chunk data + trailing \r\n ── */
        if (pos + csize + 2 > r->chunk_len) {
            /* Incomplete — rewind to chunk_start, wait for more data */
            pos = chunk_start;
            break;
        }

        /* ── Append de-chunked data to recv_buf ── */
        if (r->recv_head + csize > MJPEG_RECV_BUF_SIZE) {
            log_warning("[MjpegRX] recv_buf overflow, flushing");
            r->recv_head = 0;
        }
        memcpy(r->recv_buf + r->recv_head, r->chunk_buf + pos, csize);
        r->recv_head += csize;
        decoded_any++;
        pos += csize + 2;  /* skip data + trailing \r\n */
    }

    /* ── Remove consumed data from chunk_buf ── */
    if (pos > 0) {
        int remain = r->chunk_len - pos;
        if (remain > 0) memmove(r->chunk_buf, r->chunk_buf + pos, remain);
        r->chunk_len = remain;
    }

    return decoded_any;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Frame parser — EXACT Python test-ov-imu.py logic
 *
 *  (recv_buf has already been de-chunked by chunked_decode() above)
 *
 *  Python:
 *    s = buf.find(b'--123456789000000000000987654321')
 *    if s == -1: continue
 *    frame = buf[:s]
 *    buf = buf[s + 67:]
 *
 *    imu_start = frame.find(b'{"ax":')
 *    imu_end = frame.find(b'}', imu_start) + 1
 *    if imu_start >= 0 and imu_end > 0: ...
 *
 *    jpg_s = frame.find(b'\xff\xd8')
 *    jpg_e = frame.find(b'\xff\xd9')
 *    if jpg_s != -1 and jpg_e != -1: ...
 * ═══════════════════════════════════════════════════════════════════════ */

static void mjpeg_parse_data(MjpegReceiver* r) {
    /* HTTP headers already stripped at chunk_buf level */
    /* ── Python: s = buf.find(b'--123456789000000000000987654321') ── */
    while (r->recv_head > 0) {
        int s = -1;
        int marker_len = strlen(BOUNDARY_MARKER);  /* = 32 */
        for (int i = 0; i <= r->recv_head - marker_len; i++) {
            if (memcmp(r->recv_buf + i, BOUNDARY_MARKER, marker_len) == 0) {
                s = i;
                break;
            }
        }

        /* Python: if s == -1: continue */
        if (s < 0) return;

        /* Python: frame = buf[:s] */
        int frame_len = s;  /* everything before boundary */
        uint8_t* frame_data = r->recv_buf;

        /* ── Only process non-empty frames ── */
        if (frame_len > 0) {
            /* ── Python: imu_start = frame.find(b'{"ax":') ── */
            int imu_start = -1;
            for (int i = 0; i <= frame_len - 6; i++) {
                if (memcmp(frame_data + i, "{\"ax\":", 6) == 0) {
                    imu_start = i;
                    break;
                }
            }

            /* Python: imu_end = frame.find(b'}', imu_start) + 1
             *         if imu_start >= 0 and imu_end > 0:                   */
            if (imu_start >= 0) {
                int imu_end = -1;
                for (int i = imu_start + 6; i < frame_len; i++) {
                    if (frame_data[i] == '}') {
                        imu_end = i + 1;
                        break;
                    }
                }

                if (imu_end > 0 && imu_end - imu_start < 256) {
                    char json_buf[256];
                    int json_len = imu_end - imu_start;
                    memcpy(json_buf, frame_data + imu_start, json_len);
                    json_buf[json_len] = '\0';

                    /* Python: imu = json.loads(imu_str)
                     *         imu['ax'], imu['ay'], imu['az'],
                     *         imu['gx'], imu['gy'], imu['gz']              */
                    float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
                    if (sscanf(json_buf,
                        "{\"ax\":%f,\"ay\":%f,\"az\":%f,"
                        "\"gx\":%f,\"gy\":%f,\"gz\":%f}",
                        &ax, &ay, &az, &gx, &gy, &gz) == 6) {

                        /* ── Compute tilt from accelerometer (gravity direction) ── */
                        float roll_rad  = atan2f(ay, az);
                        float pitch_rad = atan2f(-ax, sqrtf(ay*ay + az*az));

                        /* ── Integrate gyro Z for yaw (CLOCK_MONOTONIC for µs dt) ── */
                        struct timespec imu_ts;
                        clock_gettime(CLOCK_MONOTONIC, &imu_ts);
                        double now_s = (double)imu_ts.tv_sec
                                     + (double)imu_ts.tv_nsec * 1e-9;
                        double dt = (r->last_imu_time > 0.0)
                                  ? (now_s - r->last_imu_time) : 0.04;
                        /* Clamp dt to reasonable range (avoid spikes after pause) */
                        if (dt > 0.001 && dt < 1.0) {
                            r->yaw_accum += gz * (float)dt;
                        }
                        r->last_imu_time = now_s;

                        /* ── Euler to quaternion (ZYX order: yaw→pitch→roll) ── */
                        float cr = cosf(roll_rad * 0.5f), sr = sinf(roll_rad * 0.5f);
                        float cp = cosf(pitch_rad * 0.5f), sp = sinf(pitch_rad * 0.5f);
                        float cy = cosf(r->yaw_accum * 0.5f), sy = sinf(r->yaw_accum * 0.5f);
                        float qw = cr*cp*cy + sr*sp*sy;
                        float qx = sr*cp*cy - cr*sp*sy;
                        float qy = cr*sp*cy + sr*cp*sy;
                        float qz = cr*cp*sy - sr*sp*cy;

                        /* monotonic timestamp_ms without overflow */
                        uint64_t ts_ms = (uint64_t)imu_ts.tv_sec * 1000ULL
                                       + (uint64_t)imu_ts.tv_nsec / 1000000ULL;

                        pthread_mutex_lock(&r->mutex);
                        r->latest_pose.pitch = pitch_rad;
                        r->latest_pose.roll  = roll_rad;
                        r->latest_pose.yaw   = r->yaw_accum;
                        r->latest_pose.qw = qw;
                        r->latest_pose.qx = qx;
                        r->latest_pose.qy = qy;
                        r->latest_pose.qz = qz;
                        r->latest_pose.altitude_m = 0.0f;  /* not available from accel alone */
                        r->latest_pose.temperature_c = 0.0f;
                        r->latest_pose.timestamp_ms = (uint32_t)ts_ms;
                        r->latest_pose.is_valid = true;
                        r->has_pose = true;
                        pthread_mutex_unlock(&r->mutex);

                        /* Log IMU at INFO level every 30 frames */
                        if (++r->imu_log_cnt % 30 == 0) {
                            log_info("[MjpegRX] IMU#%d | accel: ax=%.4f ay=%.4f az=%.4f m/s² | "
                                     "gyro: gx=%.4f gy=%.4f gz=%.4f rad/s | "
                                     "pose: roll=%.2f° pitch=%.2f° yaw=%.1f° | q=(%.3f,%.3f,%.3f,%.3f)",
                                     r->imu_log_cnt, ax, ay, az, gx, gy, gz,
                                     roll_rad * 57.3f, pitch_rad * 57.3f, r->yaw_accum * 57.3f,
                                     qw, qx, qy, qz);
                        }
                    }
                }
            }

            /* ── Python: jpg_s = frame.find(b'\xff\xd8')
             *         jpg_e = frame.find(b'\xff\xd9')
             *         if jpg_s != -1 and jpg_e != -1: jpg = frame[jpg_s:jpg_e+2] ── */
            int jpg_s = -1, jpg_e = -1;
            for (int i = 0; i < frame_len - 1; i++) {
                if (frame_data[i] == 0xff && frame_data[i+1] == 0xd8) {
                    jpg_s = i;
                    break;
                }
            }
            if (jpg_s >= 0) {
                for (int i = jpg_s + 2; i < frame_len - 1; i++) {
                    if (frame_data[i] == 0xff && frame_data[i+1] == 0xd9) {
                        jpg_e = i + 2;
                        break;
                    }
                }
            }

            if (jpg_s >= 0 && jpg_e > jpg_s &&
                jpg_e - jpg_s >= 100 &&
                jpg_e - jpg_s < MJPEG_MAX_FRAME_LEN) {
                int jpg_len = jpg_e - jpg_s;
                pthread_mutex_lock(&r->mutex);
                free(r->jpeg_buf);
                r->jpeg_buf = (uint8_t*)malloc(jpg_len);
                if (r->jpeg_buf) {
                    memcpy(r->jpeg_buf, frame_data + jpg_s, jpg_len);
                    r->jpeg_len = jpg_len;
                    r->frame_index++;
                    /* Store timestamp in microseconds (was seconds, lost precision) */
                    struct timespec ts;
                    clock_gettime(CLOCK_MONOTONIC, &ts);
                    r->frame_timestamp = (double)ts.tv_sec * 1e6
                                       + (double)ts.tv_nsec / 1e3;
                    if (r->frame_index == 1) {
                        log_info("[MjpegRX] First JPEG frame: %d bytes", jpg_len);
                    } else if (r->frame_index % 30 == 0) {
                        log_info("[MjpegRX] Frame #%d: %d bytes JPEG | "
                                 "recv_buf=%d/%d chunk_buf=%d/%d",
                                 r->frame_index, jpg_len,
                                 r->recv_head, MJPEG_RECV_BUF_SIZE,
                                 r->chunk_len, CHUNK_BUF_SIZE);
                    }
                }
                pthread_mutex_unlock(&r->mutex);
            }
        }

        /* ── Python: buf = buf[s + 67:] ── */
        int skip = s + BOUNDARY_SKIP;
        if (skip >= r->recv_head) {
            /* Not enough data to skip — boundary at very end of buffer */
            r->recv_head = 0;
            return;
        }
        int remaining = r->recv_head - skip;
        if (remaining > 0) {
            memmove(r->recv_buf, r->recv_buf + skip, remaining);
        }
        r->recv_head = remaining;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Background reader thread
 * ═══════════════════════════════════════════════════════════════════════ */

static bool mjpeg_connect_and_stream(MjpegReceiver* r) {
    tcp_close(&r->sock_fd);

    int fd = tcp_connect(r->esp_ip, r->esp_port);
    if (fd < 0) return false;

    if (http_send_request(fd, r->esp_ip, "/") != 0) {
        tcp_close(&fd);
        return false;
    }

    r->sock_fd = fd;
    r->connected = true;
    r->reconnect_delay_s = MJPEG_RECONNECT_INITIAL_S;
    r->recv_head = 0;
    r->chunk_len = 0;
    r->http_done = false;
    r->yaw_accum = 0.0f;
    r->last_imu_time = 0.0;
    r->imu_log_cnt = 0;
    log_info("[MjpegRX] Streaming MJPEG from %s:%d/ (Python test-ov-imu.py logic)",
             r->esp_ip, r->esp_port);
    return true;
}

static void* mjpeg_reader_thread(void* arg) {
    MjpegReceiver* r = (MjpegReceiver*)arg;
    int recv_total = 0;

    log_info("[MjpegRX] Reader thread started, connecting to %s:%d",
             r->esp_ip, r->esp_port);

    if (!mjpeg_connect_and_stream(r)) {
        log_warning("[MjpegRX] Initial connection failed, will retry");
    }

    while (r->running) {
        if (!r->connected) {
            time_t now = time(NULL);
            int wait = r->reconnect_delay_s;
            if (wait < MJPEG_RECONNECT_INITIAL_S) wait = MJPEG_RECONNECT_INITIAL_S;
            if (wait > MJPEG_RECONNECT_MAX_S) wait = MJPEG_RECONNECT_MAX_S;

            if (r->last_reconnect_attempt != 0 &&
                now < r->last_reconnect_attempt + wait) {
                usleep(500000);
                continue;
            }
            r->last_reconnect_attempt = now;

            log_info("[MjpegRX] Reconnect attempt to %s:%d (backoff=%ds)",
                     r->esp_ip, r->esp_port, wait);

            if (mjpeg_connect_and_stream(r)) {
                recv_total = 0;
            } else {
                r->reconnect_delay_s *= 2;
                if (r->reconnect_delay_s > MJPEG_RECONNECT_MAX_S)
                    r->reconnect_delay_s = MJPEG_RECONNECT_MAX_S;
            }
            continue;
        }

        /* ── Step 1: recv raw TCP data into chunk_buf ── */
        int space = CHUNK_BUF_SIZE - r->chunk_len - 1;
        if (space <= 0) {
            log_warning("[MjpegRX] chunk_buf full (%d bytes), resetting", r->chunk_len);
            r->chunk_len = 0;
            space = CHUNK_BUF_SIZE - 1;
        }

        ssize_t n = recv(r->sock_fd, r->chunk_buf + r->chunk_len, space, 0);
        if (n > 0) {
            recv_total += (int)n;
            r->chunk_len += (int)n;

            /* ── Step 2: strip HTTP response headers once ── */
            if (!r->http_done) {
                strip_http_headers(r);
                if (!r->http_done) continue;  /* need more data for headers */
            }

            /* ── Step 3: de-chunk chunk_buf → recv_buf ── */
            int decoded = chunked_decode(r);
            if (decoded < 0) {
                log_warning("[MjpegRX] Chunked decode error, resetting");
                r->connected = false;
                tcp_close(&r->sock_fd);
                r->recv_head = 0;
                r->chunk_len = 0;
                r->http_done = false;
                continue;
            }

            /* ── Step 4: parse MJPEG frames from de-chunked recv_buf ── */
            if (r->recv_head > 0) {
                mjpeg_parse_data(r);
            }
        } else if (n == 0) {
            log_warning("[MjpegRX] ESP32 closed connection (EOF) after %d bytes",
                        recv_total);
            r->connected = false;
            tcp_close(&r->sock_fd);
        } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                log_warning("[MjpegRX] recv error: %s", strerror(errno));
                r->connected = false;
                tcp_close(&r->sock_fd);
            }
        }
    }

    log_info("[MjpegRX] Reader thread exiting, %d bytes total", recv_total);
    tcp_close(&r->sock_fd);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Public API
 * ═══════════════════════════════════════════════════════════════════════ */

MjpegReceiver* mjpeg_receiver_create(const char* esp_ip, int esp_port,
                                     const char* wifi_ssid,
                                     const char* wifi_password) {
    if (!esp_ip || esp_ip[0] == '\0' || esp_port <= 0) return NULL;

    MjpegReceiver* r = (MjpegReceiver*)calloc(1, sizeof(MjpegReceiver));
    if (!r) return NULL;

    strncpy(r->esp_ip, esp_ip, sizeof(r->esp_ip) - 1);
    r->esp_port = esp_port;
    if (wifi_ssid) strncpy(r->wifi_ssid, wifi_ssid, sizeof(r->wifi_ssid) - 1);
    if (wifi_password) strncpy(r->wifi_password, wifi_password, sizeof(r->wifi_password) - 1);

    r->sock_fd = -1;
    r->connected = false;
    r->reconnect_delay_s = MJPEG_RECONNECT_INITIAL_S;
    r->has_pose = false;
    r->recv_head = 0;
    r->http_done = false;

    if (pthread_mutex_init(&r->mutex, NULL) != 0) { free(r); return NULL; }

    wifi_connect(r->wifi_ssid, r->wifi_password);

    r->running = true;
    r->shutdown = false;
    if (pthread_create(&r->reader_thread, NULL, mjpeg_reader_thread, r) != 0) {
        pthread_mutex_destroy(&r->mutex);
        free(r);
        return NULL;
    }

    log_info("[MjpegRX] Created: ESP32 at %s:%d", esp_ip, esp_port);
    return r;
}

void mjpeg_receiver_destroy(MjpegReceiver* r) {
    if (!r) return;
    r->running = false;
    r->shutdown = true;
    if (r->sock_fd >= 0) shutdown(r->sock_fd, SHUT_RDWR);
    pthread_join(r->reader_thread, NULL);
    tcp_close(&r->sock_fd);
    free(r->jpeg_buf);
    pthread_mutex_destroy(&r->mutex);
    free(r);
}

void mjpeg_receiver_update(MjpegReceiver* r) {
    (void)r;
}

bool mjpeg_receiver_get_latest_frame(MjpegReceiver* r, ArrowSourceFrame* out) {
    if (!r || !out) return false;

    bool has_data = false;
    pthread_mutex_lock(&r->mutex);

    if (r->jpeg_buf && r->jpeg_len > 0 && r->jpeg_len < MJPEG_MAX_FRAME_LEN) {
        memcpy(out->jpeg_data, r->jpeg_buf, r->jpeg_len);
        out->jpeg_len = r->jpeg_len;
        out->frame_index = r->frame_index;
        out->timestamp = r->frame_timestamp;
        has_data = true;
        free(r->jpeg_buf);
        r->jpeg_buf = NULL;
        r->jpeg_len = 0;
    }

    if (r->has_pose) {
        out->pose = r->latest_pose;
        out->has_pose = true;
        r->has_pose = false;
    } else {
        out->has_pose = false;
    }

    out->is_valid = has_data;
    pthread_mutex_unlock(&r->mutex);
    return has_data;
}

bool mjpeg_receiver_is_connected(MjpegReceiver* r) {
    return r ? r->connected : false;
}
