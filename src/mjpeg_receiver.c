#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "mjpeg_receiver.h"
#include "logger.h"
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
 * MJPEG stream parser state machine.
 *
 * An MJPEG over HTTP stream looks like:
 *
 *   HTTP/1.0 200 OK\r\n
 *   Content-Type: multipart/x-mixed-replace;boundary=123456789000000000000987654321\r\n
 *   \r\n
 *   --123456789000000000000987654321\r\n
 *   Content-Type: image/jpeg\r\n
 *   Content-Length: 12345\r\n
 *   \r\n
 *   <12345 bytes of JPEG data>
 *   --123456789000000000000987654321\r\n
 *   ...
 */

typedef enum {
    MJPEG_PARSE_HTTP_HEADER = 0,  /* reading HTTP response line + headers */
    MJPEG_PARSE_FIND_BOUNDARY,    /* scanning for next boundary marker   */
    MJPEG_PARSE_PART_HEADERS,     /* reading Content-Type/Length headers */
    MJPEG_PARSE_PART_BODY,        /* reading JPEG payload bytes          */
    MJPEG_PARSE_PART_DONE,        /* complete frame ready to consume     */
} MjpegParseState;

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

    /* ── IMU polling ── */
    bool   imu_enabled;
    int    imu_sock_fd;
    time_t last_imu_poll;

    /* ── Receive buffer ── */
    uint8_t recv_buf[MJPEG_RECV_BUF_SIZE];
    int     recv_head;    /* write position in recv_buf */
    int     recv_tail;    /* read position (parse cursor) */

    /* ── MJPEG parser state ── */
    MjpegParseState parse_state;
    char    boundary[MJPEG_BOUNDARY_MAX];
    int     boundary_len;
    int     boundary_match_pos;   /* chars matched so far (FIND_BOUNDARY) */
    int     content_length;       /* expected JPEG payload size           */
    int     content_read;         /* bytes read of current JPEG payload   */
    bool    headers_done;         /* HTTP response headers fully parsed   */

    /* ── Part body accumulation buffer ── */
    uint8_t* part_body_buf;   /* accumulates partial JPEG across recv() calls */

    /* ── Latest complete JPEG frame ── */
    uint8_t* jpeg_buf;
    size_t   jpeg_len;
    int      frame_index;
    double   frame_timestamp;

    /* ── Latest IMU data ── */
    IMUExternalPose latest_pose;
    bool   has_pose;

    /* ── Threading ── */
    pthread_t       reader_thread;
    volatile bool   running;
    volatile bool   shutdown;
    pthread_mutex_t mutex;
};

/* ═══════════════════════════════════════════════════════════════════════
 *  WiFi auto-connect (only when --wifi-ssid is explicitly passed)
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
            "nmcli dev wifi connect \"%s\" password \"%s\" 2>/dev/null",
            ssid, password);
    } else {
        snprintf(cmd, sizeof(cmd),
            "nmcli dev wifi connect \"%s\" 2>/dev/null", ssid);
    }

    int rc = system(cmd);
    if (rc != 0) {
        log_warning("[MjpegRX] nmcli wifi connect returned %d — "
                     "WiFi may already be connected or unavailable", rc);
    }
    usleep(500000);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  TCP socket helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static int tcp_connect(const char* ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        log_error("[MjpegRX] socket() failed: %s", strerror(errno));
        return -1;
    }

    /* Set non-blocking for connect (we'll switch back after) */
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        log_error("[MjpegRX] inet_pton(%s) failed", ip);
        close(fd);
        return -1;
    }

    int rc = connect(fd, (struct sockaddr*)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
        log_error("[MjpegRX] connect(%s:%d) failed: %s", ip, port, strerror(errno));
        close(fd);
        return -1;
    }

    /* Wait for connect to complete (with timeout) */
    if (rc < 0 && errno == EINPROGRESS) {
        struct timeval tv = {3, 0};  /* 3 second connect timeout */
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);
        rc = select(fd + 1, NULL, &wfds, NULL, &tv);
        if (rc <= 0) {
            log_error("[MjpegRX] connect(%s:%d) timeout", ip, port);
            close(fd);
            return -1;
        }
        int err = 0;
        socklen_t len = sizeof(err);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) {
            log_error("[MjpegRX] connect(%s:%d) async error: %s",
                      ip, port, err ? strerror(err) : "unknown");
            close(fd);
            return -1;
        }
    }

    /* Restore blocking mode */
    fcntl(fd, F_SETFL, flags);

    /* TCP_NODELAY for low-latency streaming */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    /* Set receive timeout to 1 second so we can check shutdown flag */
    struct timeval tv = {1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    log_info("[MjpegRX] Connected to %s:%d", ip, port);
    return fd;
}

static void tcp_close(int* fd) {
    if (*fd >= 0) {
        shutdown(*fd, SHUT_RDWR);
        close(*fd);
        *fd = -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  HTTP request / response
 * ═══════════════════════════════════════════════════════════════════════ */

static int http_send_request(int fd, const char* host, const char* path) {
    char req[512];
    snprintf(req, sizeof(req),
        "GET %s HTTP/1.0\r\n"
        "Host: %s\r\n"
        "Connection: close\r\n"
        "\r\n",
        path, host);
    size_t len = strlen(req);
    ssize_t n = send(fd, req, len, MSG_NOSIGNAL);
    if (n != (ssize_t)len) {
        log_error("[MjpegRX] HTTP GET %s failed: %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

/*
 * Scan recv_buf for the boundary string.
 * Returns number of bytes to skip to be positioned right after the boundary
 * (including trailing \r\n if present), or -1 if boundary not found.
 */
static int find_boundary_in_buf(const uint8_t* buf, int start, int end,
                                 const char* boundary, int blen) {
    for (int i = start; i <= end - blen; i++) {
        if (memcmp(buf + i, boundary, blen) == 0) {
            /* Skip past boundary + optional "\r\n" */
            int skip = i + blen;
            if (skip + 1 < end && buf[skip] == '\r' && buf[skip+1] == '\n')
                skip += 2;
            return skip;
        }
    }
    return -1;
}

/*
 * Extract Content-Length value from a line like "Content-Length: 12345\r\n".
 * Returns the integer value, or -1 if not found.
 */
static int parse_content_length(const uint8_t* buf, int start, int end) {
    /* Find "Content-Length:" */
    const char* marker = "Content-Length:";
    int mlen = 15;
    for (int i = start; i <= end - mlen; i++) {
        if (memcmp(buf + i, marker, mlen) == 0) {
            /* Skip optional space */
            int j = i + mlen;
            while (j < end && buf[j] == ' ') j++;
            /* Parse digits */
            int val = 0;
            while (j < end && buf[j] >= '0' && buf[j] <= '9') {
                val = val * 10 + (buf[j] - '0');
                j++;
            }
            if (val > 0 && val < MJPEG_MAX_FRAME_LEN) return val;
        }
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  MJPEG stream parser (called from reader thread)
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Process available data in recv_buf[recv_tail .. recv_head].
 * Advances recv_tail as data is consumed.
 * When a complete JPEG frame is extracted, stores it in receiver->jpeg_buf.
 */
static void mjpeg_parse_data(MjpegReceiver* r) {
    static int parse_call_count = 0;
    parse_call_count++;
    int buf_avail = r->recv_head - r->recv_tail;

    while (r->recv_tail < r->recv_head) {
        switch (r->parse_state) {

        case MJPEG_PARSE_HTTP_HEADER: {
            /* Scan for the double-CRLF that ends HTTP headers */
            const uint8_t* p = r->recv_buf + r->recv_tail;
            int remaining = r->recv_head - r->recv_tail;
            int found = -1;
            for (int i = 0; i < remaining - 3; i++) {
                if (p[i] == '\r' && p[i+1] == '\n' &&
                    p[i+2] == '\r' && p[i+3] == '\n') {
                    found = i + 4;
                    break;
                }
            }
            if (found < 0) return;  /* need more data */

            /* Extract boundary from Content-Type header */
            const char* bmarker = "boundary=";
            int bmlen = 9;
            for (int i = 0; i < found - bmlen; i++) {
                if (memcmp(p + i, bmarker, bmlen) == 0) {
                    int j = i + bmlen;
                    int blen = 0;
                    while (j + blen < found && p[j+blen] != '\r' &&
                           p[j+blen] != '\n' && p[j+blen] != ';' &&
                           blen < MJPEG_BOUNDARY_MAX - 1) {
                        blen++;
                    }
                    if (blen > 0) {
                        memcpy(r->boundary, p + j, blen);
                        r->boundary[blen] = '\0';
                        r->boundary_len = blen;
                        log_info("[MjpegRX] MJPEG boundary: %s", r->boundary);
                    }
                    break;
                }
            }

            if (r->boundary_len == 0) {
                log_error("[MjpegRX] No boundary found in HTTP response");
                /* Skip all data — connection will be reset */
                r->recv_tail = r->recv_head;
                return;
            }

            r->recv_tail += found;
            r->parse_state = MJPEG_PARSE_FIND_BOUNDARY;
            r->boundary_match_pos = 0;
            log_info("[MjpegRX] HTTP headers parsed, searching for first boundary");
            break;
        }

        case MJPEG_PARSE_FIND_BOUNDARY: {
            /* Search for "--<boundary>" in the buffer */
            char search[MJPEG_BOUNDARY_MAX + 4];
            int slen = snprintf(search, sizeof(search), "--%s", r->boundary);
            int skip = find_boundary_in_buf(r->recv_buf, r->recv_tail,
                                             r->recv_head, search, slen);
            if (skip < 0) {
                /* No boundary found — keep last 128 bytes */
                int avail = r->recv_head - r->recv_tail;
                int keep = 128;
                if (avail > keep) {
                    r->recv_tail = r->recv_head - keep;
                }
                return;
            }
            r->recv_tail = skip;
            r->parse_state = MJPEG_PARSE_PART_HEADERS;
            r->content_length = -1;
            r->content_read = 0;
            log_debug("[MjpegRX] Found boundary, reading part headers");
            break;
        }

        case MJPEG_PARSE_PART_HEADERS: {
            /* Look for "\r\n\r\n" marking end of part headers */
            const uint8_t* p = r->recv_buf + r->recv_tail;
            int remaining = r->recv_head - r->recv_tail;
            int hdr_end = -1;
            for (int i = 0; i < remaining - 3; i++) {
                if (p[i] == '\r' && p[i+1] == '\n' &&
                    p[i+2] == '\r' && p[i+3] == '\n') {
                    hdr_end = i;
                    break;
                }
            }
            if (hdr_end < 0) return;  /* need more data */

            /* Parse Content-Length */
            r->content_length = parse_content_length(p, 0, hdr_end);

            /* Skip past the double-CRLF */
            r->recv_tail += hdr_end + 4;

            if (r->content_length > 0 && r->content_length < MJPEG_MAX_FRAME_LEN) {
                r->parse_state = MJPEG_PARSE_PART_BODY;
                r->content_read = 0;
                log_debug("[MjpegRX] Part headers done, Content-Length=%d, entering PART_BODY",
                          r->content_length);
            } else {
                log_debug("[MjpegRX] Part has no valid Content-Length (%d), skipping to next boundary",
                          r->content_length);
                r->parse_state = MJPEG_PARSE_FIND_BOUNDARY;
            }
            break;
        }

        case MJPEG_PARSE_PART_BODY: {
            int available = r->recv_head - r->recv_tail;
            int needed = r->content_length - r->content_read;
            int take = (available < needed) ? available : needed;

            /* Allocate accumulation buffer on first entry */
            if (!r->part_body_buf && r->content_length > 0) {
                r->part_body_buf = (uint8_t*)malloc(r->content_length);
            }

            if (take > 0 && r->part_body_buf) {
                /* Copy into accumulation buffer (safe across recv() calls
                 * because compact_buffer resets the ring buffer) */
                memcpy(r->part_body_buf + r->content_read,
                       r->recv_buf + r->recv_tail, take);
                r->content_read += take;
                r->recv_tail += take;
            }

            if (r->content_read >= r->content_length) {
                /* Complete JPEG frame received — transfer ownership */
                pthread_mutex_lock(&r->mutex);
                free(r->jpeg_buf);
                r->jpeg_buf = r->part_body_buf;   /* transfer, don't copy */
                r->part_body_buf = NULL;
                r->jpeg_len = r->content_length;
                r->frame_index++;
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                r->frame_timestamp = (double)ts.tv_sec
                                   + (double)ts.tv_nsec / 1e9;
                pthread_mutex_unlock(&r->mutex);

                if (r->frame_index == 1) {
                    log_info("[MjpegRX] First JPEG frame extracted: %d bytes",
                             (int)r->jpeg_len);
                } else if (r->frame_index % 30 == 0) {
                    log_debug("[MjpegRX] Frame #%d: %d bytes",
                              r->frame_index, (int)r->jpeg_len);
                }
                r->parse_state = MJPEG_PARSE_FIND_BOUNDARY;
            }
            break;
        }

        case MJPEG_PARSE_PART_DONE:
        default:
            r->parse_state = MJPEG_PARSE_FIND_BOUNDARY;
            break;
        }
    }
}

/* Discard buffered data that's already been parsed (shift remaining to front).
 * Must NOT be called while in PART_BODY state — partial JPEG data is
 * accumulated in part_body_buf, not the ring buffer. */
static void mjpeg_compact_buffer(MjpegReceiver* r) {
    if (r->parse_state == MJPEG_PARSE_PART_BODY) return;
    if (r->recv_tail > 0 && r->recv_tail < r->recv_head) {
        int remaining = r->recv_head - r->recv_tail;
        memmove(r->recv_buf, r->recv_buf + r->recv_tail, remaining);
        r->recv_head = remaining;
        r->recv_tail = 0;
    } else if (r->recv_tail >= r->recv_head) {
        r->recv_head = 0;
        r->recv_tail = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  IMU data poller
 * ═══════════════════════════════════════════════════════════════════════ */

static void imu_poll(MjpegReceiver* r) {
    time_t now = time(NULL);
    if (now - r->last_imu_poll < (MJPEG_IMU_POLL_INTERVAL_MS + 999) / 1000) return;
    r->last_imu_poll = now;

    if (r->imu_sock_fd < 0) {
        r->imu_sock_fd = tcp_connect(r->esp_ip, r->esp_port);
        if (r->imu_sock_fd < 0) return;
        if (http_send_request(r->imu_sock_fd, r->esp_ip, "/imu") != 0) {
            tcp_close(&r->imu_sock_fd);
            return;
        }
    }

    char buf[512];
    ssize_t n = recv(r->imu_sock_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            tcp_close(&r->imu_sock_fd);
        }
        return;
    }
    buf[n] = '\0';

    /* Parse JSON: {"accel":[ax,ay,az],"gyro":[gx,gy,gz]} */
    float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
    int parsed = sscanf(buf,
        "{\"accel\":[%f,%f,%f],\"gyro\":[%f,%f,%f]}",
        &ax, &ay, &az, &gx, &gy, &gz);
    if (parsed == 6) {
        pthread_mutex_lock(&r->mutex);
        /* Store IMU data as an external pose.
         * Raw accel/gyro values — no quaternion available from this ESP32. */
        r->latest_pose.pitch = ay;   /* approximate tilt */
        r->latest_pose.roll  = ax;
        r->latest_pose.yaw   = gz;
        r->latest_pose.qw = 1.0f;
        r->latest_pose.qx = 0.0f;
        r->latest_pose.qy = 0.0f;
        r->latest_pose.qz = 0.0f;
        r->latest_pose.altitude_m = az;
        r->latest_pose.temperature_c = 0.0f;
        r->latest_pose.timestamp_ms = (uint32_t)(now * 1000);
        r->latest_pose.is_valid = true;
        r->has_pose = true;
        pthread_mutex_unlock(&r->mutex);
    }

    /* Close IMU connection after each poll — re-open next cycle */
    tcp_close(&r->imu_sock_fd);
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Background reader thread
 * ═══════════════════════════════════════════════════════════════════════ */

static void mjpeg_reset_parser(MjpegReceiver* r) {
    r->parse_state = MJPEG_PARSE_HTTP_HEADER;
    r->boundary_len = 0;
    r->boundary_match_pos = 0;
    r->content_length = -1;
    r->content_read = 0;
    r->recv_head = 0;
    r->recv_tail = 0;
    memset(r->boundary, 0, sizeof(r->boundary));
    free(r->part_body_buf);
    r->part_body_buf = NULL;
}

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
    mjpeg_reset_parser(r);
    log_info("[MjpegRX] Streaming MJPEG from %s:%d/", r->esp_ip, r->esp_port);
    return true;
}

static void* mjpeg_reader_thread(void* arg) {
    MjpegReceiver* r = (MjpegReceiver*)arg;
    int loop_count = 0, recv_total = 0;

    log_info("[MjpegRX] Reader thread started, connecting to %s:%d",
             r->esp_ip, r->esp_port);

    if (!mjpeg_connect_and_stream(r)) {
        log_warning("[MjpegRX] Initial connection failed, will retry");
    }

    while (r->running) {
        loop_count++;
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

            log_info("[MjpegRX] Reconnect attempt #%d to %s:%d (backoff=%ds)",
                     loop_count, r->esp_ip, r->esp_port, wait);

            if (mjpeg_connect_and_stream(r)) {
                recv_total = 0;
            } else {
                r->reconnect_delay_s *= 2;
                if (r->reconnect_delay_s > MJPEG_RECONNECT_MAX_S)
                    r->reconnect_delay_s = MJPEG_RECONNECT_MAX_S;
            }
            continue;
        }

        /* Read from socket into recv_buf */
        mjpeg_compact_buffer(r);
        int space = MJPEG_RECV_BUF_SIZE - r->recv_head - 1;
        if (space <= 0) {
            log_warning("[MjpegRX] Recv buffer full (head=%d tail=%d), resetting",
                        r->recv_head, r->recv_tail);
            r->connected = false;
            tcp_close(&r->sock_fd);
            mjpeg_reset_parser(r);
            continue;
        }

        ssize_t n = recv(r->sock_fd, r->recv_buf + r->recv_head, space, 0);
        if (n > 0) {
            recv_total += (int)n;
            r->recv_head += (int)n;
            if (loop_count == 1 || loop_count % 100 == 0) {
                log_debug("[MjpegRX] Loop#%d recv=%d total=%d parse_state=%d head=%d tail=%d",
                          loop_count, (int)n, recv_total,
                          (int)r->parse_state, r->recv_head, r->recv_tail);
            }
            mjpeg_parse_data(r);
        } else if (n == 0) {
            log_warning("[MjpegRX] ESP32 closed connection (EOF) after %d loops, %d bytes",
                        loop_count, recv_total);
            r->connected = false;
            tcp_close(&r->sock_fd);
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (loop_count <= 3) {
                    log_debug("[MjpegRX] Loop#%d recv timeout (normal)", loop_count);
                }
            } else {
                log_warning("[MjpegRX] recv error at loop#%d: %s (errno=%d)",
                            loop_count, strerror(errno), errno);
                r->connected = false;
                tcp_close(&r->sock_fd);
            }
        }

        /* Poll IMU data */
        if (r->imu_enabled && r->connected) {
            imu_poll(r);
        }
    }

    log_info("[MjpegRX] Reader thread exiting after %d loops, %d bytes total",
             loop_count, recv_total);
    tcp_close(&r->sock_fd);
    tcp_close(&r->imu_sock_fd);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Public API
 * ═══════════════════════════════════════════════════════════════════════ */

MjpegReceiver* mjpeg_receiver_create(const char* esp_ip, int esp_port,
                                     const char* wifi_ssid,
                                     const char* wifi_password) {
    if (!esp_ip || esp_ip[0] == '\0' || esp_port <= 0) {
        log_error("[MjpegRX] Invalid parameters: ip=%s port=%d",
                  esp_ip ? esp_ip : "(null)", esp_port);
        return NULL;
    }

    MjpegReceiver* r = (MjpegReceiver*)calloc(1, sizeof(MjpegReceiver));
    if (!r) return NULL;

    strncpy(r->esp_ip, esp_ip, sizeof(r->esp_ip) - 1);
    r->esp_port = esp_port;

    if (wifi_ssid) strncpy(r->wifi_ssid, wifi_ssid, sizeof(r->wifi_ssid) - 1);
    if (wifi_password) strncpy(r->wifi_password, wifi_password, sizeof(r->wifi_password) - 1);

    r->sock_fd = -1;
    r->imu_sock_fd = -1;
    r->imu_enabled = true;
    r->connected = false;
    r->reconnect_delay_s = MJPEG_RECONNECT_INITIAL_S;
    r->last_reconnect_attempt = 0;
    r->has_pose = false;

    mjpeg_reset_parser(r);

    if (pthread_mutex_init(&r->mutex, NULL) != 0) {
        free(r);
        return NULL;
    }

    /* WiFi auto-connect (no-op if ssid is NULL) */
    wifi_connect(r->wifi_ssid, r->wifi_password);

    /* Start background reader thread */
    r->running = true;
    r->shutdown = false;
    if (pthread_create(&r->reader_thread, NULL, mjpeg_reader_thread, r) != 0) {
        log_error("[MjpegRX] Failed to create reader thread");
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

    /* Wake up reader thread by shutting down socket */
    if (r->sock_fd >= 0) {
        shutdown(r->sock_fd, SHUT_RDWR);
    }

    pthread_join(r->reader_thread, NULL);

    tcp_close(&r->sock_fd);
    tcp_close(&r->imu_sock_fd);
    free(r->jpeg_buf);
    free(r->part_body_buf);
    pthread_mutex_destroy(&r->mutex);
    free(r);
}

void mjpeg_receiver_update(MjpegReceiver* r) {
    /* The reader thread handles all I/O continuously.
     * This is a no-op — kept for API compatibility with arrow_receiver. */
    (void)r;
}

bool mjpeg_receiver_get_latest_frame(MjpegReceiver* r, ArrowSourceFrame* out) {
    if (!r || !out) return false;

    bool has_data = false;
    static int get_call_count = 0, get_ok_count = 0;
    get_call_count++;

    pthread_mutex_lock(&r->mutex);

    if (r->jpeg_buf && r->jpeg_len > 0 && r->jpeg_len < MJPEG_MAX_FRAME_LEN) {
        memcpy(out->jpeg_data, r->jpeg_buf, r->jpeg_len);
        out->jpeg_len = r->jpeg_len;
        out->frame_index = r->frame_index;
        out->timestamp = r->frame_timestamp;
        has_data = true;
        get_ok_count++;

        if (r->frame_index <= 3 || r->frame_index % 50 == 0) {
            log_info("[MjpegRX] get_latest_frame #%d: frame_idx=%d size=%zu (ok=%d/%d calls)",
                     get_call_count, r->frame_index, r->jpeg_len,
                     get_ok_count, get_call_count);
        }

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
