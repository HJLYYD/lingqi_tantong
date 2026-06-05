#define _DEFAULT_SOURCE

#include "arrow_receiver.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ARROW_STATE_IDLE        0
#define ARROW_STATE_HEADER      1
#define ARROW_STATE_PAYLOAD     2
#define ARROW_STATE_CRC         3
#define ARROW_STATE_END_MAGIC   4
#define ARROW_STATE_COMPLETE    5

#define ARROW_CRC_POLY          0x1021
#define ARROW_CRC_INIT          0xFFFF

#define ARROW_RECONNECT_INITIAL_S   1
#define ARROW_RECONNECT_MAX_S       16

struct ArrowReceiver {
    int uart_fd;
    int uart_fd_secondary;
    char uart_path[256];
    char uart_path_secondary[256];
    int baud_rate;
    bool connected;
    bool dual_link_active;

    int reconnect_delay_s;
    time_t last_reconnect_attempt;

    uint8_t rx_buffer[ARROW_UART_BUFFER_SIZE];
    int rx_head;
    int rx_tail;

    int state;
    uint8_t header_buf[9];
    int header_pos;

    uint8_t* payload_buf;
    uint32_t payload_len;
    uint32_t payload_pos;

    uint8_t frame_type;
    uint32_t frame_seq;
    uint32_t frame_timestamp;

    uint8_t* jpeg_buf;
    size_t jpeg_len;
    int frame_index;
    IMUExternalPose latest_pose;
    bool has_pose;

    pthread_mutex_t mutex;
};

static uint16_t crc16_ccitt_update(uint16_t crc, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ ARROW_CRC_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

static int uart_configure(int fd, int baud_rate) {
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        return -1;
    }

    speed_t speed;
    switch (baud_rate) {
        case 3000000: speed = B3000000; break;
        case 921600:  speed = B921600;  break;
        case 460800:  speed = B460800;  break;
        case 230400:  speed = B230400;  break;
        case 115200:  speed = B115200;  break;
        default:      speed = B115200;  break;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        return -1;
    }

    return 0;
}

static int uart_open(const char* path, int baud_rate) {
    int fd = open(path, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        return -1;
    }

    if (uart_configure(fd, baud_rate) != 0) {
        close(fd);
        return -1;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags & ~O_NDELAY);

    return fd;
}

static void uart_close(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}

static int ring_buffer_available(const ArrowReceiver* receiver) {
    if (receiver->rx_head >= receiver->rx_tail) {
        return receiver->rx_head - receiver->rx_tail;
    }
    return ARROW_UART_BUFFER_SIZE - receiver->rx_tail + receiver->rx_head;
}

static int ring_buffer_free(const ArrowReceiver* receiver) {
    return ARROW_UART_BUFFER_SIZE - 1 - ring_buffer_available(receiver);
}

static void ring_buffer_write(ArrowReceiver* receiver, uint8_t byte) {
    receiver->rx_buffer[receiver->rx_head] = byte;
    receiver->rx_head = (receiver->rx_head + 1) % ARROW_UART_BUFFER_SIZE;
}

static int ring_buffer_read(ArrowReceiver* receiver, uint8_t* byte) {
    if (ring_buffer_available(receiver) == 0) {
        return -1;
    }
    *byte = receiver->rx_buffer[receiver->rx_tail];
    receiver->rx_tail = (receiver->rx_tail + 1) % ARROW_UART_BUFFER_SIZE;
    return 0;
}

static int uart_read_into_buffer(ArrowReceiver* receiver) {
    uint8_t tmp[256];
    int n = read(receiver->uart_fd, tmp, sizeof(tmp));
    if (n <= 0) {
        return n;
    }
    for (int i = 0; i < n; i++) {
        if (ring_buffer_free(receiver) > 0) {
            ring_buffer_write(receiver, tmp[i]);
        }
    }
    return n;
}

static void quaternion_to_euler(float qw, float qx, float qy, float qz,
                                float* pitch, float* roll, float* yaw) {
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    *roll = (float)(atan2(sinr_cosp, cosr_cosp) * 180.0 / M_PI);

    double sinp = 2.0 * (qw * qy - qz * qx);
    if (sinp > 1.0) {
        *pitch = 90.0f;
    } else if (sinp < -1.0) {
        *pitch = -90.0f;
    } else {
        *pitch = (float)(asin(sinp) * 180.0 / M_PI);
    }

    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    *yaw = (float)(atan2(siny_cosp, cosy_cosp) * 180.0 / M_PI);
}

static void dispatch_frame(ArrowReceiver* receiver) {
    receiver->state = ARROW_STATE_IDLE;

    if (receiver->frame_type == ARROW_TYPE_JPEG) {
        pthread_mutex_lock(&receiver->mutex);
        free(receiver->jpeg_buf);
        receiver->jpeg_buf = (uint8_t*)malloc(receiver->payload_len);
        receiver->jpeg_len = 0;
        if (receiver->jpeg_buf) {
            memcpy(receiver->jpeg_buf, receiver->payload_buf, receiver->payload_len);
            receiver->jpeg_len = receiver->payload_len;
            receiver->frame_index = (int)receiver->frame_seq;
        }
        pthread_mutex_unlock(&receiver->mutex);
    } else if (receiver->frame_type == ARROW_TYPE_IMU_POSE) {
        if (receiver->payload_len >= 24) {
            float qw, qx, qy, qz, altitude, temp;
            memcpy(&qw, receiver->payload_buf, 4);
            memcpy(&qx, receiver->payload_buf + 4, 4);
            memcpy(&qy, receiver->payload_buf + 8, 4);
            memcpy(&qz, receiver->payload_buf + 12, 4);
            memcpy(&altitude, receiver->payload_buf + 16, 4);
            memcpy(&temp, receiver->payload_buf + 20, 4);

            float pitch, roll, yaw;
            quaternion_to_euler(qw, qx, qy, qz, &pitch, &roll, &yaw);

            pthread_mutex_lock(&receiver->mutex);
            receiver->latest_pose.qw = qw;
            receiver->latest_pose.qx = qx;
            receiver->latest_pose.qy = qy;
            receiver->latest_pose.qz = qz;
            receiver->latest_pose.pitch = pitch;
            receiver->latest_pose.roll = roll;
            receiver->latest_pose.yaw = yaw;
            receiver->latest_pose.altitude_m = altitude;
            receiver->latest_pose.temperature_c = temp;
            receiver->latest_pose.timestamp_ms = receiver->frame_timestamp;
            receiver->has_pose = true;
            pthread_mutex_unlock(&receiver->mutex);
        }
    }

    free(receiver->payload_buf);
    receiver->payload_buf = NULL;
    receiver->payload_len = 0;
    receiver->payload_pos = 0;
    receiver->header_pos = 0;
}

static void process_byte(ArrowReceiver* receiver, uint8_t byte) {
    switch (receiver->state) {
    case ARROW_STATE_IDLE:
        if (receiver->header_pos == 0 && byte == ARROW_FRAME_MAGIC_1) {
            receiver->header_pos = 1;
        } else if (receiver->header_pos == 1 && byte == ARROW_FRAME_MAGIC_2) {
            receiver->header_pos = 0;
            receiver->state = ARROW_STATE_HEADER;
            receiver->header_pos = 0;
        } else {
            receiver->header_pos = 0;
        }
        break;

    case ARROW_STATE_HEADER:
        receiver->header_buf[receiver->header_pos++] = byte;
        if (receiver->header_pos >= 9) {
            receiver->frame_type = receiver->header_buf[0];
            receiver->frame_seq = ((uint32_t)receiver->header_buf[1] << 8) |
                                  ((uint32_t)receiver->header_buf[2]);
            receiver->frame_timestamp = ((uint32_t)receiver->header_buf[3] << 24) |
                                        ((uint32_t)receiver->header_buf[4] << 16) |
                                        ((uint32_t)receiver->header_buf[5] << 8) |
                                        ((uint32_t)receiver->header_buf[6]);
            receiver->payload_len = ((uint32_t)receiver->header_buf[7] << 8) |
                                    ((uint32_t)receiver->header_buf[8]);
            receiver->header_pos = 0;

            if (receiver->payload_len > 0 && receiver->payload_len <= (1024 * 1024)) {
                receiver->payload_buf = (uint8_t*)malloc(receiver->payload_len);
                if (receiver->payload_buf) {
                    receiver->payload_pos = 0;
                    receiver->state = ARROW_STATE_PAYLOAD;
                } else {
                    receiver->state = ARROW_STATE_IDLE;
                }
            } else {
                receiver->state = ARROW_STATE_IDLE;
            }
        }
        break;

    case ARROW_STATE_PAYLOAD:
        receiver->payload_buf[receiver->payload_pos++] = byte;
        if (receiver->payload_pos >= receiver->payload_len) {
            receiver->state = ARROW_STATE_CRC;
            receiver->header_pos = 0;
        }
        break;

    case ARROW_STATE_CRC: {
        receiver->header_buf[receiver->header_pos++] = byte;
        if (receiver->header_pos >= 2) {
            uint16_t received_crc = ((uint16_t)receiver->header_buf[0] << 8) |
                                    ((uint16_t)receiver->header_buf[1]);

            uint16_t computed_crc;

            uint8_t hdr[11];
            hdr[0] = ARROW_FRAME_MAGIC_1;
            hdr[1] = ARROW_FRAME_MAGIC_2;
            hdr[2] = receiver->frame_type;
            hdr[3] = (uint8_t)(receiver->frame_seq >> 8);
            hdr[4] = (uint8_t)(receiver->frame_seq);
            hdr[5] = (uint8_t)(receiver->frame_timestamp >> 24);
            hdr[6] = (uint8_t)(receiver->frame_timestamp >> 16);
            hdr[7] = (uint8_t)(receiver->frame_timestamp >> 8);
            hdr[8] = (uint8_t)(receiver->frame_timestamp);
            hdr[9] = (uint8_t)(receiver->payload_len >> 8);
            hdr[10] = (uint8_t)(receiver->payload_len);

            computed_crc = crc16_ccitt_update(ARROW_CRC_INIT, hdr, 11);
            computed_crc = crc16_ccitt_update(computed_crc, receiver->payload_buf, receiver->payload_len);

            receiver->header_pos = 0;
            if (received_crc == computed_crc) {
                receiver->state = ARROW_STATE_END_MAGIC;
            } else {
                free(receiver->payload_buf);
                receiver->payload_buf = NULL;
                receiver->payload_len = 0;
                receiver->payload_pos = 0;
                receiver->state = ARROW_STATE_IDLE;
            }
        }
        break;
    }

    case ARROW_STATE_END_MAGIC:
        receiver->header_buf[receiver->header_pos++] = byte;
        if (receiver->header_pos >= 2) {
            if (receiver->header_buf[0] == ARROW_FRAME_MAGIC_2 &&
                receiver->header_buf[1] == ARROW_FRAME_MAGIC_1) {
                receiver->state = ARROW_STATE_COMPLETE;
            } else {
                free(receiver->payload_buf);
                receiver->payload_buf = NULL;
                receiver->payload_len = 0;
                receiver->payload_pos = 0;
                receiver->state = ARROW_STATE_IDLE;
            }
        }
        break;

    case ARROW_STATE_COMPLETE:
        dispatch_frame(receiver);
        break;
    }
}

ArrowReceiver* arrow_receiver_create(const char* uart_dev_path, int baud_rate) {
    ArrowReceiver* receiver = (ArrowReceiver*)calloc(1, sizeof(ArrowReceiver));
    if (!receiver) return NULL;

    strncpy(receiver->uart_path, uart_dev_path, sizeof(receiver->uart_path) - 1);
    receiver->baud_rate = baud_rate;
    receiver->state = ARROW_STATE_IDLE;
    receiver->rx_head = 0;
    receiver->rx_tail = 0;
    receiver->has_pose = false;

    if (pthread_mutex_init(&receiver->mutex, NULL) != 0) {
        free(receiver);
        return NULL;
    }

    receiver->uart_fd = -1;
    receiver->uart_fd_secondary = -1;

    for (int retry = 0; retry < ARROW_UART_RETRY_COUNT; retry++) {
        receiver->uart_fd = uart_open(uart_dev_path, baud_rate);
        if (receiver->uart_fd >= 0) break;
        log_warning("UART open attempt %d/%d failed for %s, retrying...",
                    retry + 1, ARROW_UART_RETRY_COUNT, uart_dev_path);
        usleep(ARROW_UART_RETRY_DELAY_US);
    }

    if (receiver->uart_fd < 0) {
        log_error("Failed to open UART device after %d attempts: %s", ARROW_UART_RETRY_COUNT, uart_dev_path);
        pthread_mutex_destroy(&receiver->mutex);
        free(receiver);
        return NULL;
    }

    receiver->connected = true;
    receiver->dual_link_active = false;
    receiver->reconnect_delay_s = ARROW_RECONNECT_INITIAL_S;
    receiver->last_reconnect_attempt = 0;
    log_info("ArrowReceiver created on %s at %d baud", uart_dev_path, baud_rate);
    return receiver;
}

static void arrow_uart_reset_state(ArrowReceiver* receiver) {
    receiver->state = ARROW_STATE_IDLE;
    receiver->header_pos = 0;
    receiver->payload_pos = 0;
    receiver->payload_len = 0;
    if (receiver->payload_buf) {
        free(receiver->payload_buf);
        receiver->payload_buf = NULL;
    }
    receiver->rx_head = 0;
    receiver->rx_tail = 0;
}

static bool arrow_uart_try_reconnect(ArrowReceiver* receiver) {
    time_t now = time(NULL);
    int wait = receiver->reconnect_delay_s;
    if (wait < ARROW_RECONNECT_INITIAL_S) wait = ARROW_RECONNECT_INITIAL_S;
    if (wait > ARROW_RECONNECT_MAX_S) wait = ARROW_RECONNECT_MAX_S;
    if (receiver->last_reconnect_attempt != 0 &&
        now < receiver->last_reconnect_attempt + wait) {
        return false;
    }
    receiver->last_reconnect_attempt = now;

    if (receiver->uart_fd >= 0) {
        close(receiver->uart_fd);
        receiver->uart_fd = -1;
    }

    int fd = uart_open(receiver->uart_path, receiver->baud_rate);
    if (fd < 0) {
        log_warning("Arrow UART reconnect attempt to %s failed (backoff=%ds)",
                    receiver->uart_path, wait);
        receiver->reconnect_delay_s *= 2;
        if (receiver->reconnect_delay_s > ARROW_RECONNECT_MAX_S)
            receiver->reconnect_delay_s = ARROW_RECONNECT_MAX_S;
        return false;
    }

    tcflush(fd, TCIOFLUSH);
    receiver->uart_fd = fd;
    receiver->connected = true;
    receiver->reconnect_delay_s = ARROW_RECONNECT_INITIAL_S;
    arrow_uart_reset_state(receiver);
    log_info("Arrow UART reconnected to %s at %d baud", receiver->uart_path, receiver->baud_rate);
    return true;
}

void arrow_receiver_destroy(ArrowReceiver* receiver) {
    if (!receiver) return;

    uart_close(receiver->uart_fd);
    if (receiver->uart_fd_secondary >= 0) {
        uart_close(receiver->uart_fd_secondary);
    }
    free(receiver->jpeg_buf);
    free(receiver->payload_buf);
    pthread_mutex_destroy(&receiver->mutex);
    free(receiver);
}

ARFramePair arrow_receiver_get_pair(ArrowReceiver* receiver) {
    ARFramePair pair;
    memset(&pair, 0, sizeof(pair));

    if (!receiver) return pair;

    int n = uart_read_into_buffer(receiver);
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        receiver->connected = false;
    }

    uint8_t byte;
    while (ring_buffer_read(receiver, &byte) == 0) {
        process_byte(receiver, byte);
    }

    pthread_mutex_lock(&receiver->mutex);

    if (receiver->jpeg_buf && receiver->jpeg_len > 0) {
        pair.jpeg_buf = (uint8_t*)malloc(receiver->jpeg_len);
        if (pair.jpeg_buf) {
            memcpy(pair.jpeg_buf, receiver->jpeg_buf, receiver->jpeg_len);
            pair.jpeg_len = receiver->jpeg_len;
        }
    }

    pair.frame_index = receiver->frame_index;

    if (receiver->has_pose) {
        pair.pose = receiver->latest_pose;
        pair.has_pose = true;
    }

    pthread_mutex_unlock(&receiver->mutex);

    return pair;
}

bool arrow_receiver_is_connected(ArrowReceiver* receiver) {
    return receiver ? receiver->connected : false;
}

void arrow_receiver_update(ArrowReceiver* receiver) {
    if (!receiver) return;
    if (!receiver->connected) {
        arrow_uart_try_reconnect(receiver);
        if (!receiver->connected) return;
    }

    int n = uart_read_into_buffer(receiver);
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        receiver->connected = false;
        log_warning("Arrow UART disconnected: %s (errno=%d %s)",
                    receiver->uart_path, errno, strerror(errno));
        arrow_uart_reset_state(receiver);
        return;
    }
    if (n == 0) {
        receiver->connected = false;
        log_warning("Arrow UART EOF on %s (TX end closed)", receiver->uart_path);
        arrow_uart_reset_state(receiver);
        return;
    }

    if (receiver->dual_link_active && receiver->uart_fd_secondary >= 0) {
        uint8_t tmp[256];
        int n2 = read(receiver->uart_fd_secondary, tmp, sizeof(tmp));
        if (n2 > 0) {
            for (int i = 0; i < n2; i++) {
                if (ring_buffer_free(receiver) > 0) {
                    ring_buffer_write(receiver, tmp[i]);
                }
            }
        } else if (n2 < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            log_warning("Arrow secondary UART error on %s, disabling dual-link",
                        receiver->uart_path_secondary);
            close(receiver->uart_fd_secondary);
            receiver->uart_fd_secondary = -1;
            receiver->dual_link_active = false;
        }
    }

    uint8_t byte;
    while (ring_buffer_read(receiver, &byte) == 0) {
        process_byte(receiver, byte);
    }
}

bool arrow_receiver_get_latest_frame(ArrowReceiver* receiver, ArrowSourceFrame* out_frame) {
    if (!receiver || !out_frame) return false;

    bool has_data = false;

    pthread_mutex_lock(&receiver->mutex);

    if (receiver->jpeg_buf && receiver->jpeg_len > 0 && receiver->jpeg_len < ARROW_MAX_FRAME_LEN) {
        memcpy(out_frame->jpeg_data, receiver->jpeg_buf, receiver->jpeg_len);
        out_frame->jpeg_len = receiver->jpeg_len;
        out_frame->frame_index = receiver->frame_index;
        out_frame->timestamp = (double)receiver->frame_timestamp / 1000.0;
        has_data = true;

        free(receiver->jpeg_buf);
        receiver->jpeg_buf = NULL;
        receiver->jpeg_len = 0;
    }

    if (receiver->has_pose) {
        out_frame->pose = receiver->latest_pose;
        out_frame->has_pose = true;
        receiver->has_pose = false;
    } else {
        out_frame->has_pose = false;
    }

    out_frame->is_valid = has_data;

    pthread_mutex_unlock(&receiver->mutex);

    return has_data;
}

int arrow_receiver_add_secondary_link(ArrowReceiver* receiver, const char* uart_dev_path, int baud_rate) {
    if (!receiver || !uart_dev_path) return -1;

    int fd = uart_open(uart_dev_path, baud_rate);
    if (fd < 0) {
        log_error("Failed to open secondary UART: %s", uart_dev_path);
        return -1;
    }

    receiver->uart_fd_secondary = fd;
    strncpy(receiver->uart_path_secondary, uart_dev_path, sizeof(receiver->uart_path_secondary) - 1);
    receiver->dual_link_active = true;

    log_info("Secondary UART link active: %s @ %d bps (dual-link: frame on %s, ctrl on %s)",
             uart_dev_path, baud_rate, receiver->uart_path, uart_dev_path);
    return 0;
}