/*
 * CoapReceiver — K1 端 CoAP/UDP 接收器
 *
 * 对应 ESP32 新协议 (coap_server.h + ov-imu-pwm.ino):
 *   - UDP 端口 5683
 *   - /stream → Block2 分块 JPEG 流 (CoAP NON, 纯 JPEG)
 *   - /imu    → 独立 JSON 轮询 (1Hz)
 *
 * 协议参考: ov-imu-pwm.py (Python 测试客户端)
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "coap_receiver.h"
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

/* ═══════════════════════════════════════════════════════════════════════
 *  Block2 帧重组缓冲
 *
 *  v2 协议变更 (ov-imu-pwm(1).py JpegStreamAssembler):
 *   - 支持乱序块 (block_num 不要求严格递增)
 *   - ETag 在所有块中都存在 (不仅 block 0)
 *   - Size2 仅在 block 0 中 (可选, 用于预知总大小)
 *   - 重组: 当收到 more=false 的块时, 检查是否收集齐所有块
 * ═══════════════════════════════════════════════════════════════════════ */

#define FRAME_BUF_CAPACITY      (128 * 1024)   /* 128KB，足够 VGA JPEG */
#define MAX_BLOCKS_PER_FRAME    128             /* 128KB / 1KB = 128 */
#define BLOCK_MAP_WORDS         4               /* 128 bits / 32 */
#define IMU_POLL_INTERVAL_MS    1000
#define STREAM_KEEPALIVE_S      8               /* 每 8 秒发送 keepalive */
#define STREAM_RESUB_INTERVAL_S 25              /* 25 秒无数据则重订阅 */
#define FRAME_TIMEOUT_MS        4000            /* 帧重组超时 (Python: 4s) */

typedef struct {
    uint32_t etag;               /* 当前帧 ETag (所有块相同) */
    uint32_t total_size;         /* Size2 总大小 (0 = 未知) */
    uint8_t  szx;                /* 块大小指数 */
    uint32_t total_blocks;       /* 期望总块数 (0 = 未知, 最后一块到达时确定) */
    uint32_t blocks_received;    /* 已收到块的总数 */
    uint32_t block_map[BLOCK_MAP_WORDS]; /* bitmap 标记哪些块号已收到 */
    uint8_t* data;               /* 重组缓冲区 */
    size_t   capacity;
    double   start_time_ms;      /* 帧开始时间 */
    bool     active;
    bool     final_seen;         /* 已收到 more=false 的块 */
} Block2Assembly;

/* ═══════════════════════════════════════════════════════════════════════
 *  CoAP Receiver 结构体
 * ═══════════════════════════════════════════════════════════════════════ */

struct CoapReceiver {
    /* ── 连接 ── */
    char   esp_ip[64];
    int    esp_port;
    char   wifi_ssid[64];
    char   wifi_password[64];
    int    sock_fd;
    bool   connected;

    int    reconnect_delay_s;
    time_t last_reconnect_attempt;

    /* ── 帧重组 ── */
    Block2Assembly assembly;

    /* ── 最新 JPEG 帧 ── */
    uint8_t* jpeg_buf;
    size_t   jpeg_len;
    int      frame_index;
    double   frame_timestamp;

    /* ── IMU 数据 ── */
    IMUExternalPose latest_pose;
    bool   has_pose;
    float  yaw_accum;
    double last_imu_time;
    int    imu_log_cnt;
    double last_imu_poll_ms;
    uint16_t imu_msg_id;

    /* ── Stream 订阅 ── */
    uint16_t stream_msg_id;
    double   last_stream_sub_ms;

    /* ── 原始 IMU 数据回调 ── */
    CoapImuRawCallback imu_raw_cb;
    void*              imu_raw_user;

    /* ── 线程 ── */
    pthread_t       reader_thread;
    volatile bool   running;
    volatile bool   shutdown;
    pthread_mutex_t mutex;
};

/* ═══════════════════════════════════════════════════════════════════════
 *  WiFi 自动连接 (nmcli)
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * WiFi auto-connect with TIMEOUT guards.
 *
 * Design:
 *   - CoAP/UDP only needs IP connectivity to ESP32 — no special WiFi setup required.
 *   - All system() calls are TIME-BOUND (timeout wrapper) so the realtime pipeline
 *     never blocks indefinitely during init.
 *   - Pre-connection is strongly recommended: connect before running lingqi_tantong.
 *
 * Detection order (fastest → slowest):
 *   1. iw dev $iface info | grep 'ssid <name>'   — kernel-level, no nmcli, ~1ms
 *   2. nmcli -t -f ACTIVE,SSID dev wifi           — WiFi scan cache, ~100ms
 *   3. nmcli -t -f NAME connection show --active  — connection profiles, ~100ms
 *
 * Connection:
 *   4. nmcli dev wifi connect <ssid>              — with timeout, ~2-15s
 */

/* ── Shell-safe quoting for SSID ──
 * Replaces single-quotes in the SSID with the sequence: '\'' (end quote,
 * escaped literal quote, resume quote) so the value is safe inside
 * single-quoted shell strings. */
static void shell_quote_ssid(const char* ssid, char* out, size_t out_sz) {
    size_t j = 0;
    for (size_t i = 0; ssid[i] && j + 4 < out_sz; i++) {
        if (ssid[i] == '\'') {
            if (j + 4 >= out_sz) break;
            out[j++] = '\''; out[j++] = '\\'; out[j++] = '\''; out[j++] = '\'';
        } else {
            out[j++] = ssid[i];
        }
    }
    out[j] = '\0';
}

#define WIFI_CMD_TIMEOUT_S  15

static bool wifi_connect(const char* ssid, const char* password) {
    if (!ssid || ssid[0] == '\0') {
        log_info("[CoapRX] No WiFi SSID given, assuming user pre-connected");
        return true;
    }

    /* Shell-safe copy of SSID */
    char safe_ssid[128];
    shell_quote_ssid(ssid, safe_ssid, sizeof(safe_ssid));

    bool already_connected = false;

    /* ── Method 1: iw dev (kernel-level, no nmcli dependency, ~1ms) ── */
    {
        char check[512];
        snprintf(check, sizeof(check),
            "for iface in $(iw dev 2>/dev/null | awk '/Interface/{print $2}'); do "
            "iw dev \"$iface\" info 2>/dev/null | grep -q 'ssid %s' && exit 0; "
            "done; exit 1", safe_ssid);
        if (system(check) == 0) {
            already_connected = true;
            log_info("[CoapRX] WiFi check (iw dev): already on '%s'", ssid);
        }
    }

    /* ── Method 2: nmcli WiFi scan cache (SSID match, not connection name) ── */
    if (!already_connected) {
        char check[256];
        snprintf(check, sizeof(check),
            "timeout 3 nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep -q '^yes:%s$'",
            safe_ssid);
        if (system(check) == 0) {
            already_connected = true;
            log_info("[CoapRX] WiFi check (nmcli scan): already on '%s'", ssid);
        }
    }

    /* ── Method 3: nmcli connection profiles (fallback — profile name may ≠ SSID) ── */
    if (!already_connected) {
        char check[256];
        snprintf(check, sizeof(check),
            "timeout 3 nmcli -t -f NAME connection show --active 2>/dev/null | grep -qxF '%s'",
            safe_ssid);
        if (system(check) == 0) {
            already_connected = true;
            log_info("[CoapRX] WiFi check (nmcli profile): already on '%s'", ssid);
        }
    }

    if (already_connected) {
        log_info("[CoapRX] WiFi already connected to '%s' — skipping connect", ssid);
        return true;
    }

    /* ── Method 4: Connect (with timeout guard) ── */
    log_info("[CoapRX] Connecting to WiFi '%s' (timeout=%ds)...",
             ssid, WIFI_CMD_TIMEOUT_S);

    char cmd[512];
    if (password && password[0] != '\0') {
        /* nmcli handles password securely internally; the shell only sees the
         * password as a quoted argument.  Use single quotes around both. */
        char safe_pw[128];
        shell_quote_ssid(password, safe_pw, sizeof(safe_pw));
        snprintf(cmd, sizeof(cmd),
            "timeout %d nmcli dev wifi connect '%s' password '%s' 2>&1",
            WIFI_CMD_TIMEOUT_S, safe_ssid, safe_pw);
    } else {
        snprintf(cmd, sizeof(cmd),
            "timeout %d nmcli dev wifi connect '%s' 2>&1",
            WIFI_CMD_TIMEOUT_S, safe_ssid);
    }

    int rc = system(cmd);
    if (rc == 124) {
        /* timeout(1) exit code 124 = killed by SIGTERM after timeout */
        log_warning("[CoapRX] WiFi connect to '%s' timed out after %ds — "
                    "nmcli/D-Bus may be hung. UDP will still be attempted. "
                    "Tip: pre-connect before running lingqi_tantong.",
                    ssid, WIFI_CMD_TIMEOUT_S);
    } else if (rc == 127) {
        /* shell returns 127 = command not found */
        log_warning("[CoapRX] 'nmcli' not found — WiFi connection skipped. "
                    "UDP packets to %s will be sent on existing interface. "
                    "Install: sudo apt install network-manager",
                    ssid);
    } else if (rc != 0) {
        /* nmcli failed — try sudo */
        if (password && password[0] != '\0') {
            char safe_pw[128];
            shell_quote_ssid(password, safe_pw, sizeof(safe_pw));
            snprintf(cmd, sizeof(cmd),
                "timeout %d sudo nmcli dev wifi connect '%s' password '%s' 2>&1",
                WIFI_CMD_TIMEOUT_S, safe_ssid, safe_pw);
        } else {
            snprintf(cmd, sizeof(cmd),
                "timeout %d sudo nmcli dev wifi connect '%s' 2>&1",
                WIFI_CMD_TIMEOUT_S, safe_ssid);
        }
        int rc2 = system(cmd);
        if (rc2 == 124) {
            log_warning("[CoapRX] WiFi sudo connect timed out after %ds", WIFI_CMD_TIMEOUT_S);
        } else if (rc2 != 0) {
            log_warning("[CoapRX] WiFi connect failed (rc=%d, sudo rc=%d) for '%s'. "
                        "UDP may not reach ESP32. Pre-connect manually.",
                        rc, rc2, ssid);
        } else {
            log_info("[CoapRX] WiFi connected to '%s' (via sudo)", ssid);
        }
    } else {
        log_info("[CoapRX] WiFi connected to '%s'", ssid);
    }

    return true;
}
#undef WIFI_CMD_TIMEOUT_S

/* ═══════════════════════════════════════════════════════════════════════
 *  UDP socket 工具
 * ═══════════════════════════════════════════════════════════════════════ */

static int udp_create_socket(void) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        log_error("[CoapRX] socket() failed: %s", strerror(errno));
        return -1;
    }

    /* 设置接收超时 500ms，让线程可以定期检查 running 标志 */
    struct timeval tv = {0, 500000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    /* 增大接收缓冲区 */
    int rcvbuf = 256 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    return fd;
}

static void udp_close(int* fd) {
    if (*fd >= 0) { close(*fd); *fd = -1; }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  CoAP 协议编码（构建请求包）
 * ═══════════════════════════════════════════════════════════════════════ */

static uint16_t read_u16_be(const uint8_t* p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

static void write_u16_be(uint8_t* p, uint16_t v) {
    p[0] = (v >> 8) & 0xFF;
    p[1] = v & 0xFF;
}

/*
 * 编码 CoAP option (delta-length 格式)
 * 返回写入的字节数，失败返回 -1
 */
static int encode_option(uint8_t* out, int prev_opt, int opt_num,
                         const uint8_t* val, int val_len) {
    int delta = opt_num - prev_opt;
    int pos = 0;

    if (delta <= 12 && val_len <= 12) {
        out[pos++] = (uint8_t)((delta << 4) | val_len);
    } else if (delta <= 12 && val_len <= 255) {
        out[pos++] = (uint8_t)((delta << 4) | 13);
        out[pos++] = (uint8_t)(val_len - 13);
    } else if (delta <= 255 && val_len <= 12) {
        out[pos++] = (uint8_t)((13 << 4) | val_len);
        out[pos++] = (uint8_t)(delta - 13);
    } else {
        return -1;  /* 不支持长扩展 */
    }

    memcpy(out + pos, val, val_len);
    return pos + val_len;
}

/*
 * 构建 CoAP GET 请求
 *    path:  如 "/stream" 或 "/imu"
 *    msg_id: 消息 ID
 *    token:  token 字节 (4 字节)
 *    out:    输出缓冲区
 *    out_max: 输出缓冲区大小
 * 返回实际包长度，失败返回 0
 */
static int build_coap_get(const char* path, uint16_t msg_id,
                          const uint8_t token[4], uint8_t* out, int out_max) {
    /* 解析路径段: "/stream" → ["stream"], "/imu" → ["imu"] */
    char segments[4][32];
    int num_segs = 0;
    const char* p = path;
    while (*p == '/') p++;
    if (*p == '\0') return 0;

    const char* start = p;
    while (*p && num_segs < 4) {
        if (*p == '/') {
            int len = (int)(p - start);
            if (len > 0 && len < 32) {
                memcpy(segments[num_segs], start, len);
                segments[num_segs][len] = '\0';
                num_segs++;
            }
            start = p + 1;
        }
        p++;
    }
    if (*start) {
        int len = (int)(p - start);
        if (len > 0 && len < 32) {
            memcpy(segments[num_segs], start, len);
            segments[num_segs][len] = '\0';
            num_segs++;
        }
    }

    /* 构建 option 字段 */
    uint8_t opts[256];
    int opt_len = 0;
    int prev_opt = 0;

    for (int i = 0; i < num_segs; i++) {
        int seg_len = (int)strlen(segments[i]);
        int n = encode_option(opts + opt_len, prev_opt, COAP_OPT_URI_PATH,
                              (const uint8_t*)segments[i], seg_len);
        if (n < 0 || opt_len + n > (int)sizeof(opts)) return 0;
        opt_len += n;
        prev_opt = COAP_OPT_URI_PATH;
    }

    /* CoAP 头: Ver(2bit)=1, Type(2bit)=CON(0), TKL(4bit)=4 */
    int pos = 0;
    /* BUGFIX: check for header(4) + token(TKL=4) = 8 bytes total */
    if (pos + 4 + 4 > out_max) return 0;
    out[pos++] = (1 << 6) | (COAP_TYPE_CON << 4) | 4;  /* TKL=4 */
    out[pos++] = COAP_CODE_GET;
    write_u16_be(out + pos, msg_id);
    pos += 2;
    memcpy(out + pos, token, 4);
    pos += 4;

    /* Options */
    if (opt_len > 0) {
        if (pos + opt_len > out_max) return 0;
        memcpy(out + pos, opts, opt_len);
        pos += opt_len;
    }

    return pos;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  CoAP 协议解码（解析响应包）
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t  token[8];
    int      token_len;
    int      opts_present;       /* bitmap of option numbers found */
    uint32_t etag;
    uint32_t block2_num;
    bool     block2_more;
    uint8_t  block2_szx;
    uint32_t size2;
    uint8_t  content_format;
    const uint8_t* payload;
    int      payload_len;
} CoapResponse;

/*
 * 解析 CoAP 响应包
 * 返回 true 表示解析成功
 */
static bool parse_coap_response(const uint8_t* data, int len, CoapResponse* resp) {
    if (len < 4) return false;

    memset(resp, 0, sizeof(*resp));

    int tkl = data[0] & 0x0F;
    if (tkl > 8) return false;

    resp->token_len = tkl;

    int pos = 4;
    for (int i = 0; i < tkl; i++) {
        if (pos >= len) return false;
        resp->token[i] = data[pos++];
    }

    /* 解析 options */
    int opt_num = 0;
    while (pos < len) {
        if (data[pos] == 0xFF) {
            pos++;
            resp->payload = data + pos;
            resp->payload_len = len - pos;
            break;
        }

        if (pos >= len) break;
        uint8_t hdr = data[pos++];
        int delta = (hdr >> 4) & 0x0F;
        int olen  = hdr & 0x0F;

        if (delta == 13) {
            if (pos >= len) return false;
            delta = data[pos++] + 13;
        } else if (delta == 14) {
            if (pos + 1 >= len) return false;
            delta = read_u16_be(data + pos) + 269;
            pos += 2;
        } else if (delta == 15) {
            return false;
        }

        if (olen == 13) {
            if (pos >= len) return false;
            olen = data[pos++] + 13;
        } else if (olen == 14) {
            if (pos + 1 >= len) return false;
            olen = read_u16_be(data + pos) + 269;
            pos += 2;
        } else if (olen == 15) {
            return false;
        }

        opt_num += delta;
        if (pos + olen > len) return false;

        resp->opts_present |= (1 << opt_num);

        /* 解析已知 option 值 */
        if (opt_num == COAP_OPT_ETAG && olen >= 1 && olen <= 4) {
            resp->etag = 0;
            for (int i = 0; i < olen; i++)
                resp->etag = (resp->etag << 8) | data[pos + i];
        } else if (opt_num == COAP_OPT_BLOCK2 && olen >= 1) {
            uint32_t val = 0;
            for (int i = 0; i < olen; i++)
                val = (val << 8) | data[pos + i];
            resp->block2_num = val >> 4;
            resp->block2_more = (val & 0x08) != 0;
            resp->block2_szx = val & 0x07;
        } else if (opt_num == COAP_OPT_SIZE2 && olen >= 1 && olen <= 4) {
            resp->size2 = 0;
            for (int i = 0; i < olen; i++)
                resp->size2 = (resp->size2 << 8) | data[pos + i];
        } else if (opt_num == COAP_OPT_CONTENT_FMT && olen >= 1) {
            resp->content_format = data[pos];
        }

        pos += olen;
    }

    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Block2 帧重组
 * ═══════════════════════════════════════════════════════════════════════ */

static void assembly_init(Block2Assembly* as) {
    as->etag = 0;
    as->total_size = 0;
    as->szx = COAP_DEFAULT_SZX;
    as->total_blocks = 0;
    as->blocks_received = 0;
    memset(as->block_map, 0, sizeof(as->block_map));
    as->data = (uint8_t*)malloc(FRAME_BUF_CAPACITY);
    as->capacity = FRAME_BUF_CAPACITY;
    as->start_time_ms = 0;
    as->active = false;
    as->final_seen = false;
}

static void assembly_reset(Block2Assembly* as) {
    as->etag = 0;
    as->total_size = 0;
    as->total_blocks = 0;
    as->blocks_received = 0;
    memset(as->block_map, 0, sizeof(as->block_map));
    as->start_time_ms = 0;
    as->active = false;
    as->final_seen = false;
    /* 保留 data 缓冲区不释放 */
}

static void assembly_destroy(Block2Assembly* as) {
    free(as->data);
    as->data = NULL;
    as->capacity = 0;
}

/* bitmap helper: mark block_num as received */
static inline void block_map_set(uint32_t* map, uint32_t idx) {
    map[idx / 32] |= (1u << (idx % 32));
}

static inline bool block_map_test(const uint32_t* map, uint32_t idx) {
    return (map[idx / 32] & (1u << (idx % 32))) != 0;
}

/*
 * 处理一个 Block2 块 (v2 协议: 支持乱序, ETag 全局)
 * 返回:
 *   -1 = 忽略（重复/不属于当前帧）
 *    0 = 已存储，等待更多块
 *    1 = 帧组装完成（完整 JPEG 数据在 as->data 中，长度在 as->total_size）
 */
static int assembly_add_block(Block2Assembly* as, const CoapResponse* resp,
                               double now_ms) {
    uint32_t block_num  = resp->block2_num;
    bool     block_more = resp->block2_more;
    uint8_t  block_szx  = resp->block2_szx;
    int      chunk_len  = resp->payload_len;
    uint32_t block_etag = resp->etag;  /* v2: 所有块都有 ETag */

    size_t block_size = COAP_BLOCK_SIZE(block_szx);

    /* ── 新帧检测: block 0 到达 or ETag 变化 ── */
    bool new_frame = (block_num == 0);
    if (!new_frame && as->active && block_etag != 0 && block_etag != as->etag) {
        /* ETag 变化 → 开始新帧 (丢弃旧帧) */
        log_debug("[CoapRX] ETag changed %u→%u, starting new frame",
                  (unsigned)as->etag, (unsigned)block_etag);
        new_frame = true;
    }

    if (new_frame) {
        assembly_reset(as);
        as->etag = block_etag;
        as->total_size = resp->size2;  /* block 0 带有 Size2 */
        as->szx = block_szx;
        as->start_time_ms = now_ms;
        as->active = true;
        as->final_seen = false;

        /* 扩展缓冲区 */
        if (as->total_size > as->capacity) {
            size_t new_cap = as->total_size + 4096;
            uint8_t* new_data = (uint8_t*)realloc(as->data, new_cap);
            if (!new_data) {
                log_error("[CoapRX] Failed to realloc frame buffer to %zu", new_cap);
                assembly_reset(as);
                return -1;
            }
            as->data = new_data;
            as->capacity = new_cap;
        }
    }

    /* ── 检查帧有效性 ── */
    if (!as->active) return -1;
    if (block_etag != 0 && block_etag != as->etag) return -1;

    /* ── 检查块号范围 ── */
    if (block_num >= MAX_BLOCKS_PER_FRAME) {
        log_warning("[CoapRX] Block %u exceeds max %d", block_num, MAX_BLOCKS_PER_FRAME);
        return -1;
    }

    /* ── 跳过重复块 ── */
    if (block_map_test(as->block_map, block_num)) return -1;

    /* ── 检查块大小 (非最后块必须是完整块大小) ── */
    if (block_more && chunk_len != (int)block_size) {
        log_warning("[CoapRX] Block %u: expected %zu bytes, got %d (more=1)",
                    block_num, block_size, chunk_len);
        return -1;
    }

    /* ── 存储块数据 ── */
    size_t offset = block_num * block_size;
    if (offset + chunk_len > as->capacity) {
        log_error("[CoapRX] Block %u: offset %zu + len %d > capacity %zu",
                  block_num, offset, chunk_len, as->capacity);
        assembly_reset(as);
        return -1;
    }

    memcpy(as->data + offset, resp->payload, chunk_len);
    block_map_set(as->block_map, block_num);
    as->blocks_received++;

    /* ── 记录最后一块 ── */
    if (!block_more) {
        as->final_seen = true;
        /* 从最后一块推导帧总大小 */
        as->total_blocks = block_num + 1;
        if (as->total_size == 0) {
            as->total_size = offset + chunk_len;
        }
    }

    /* ── 检查帧是否完整 ── */
    if (as->final_seen) {
        uint32_t expected = as->total_blocks;
        if (expected == 0) {
            /* 异常: 只收到一个 more=false 的块 (单块帧) */
            expected = 1;
            as->total_blocks = 1;
            as->total_size = chunk_len;
        }

        if (as->blocks_received >= expected) {
            as->active = false;
            return 1;  /* 帧完整 */
        }

        /* 已知总块数但未收齐 — 继续等待 (不超时处理) */
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  IMU JSON 解析
 * ═══════════════════════════════════════════════════════════════════════ */

static bool parse_imu_json(const uint8_t* payload, int payload_len,
                           float* ax, float* ay, float* az,
                           float* gx, float* gy, float* gz) {
    if (!payload || payload_len <= 0 || payload_len > 256) return false;

    char buf[256];
    int copy = payload_len < (int)sizeof(buf) - 1 ? payload_len : (int)sizeof(buf) - 1;
    memcpy(buf, payload, copy);
    buf[copy] = '\0';

    return sscanf(buf,
        "{\"ax\":%f,\"ay\":%f,\"az\":%f,"
        "\"gx\":%f,\"gy\":%f,\"gz\":%f}",
        ax, ay, az, gx, gy, gz) == 6;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  发送 CoAP 请求
 * ═══════════════════════════════════════════════════════════════════════ */

static bool coap_send_request(CoapReceiver* r, const char* path,
                              uint16_t msg_id, const uint8_t token[4]) {
    if (r->sock_fd < 0) return false;

    uint8_t tx[COAP_MAX_PACKET];
    int tx_len = build_coap_get(path, msg_id, token, tx, sizeof(tx));
    if (tx_len <= 0) return false;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)r->esp_port);
    if (inet_pton(AF_INET, r->esp_ip, &addr.sin_addr) != 1) {
        log_error("[CoapRX] inet_pton(%s) failed", r->esp_ip);
        return false;
    }

    ssize_t n = sendto(r->sock_fd, tx, tx_len, 0,
                       (struct sockaddr*)&addr, sizeof(addr));
    if (n != tx_len) {
        log_warning("[CoapRX] sendto %s failed: %s", path, strerror(errno));
        return false;
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  处理接收到的 CoAP 数据包
 * ═══════════════════════════════════════════════════════════════════════ */

static void process_coap_packet(CoapReceiver* r, const uint8_t* data, int len,
                                double now_ms) {
    CoapResponse resp;
    if (!parse_coap_response(data, len, &resp)) return;

    /* ── 判断是 stream 响应还是 imu 响应 ── */
    bool is_stream = (resp.token_len == 4 &&
                      memcmp(resp.token, (const uint8_t[]){
                          (COAP_STREAM_TOKEN >> 24) & 0xFF,
                          (COAP_STREAM_TOKEN >> 16) & 0xFF,
                          (COAP_STREAM_TOKEN >> 8) & 0xFF,
                          COAP_STREAM_TOKEN & 0xFF}, 4) == 0);
    bool is_imu = (resp.token_len == 4 &&
                   memcmp(resp.token, (const uint8_t[]){
                       (COAP_IMU_TOKEN >> 24) & 0xFF,
                       (COAP_IMU_TOKEN >> 16) & 0xFF,
                       (COAP_IMU_TOKEN >> 8) & 0xFF,
                       COAP_IMU_TOKEN & 0xFF}, 4) == 0);

    /* ── Stream 响应: Block2 分块 ── */
    if (is_stream && (resp.opts_present & (1 << COAP_OPT_BLOCK2))) {
        int result = assembly_add_block(&r->assembly, &resp, now_ms);

        if (result == 1) {
            /* 帧组装完成 */
            size_t jpg_len = r->assembly.total_size;

            /* 验证 JPEG 头 */
            if (jpg_len < 2 || jpg_len > FRAME_MAX_JPEG_LEN ||
                r->assembly.data[0] != 0xFF || r->assembly.data[1] != 0xD8) {
                log_warning("[CoapRX] Assembled frame invalid: %zu bytes, "
                           "header=%02x%02x", jpg_len,
                           r->assembly.data[0], r->assembly.data[1]);
                return;
            }

            pthread_mutex_lock(&r->mutex);
            free(r->jpeg_buf);
            r->jpeg_buf = (uint8_t*)malloc(jpg_len);
            if (r->jpeg_buf) {
                memcpy(r->jpeg_buf, r->assembly.data, jpg_len);
                r->jpeg_len = jpg_len;
                r->frame_index++;
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                r->frame_timestamp = (double)ts.tv_sec * 1e6
                                   + (double)ts.tv_nsec / 1e3;

                if (r->frame_index == 1) {
                    log_info("[CoapRX] First JPEG frame: %zu bytes, %u blocks",
                             jpg_len, (unsigned)r->assembly.blocks_received);
                } else if (r->frame_index % 30 == 0) {
                    log_info("[CoapRX] Frame #%d: %zu bytes JPEG, "
                             "%u blocks (etag=%u)",
                             r->frame_index, jpg_len,
                             (unsigned)r->assembly.blocks_received,
                             (unsigned)r->assembly.etag);
                }
            }
            pthread_mutex_unlock(&r->mutex);
        }
        return;
    }

    /* ── IMU 响应: JSON payload ── */
    if (is_imu && resp.payload_len > 0) {
        float ax, ay, az, gx, gy, gz;
        if (parse_imu_json(resp.payload, resp.payload_len,
                           &ax, &ay, &az, &gx, &gy, &gz)) {

            /* 原始数据回调: 馈送到 Madgwick 相机滤波器 */
            if (r->imu_raw_cb) {
                float accel[3] = {ax, ay, az};
                float gyro[3]  = {gx, gy, gz};
                r->imu_raw_cb(r->imu_raw_user, accel, gyro);
            }

            /* 计算 tilt (roll/pitch from accelerometer) */
            float roll_rad  = atan2f(ay, az);
            float pitch_rad = atan2f(-ax, sqrtf(ay*ay + az*az));

            /* 积分 gyro Z 得到 yaw */
            struct timespec imu_ts;
            clock_gettime(CLOCK_MONOTONIC, &imu_ts);
            double now_s = (double)imu_ts.tv_sec
                         + (double)imu_ts.tv_nsec * 1e-9;
            double dt = (r->last_imu_time > 0.0)
                      ? (now_s - r->last_imu_time) : 0.04;
            if (dt > 0.001 && dt < 1.0) {
                r->yaw_accum += gz * (float)dt;
            }
            r->last_imu_time = now_s;

            /* Euler → Quaternion (ZYX) */
            float cr = cosf(roll_rad * 0.5f), sr = sinf(roll_rad * 0.5f);
            float cp = cosf(pitch_rad * 0.5f), sp = sinf(pitch_rad * 0.5f);
            float cy = cosf(r->yaw_accum * 0.5f), sy = sinf(r->yaw_accum * 0.5f);
            float qw = cr*cp*cy + sr*sp*sy;
            float qx = sr*cp*cy - cr*sp*sy;
            float qy = cr*sp*cy + sr*cp*sy;
            float qz = cr*cp*sy - sr*sp*cy;

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
            r->latest_pose.altitude_m = 0.0f;
            r->latest_pose.temperature_c = 0.0f;
            r->latest_pose.timestamp_ms = (uint32_t)ts_ms;
            r->latest_pose.is_valid = true;
            r->has_pose = true;
            pthread_mutex_unlock(&r->mutex);

            if (++r->imu_log_cnt % 30 == 0) {
                log_info("[CoapRX] IMU#%d | accel: ax=%.4f ay=%.4f az=%.4f m/s² | "
                         "gyro: gx=%.4f gy=%.4f gz=%.4f rad/s | "
                         "pose: roll=%.2f° pitch=%.2f° yaw=%.1f°",
                         r->imu_log_cnt, ax, ay, az, gx, gy, gz,
                         roll_rad * 57.3f, pitch_rad * 57.3f,
                         r->yaw_accum * 57.3f);
            }
        }
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  后台接收线程
 * ═══════════════════════════════════════════════════════════════════════ */

static bool coap_connect_and_subscribe(CoapReceiver* r) {
    udp_close(&r->sock_fd);

    int fd = udp_create_socket();
    if (fd < 0) return false;

    r->sock_fd = fd;
    r->connected = true;
    r->reconnect_delay_s = COAP_RECONNECT_INITIAL_S;
    r->yaw_accum = 0.0f;
    r->last_imu_time = 0.0;
    r->imu_log_cnt = 0;
    r->last_imu_poll_ms = 0;
    r->last_stream_sub_ms = 0;
    r->imu_msg_id = 0x2000;
    r->stream_msg_id = 0x1000;
    assembly_reset(&r->assembly);

    /* 构造 token */
    uint8_t stream_token[4] = {
        (uint8_t)((COAP_STREAM_TOKEN >> 24) & 0xFF),
        (uint8_t)((COAP_STREAM_TOKEN >> 16) & 0xFF),
        (uint8_t)((COAP_STREAM_TOKEN >> 8) & 0xFF),
        (uint8_t)(COAP_STREAM_TOKEN & 0xFF)
    };

    /* 订阅 /stream */
    if (coap_send_request(r, "/stream", r->stream_msg_id, stream_token)) {
        log_info("[CoapRX] Subscribed to /stream (msg_id=%u)",
                 (unsigned)r->stream_msg_id);
        r->stream_msg_id = (r->stream_msg_id + 1) & 0xFFFF;
    }

    log_info("[CoapRX] Connected to %s:%d (CoAP/UDP)",
             r->esp_ip, r->esp_port);
    return true;
}

static void* coap_reader_thread(void* arg) {
    CoapReceiver* r = (CoapReceiver*)arg;

    log_info("[CoapRX] Reader thread started, connecting to %s:%d",
             r->esp_ip, r->esp_port);

    if (!coap_connect_and_subscribe(r)) {
        log_warning("[CoapRX] Initial connection failed, will retry");
    }

    uint8_t rx_buf[COAP_MAX_PACKET];

    while (r->running) {
        if (!r->connected) {
            time_t now = time(NULL);
            int wait = r->reconnect_delay_s;
            if (wait < COAP_RECONNECT_INITIAL_S) wait = COAP_RECONNECT_INITIAL_S;
            if (wait > COAP_RECONNECT_MAX_S) wait = COAP_RECONNECT_MAX_S;

            if (r->last_reconnect_attempt != 0 &&
                now < r->last_reconnect_attempt + wait) {
                usleep(500000);
                continue;
            }
            r->last_reconnect_attempt = now;

            log_info("[CoapRX] Reconnect attempt to %s:%d (backoff=%ds)",
                     r->esp_ip, r->esp_port, wait);

            if (coap_connect_and_subscribe(r)) {
                /* OK */;
            } else {
                r->reconnect_delay_s *= 2;
                if (r->reconnect_delay_s > COAP_RECONNECT_MAX_S)
                    r->reconnect_delay_s = COAP_RECONNECT_MAX_S;
            }
            continue;
        }

        /* ── 计算当前时间 ── */
        struct timespec ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        double now_ms = (double)ts_now.tv_sec * 1000.0
                      + (double)ts_now.tv_nsec / 1000000.0;

        /* ── Keepalive: 每 STREAM_KEEPALIVE_S 发送 stream 订阅以保持 ESP32 活跃 ── */
        if (now_ms - r->last_stream_sub_ms > STREAM_KEEPALIVE_S * 1000.0) {
            uint8_t stream_token[4] = {
                (uint8_t)((COAP_STREAM_TOKEN >> 24) & 0xFF),
                (uint8_t)((COAP_STREAM_TOKEN >> 16) & 0xFF),
                (uint8_t)((COAP_STREAM_TOKEN >> 8) & 0xFF),
                (uint8_t)(COAP_STREAM_TOKEN & 0xFF)
            };
            if (coap_send_request(r, "/stream", r->stream_msg_id, stream_token)) {
                r->stream_msg_id = (r->stream_msg_id + 1) & 0xFFFF;
                r->last_stream_sub_ms = now_ms;
            }
        }

        /* ── 定期轮询 /imu ── */
        if (now_ms - r->last_imu_poll_ms > IMU_POLL_INTERVAL_MS) {
            uint8_t imu_token[4] = {
                (uint8_t)((COAP_IMU_TOKEN >> 24) & 0xFF),
                (uint8_t)((COAP_IMU_TOKEN >> 16) & 0xFF),
                (uint8_t)((COAP_IMU_TOKEN >> 8) & 0xFF),
                (uint8_t)(COAP_IMU_TOKEN & 0xFF)
            };
            if (coap_send_request(r, "/imu", r->imu_msg_id, imu_token)) {
                r->imu_msg_id = (r->imu_msg_id + 1) & 0xFFFF;
                r->last_imu_poll_ms = now_ms;
            }
        }

        /* ── 检查帧重组超时 ── */
        if (r->assembly.active &&
            now_ms - r->assembly.start_time_ms > FRAME_TIMEOUT_MS) {
            log_warning("[CoapRX] Frame assembly timeout (etag=%u, got=%u/%u blocks)",
                        (unsigned)r->assembly.etag,
                        (unsigned)r->assembly.blocks_received,
                        (unsigned)r->assembly.total_blocks);
            assembly_reset(&r->assembly);
        }

        /* ── 接收 UDP 数据包 ── */
        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        ssize_t n = recvfrom(r->sock_fd, rx_buf, sizeof(rx_buf), 0,
                             (struct sockaddr*)&from_addr, &from_len);

        if (n > 0) {
            process_coap_packet(r, rx_buf, (int)n, now_ms);
        } else if (n < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                log_warning("[CoapRX] recvfrom error: %s", strerror(errno));
                r->connected = false;
                udp_close(&r->sock_fd);
            }
        }
    }

    log_info("[CoapRX] Reader thread exiting");
    udp_close(&r->sock_fd);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  公开 API
 * ═══════════════════════════════════════════════════════════════════════ */

CoapReceiver* coap_receiver_create(const char* esp_ip, int esp_port,
                                   const char* wifi_ssid,
                                   const char* wifi_password) {
    if (!esp_ip || esp_ip[0] == '\0' || esp_port <= 0) return NULL;

    CoapReceiver* r = (CoapReceiver*)calloc(1, sizeof(CoapReceiver));
    if (!r) return NULL;

    strncpy(r->esp_ip, esp_ip, sizeof(r->esp_ip) - 1);
    r->esp_port = esp_port;
    if (wifi_ssid) strncpy(r->wifi_ssid, wifi_ssid, sizeof(r->wifi_ssid) - 1);
    if (wifi_password) strncpy(r->wifi_password, wifi_password, sizeof(r->wifi_password) - 1);

    r->sock_fd = -1;
    r->connected = false;
    r->reconnect_delay_s = COAP_RECONNECT_INITIAL_S;
    r->has_pose = false;

    assembly_init(&r->assembly);

    if (pthread_mutex_init(&r->mutex, NULL) != 0) {
        assembly_destroy(&r->assembly);
        free(r);
        return NULL;
    }

    wifi_connect(r->wifi_ssid, r->wifi_password);

    r->running = true;
    r->shutdown = false;
    if (pthread_create(&r->reader_thread, NULL, coap_reader_thread, r) != 0) {
        pthread_mutex_destroy(&r->mutex);
        assembly_destroy(&r->assembly);
        free(r);
        return NULL;
    }

    log_info("[CoapRX] Created: ESP32 at %s:%d (CoAP/UDP)", esp_ip, esp_port);
    return r;
}

void coap_receiver_destroy(CoapReceiver* r) {
    if (!r) return;
    log_info("[CoapRX] Destroy: signaling reader thread to stop");
    log_flush();
    r->running = false;
    r->shutdown = true;
    log_info("[CoapRX] Destroy: joining reader thread");
    log_flush();
    pthread_join(r->reader_thread, NULL);
    log_info("[CoapRX] Destroy: reader thread joined, closing socket");
    log_flush();
    udp_close(&r->sock_fd);
    log_info("[CoapRX] Destroy: freeing assembly buffer");
    log_flush();
    assembly_destroy(&r->assembly);
    log_info("[CoapRX] Destroy: freeing jpeg buffer");
    log_flush();
    free(r->jpeg_buf);
    pthread_mutex_destroy(&r->mutex);
    free(r);
    log_info("[CoapRX] Destroy: complete");
    log_flush();
}

void coap_receiver_update(CoapReceiver* r) {
    (void)r;
}

bool coap_receiver_get_latest_frame(CoapReceiver* r, ArrowSourceFrame* out) {
    if (!r || !out) return false;

    bool has_data = false;
    pthread_mutex_lock(&r->mutex);

    if (r->jpeg_buf && r->jpeg_len > 0 && r->jpeg_len < FRAME_MAX_JPEG_LEN) {
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

bool coap_receiver_is_connected(CoapReceiver* r) {
    return r ? r->connected : false;
}

void coap_receiver_set_imu_raw_callback(CoapReceiver* r,
                                         CoapImuRawCallback cb, void* user) {
    if (r) {
        r->imu_raw_cb = cb;
        r->imu_raw_user = user;
    }
}
