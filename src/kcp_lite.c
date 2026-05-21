#include "kcp_lite.h"
#include <stdlib.h>
#include <string.h>

/*
 * KCP-lite 协议偏差说明 (相对于标准 IKCP/skywind3000/kcp):
 *
 * 1. 拥塞控制简化: 标准 KCP 实现基于 TCP BBR 的拥塞控制，包括 cwnd 增长/缩减策略。
 *    KCP-lite 仅保留了基础的滑动窗口流控，未实现完整的拥塞避免算法。
 *    在低丢包率局域网环境下运行良好，但跨广域网时吞吐量可能偏离预期。
 *
 * 2. 快速重传阈值: 标准 KCP 使用 fastresend (默认 2) 控制快速重传触发条件。
 *    KCP-lite 简化为此逻辑，重传判定基于固定 RTO 计算而非自适应。
 *
 * 3. 流控窗口更新: 标准 KCP 通过 IKCP_CMD_WASK/IKCP_CMD_WINS 命令
 *    动态通告远端窗口。KCP-lite 初始通告窗口后不再更新。
 *
 * 4. MTU 碎片化: 标准 KCP 支持超过 MTU 的数据分段(fragmentation),
 *    KCP-lite 当前版本不处理跨 MTU 数据分段。
 *
 * 兼容性: KCP-lite 数据包格式与标准 IKCP 基本兼容(使用相同的 cmd 编码),
 *         但在拥塞控制和流控行为上存在差异。如需与标准 KCP 实现对等通信,
 *         建议直接使用 https://github.com/skywind3000/kcp 完整实现。
 *
 * 适用场景: 本项目用于 Muse Pi Pro (K1) 与 ESP32P4 之间的本地 UART/串口
 *          可靠传输,局域网环境,延迟 <5ms,丢包率可忽略。此场景下 KCP-lite
 *          的简化实现足够使用。
 */

#define IKCP_CMD_PUSH  81
#define IKCP_CMD_ACK   82
#define IKCP_CMD_WASK  83
#define IKCP_CMD_WINS  84

#define IKCP_OVERHEAD  24

typedef struct ikcp_seg {
    uint32_t conv;
    uint8_t  cmd;
    uint8_t  frg;
    uint16_t wnd;
    uint32_t ts;
    uint32_t sn;
    uint32_t una;
    uint32_t len;
    uint8_t* data;
    uint32_t xmit;
    uint32_t resendts;
    uint32_t rto;
    uint32_t fastack;
    struct ikcp_seg* next;
} ikcp_seg;

struct kcp_lite_t {
    uint32_t conv;
    int mtu;
    int mss;
    int state;
    uint32_t snd_una;
    uint32_t snd_nxt;
    uint32_t rcv_nxt;
    int snd_wnd;
    int rcv_wnd;
    uint32_t rmt_wnd;
    uint32_t cwnd;
    uint32_t probe;
    uint32_t current;
    uint32_t interval;
    uint32_t ts_flush;
    int nodelay;
    uint32_t ts_updated;
    uint32_t rx_rto;
    uint32_t rx_minrto;
    int resend;
    int nc;
    ikcp_seg* snd_queue;
    ikcp_seg* snd_queue_tail;
    ikcp_seg* rcv_queue;
    ikcp_seg* rcv_queue_tail;
    ikcp_seg* snd_buf;
    ikcp_seg* snd_buf_tail;
    ikcp_seg* rcv_buf;
    uint8_t* output_buf;
    int (*output)(const uint8_t* buf, int len, void* user);
    void* user;
};

static uint32_t ikcp_get_uint32(const uint8_t* p)
{
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void ikcp_set_uint32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static uint16_t ikcp_get_uint16(const uint8_t* p)
{
    return ((uint16_t)p[0]) | ((uint16_t)p[1] << 8);
}

static void ikcp_set_uint16(uint8_t* p, uint16_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
}

static ikcp_seg* ikcp_seg_alloc(int size)
{
    ikcp_seg* seg = (ikcp_seg*)calloc(1, sizeof(ikcp_seg));
    if (!seg) return NULL;
    if (size > 0) {
        seg->data = (uint8_t*)malloc((size_t)size);
        if (!seg->data) {
            free(seg);
            return NULL;
        }
    }
    return seg;
}

static void ikcp_seg_free(ikcp_seg* seg)
{
    if (seg) {
        free(seg->data);
        free(seg);
    }
}

static void ikcp_queue_push(ikcp_seg** head, ikcp_seg** tail, ikcp_seg* seg)
{
    seg->next = NULL;
    if (*tail) {
        (*tail)->next = seg;
    } else {
        *head = seg;
    }
    *tail = seg;
}

static ikcp_seg* ikcp_queue_pop(ikcp_seg** head, ikcp_seg** tail)
{
    ikcp_seg* seg = *head;
    if (seg) {
        *head = seg->next;
        if (!*head) {
            *tail = NULL;
        }
    }
    return seg;
}

static void ikcp_queue_clear(ikcp_seg** head, ikcp_seg** tail)
{
    ikcp_seg* cur = *head;
    while (cur) {
        ikcp_seg* next = cur->next;
        ikcp_seg_free(cur);
        cur = next;
    }
    *head = NULL;
    if (tail) *tail = NULL;
}

static void ikcp_snd_buf_remove(ikcp_seg** head, ikcp_seg** tail, ikcp_seg* target)
{
    ikcp_seg* prev = NULL;
    ikcp_seg* cur = *head;
    while (cur) {
        if (cur == target) {
            if (prev) {
                prev->next = cur->next;
            } else {
                *head = cur->next;
            }
            if (*tail == cur) {
                *tail = prev;
            }
            ikcp_seg_free(cur);
            return;
        }
        prev = cur;
        cur = cur->next;
    }
}

static int ikcp_encode_seg(uint8_t* buf, const ikcp_seg* seg)
{
    ikcp_set_uint32(buf, seg->conv);
    buf[4] = seg->cmd;
    buf[5] = seg->frg;
    ikcp_set_uint16(buf + 6, seg->wnd);
    ikcp_set_uint32(buf + 8, seg->ts);
    ikcp_set_uint32(buf + 12, seg->sn);
    ikcp_set_uint32(buf + 16, seg->una);
    ikcp_set_uint32(buf + 20, seg->len);
    if (seg->len > 0 && seg->data) {
        memcpy(buf + IKCP_OVERHEAD, seg->data, seg->len);
    }
    return IKCP_OVERHEAD + (int)seg->len;
}

static uint32_t ikcp_time_diff(uint32_t later, uint32_t earlier)
{
    return later - earlier;
}

static int ikcp_output(kcp_lite_t* kcp, const uint8_t* data, int len)
{
    if (kcp->output) {
        return kcp->output(data, len, kcp->user);
    }
    return 0;
}

void kcp_lite_set_output(kcp_lite_t* kcp, int (*output)(const uint8_t* buf, int len, void* user), void* user)
{
    kcp->output = output;
    kcp->user = user;
}

kcp_lite_t* kcp_lite_create(uint32_t conv_id, kcp_lite_config_t config)
{
    kcp_lite_t* kcp = (kcp_lite_t*)calloc(1, sizeof(kcp_lite_t));
    if (!kcp) return NULL;

    kcp->conv = conv_id;
    kcp->mtu = config.mtu > 0 ? config.mtu : 1400;
    kcp->mss = kcp->mtu - IKCP_OVERHEAD;
    kcp->snd_wnd = config.snd_wnd > 0 ? config.snd_wnd : 32;
    kcp->rcv_wnd = config.rcv_wnd > 0 ? config.rcv_wnd : 32;
    kcp->rmt_wnd = 32;
    kcp->cwnd = 0;
    kcp->probe = 0;
    kcp->interval = (uint32_t)(config.interval_ms > 0 ? config.interval_ms : 10);
    kcp->nodelay = config.nodelay;
    kcp->ts_updated = 0;
    kcp->rx_rto = 200;
    kcp->rx_minrto = 100;
    kcp->resend = config.resend > 0 ? config.resend : 2;
    kcp->nc = config.nc;

    kcp->output_buf = (uint8_t*)malloc((size_t)(kcp->mtu + IKCP_OVERHEAD) * 3);
    if (!kcp->output_buf) {
        free(kcp);
        return NULL;
    }

    return kcp;
}

void kcp_lite_destroy(kcp_lite_t* kcp)
{
    if (!kcp) return;
    ikcp_queue_clear(&kcp->snd_queue, &kcp->snd_queue_tail);
    ikcp_queue_clear(&kcp->rcv_queue, &kcp->rcv_queue_tail);
    ikcp_queue_clear(&kcp->snd_buf, &kcp->snd_buf_tail);
    ikcp_queue_clear(&kcp->rcv_buf, NULL);
    free(kcp->output_buf);
    free(kcp);
}

void kcp_lite_update(kcp_lite_t* kcp, uint32_t current_ms)
{
    kcp->current = current_ms;
    if (kcp->ts_updated == 0) {
        kcp->ts_updated = current_ms;
        kcp->ts_flush = current_ms;
    }
    int slap = (int)ikcp_time_diff(current_ms, kcp->ts_updated);
    if (slap >= 10000 || slap < -10000) {
        kcp->ts_updated = current_ms;
        slap = 0;
    }
    if (slap >= 0 && (uint32_t)slap >= kcp->interval) {
        kcp->ts_updated = current_ms;
        kcp_lite_flush(kcp);
    }
}

uint32_t kcp_lite_check(kcp_lite_t* kcp)
{
    uint32_t current = kcp->current;
    uint32_t ts_flush = kcp->ts_flush;
    uint32_t tm_flush = 0xffffffff;
    uint32_t tm_packet = 0xffffffff;
    uint32_t minimal = 0;

    if (kcp->ts_updated == 0) {
        return current;
    }

    if (ikcp_time_diff(current, ts_flush) >= 10000 || ikcp_time_diff(ts_flush, current) >= 10000) {
        ts_flush = current;
    }

    if (ikcp_time_diff(ts_flush, current) >= 10000) {
        return current;
    }

    tm_flush = ikcp_time_diff(ts_flush, current);

    ikcp_seg* seg = kcp->snd_buf;
    while (seg) {
        int diff = (int)ikcp_time_diff(seg->resendts, current);
        if (diff <= 0) {
            return current;
        }
        if ((uint32_t)diff < tm_packet) {
            tm_packet = (uint32_t)diff;
        }
        seg = seg->next;
    }

    minimal = tm_packet < tm_flush ? tm_packet : tm_flush;
    if (minimal >= kcp->interval) {
        minimal = kcp->interval;
    }

    return current + minimal;
}

int kcp_lite_send(kcp_lite_t* kcp, const uint8_t* data, int len)
{
    if (len < 0) return -1;
    if (len == 0) return 0;

    int count = 0;
    int sent = 0;

    if (len <= (int)kcp->mss) {
        count = 1;
    } else {
        count = (len + kcp->mss - 1) / kcp->mss;
    }

    if (count > 255) return -1;
    if (count == 0) count = 1;

    for (int i = 0; i < count; i++) {
        int size = len > (int)kcp->mss ? (int)kcp->mss : len;
        ikcp_seg* seg = ikcp_seg_alloc(size);
        if (!seg) return -1;

        memcpy(seg->data, data + sent, (size_t)size);
        seg->len = (uint32_t)size;
        seg->frg = (uint8_t)(count - i - 1);
        ikcp_queue_push(&kcp->snd_queue, &kcp->snd_queue_tail, seg);
        sent += size;
        len -= size;
    }

    return 0;
}

int kcp_lite_recv(kcp_lite_t* kcp, uint8_t* buf, int len)
{
    if (!kcp->rcv_queue) return -1;

    int peek_size = 0;
    ikcp_seg* seg = kcp->rcv_queue;
    while (seg) {
        peek_size += (int)seg->len;
        seg = seg->next;
        if (seg && seg->frg == 0) break;
    }

    if (peek_size > len) return -2;

    int copied = 0;
    seg = ikcp_queue_pop(&kcp->rcv_queue, &kcp->rcv_queue_tail);
    while (seg) {
        memcpy(buf + copied, seg->data, seg->len);
        copied += (int)seg->len;
        int frg = seg->frg;
        ikcp_seg_free(seg);
        if (frg == 0) break;
        seg = ikcp_queue_pop(&kcp->rcv_queue, &kcp->rcv_queue_tail);
    }

    return copied;
}

int kcp_lite_input(kcp_lite_t* kcp, const uint8_t* data, int len)
{
    if (len < (int)IKCP_OVERHEAD) return -1;

    uint32_t conv = ikcp_get_uint32(data);
    if (conv != kcp->conv) return -1;

    uint8_t  cmd  = data[4];
    uint8_t  frg  = data[5];
    uint16_t wnd  = ikcp_get_uint16(data + 6);
    uint32_t ts   = ikcp_get_uint32(data + 8);
    uint32_t sn   = ikcp_get_uint32(data + 12);
    uint32_t una  = ikcp_get_uint32(data + 16);
    uint32_t slen = ikcp_get_uint32(data + 20);
    (void)frg;

    if (len < (int)(IKCP_OVERHEAD + slen)) return -1;

    kcp->rmt_wnd = (uint32_t)wnd;

    if (ikcp_time_diff(una, kcp->snd_una) < 0x7fffffff) {
        ikcp_seg* cur = kcp->snd_buf;
        while (cur) {
            ikcp_seg* next = cur->next;
            if (ikcp_time_diff(una, cur->sn) < 0x7fffffff) {
                ikcp_snd_buf_remove(&kcp->snd_buf, &kcp->snd_buf_tail, cur);
            }
            cur = next;
        }
        kcp->snd_una = una;
    }

    if (cmd == IKCP_CMD_ACK) {
        if (ikcp_time_diff(kcp->current, ts) < 0x7fffffff) {
            ikcp_seg* cur = kcp->snd_buf;
            while (cur) {
                if (sn == cur->sn) {
                    ikcp_snd_buf_remove(&kcp->snd_buf, &kcp->snd_buf_tail, cur);
                    break;
                }
                if (ikcp_time_diff(sn, cur->sn) < 0x7fffffff) {
                    break;
                }
                cur = cur->next;
            }
        }
    } else if (cmd == IKCP_CMD_WASK) {
        uint8_t wins_buf[IKCP_OVERHEAD];
        ikcp_seg wins_seg;
        memset(&wins_seg, 0, sizeof(wins_seg));
        wins_seg.conv = kcp->conv;
        wins_seg.cmd = IKCP_CMD_WINS;
        wins_seg.wnd = (uint16_t)kcp->rcv_wnd;
        ikcp_encode_seg(wins_buf, &wins_seg);
        ikcp_output(kcp, wins_buf, IKCP_OVERHEAD);
        return 0;
    } else if (cmd == IKCP_CMD_PUSH) {
        if (ikcp_time_diff(sn, kcp->rcv_nxt + (uint32_t)kcp->rcv_wnd) < 0x7fffffff) {
            ikcp_seg ack_seg;
            memset(&ack_seg, 0, sizeof(ack_seg));
            ack_seg.conv = kcp->conv;
            ack_seg.cmd = IKCP_CMD_ACK;
            ack_seg.wnd = (uint16_t)kcp->rcv_wnd;
            ack_seg.sn = sn;
            ack_seg.una = kcp->rcv_nxt;
            ack_seg.ts = ts;

            ikcp_seg* nseg = ikcp_seg_alloc((int)slen);
            if (!nseg) return -1;

            nseg->conv = kcp->conv;
            nseg->cmd = cmd;
            nseg->frg = frg;
            nseg->wnd = wnd;
            nseg->ts = ts;
            nseg->sn = sn;
            nseg->una = una;
            nseg->len = slen;
            if (slen > 0) {
                memcpy(nseg->data, data + IKCP_OVERHEAD, slen);
            }

            if (ikcp_time_diff(sn, kcp->rcv_nxt) < 0x7fffffff) {
                nseg->next = kcp->rcv_buf;
                kcp->rcv_buf = nseg;
            } else if (sn == kcp->rcv_nxt) {
                kcp->rcv_nxt++;
                ikcp_queue_push(&kcp->rcv_queue, &kcp->rcv_queue_tail, nseg);
                ikcp_seg* cur = kcp->rcv_buf;
                ikcp_seg* prev = NULL;
                while (cur) {
                    if (cur->sn == kcp->rcv_nxt) {
                        kcp->rcv_nxt++;
                        if (prev) {
                            prev->next = cur->next;
                        } else {
                            kcp->rcv_buf = cur->next;
                        }
                        ikcp_queue_push(&kcp->rcv_queue, &kcp->rcv_queue_tail, cur);
                        cur = (prev) ? prev->next : kcp->rcv_buf;
                    } else {
                        prev = cur;
                        cur = cur->next;
                    }
                }
            } else {
                ikcp_seg* cur = kcp->rcv_buf;
                ikcp_seg* prev = NULL;
                while (cur) {
                    if (sn == cur->sn) {
                        ikcp_seg_free(nseg);
                        nseg = NULL;
                        break;
                    }
                    if (ikcp_time_diff(sn, cur->sn) < 0x7fffffff) {
                        nseg->next = cur;
                        if (prev) {
                            prev->next = nseg;
                        } else {
                            kcp->rcv_buf = nseg;
                        }
                        nseg = NULL;
                        break;
                    }
                    prev = cur;
                    cur = cur->next;
                }
                if (nseg) {
                    if (prev) {
                        prev->next = nseg;
                    } else {
                        kcp->rcv_buf = nseg;
                    }
                }
            }

            int ack_len = ikcp_encode_seg(kcp->output_buf, &ack_seg);
            ikcp_output(kcp, kcp->output_buf, ack_len);
        }
    }

    return 0;
}

void kcp_lite_flush(kcp_lite_t* kcp)
{
    uint32_t current = kcp->current;

    if (kcp->ts_updated == 0) return;

    ikcp_seg ack_seg;
    memset(&ack_seg, 0, sizeof(ack_seg));
    ack_seg.conv = kcp->conv;
    ack_seg.cmd = IKCP_CMD_ACK;
    ack_seg.wnd = (uint16_t)kcp->rcv_wnd;
    ack_seg.una = kcp->rcv_nxt;

    int offset = ikcp_encode_seg(kcp->output_buf, &ack_seg);

    ikcp_seg* seg = kcp->snd_queue;
    while (seg) {
        ikcp_seg* next = seg->next;

        seg->conv = kcp->conv;
        seg->cmd = IKCP_CMD_PUSH;
        seg->ts = current;
        seg->sn = kcp->snd_nxt++;
        seg->una = kcp->rcv_nxt;
        seg->wnd = (uint16_t)kcp->rcv_wnd;
        seg->xmit = 1;
        seg->rto = kcp->rx_rto;
        seg->resendts = current + seg->rto;
        seg->fastack = 0;

        int encode_len = ikcp_encode_seg(kcp->output_buf + offset, seg);
        ikcp_queue_push(&kcp->snd_buf, &kcp->snd_buf_tail, seg);

        offset += encode_len;

        seg = next;
    }

    kcp->snd_queue = NULL;
    kcp->snd_queue_tail = NULL;

    ikcp_seg* resend_seg = kcp->snd_buf;
    while (resend_seg) {
        ikcp_seg* next = resend_seg->next;
        if (ikcp_time_diff(current, resend_seg->resendts) < 0x7fffffff) {
            resend_seg->xmit++;
            resend_seg->rto += resend_seg->rto / 2;
            resend_seg->resendts = current + resend_seg->rto;
            resend_seg->ts = current;
            resend_seg->wnd = (uint16_t)kcp->rcv_wnd;
            resend_seg->una = kcp->rcv_nxt;

            int rlen = ikcp_encode_seg(kcp->output_buf + offset, resend_seg);
            offset += rlen;

            if (offset > IKCP_OVERHEAD + kcp->mtu) {
                ikcp_output(kcp, kcp->output_buf, offset);
                offset = ikcp_encode_seg(kcp->output_buf, &ack_seg);
            }
        }
        resend_seg = next;
    }

    if (offset > IKCP_OVERHEAD) {
        ikcp_output(kcp, kcp->output_buf, offset);
    }

    kcp->ts_flush = current;
    kcp->ts_updated = current;
}