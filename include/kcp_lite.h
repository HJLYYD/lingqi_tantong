#ifndef KCP_LITE_H
#define KCP_LITE_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kcp_lite_t kcp_lite_t;

typedef struct {
    int mtu;
    int snd_wnd;
    int rcv_wnd;
    int nodelay;
    int interval_ms;
    int resend;
    int nc;
} kcp_lite_config_t;

kcp_lite_t* kcp_lite_create(uint32_t conv_id, kcp_lite_config_t config);
void kcp_lite_destroy(kcp_lite_t* kcp);

void kcp_lite_set_output(kcp_lite_t* kcp, int (*output)(const uint8_t* buf, int len, void* user), void* user);

int kcp_lite_send(kcp_lite_t* kcp, const uint8_t* data, int len);
int kcp_lite_recv(kcp_lite_t* kcp, uint8_t* buf, int len);
int kcp_lite_input(kcp_lite_t* kcp, const uint8_t* data, int len);

void kcp_lite_update(kcp_lite_t* kcp, uint32_t current_ms);
uint32_t kcp_lite_check(kcp_lite_t* kcp);
void kcp_lite_flush(kcp_lite_t* kcp);

#ifdef __cplusplus
}
#endif

#endif