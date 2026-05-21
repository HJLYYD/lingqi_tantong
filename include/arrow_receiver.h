#ifndef ARROW_RECEIVER_H
#define ARROW_RECEIVER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARROW_UART_BUFFER_SIZE 65536
#define ARROW_FRAME_MAGIC_1         0xA5
#define ARROW_FRAME_MAGIC_2         0x5A
#define ARROW_HEADER_SIZE           9
#define ARROW_UART_RETRY_COUNT      5
#define ARROW_UART_RETRY_DELAY_US   200000

#define ARROW_TYPE_JPEG             0x01
#define ARROW_TYPE_IMU_POSE         0x02

typedef struct ArrowReceiver ArrowReceiver;

ArrowReceiver* arrow_receiver_create(const char* uart_dev_path, int baud_rate);
void arrow_receiver_destroy(ArrowReceiver* receiver);
ARFramePair arrow_receiver_get_pair(ArrowReceiver* receiver);
bool arrow_receiver_is_connected(ArrowReceiver* receiver);

void arrow_receiver_update(ArrowReceiver* receiver);
bool arrow_receiver_get_latest_frame(ArrowReceiver* receiver, ArrowSourceFrame* out_frame);
int  arrow_receiver_add_secondary_link(ArrowReceiver* receiver, const char* uart_dev_path, int baud_rate);

#ifdef __cplusplus
}
#endif

#endif