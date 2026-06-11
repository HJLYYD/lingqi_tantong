#ifndef DISPLAY_OUTPUT_H
#define DISPLAY_OUTPUT_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DisplayOutput — unified real-time visualization output module.
 *
 * Routes rendered RGB24 visualization buffers to multiple output channels
 * simultaneously:
 *
 *   visualizer → vis_buffer (RGB24)
 *                    │
 *        ┌───────────┼───────────┐
 *        ▼           ▼           ▼
 *   Framebuffer   RTSP/UDP    Video File
 *   (/dev/fb0)    Stream      (MP4)
 *
 * Each output channel runs in its own thread to avoid blocking the
 * inference pipeline.  The write_frame() call is non-blocking — frames
 * are pushed into a lock-free ring buffer and consumed asynchronously.
 */

#define DISPLAY_MAX_OUTPUTS      4
#define DISPLAY_RING_SIZE        8
#define DISPLAY_FB_PATH_MAX      64
#define DISPLAY_URL_MAX          256

/* ── Output channel types ── */
typedef enum {
    DISPLAY_CHANNEL_NONE       = 0,
    DISPLAY_CHANNEL_FRAMEBUFFER = 1 << 0,   /* /dev/fb0 direct pixel write */
    DISPLAY_CHANNEL_RTSP       = 1 << 1,    /* RTSP streaming via ffmpeg   */
    DISPLAY_CHANNEL_UDP_TS     = 1 << 2,    /* UDP MPEG-TS via ffmpeg      */
    DISPLAY_CHANNEL_VIDEO_FILE = 1 << 3,    /* MP4 file via ffmpeg pipe    */
} DisplayChannelType;

typedef struct DisplayOutput DisplayOutput;

/*
 * Create a display output manager.
 *
 *   width, height  — frame dimensions (should match vis_buffer)
 *   fps            — target framerate (used by ffmpeg encoders)
 *   channels       — bitmask of DisplayChannelType flags
 *   fb_path        — framebuffer device path (e.g. "/dev/fb0"), may be NULL
 *   stream_url      — RTSP/UDP destination URL, may be NULL
 *   video_path     — output MP4 file path, may be NULL
 *
 * Returns NULL on failure (logs the reason).
 */
DisplayOutput* display_output_create(int width, int height, float fps,
                                     uint32_t channels,
                                     const char* fb_path,
                                     const char* stream_url,
                                     const char* video_path);

/*
 * Push a rendered frame to all enabled output channels.
 *
 * Non-blocking: the frame is copied into a ring buffer and consumed
 * by writer threads.  If the ring buffer is full the oldest un-consumed
 * frame is dropped (writer is falling behind the pipeline).
 *
 *   rgb_data  — RGB24 pixel buffer (width * height * 3 bytes)
 *
 * Returns 0 on success, -1 if no channels are enabled or data is NULL.
 */
int  display_output_write_frame(DisplayOutput* d, const uint8_t* rgb_data);

/*
 * Flush all pending frames and shut down writer threads.
 * Blocks until all channels have finished writing.
 */
void display_output_destroy(DisplayOutput* d);

/*
 * Query whether any channel is still active (has not encountered a fatal error).
 * If all channels are dead, the caller may choose to exit early.
 */
bool display_output_is_alive(const DisplayOutput* d);

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_OUTPUT_H */
