#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "display_output.h"
#include "video_writer.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <sys/mman.h>

#ifdef HAS_SDL2
#include "display_sdl2.h"
#endif
#include <sys/ioctl.h>
#include <linux/fb.h>

/* ── Ring buffer slot ── */
typedef struct {
    uint8_t* data;
    size_t    size;
    bool      filled;
    int       frame_index;
} DisplaySlot;

/* ── Framebuffer channel ── */
typedef struct {
    int      fd;
    uint8_t* fb_mmap;
    size_t   fb_size;
    int      fb_width;
    int      fb_height;
    int      fb_bpp;         /* bits per pixel */
    int      fb_line_len;    /* bytes per scanline */
    bool     fb_is_bgr;      /* true if framebuffer expects BGR(A) order */
    bool     fb_has_alpha;   /* true if 32 bpp with alpha channel */
} DisplayFB;

/* ── FFmpeg pipe channel (shared by RTSP, UDP-TS) ── */
typedef struct {
    FILE*    pipe;
    bool     pipe_broken;
    char     cmd[1024];
} DisplayFFmpeg;

struct DisplayOutput {
    /* ── Configuration ── */
    int      width;
    int      height;
    float    fps;
    uint32_t channel_mask;

    /* ── Ring buffer ── */
    DisplaySlot slots[DISPLAY_RING_SIZE];
    int         write_idx;
    int         fb_read_idx;
    int         ffmpeg_read_idx;
    int         file_read_idx;
    int         active_slots;
    pthread_mutex_t ring_mutex;
    pthread_cond_t  data_cond;

    /* ── Channels ── */
    DisplayFB      fb;
    DisplayFFmpeg  ffmpeg;
    VideoWriter*   video_writer;

    /* ── Threads ── */
    pthread_t fb_thread;
    pthread_t ffmpeg_thread;
    pthread_t file_thread;
    bool      fb_thread_active;
    bool      ffmpeg_thread_active;
    bool      file_thread_active;
    volatile bool running;
    volatile bool shutdown;
};

/* ═══════════════════════════════════════════════════════════════════════
 *  Internal helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static void ring_init(DisplayOutput* d, size_t frame_bytes) {
    for (int i = 0; i < DISPLAY_RING_SIZE; i++) {
        d->slots[i].data = (uint8_t*)calloc(1, frame_bytes);
        d->slots[i].size = frame_bytes;
        d->slots[i].filled = false;
        d->slots[i].frame_index = -1;
    }
    d->write_idx = 0;
    d->fb_read_idx = 0;
    d->ffmpeg_read_idx = 0;
    d->file_read_idx = 0;
    d->active_slots = 0;
}

static void ring_destroy(DisplayOutput* d) {
    for (int i = 0; i < DISPLAY_RING_SIZE; i++) {
        free(d->slots[i].data);
        d->slots[i].data = NULL;
    }
}

/*
 * Latest-frame consumer: grab the most recently written filled slot
 * that hasn't been consumed yet (tracked per-channel via frame_index).
 */
static int ring_consume_latest(DisplayOutput* d, int* last_seen_frame,
                               uint8_t** out_data, size_t* out_size) {
    int best_slot = -1;
    int best_frame = *last_seen_frame;

    for (int i = 0; i < DISPLAY_RING_SIZE; i++) {
        DisplaySlot* s = &d->slots[i];
        if (!s->filled) continue;
        if (s->frame_index > best_frame) {
            best_frame = s->frame_index;
            best_slot = i;
        }
    }

    if (best_slot >= 0) {
        *last_seen_frame = best_frame;
        *out_data = d->slots[best_slot].data;
        *out_size = d->slots[best_slot].size;
        return 0;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Framebuffer
 * ═══════════════════════════════════════════════════════════════════════ */

static bool fb_open(DisplayOutput* d, const char* fb_path) {
    if (!fb_path || fb_path[0] == '\0') return false;

    /* O_NONBLOCK prevents hanging on DRM/KMS drivers that wait for
     * a physical display to be connected before completing open(). */
    d->fb.fd = open(fb_path, O_RDWR | O_NONBLOCK);
    if (d->fb.fd < 0) {
        log_warning("[Display] Cannot open framebuffer %s: %s", fb_path, strerror(errno));
        return false;
    }

    /* Clear O_NONBLOCK for normal read/write operations */
    int flags = fcntl(d->fb.fd, F_GETFL, 0);
    fcntl(d->fb.fd, F_SETFL, flags & ~O_NONBLOCK);

    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;

    if (ioctl(d->fb.fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        log_error("[Display] FBIOGET_VSCREENINFO failed: %s", strerror(errno));
        close(d->fb.fd);
        d->fb.fd = -1;
        return false;
    }

    if (ioctl(d->fb.fd, FBIOGET_FSCREENINFO, &finfo) < 0) {
        log_error("[Display] FBIOGET_FSCREENINFO failed: %s", strerror(errno));
        close(d->fb.fd);
        d->fb.fd = -1;
        return false;
    }

    d->fb.fb_width   = (int)vinfo.xres;
    d->fb.fb_height  = (int)vinfo.yres;
    d->fb.fb_bpp     = (int)vinfo.bits_per_pixel;
    d->fb.fb_line_len = (int)finfo.line_length;

    /* Detect pixel format from R/G/B offsets */
    d->fb.fb_is_bgr = (vinfo.red.offset > vinfo.blue.offset);
    d->fb.fb_has_alpha = (vinfo.transp.offset != 0 || vinfo.bits_per_pixel == 32);

    d->fb.fb_size = (size_t)finfo.smem_len;
    d->fb.fb_mmap = (uint8_t*)mmap(NULL, d->fb.fb_size,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED, d->fb.fd, 0);
    if (d->fb.fb_mmap == MAP_FAILED) {
        log_error("[Display] mmap framebuffer failed: %s", strerror(errno));
        close(d->fb.fd);
        d->fb.fd = -1;
        return false;
    }

    log_info("[Display] Framebuffer %s: %dx%d %dbpp %s, line=%d, size=%zuKB",
             fb_path, d->fb.fb_width, d->fb.fb_height, d->fb.fb_bpp,
             d->fb.fb_is_bgr ? "BGR" : "RGB",
             d->fb.fb_line_len, d->fb.fb_size / 1024);
    return true;
}

static void fb_close(DisplayFB* fb) {
    if (fb->fb_mmap && fb->fb_mmap != MAP_FAILED) {
        munmap(fb->fb_mmap, fb->fb_size);
    }
    if (fb->fd >= 0) {
        close(fb->fd);
    }
    memset(fb, 0, sizeof(*fb));
    fb->fd = -1;
}

/*
 * Convert RGB24 to framebuffer pixel format and blit.
 *
 * Handles:
 *   - 32 bpp BGRA (most common on ARM/RISC-V embedded GPUs)
 *   - 24 bpp BGR
 *   - 16 bpp RGB565
 *   - Center/squash if frame size differs from fb size
 */
static void fb_blit(const DisplayOutput* d, const uint8_t* rgb24) {
    const DisplayFB* fb = &d->fb;
    int src_w = d->width;
    int src_h = d->height;

    /* Compute destination region: center or scale-to-fit */
    float scale = 1.0f;
    int dst_w = src_w;
    int dst_h = src_h;
    int dst_x0 = 0;
    int dst_y0 = 0;

    if (fb->fb_width < src_w || fb->fb_height < src_h) {
        /* Scale down to fit */
        float sx = (float)fb->fb_width  / (float)src_w;
        float sy = (float)fb->fb_height / (float)src_h;
        scale = (sx < sy) ? sx : sy;
        dst_w = (int)(src_w * scale);
        dst_h = (int)(src_h * scale);
        dst_x0 = (fb->fb_width  - dst_w) / 2;
        dst_y0 = (fb->fb_height - dst_h) / 2;
    } else {
        /* Center */
        dst_x0 = (fb->fb_width  - src_w) / 2;
        dst_y0 = (fb->fb_height - src_h) / 2;
    }

    int bpp_bytes = fb->fb_bpp / 8;
    if (bpp_bytes < 2) bpp_bytes = 2;

    for (int y = 0; y < dst_h; y++) {
        int src_y = (int)(y / scale);
        if (src_y < 0) src_y = 0;
        if (src_y >= src_h) src_y = src_h - 1;

        for (int x = 0; x < dst_w; x++) {
            int src_x = (int)(x / scale);
            if (src_x < 0) src_x = 0;
            if (src_x >= src_w) src_x = src_w - 1;

            const uint8_t* sp = rgb24 + (src_y * src_w + src_x) * 3;
            uint8_t r = sp[0];
            uint8_t g = sp[1];
            uint8_t b = sp[2];

            int dx = dst_x0 + x;
            int dy = dst_y0 + y;
            int fb_off = dy * fb->fb_line_len + dx * bpp_bytes;

            if (fb_off + bpp_bytes > (int)fb->fb_size) continue;

            if (fb->fb_bpp == 32) {
                if (fb->fb_is_bgr) {
                    fb->fb_mmap[fb_off + 0] = b;
                    fb->fb_mmap[fb_off + 1] = g;
                    fb->fb_mmap[fb_off + 2] = r;
                    fb->fb_mmap[fb_off + 3] = 255;  /* alpha */
                } else {
                    fb->fb_mmap[fb_off + 0] = r;
                    fb->fb_mmap[fb_off + 1] = g;
                    fb->fb_mmap[fb_off + 2] = b;
                    fb->fb_mmap[fb_off + 3] = 255;
                }
            } else if (fb->fb_bpp == 24) {
                if (fb->fb_is_bgr) {
                    fb->fb_mmap[fb_off + 0] = b;
                    fb->fb_mmap[fb_off + 1] = g;
                    fb->fb_mmap[fb_off + 2] = r;
                } else {
                    fb->fb_mmap[fb_off + 0] = r;
                    fb->fb_mmap[fb_off + 1] = g;
                    fb->fb_mmap[fb_off + 2] = b;
                }
            } else if (fb->fb_bpp == 16) {
                /* RGB565 */
                uint16_t rgb565 = (uint16_t)((r >> 3) << 11) |
                                  (uint16_t)((g >> 2) << 5)  |
                                  (uint16_t)(b >> 3);
                fb->fb_mmap[fb_off + 0] = (uint8_t)(rgb565 & 0xFF);
                fb->fb_mmap[fb_off + 1] = (uint8_t)(rgb565 >> 8);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  FFmpeg pipe (RTSP / UDP-TS)
 * ═══════════════════════════════════════════════════════════════════════ */

static bool ffmpeg_open(DisplayFFmpeg* ff, int width, int height, float fps,
                        const char* stream_url) {
    if (!stream_url || stream_url[0] == '\0') return false;

    /*
     * Determine output format from URL prefix.
     *   rtsp://...  → RTSP (requires a media server like mediamtx)
     *   udp://...   → raw MPEG-TS over UDP (no server needed)
     *   rtmp://...  → RTMP (requires nginx-rtmp or similar)
     *
     * For simplicity, we detect the protocol and pick the right ffmpeg args.
     *
     * NOTE: RTSP output requires a running media server (e.g. MediaMTX).
     * UDP MPEG-TS is simpler — just point ffplay/VLC at udp://<addr>:<port>.
     */
    const char* fmt = NULL;
    const char* mux_opts = "";

    if (strncmp(stream_url, "rtsp://", 7) == 0) {
        fmt = "rtsp";
        mux_opts = "-rtsp_transport tcp";
    } else if (strncmp(stream_url, "udp://", 6) == 0) {
        fmt = "mpegts";
        mux_opts = "";
    } else if (strncmp(stream_url, "rtmp://", 7) == 0) {
        fmt = "flv";
        mux_opts = "";
    } else {
        log_error("[Display] Unsupported stream URL protocol: %s", stream_url);
        return false;
    }

    snprintf(ff->cmd, sizeof(ff->cmd),
        "ffmpeg -y -loglevel quiet "
        "-f rawvideo -vcodec rawvideo "
        "-s %dx%d -pix_fmt rgb24 -r %.0f "
        "-i - "
        "-c:v libx264 -preset ultrafast -tune zerolatency -crf 28 "
        "-pix_fmt yuv420p "
        "%s -f %s \"%s\"",
        width, height, fps,
        mux_opts, fmt, stream_url);

    log_info("[Display] Launching ffmpeg pipe: %s", ff->cmd);

    void (*old_sigpipe)(int) = signal(SIGPIPE, SIG_IGN);
    ff->pipe = popen(ff->cmd, "w");
    signal(SIGPIPE, old_sigpipe);

    if (!ff->pipe) {
        log_error("[Display] Failed to launch ffmpeg pipe");
        return false;
    }

    ff->pipe_broken = false;
    return true;
}

static void ffmpeg_close(DisplayFFmpeg* ff) {
    if (!ff->pipe) return;
    if (!ff->pipe_broken) {
        fflush(ff->pipe);
    }
    int rc = pclose(ff->pipe);
    if (rc != 0) {
        log_info("[Display] ffmpeg stream pipe closed (exit code %d)", rc);
    }
    ff->pipe = NULL;
}

static int ffmpeg_write(DisplayFFmpeg* ff, const uint8_t* data, size_t size) {
    if (!ff->pipe || ff->pipe_broken) return -1;

    clearerr(ff->pipe);
    size_t written = fwrite(data, 1, size, ff->pipe);
    if (written != size) {
        if (ferror(ff->pipe)) {
            log_warning("[Display] ffmpeg pipe write failed — stream broken");
            ff->pipe_broken = true;
        }
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Per-channel writer threads
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── SDL2 desktop-window writer (replaces fbdev when desktop is active) ── */

#ifdef HAS_SDL2
static void* sdl2_writer_thread(void* arg) {
    DisplayOutput* d = (DisplayOutput*)arg;
    int last_frame = -1;
    int frame_count = 0;

    log_info("[Display] SDL2 window thread started");

    while (d->running && sdl2_display_is_running()) {
        uint8_t* data = NULL;
        size_t   size = 0;

        pthread_mutex_lock(&d->ring_mutex);
        int rc = ring_consume_latest(d, &last_frame, &data, &size);
        pthread_mutex_unlock(&d->ring_mutex);

        if (rc == 0 && data) {
            sdl2_display_frame(data, d->width, d->height);
            frame_count++;
            if (frame_count % 30 == 0) {
                log_info("[Display] SDL2: rendered %d frames | %dx%d",
                         frame_count, d->width, d->height);
            }
        } else {
            SDL_Delay(10);  /* no new frame — keep event loop alive */
        }
    }

    log_info("[Display] SDL2 window exiting (%d frames)", frame_count);
    sdl2_display_close();
    return NULL;
}
#endif /* HAS_SDL2 */

/* ── Raw framebuffer (/dev/fb0) fallback writer ── */

static void* fb_writer_thread(void* arg) {
    DisplayOutput* d = (DisplayOutput*)arg;
    int last_frame = -1;
    int frame_count = 0;

    log_info("[Display] Framebuffer writer thread started");
    log_info("[Display] fb: output %dx%d → framebuffer %dx%d",
             d->width, d->height, d->fb.fb_width, d->fb.fb_height);

    while (d->running) {
        uint8_t* data = NULL;
        size_t   size = 0;

        pthread_mutex_lock(&d->ring_mutex);
        int rc = ring_consume_latest(d, &last_frame, &data, &size);
        pthread_mutex_unlock(&d->ring_mutex);

        if (rc == 0 && data) {
            fb_blit(d, data);
            frame_count++;
            if (frame_count % 30 == 0) {
                int offset_x = (d->fb.fb_width  > d->width)  ? (d->fb.fb_width  - d->width)  / 2 : 0;
                int offset_y = (d->fb.fb_height > d->height) ? (d->fb.fb_height - d->height) / 2 : 0;
                log_info("[Display] fb: wrote %d frames | %dx%d→%dx%d offset=(%d,%d)",
                         frame_count, d->width, d->height,
                         d->fb.fb_width, d->fb.fb_height, offset_x, offset_y);
            }
        } else {
            /* No new frame — short sleep to avoid busy-waiting */
            struct timespec ts = {0, 16666667L};  /* ~16ms (~60 Hz max) */
            nanosleep(&ts, NULL);
        }
    }

    log_info("[Display] Framebuffer writer exiting (%d frames)", frame_count);
    return NULL;
}

static void* ffmpeg_writer_thread(void* arg) {
    DisplayOutput* d = (DisplayOutput*)arg;
    int last_frame = -1;
    int frame_count = 0;
    size_t frame_bytes = (size_t)d->width * d->height * 3;

    log_info("[Display] FFmpeg stream writer thread started");

    while (d->running) {
        uint8_t* data = NULL;
        size_t   size = 0;

        pthread_mutex_lock(&d->ring_mutex);
        int rc = ring_consume_latest(d, &last_frame, &data, &size);
        pthread_mutex_unlock(&d->ring_mutex);

        if (rc == 0 && data && !d->ffmpeg.pipe_broken) {
            ffmpeg_write(&d->ffmpeg, data, frame_bytes);
            frame_count++;

            /* Periodic flush to keep latency low */
            if (frame_count % 15 == 0) {
                fflush(d->ffmpeg.pipe);
            }
        } else {
            struct timespec ts = {0, 10000000L};  /* 10ms */
            nanosleep(&ts, NULL);
        }
    }

    log_info("[Display] FFmpeg stream writer exiting (%d frames)", frame_count);
    return NULL;
}

static void* file_writer_thread(void* arg) {
    DisplayOutput* d = (DisplayOutput*)arg;
    int last_frame = -1;
    int frame_count = 0;

    log_info("[Display] Video file writer thread started");

    while (d->running) {
        uint8_t* data = NULL;
        size_t   size = 0;

        pthread_mutex_lock(&d->ring_mutex);
        int rc = ring_consume_latest(d, &last_frame, &data, &size);
        pthread_mutex_unlock(&d->ring_mutex);

        if (rc == 0 && data && d->video_writer) {
            video_writer_write_frame(d->video_writer, data);
            frame_count++;
        } else {
            struct timespec ts = {0, 10000000L};
            nanosleep(&ts, NULL);
        }
    }

    log_info("[Display] Video file writer exiting (%d frames)", frame_count);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Public API
 * ═══════════════════════════════════════════════════════════════════════ */

DisplayOutput* display_output_create(int width, int height, float fps,
                                     uint32_t channels,
                                     const char* fb_path,
                                     const char* stream_url,
                                     const char* video_path) {
    if (width <= 0 || height <= 0 || channels == DISPLAY_CHANNEL_NONE) {
        return NULL;
    }

    DisplayOutput* d = (DisplayOutput*)calloc(1, sizeof(DisplayOutput));
    if (!d) return NULL;

    d->width  = width;
    d->height = height;
    d->fps    = fps > 0.0f ? fps : 15.0f;
    d->channel_mask = channels;
    d->running = true;
    d->shutdown = false;
    d->fb.fd = -1;  /* avoid close(0) when SDL2 is active (fd=0 from calloc) */

    size_t frame_bytes = (size_t)width * height * 3;
    ring_init(d, frame_bytes);

    pthread_mutex_init(&d->ring_mutex, NULL);
    pthread_cond_init(&d->data_cond, NULL);

    /* ── Display channel: prefer SDL2 desktop window, fallback to fbdev ── */
    if (channels & DISPLAY_CHANNEL_FRAMEBUFFER) {
        bool display_ok = false;

        /* Detect headless session (SSH only, no desktop).
         * In a headless session, framebuffer open() can hang the K1 DRM driver
         * waiting for a display that isn't connected. Skip it proactively. */
        bool headless = (getenv("DISPLAY") == NULL && getenv("WAYLAND_DISPLAY") == NULL);

#ifdef HAS_SDL2
        /* Try SDL2 first — needs a real desktop (X11/Wayland) */
        if (!headless) {
            log_info("[Display] Desktop detected, trying SDL2...");
            if (sdl2_display_init(width, height)) {
                if (pthread_create(&d->fb_thread, NULL, sdl2_writer_thread, d) == 0) {
                    d->fb_thread_active = true;
                    display_ok = true;
                    log_info("[Display] Using SDL2 window (desktop visible)");
                } else {
                    log_warning("[Display] SDL2 thread creation failed, falling back");
                    sdl2_display_close();
                }
            }
        }
#endif

        /* fbdev fallback: use O_NONBLOCK to avoid hanging on DRM drivers
         * that wait indefinitely for a physical display connection. */
        if (!display_ok && fb_open(d, fb_path)) {
            pthread_create(&d->fb_thread, NULL, fb_writer_thread, d);
            d->fb_thread_active = true;
            display_ok = true;
        }

        if (!display_ok) {
            if (headless) {
                log_info("[Display] Headless SSH session — display output disabled "
                         "(set DISPLAY=:0 or connect a monitor for fbdev)");
            } else {
                log_warning("[Display] No display backend available — disabling");
            }
            d->channel_mask &= ~DISPLAY_CHANNEL_FRAMEBUFFER;
        }
    }

    /* ── FFmpeg streaming channel (RTSP or UDP-TS) ── */
    if (channels & (DISPLAY_CHANNEL_RTSP | DISPLAY_CHANNEL_UDP_TS)) {
        if (ffmpeg_open(&d->ffmpeg, width, height, fps, stream_url)) {
            pthread_create(&d->ffmpeg_thread, NULL, ffmpeg_writer_thread, d);
            d->ffmpeg_thread_active = true;
        } else {
            log_warning("[Display] FFmpeg stream unavailable — disabling");
            d->channel_mask &= ~(DISPLAY_CHANNEL_RTSP | DISPLAY_CHANNEL_UDP_TS);
        }
    }

    /* ── Video file channel ── */
    if (channels & DISPLAY_CHANNEL_VIDEO_FILE) {
        d->video_writer = video_writer_create(video_path, width, height, fps);
        if (d->video_writer) {
            pthread_create(&d->file_thread, NULL, file_writer_thread, d);
            d->file_thread_active = true;
        } else {
            log_warning("[Display] Video file output unavailable — disabling");
            d->channel_mask &= ~DISPLAY_CHANNEL_VIDEO_FILE;
        }
    }

    /* Log what's actually active */
    {
        char desc[256] = {0};
        if (d->fb_thread_active)    strcat(desc, "FB ");
        if (d->ffmpeg_thread_active) strcat(desc, "STREAM ");
        if (d->file_thread_active)   strcat(desc, "FILE ");
        if (desc[0] == '\0') strcpy(desc, "(none)");
        log_info("[Display] Output channels: %s | %dx%d @ %.1f FPS",
                 desc, width, height, fps);
    }

    return d;
}

int display_output_write_frame(DisplayOutput* d, const uint8_t* rgb_data) {
    if (!d || !rgb_data) return -1;
    if (!d->fb_thread_active && !d->ffmpeg_thread_active && !d->file_thread_active)
        return -1;

    size_t frame_bytes = (size_t)d->width * d->height * 3;
    static int global_frame_counter = 0;
    int fc = __sync_fetch_and_add(&global_frame_counter, 1);

    pthread_mutex_lock(&d->ring_mutex);

    /* Write into current slot, overwriting oldest data */
    DisplaySlot* slot = &d->slots[d->write_idx];
    memcpy(slot->data, rgb_data, frame_bytes);
    slot->filled = true;
    slot->frame_index = fc;
    d->write_idx = (d->write_idx + 1) % DISPLAY_RING_SIZE;
    if (d->active_slots < DISPLAY_RING_SIZE) {
        d->active_slots++;
    }

    pthread_cond_broadcast(&d->data_cond);
    pthread_mutex_unlock(&d->ring_mutex);

    return 0;
}

void display_output_destroy(DisplayOutput* d) {
    if (!d) return;

    /* Signal threads to stop */
    d->running = false;
    pthread_cond_broadcast(&d->data_cond);

    /* Join threads */
    if (d->fb_thread_active) {
        pthread_join(d->fb_thread, NULL);
        d->fb_thread_active = false;
    }
    if (d->ffmpeg_thread_active) {
        pthread_join(d->ffmpeg_thread, NULL);
        d->ffmpeg_thread_active = false;
    }
    if (d->file_thread_active) {
        pthread_join(d->file_thread, NULL);
        d->file_thread_active = false;
    }

    /* Close channels */
    fb_close(&d->fb);
    ffmpeg_close(&d->ffmpeg);
    if (d->video_writer) {
        video_writer_destroy(d->video_writer);
        d->video_writer = NULL;
    }

    /* Clean up ring buffer */
    ring_destroy(d);
    pthread_mutex_destroy(&d->ring_mutex);
    pthread_cond_destroy(&d->data_cond);

    free(d);
}

bool display_output_is_alive(const DisplayOutput* d) {
    if (!d) return false;
    /* Framebuffer rarely fails once opened */
    if (d->ffmpeg_thread_active && d->ffmpeg.pipe_broken) return false;
    return true;
}
