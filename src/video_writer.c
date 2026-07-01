/**
 * video_writer.c — Async video writer with dedicated encoding thread.
 *
 * Architecture: SPSC ring buffer → encoding thread → ffmpeg pipe.
 *
 * The main inference thread calls video_writer_write_frame() which copies
 * the frame into a pre-allocated ring buffer and returns immediately (never
 * blocks).  A background encoding thread drains the ring buffer and feeds
 * frames to ffmpeg at whatever rate the encoder can sustain.
 *
 * This decouples inference speed from encoding speed — critical on K1 where
 * software H.264 can only manage ~2-3 FPS at 720×1280.
 */

#include "video_writer.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>

/* ── Ring buffer ──────────────────────────────────────────────────────────
 * Small ring buffer (8 slots × 2.76 MB = ~22 MB).  The main thread writes
 * into the next slot and advances write_idx.  The encoding thread waits for
 * new frames and writes them to ffmpeg.  If the ring is full (encoder far
 * behind), the oldest unread frame is overwritten. */

#define VW_RING_SIZE  8

struct VideoWriter {
    /* ── Ring buffer (SPSC: main thread produces, encoder consumes) ── */
    uint8_t* ring_frames[VW_RING_SIZE];   /* pre-allocated frame buffers */
    int      ring_write_idx;              /* next slot to fill (main thread) */
    int      ring_read_idx;               /* next slot to drain (encoder thread) */
    int      ring_pending;                /* unread frames in ring */

    /* ── Encoding thread ── */
    pthread_t      enc_thread;
    pthread_mutex_t enc_mutex;
    pthread_cond_t  enc_cond;
    bool            enc_running;

    /* ── ffmpeg pipe ── */
    FILE* pipe;
    bool  pipe_broken;

    /* ── Metadata ── */
    int   width;
    int   height;
    float fps;
    int   frame_count;        /* frames pushed to ring */
    int   frames_encoded;     /* frames actually written to ffmpeg */
    int   frames_dropped;     /* frames overwritten (ring overflow) */
};

/* ── Encoding thread ──────────────────────────────────────────────────── */

static void* video_writer_thread(void* arg) {
    VideoWriter* vw = (VideoWriter*)arg;
    size_t frame_bytes = (size_t)vw->width * vw->height * 3;

    /* BUGFIX: staging buffer to avoid data race with producer.
     * Copy frame under mutex, then fwrite from staging outside.
     * This eliminates the known race where producer memcpy() could
     * overwrite the ring slot while fwrite() was reading it. */
    uint8_t* staging = (uint8_t*)malloc(frame_bytes);
    if (!staging) {
        log_critical("[VideoWriter] Failed to allocate staging buffer (%zu bytes)", frame_bytes);
        return NULL;
    }

    while (true) {
        pthread_mutex_lock(&vw->enc_mutex);

        while (vw->ring_pending == 0 && vw->enc_running) {
            pthread_cond_wait(&vw->enc_cond, &vw->enc_mutex);
        }

        if (!vw->enc_running && vw->ring_pending == 0) {
            pthread_mutex_unlock(&vw->enc_mutex);
            break;
        }

        /* Copy frame to staging under mutex, then release */
        memcpy(staging, vw->ring_frames[vw->ring_read_idx], frame_bytes);
        vw->ring_read_idx = (vw->ring_read_idx + 1) % VW_RING_SIZE;
        vw->ring_pending--;
        pthread_mutex_unlock(&vw->enc_mutex);

        if (vw->pipe_broken || !vw->pipe) {
            break;
        }

        clearerr(vw->pipe);
        size_t written = fwrite(staging, 1, frame_bytes, vw->pipe);
        if (written != frame_bytes) {
            if (ferror(vw->pipe)) {
                log_warning("[VideoWriter] ffmpeg pipe write error — encoding thread exiting");
                vw->pipe_broken = true;
                break;
            }
        } else {
            vw->frames_encoded++;
        }
    }

    free(staging);
    return NULL;
}

/* ── Public API ───────────────────────────────────────────────────────── */

VideoWriter* video_writer_create(const char* output_path, int width, int height, float fps) {
    if (!output_path || width <= 0 || height <= 0 || fps <= 0.0f) return NULL;

    VideoWriter* vw = (VideoWriter*)calloc(1, sizeof(VideoWriter));
    if (!vw) return NULL;

    vw->width  = width;
    vw->height = height;
    vw->fps    = fps;
    size_t frame_bytes = (size_t)width * height * 3;

    /* ── Pre-allocate ring buffer frames ── */
    for (int i = 0; i < VW_RING_SIZE; i++) {
        vw->ring_frames[i] = (uint8_t*)malloc(frame_bytes);
        if (!vw->ring_frames[i]) {
            for (int j = 0; j < i; j++) free(vw->ring_frames[j]);
            free(vw);
            return NULL;
        }
    }

    /* ── Launch ffmpeg pipe ──
     * Software libx264 — the K1 VPU hardware encoder (h264_v4l2m2m) has
     * known driver defects in the current BSP (MVX pixel-format rejection,
     * NV12 color corruption, V2D segfault).  See SpacemiT forum topics
     * #1299 and #1336.  Software encoding via async thread is the reliable
     * path until the BSP is updated.
     *
     * VFR (variable frame rate) with -use_wallclock_as_timestamps:
     * each frame gets its real arrival-time PTS instead of a synthetic
     * evenly-spaced one.  This ensures output duration ≈ session duration
     * even when the pipeline runs far below the nominal camera FPS.
     * -r <fps> is kept as the input timebase hint; -vsync passthrough
     * preserves the wallclock timestamps without CFR duplication/drop.
     *
     * SIGPIPE is kept ignored for the entire pipe lifetime so the encoding
     * thread detects ffmpeg exit via ferror() rather than being killed. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -loglevel error "
        "-f rawvideo -vcodec rawvideo "
        "-s %dx%d -pix_fmt rgb24 "
        "-use_wallclock_as_timestamps 1 "
        "-r %.2f "
        "-i - "
        "-c:v libx264 -preset ultrafast -crf 28 -tune zerolatency "
        "-pix_fmt yuv420p "
        "-vsync passthrough "
        "\"%s\" 2>/tmp/vw_ffmpeg.log",
        width, height, vw->fps, output_path);

    log_info("[VideoWriter] launching encoding thread → %s", output_path);

    /* Keep SIGPIPE ignored for the encoding thread lifetime.
     * Restoring SIG_DFL would kill the process when ffmpeg exits. */
    signal(SIGPIPE, SIG_IGN);
    vw->pipe = popen(cmd, "w");

    if (!vw->pipe) {
        log_error("[VideoWriter] failed to launch ffmpeg");
        for (int j = 0; j < VW_RING_SIZE; j++) free(vw->ring_frames[j]);
        free(vw);
        return NULL;
    }

    /* ── Start encoding thread ── */
    pthread_mutex_init(&vw->enc_mutex, NULL);
    pthread_cond_init(&vw->enc_cond, NULL);
    vw->enc_running = true;

    int ret = pthread_create(&vw->enc_thread, NULL, video_writer_thread, vw);
    if (ret != 0) {
        log_error("[VideoWriter] failed to create encoding thread: %s", strerror(ret));
        pclose(vw->pipe);
        pthread_mutex_destroy(&vw->enc_mutex);
        pthread_cond_destroy(&vw->enc_cond);
        for (int j = 0; j < VW_RING_SIZE; j++) free(vw->ring_frames[j]);
        free(vw);
        return NULL;
    }

    log_info("[VideoWriter] ready — %dx%d async (%d-slot ring buffer)", width, height, VW_RING_SIZE);
    return vw;
}

int video_writer_write_frame(VideoWriter* vw, const uint8_t* frame_data) {
    if (!vw || !frame_data) return -1;
    if (!vw->enc_running) return -2;

    size_t frame_bytes = (size_t)vw->width * vw->height * 3;

    pthread_mutex_lock(&vw->enc_mutex);

    if (vw->ring_pending >= VW_RING_SIZE) {
        /* Ring full — encoder is far behind.  Overwrite the oldest
         * unread frame (the one at ring_read_idx).  The encoder
         * thread hasn't read it yet, so we advance read_idx past it. */
        vw->ring_read_idx = (vw->ring_read_idx + 1) % VW_RING_SIZE;
        vw->ring_pending--;
        vw->frames_dropped++;
    }

    /* Copy frame into the ring (defensive: clamp slot to valid range) */
    if (vw->ring_write_idx < 0 || vw->ring_write_idx >= VW_RING_SIZE) {
        log_warning("[VideoWriter] ring_write_idx=%d out of bounds, resetting to 0", vw->ring_write_idx);
        vw->ring_write_idx = 0;
    }
    int slot = vw->ring_write_idx;
    memcpy(vw->ring_frames[slot], frame_data, frame_bytes);
    vw->ring_write_idx = (vw->ring_write_idx + 1) % VW_RING_SIZE;
    vw->ring_pending++;
    vw->frame_count++;

    pthread_cond_signal(&vw->enc_cond);
    pthread_mutex_unlock(&vw->enc_mutex);

    return vw->ring_pending < VW_RING_SIZE ? 0 : 1;  /* 1 = backpressure active */
}

int video_writer_flush(VideoWriter* vw) {
    if (!vw) return -1;

    pthread_mutex_lock(&vw->enc_mutex);
    /* Wait for encoder to drain the ring (with timeout) */
    int waited = 0;
    while (vw->ring_pending > 0 && waited < 300) {
        pthread_mutex_unlock(&vw->enc_mutex);
        usleep(100000);  /* 100ms */
        pthread_mutex_lock(&vw->enc_mutex);
        waited++;
    }
    pthread_mutex_unlock(&vw->enc_mutex);

    if (vw->pipe && !vw->pipe_broken) {
        fflush(vw->pipe);
    }
    return vw->ring_pending;
}

void video_writer_destroy(VideoWriter* vw) {
    if (!vw) return;

    log_info("[VideoWriter] Destroy: stopping encoder thread");
    log_flush();

    /* Signal encoder thread to shut down */
    pthread_mutex_lock(&vw->enc_mutex);
    vw->enc_running = false;
    pthread_cond_signal(&vw->enc_cond);
    pthread_mutex_unlock(&vw->enc_mutex);

    log_info("[VideoWriter] Destroy: joining encoder thread");
    log_flush();

    /* Wait for encoder thread to finish */
    pthread_join(vw->enc_thread, NULL);

    log_info("[VideoWriter] Destroy: closing ffmpeg pipe");
    log_flush();

    /* Close ffmpeg pipe */
    if (vw->pipe) {
        if (!vw->pipe_broken) {
            fflush(vw->pipe);
        }
        int rc = pclose(vw->pipe);
        if (rc != 0) {
            log_debug("[VideoWriter] ffmpeg exited with code %d", rc);
        }
    }

    /* Free ring buffer */
    for (int i = 0; i < VW_RING_SIZE; i++) {
        free(vw->ring_frames[i]);
    }

    pthread_mutex_destroy(&vw->enc_mutex);
    pthread_cond_destroy(&vw->enc_cond);

    log_info("[VideoWriter] finalized — %d pushed, %d encoded, %d dropped",
             vw->frame_count, vw->frames_encoded, vw->frames_dropped);
    log_flush();

    free(vw);
}
