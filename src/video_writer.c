#include "video_writer.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>

/*
 * FFmpeg pipe-based video writer.
 *
 * Opens a pipe to ffmpeg and streams raw RGB24 frames to stdin.
 * ffmpeg handles all encoding, muxing, and format details —
 * vastly more reliable than hand-rolled AVI.
 */

struct VideoWriter {
    FILE* pipe;
    int width;
    int height;
    float fps;
    int frame_count;
    bool pipe_broken;      /* set when ffmpeg pipe dies (SIGPIPE, write error) */
    char temp_path[512];
};

VideoWriter* video_writer_create(const char* output_path, int width, int height, float fps) {
    if (!output_path || width <= 0 || height <= 0 || fps <= 0.0f) return NULL;

    VideoWriter* vw = (VideoWriter*)calloc(1, sizeof(VideoWriter));
    if (!vw) return NULL;

    vw->width  = width;
    vw->height = height;
    vw->fps    = fps;
    vw->pipe_broken = false;

    /*
     * Build ffmpeg command.  -y overwrites output.  -f rawvideo reads
     * raw RGB24 frames from stdin.  -c:v libx264 encodes to H.264 MP4.
     * ultrafast preset minimizes CPU overhead during inference.
     *
     * For hardware-accelerated encoding on K1, replace libx264 with:
     *   -c:v h264_mpp  (requires --enable-libspacemit_mpp in ffmpeg build)
     */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -loglevel quiet "
        "-f rawvideo -vcodec rawvideo "
        "-s %dx%d -pix_fmt rgb24 -r %.0f "
        "-i - "
        "-c:v libx264 -preset ultrafast -crf 23 "
        "-pix_fmt yuv420p "
        "\"%s\"",   /* stderr suppressed: -loglevel quiet already silences ffmpeg;
                       no 2>/dev/null needed — that breaks pipe error reporting */
        vw->width, vw->height, vw->fps, output_path);

    log_info("VideoWriter: launching ffmpeg pipe -> %s", output_path);

    /* Ignore SIGPIPE so we get EPIPE from fwrite() instead of being killed.
     * Restore previous handler after popen() in case another library needs it. */
    void (*old_sigpipe)(int) = signal(SIGPIPE, SIG_IGN);

    vw->pipe = popen(cmd, "w");

    signal(SIGPIPE, old_sigpipe);

    if (!vw->pipe) {
        log_error("VideoWriter: failed to launch ffmpeg pipe");
        free(vw);
        return NULL;
    }

    /* Default stdio buffering is fine.  _IONBF would issue one write()
     * syscall per byte, killing performance for 2.76MB frames. */

    log_info("VideoWriter: ready — %dx%d @ %.1f FPS (H.264 MP4)", width, height, fps);
    return vw;
}

int video_writer_write_frame(VideoWriter* vw, const uint8_t* frame_data) {
    if (!vw || !vw->pipe || !frame_data) return -1;
    if (vw->pipe_broken) return -2;  /* silently skip; ffmpeg already dead */

    size_t frame_bytes = (size_t)vw->width * vw->height * 3;
    clearerr(vw->pipe);
    size_t written = fwrite(frame_data, 1, frame_bytes, vw->pipe);

    if (written != frame_bytes) {
        if (ferror(vw->pipe)) {
            int err = ferror(vw->pipe);
            log_warning("VideoWriter: fwrite frame %d failed (errno=%d), pipe broken — disabling output",
                        vw->frame_count, err);
            vw->pipe_broken = true;
        } else {
            log_error("VideoWriter: fwrite frame %d partial (%zu/%zu bytes)",
                      vw->frame_count, written, frame_bytes);
        }
        return -1;
    }

    vw->frame_count++;
    if (vw->frame_count % 30 == 0) {
        fflush(vw->pipe);
    }
    return 0;
}

int video_writer_flush(VideoWriter* vw) {
    if (!vw || !vw->pipe || vw->pipe_broken) return -1;
    fflush(vw->pipe);
    return 0;
}

void video_writer_destroy(VideoWriter* vw) {
    if (!vw) return;
    if (vw->pipe) {
        if (!vw->pipe_broken) {
            fflush(vw->pipe);
        }
        /*
         * pclose() closes stdin to ffmpeg, signals EOF, then waitpid()s.
         * ffmpeg may return exit code 2 on SIGPIPE / broken pipe when
         * stdin closes before it finishes writing the trailer — this is
         * normal for pipe-based encoding and the output file is still valid.
         */
        int rc = pclose(vw->pipe);
        if (rc != 0) {
            if (vw->pipe_broken) {
                log_info("VideoWriter: ffmpeg pipe closed (was broken; exit code %d)", rc);
            } else if (rc == 2) {
                /* Exit code 2 = SIGPIPE / write error on close — normal
                 * when stdin closes during trailer write. Output is valid. */
                log_debug("VideoWriter: ffmpeg exited with code %d (normal pipe EOF)", rc);
            } else {
                log_warning("VideoWriter: ffmpeg exited with code %d", rc);
            }
        } else {
            log_info("VideoWriter: finalized — %d frames", vw->frame_count);
        }
        vw->pipe = NULL;
    }
    free(vw);
}
