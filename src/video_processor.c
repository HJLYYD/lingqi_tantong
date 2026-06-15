#define _DEFAULT_SOURCE

#include "video_processor.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef HAS_V4L2
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#endif

#define VP_V4L2_BUFFER_COUNT    4

struct VideoProcessor {
    char input_path[MAX_PATH_LEN];
    int target_width;
    int target_height;
    bool normalize;
    bool is_opened;
    int original_width;
    int original_height;
    float fps;
    int frame_count;
    int total_frames;
    int frame_index;
    uint8_t* frame_buffer;
    size_t buffer_size;
    int source_type;

    /* FFmpeg pipe for decoding video files to raw RGB24.
     * Opened by video_processor_open() for VP_SOURCE_FILE,
     * read frame-by-frame in video_processor_read_frame_raw(). */
    FILE* decode_pipe;

#ifdef HAS_V4L2
    int v4l2_fd;
    uint8_t* v4l2_buffers[VP_V4L2_BUFFER_COUNT];
    size_t v4l2_buffer_sizes[VP_V4L2_BUFFER_COUNT];
    int v4l2_buffer_count;
    uint32_t v4l2_pixelformat;
    bool v4l2_streaming;
#endif
};

VideoProcessor* video_processor_create(const char* input_path, int target_width, int target_height, bool normalize) {
    VideoProcessor* vp = (VideoProcessor*)calloc(1, sizeof(VideoProcessor));
    if (!vp) return NULL;

    if (input_path) {
        strncpy(vp->input_path, input_path, MAX_PATH_LEN - 1);
        vp->input_path[MAX_PATH_LEN - 1] = '\0';
    }
    vp->target_width = target_width;
    vp->target_height = target_height;
    vp->normalize = normalize;
    vp->is_opened = false;
    vp->frame_count = 0;
    vp->frame_index = 0;
    vp->total_frames = 0;
    vp->fps = 30.0f;
    vp->frame_buffer = NULL;
    vp->buffer_size = 0;
    vp->source_type = VP_SOURCE_FILE;

#ifdef HAS_V4L2
    vp->v4l2_fd = -1;
    vp->v4l2_buffer_count = 0;
    vp->v4l2_streaming = false;
    for (int i = 0; i < VP_V4L2_BUFFER_COUNT; i++) {
        vp->v4l2_buffers[i] = NULL;
        vp->v4l2_buffer_sizes[i] = 0;
    }
#endif

    return vp;
}

#ifdef HAS_V4L2
static void v4l2_close_stream(VideoProcessor* vp) {
    if (vp->v4l2_fd < 0) return;
    if (vp->v4l2_streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(vp->v4l2_fd, VIDIOC_STREAMOFF, &type);
        vp->v4l2_streaming = false;
    }
    for (int i = 0; i < vp->v4l2_buffer_count; i++) {
        if (vp->v4l2_buffers[i] && vp->v4l2_buffer_sizes[i] > 0) {
            munmap(vp->v4l2_buffers[i], vp->v4l2_buffer_sizes[i]);
        }
        vp->v4l2_buffers[i] = NULL;
        vp->v4l2_buffer_sizes[i] = 0;
    }
    vp->v4l2_buffer_count = 0;
    close(vp->v4l2_fd);
    vp->v4l2_fd = -1;
}

static void yuyv_to_rgb(const uint8_t* src, int width, int height, uint8_t* dst) {
    int n_pairs = (width * height) / 2;
    for (int i = 0; i < n_pairs; i++) {
        int y0 = src[4 * i + 0];
        int u  = src[4 * i + 1];
        int y1 = src[4 * i + 2];
        int v  = src[4 * i + 3];

        int c0 = y0 - 16, c1 = y1 - 16;
        int d = u - 128, e = v - 128;

        int r0 = (298 * c0 + 409 * e + 128) >> 8;
        int g0 = (298 * c0 - 100 * d - 208 * e + 128) >> 8;
        int b0 = (298 * c0 + 516 * d + 128) >> 8;

        int r1 = (298 * c1 + 409 * e + 128) >> 8;
        int g1 = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
        int b1 = (298 * c1 + 516 * d + 128) >> 8;

        dst[6 * i + 0] = (uint8_t)UTILS_CLAMP(r0, 0, 255);
        dst[6 * i + 1] = (uint8_t)UTILS_CLAMP(g0, 0, 255);
        dst[6 * i + 2] = (uint8_t)UTILS_CLAMP(b0, 0, 255);
        dst[6 * i + 3] = (uint8_t)UTILS_CLAMP(r1, 0, 255);
        dst[6 * i + 4] = (uint8_t)UTILS_CLAMP(g1, 0, 255);
        dst[6 * i + 5] = (uint8_t)UTILS_CLAMP(b1, 0, 255);
    }
}

static int v4l2_open_stream(VideoProcessor* vp, const char* device_path,
                            int width, int height, float fps, const char* preferred_format) {
    int fd = open(device_path, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        log_error("V4L2: open(%s) failed: %s", device_path, strerror(errno));
        return -1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        log_error("V4L2: VIDIOC_QUERYCAP failed: %s", strerror(errno));
        close(fd);
        return -1;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        !(cap.capabilities & V4L2_CAP_STREAMING)) {
        log_error("V4L2: device %s lacks capture/streaming capability (caps=0x%x)",
                  device_path, cap.capabilities);
        close(fd);
        return -1;
    }

    log_info("V4L2: device %s (%s, %s)", device_path,
             cap.card[0] ? (const char*)cap.card : "unknown",
             cap.driver[0] ? (const char*)cap.driver : "unknown");

    uint32_t want_fmt = V4L2_PIX_FMT_MJPEG;
    if (preferred_format) {
        if (strcasecmp(preferred_format, "YUYV") == 0 ||
            strcasecmp(preferred_format, "YUV") == 0) {
            want_fmt = V4L2_PIX_FMT_YUYV;
        } else if (strcasecmp(preferred_format, "RGB24") == 0 ||
                   strcasecmp(preferred_format, "RGB") == 0) {
            want_fmt = V4L2_PIX_FMT_RGB24;
        }
    }

    struct v4l2_format vfmt;
    memset(&vfmt, 0, sizeof(vfmt));
    vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vfmt.fmt.pix.width = (uint32_t)width;
    vfmt.fmt.pix.height = (uint32_t)height;
    vfmt.fmt.pix.pixelformat = want_fmt;
    vfmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(fd, VIDIOC_S_FMT, &vfmt) != 0) {
        log_error("V4L2: VIDIOC_S_FMT failed: %s", strerror(errno));
        close(fd);
        return -1;
    }
    if (vfmt.fmt.pix.pixelformat != want_fmt) {
        uint32_t got = vfmt.fmt.pix.pixelformat;
        log_warning("V4L2: requested pixfmt 0x%x, driver chose 0x%x", want_fmt, got);
    }
    vp->v4l2_pixelformat = vfmt.fmt.pix.pixelformat;
    vp->original_width = (int)vfmt.fmt.pix.width;
    vp->original_height = (int)vfmt.fmt.pix.height;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_PARM, &parm) == 0 &&
        (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) {
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = (fps > 0.0f) ? (uint32_t)fps : 30;
        if (ioctl(fd, VIDIOC_S_PARM, &parm) != 0) {
            log_warning("V4L2: VIDIOC_S_PARM failed: %s", strerror(errno));
        } else {
            vp->fps = (float)parm.parm.capture.timeperframe.denominator /
                      (float)parm.parm.capture.timeperframe.numerator;
        }
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = VP_V4L2_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0) {
        log_error("V4L2: VIDIOC_REQBUFS failed: %s", strerror(errno));
        close(fd);
        return -1;
    }
    if (req.count < 2) {
        log_error("V4L2: insufficient mmap buffers (%u)", req.count);
        close(fd);
        return -1;
    }

    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
            log_error("V4L2: VIDIOC_QUERYBUF[%u] failed: %s", i, strerror(errno));
            close(fd);
            return -1;
        }
        void* ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (ptr == MAP_FAILED) {
            log_error("V4L2: mmap[%u] failed: %s", i, strerror(errno));
            close(fd);
            return -1;
        }
        vp->v4l2_buffers[i] = (uint8_t*)ptr;
        vp->v4l2_buffer_sizes[i] = buf.length;
        if (ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            log_error("V4L2: VIDIOC_QBUF[%u] failed: %s", i, strerror(errno));
            close(fd);
            return -1;
        }
    }
    vp->v4l2_buffer_count = (int)req.count;

    enum v4l2_buf_type stype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &stype) != 0) {
        log_error("V4L2: VIDIOC_STREAMON failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    vp->v4l2_fd = fd;
    vp->v4l2_streaming = true;
    vp->source_type = VP_SOURCE_CAMERA;
    return 0;
}

static FrameData* v4l2_read_frame(VideoProcessor* vp) {
    if (!vp || vp->v4l2_fd < 0 || !vp->v4l2_streaming) return NULL;

    fd_set fds;
    struct timeval tv;
    FD_ZERO(&fds);
    FD_SET(vp->v4l2_fd, &fds);
    tv.tv_sec = 0;
    tv.tv_usec = 200000;

    int r = select(vp->v4l2_fd + 1, &fds, NULL, NULL, &tv);
    if (r <= 0) {
        if (r == 0) log_debug("V4L2: select timeout");
        else log_warning("V4L2: select error: %s", strerror(errno));
        return NULL;
    }

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(vp->v4l2_fd, VIDIOC_DQBUF, &buf) != 0) {
        if (errno != EAGAIN) log_warning("V4L2: VIDIOC_DQBUF failed: %s", strerror(errno));
        return NULL;
    }
    if (buf.index >= (uint32_t)vp->v4l2_buffer_count) {
        log_error("V4L2: DQBUF returned bad index %u", buf.index);
        return NULL;
    }

    int out_w = vp->target_width > 0 ? vp->target_width : vp->original_width;
    int out_h = vp->target_height > 0 ? vp->target_height : vp->original_height;
    if (out_w <= 0 || out_h <= 0) {
        ioctl(vp->v4l2_fd, VIDIOC_QBUF, &buf);
        return NULL;
    }

    size_t out_bytes = (size_t)out_w * out_h * 3;
    FrameData* frame = (FrameData*)malloc(sizeof(FrameData));
    if (!frame) {
        ioctl(vp->v4l2_fd, VIDIOC_QBUF, &buf);
        return NULL;
    }
    frame->data = (uint8_t*)malloc(out_bytes);
    if (!frame->data) {
        free(frame);
        ioctl(vp->v4l2_fd, VIDIOC_QBUF, &buf);
        return NULL;
    }

    const uint8_t* payload = vp->v4l2_buffers[buf.index];
    size_t payload_len = buf.bytesused;
    int decode_ok = 0;

    if (vp->v4l2_pixelformat == V4L2_PIX_FMT_MJPEG) {
        if (soft_jpeg_decode_to_rgb(payload, payload_len, frame->data, out_w, out_h) != 0) {
            log_warning("V4L2: MJPEG decode failed for frame %d", vp->frame_index);
            decode_ok = -1;
        }
    } else if (vp->v4l2_pixelformat == V4L2_PIX_FMT_YUYV) {
        uint8_t* rgb_native = (uint8_t*)malloc((size_t)vp->original_width * vp->original_height * 3);
        if (rgb_native) {
            yuyv_to_rgb(payload, vp->original_width, vp->original_height, rgb_native);
            if (vp->original_width == out_w && vp->original_height == out_h) {
                memcpy(frame->data, rgb_native, out_bytes);
            } else {
                utils_resize_image(rgb_native, vp->original_width, vp->original_height,
                                   frame->data, out_w, out_h, 3);
            }
            free(rgb_native);
        } else {
            decode_ok = -1;
        }
    } else if (vp->v4l2_pixelformat == V4L2_PIX_FMT_RGB24) {
        if (vp->original_width == out_w && vp->original_height == out_h &&
            payload_len >= out_bytes) {
            memcpy(frame->data, payload, out_bytes);
        } else {
            utils_resize_image(payload, vp->original_width, vp->original_height,
                               frame->data, out_w, out_h, 3);
        }
    } else {
        log_error("V4L2: unsupported pixelformat 0x%x", vp->v4l2_pixelformat);
        decode_ok = -1;
    }

    ioctl(vp->v4l2_fd, VIDIOC_QBUF, &buf);

    if (decode_ok != 0) {
        free(frame->data);
        free(frame);
        return NULL;
    }

    frame->width = out_w;
    frame->height = out_h;
    frame->channels = 3;
    frame->frame_index = vp->frame_index++;
    frame->timestamp = vp->fps > 0.0f ? (double)frame->frame_index / vp->fps : 0.0;
    vp->frame_count++;
    return frame;
}
#endif

void video_processor_destroy(VideoProcessor* vp) {
    if (!vp) return;
    video_processor_close(vp);
    if (vp->frame_buffer) {
        free(vp->frame_buffer);
    }
    free(vp);
}

/*
 * Probe video metadata using ffprobe.
 * Returns 0 on success, fills width, height, fps, total_frames.
 * total_frames may be 0 if not available (streaming/unknown duration).
 */
static int ffprobe_video_info(const char* path, int* width, int* height,
                               float* fps, int* total_frames) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1:nokey=1 \"%s\" 2>/dev/null",
        path);

    FILE* fp = popen(cmd, "r");
    if (!fp) {
        log_error("ffprobe: failed to execute (is ffmpeg installed?)");
        return -1;
    }

    char line[128];
    int count = 0;
    int w = 0, h = 0, nf = 0;
    float f = 30.0f;

    while (fgets(line, sizeof(line), fp) && count < 4) {
        /* Strip trailing newline / CR */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        switch (count) {
        case 0: w = atoi(line); break;
        case 1: h = atoi(line); break;
        case 2: {
            int num = 30, den = 1;
            if (sscanf(line, "%d/%d", &num, &den) == 2 && den > 0)
                f = (float)num / (float)den;
            break;
        }
        case 3:
            if (strcmp(line, "N/A") != 0 && strlen(line) > 0)
                nf = atoi(line);
            break;
        }
        count++;
    }
    pclose(fp);

    if (w <= 0 || h <= 0) {
        log_error("ffprobe: failed to parse video dimensions for %s", path);
        return -1;
    }

    *width = w;
    *height = h;
    *fps = f;
    *total_frames = nf;

    log_info("ffprobe: %dx%d @ %.2f fps, %d frames", w, h, (double)f, nf);
    return 0;
}

int video_processor_open(VideoProcessor* vp, const char* input_path) {
    if (!vp) return VP_ERROR;

    if (vp->is_opened) {
        log_error("VideoProcessor: already opened — call video_processor_close() first");
        return VP_ERROR;
    }

    if (input_path) {
        strncpy(vp->input_path, input_path, MAX_PATH_LEN - 1);
        vp->input_path[MAX_PATH_LEN - 1] = '\0';
    }

    if (strlen(vp->input_path) == 0) {
        log_error("No input path specified");
        return VP_ERROR;
    }

    /* ── Probe video metadata via ffprobe ── */
    int width = 0, height = 0, total_frames = 0;
    float fps = 30.0f;
    if (ffprobe_video_info(vp->input_path, &width, &height, &fps, &total_frames) != 0) {
        log_error("Failed to probe video metadata: %s", vp->input_path);
        return VP_ERROR;
    }

    vp->original_width  = width;
    vp->original_height = height;
    vp->fps             = fps;
    vp->total_frames    = total_frames;

    int w = vp->target_width  > 0 ? vp->target_width  : vp->original_width;
    int h = vp->target_height > 0 ? vp->target_height : vp->original_height;

    vp->buffer_size  = (size_t)w * h * 3;
    vp->frame_buffer = (uint8_t*)malloc(vp->buffer_size);
    if (!vp->frame_buffer) {
        log_error("Failed to allocate frame buffer (%zux%zux3)", (size_t)w, (size_t)h);
        return VP_ERROR;
    }

    /* ── Pre-flight: check ffmpeg is available ── */
    {
        FILE* test = popen("ffmpeg -version 2>/dev/null", "r");
        if (!test) {
            log_error("VideoProcessor: ffmpeg not found. Install with: sudo apt install ffmpeg");
            free(vp->frame_buffer);
            vp->frame_buffer = NULL;
            return VP_ERROR;
        }
        pclose(test);
    }

    /* ── Open FFmpeg decode pipe ──
     * ffmpeg decodes the input video to raw RGB24 and writes to stdout.
     * This is the same pattern as video_writer.c (which uses ffmpeg for
     * encoding).  -an -sn skip audio/subtitle streams for speed. */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -v quiet -i \"%s\" -an -sn -f rawvideo -pix_fmt rgb24 "
        "-vcodec rawvideo -",
        vp->input_path);

    log_info("VideoProcessor: launching ffmpeg decoder for %s", vp->input_path);
    log_debug("  cmd: %s", cmd);
    vp->decode_pipe = popen(cmd, "r");
    if (!vp->decode_pipe) {
        log_error("VideoProcessor: failed to launch ffmpeg decode pipe (errno=%d: %s)",
                  errno, strerror(errno));
        free(vp->frame_buffer);
        vp->frame_buffer = NULL;
        return VP_ERROR;
    }

    /* Default stdio buffering (~8KB) is fine for sequential 2.76MB frame reads.
     * DO NOT use _IONBF here — it would issue one read() syscall per byte,
     * turning a ~2ms fread into a 5-second catastrophe. */

    vp->is_opened   = true;
    vp->frame_index = 0;
    vp->frame_count = 0;
    vp->source_type = VP_SOURCE_FILE;

    log_info("Video opened: %s", vp->input_path);
    log_info("  Resolution: %dx%d", vp->original_width, vp->original_height);
    log_info("  FPS: %.2f", (double)vp->fps);
    if (vp->total_frames > 0)
        log_info("  Total frames: %d", vp->total_frames);

    return VP_OK;
}

void video_processor_close(VideoProcessor* vp) {
    if (!vp) return;
    vp->is_opened = false;

    /* Close ffmpeg decode pipe (file source) */
    if (vp->decode_pipe) {
        int rc = pclose(vp->decode_pipe);
        if (rc != 0 && rc != 2) {
            /* Exit code 2 = SIGPIPE on close — normal for pipe decode */
            log_warning("VideoProcessor: ffmpeg decoder exited with code %d", rc);
        }
        vp->decode_pipe = NULL;
    }

#ifdef HAS_V4L2
    v4l2_close_stream(vp);
#endif

    if (vp->frame_buffer) {
        free(vp->frame_buffer);
        vp->frame_buffer = NULL;
        vp->buffer_size = 0;
    }

    log_info("Video capture closed");
}

bool video_processor_is_opened(const VideoProcessor* vp) {
    return vp ? vp->is_opened : false;
}

FrameData* video_processor_read_frame_raw(VideoProcessor* vp) {
    if (!vp || !vp->is_opened) return NULL;

#ifdef HAS_V4L2
    if (vp->source_type == VP_SOURCE_CAMERA && vp->v4l2_fd >= 0) {
        return v4l2_read_frame(vp);
    }
#endif

    /* ── File source: read one raw RGB24 frame from ffmpeg pipe ── */
    if (vp->total_frames > 0 && vp->frame_index >= vp->total_frames) {
        log_debug("Video EOF: frame %d of %d", vp->frame_index, vp->total_frames);
        return NULL;
    }

    int w = vp->target_width  > 0 ? vp->target_width  : vp->original_width;
    int h = vp->target_height > 0 ? vp->target_height : vp->original_height;
    size_t frame_bytes = (size_t)w * h * 3;

    if (!vp->decode_pipe) {
        log_error("VideoProcessor: decode pipe not open");
        return NULL;
    }

    /* Read exactly one raw RGB24 frame from ffmpeg's stdout */
    size_t n = fread(vp->frame_buffer, 1, frame_bytes, vp->decode_pipe);
    if (n != frame_bytes) {
        if (ferror(vp->decode_pipe)) {
            log_error("VideoProcessor: fread frame %d failed (pipe error)", vp->frame_index);
        }
        /* EOF reached or incomplete frame — ffmpeg finished */
        return NULL;
    }

    FrameData* frame = (FrameData*)malloc(sizeof(FrameData));
    if (!frame) {
        log_error("Failed to allocate FrameData");
        return NULL;
    }

    frame->data = (uint8_t*)malloc(frame_bytes);
    if (!frame->data) {
        log_error("Failed to allocate frame pixel buffer (%zu bytes)", frame_bytes);
        free(frame);
        return NULL;
    }

    memcpy(frame->data, vp->frame_buffer, frame_bytes);
    frame->width       = w;
    frame->height      = h;
    frame->channels    = 3;
    frame->frame_index = vp->frame_index++;
    frame->timestamp   = (double)frame->frame_index / (vp->fps > 0.0f ? vp->fps : 30.0f);

    vp->frame_count++;
    return frame;
}

FrameData* video_processor_read_frame(VideoProcessor* vp) {
    return video_processor_read_frame_raw(vp);
}

void frame_data_destroy(FrameData* frame) {
    if (!frame) return;
    if (frame->data) {
        free(frame->data);
    }
    free(frame);
}

float video_processor_get_fps(const VideoProcessor* vp) {
    return vp ? vp->fps : 0.0f;
}

int video_processor_get_frame_count(const VideoProcessor* vp) {
    return vp ? vp->frame_count : 0;
}

int video_processor_get_width(const VideoProcessor* vp) {
    return vp ? (vp->target_width > 0 ? vp->target_width : vp->original_width) : 0;
}

int video_processor_get_height(const VideoProcessor* vp) {
    return vp ? (vp->target_height > 0 ? vp->target_height : vp->original_height) : 0;
}

VideoProcessor* video_processor_create_from_camera(const char* device_path, int width, int height, float fps, const char* camera_format) {
    VideoProcessor* vp = (VideoProcessor*)calloc(1, sizeof(VideoProcessor));
    if (!vp) return NULL;

    if (device_path) {
        strncpy(vp->input_path, device_path, MAX_PATH_LEN - 1);
        vp->input_path[MAX_PATH_LEN - 1] = '\0';
    }

    vp->target_width = width;
    vp->target_height = height;
    vp->normalize = false;
    vp->is_opened = false;
    vp->fps = fps;
    vp->frame_count = 0;
    vp->frame_index = 0;
    vp->total_frames = 0;
    vp->buffer_size = (size_t)width * height * 3;
    vp->frame_buffer = (uint8_t*)malloc(vp->buffer_size);
    if (!vp->frame_buffer) {
        free(vp);
        return NULL;
    }
    vp->source_type = VP_SOURCE_CAMERA;

#ifdef HAS_V4L2
    vp->v4l2_fd = -1;
    vp->v4l2_buffer_count = 0;
    vp->v4l2_streaming = false;
    if (device_path && v4l2_open_stream(vp, device_path, width, height, fps, camera_format) != 0) {
        log_error("V4L2 capture init failed for %s", device_path);
        free(vp->frame_buffer);
        free(vp);
        return NULL;
    }
    vp->is_opened = true;
    log_info("V4L2 camera streaming: %s %dx%d@%.1ffps [pixfmt=0x%x, buffers=%d]",
             device_path, vp->original_width, vp->original_height,
             (double)vp->fps, vp->v4l2_pixelformat, vp->v4l2_buffer_count);
#else
    (void)camera_format;
    log_error("V4L2 support not compiled in (CMake build did not detect <linux/videodev2.h>)");
    free(vp->frame_buffer);
    free(vp);
    return NULL;
#endif

    return vp;
}

int video_processor_get_source_type(const VideoProcessor* vp) {
    if (!vp) return VP_SOURCE_FILE;
    if (vp->source_type == VP_SOURCE_CAMERA) return VP_SOURCE_CAMERA;
    if (vp->source_type == VP_SOURCE_ARROW) return VP_SOURCE_ARROW;
    if (vp->input_path[0]) {
        if (strncmp(vp->input_path, "/dev/video", 10) == 0) return VP_SOURCE_CAMERA;
        if (strncmp(vp->input_path, "arrow:", 6) == 0) return VP_SOURCE_ARROW;
    }
    return VP_SOURCE_FILE;
}
