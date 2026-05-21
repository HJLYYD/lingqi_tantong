#include "video_writer.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * AVI 写入器崩溃恢复说明:
 *
 * Incremental 模式 (默认) 在 destroy() 时回写 RIFF/AVI 头部中的总帧数,
 * 以及 LIST hdrl 和 LIST strl 的大小字段。如果程序在 write_frame() 期间
 * 崩溃 (SIGSEGV, SIGKILL, 断电等), 这些头部字段将保持为 0, 导致生成的
 * .avi 文件无法被标准播放器 (VLC/ffplay) 识别或播放。
 *
 * 恢复建议:
 *   1. 短时录制 (<10min): 使用 buffered 模式 (incremental_mode=false),
 *      在 flush() 时一次性写入完整头部, 但需要足够内存缓冲所有帧。
 *   2. 对已损坏的 .avi 文件: 可用 ffmpeg 进行头部修复:
 *        ffmpeg -i damaged.avi -c copy repaired.avi
 *      或使用十六进制编辑器手动补全 RIFF 大小字段。
 *   3. 长期改进: 周期性回写头部 (每 N 帧), 或在文件末尾写入 idx1 索引块
 *      (OpenDML AVI 2.0 规范), 使播放器可通过索引定位帧数据。
 *
 * AVI 格式: Raw RGB24, 无压缩 (bi_compression=0), Bottom-Up (BITMAPINFOHEADER bi_height > 0)
 */

#define MAX_FRAMES_IN_BUFFER 64

struct VideoWriter {
    char output_path[MAX_PATH_LEN];
    int width;
    int height;
    float fps;
    int frame_count;
    uint8_t** frames;
    int* frame_sizes;
    int buffer_count;
    FILE* file;
    bool incremental_mode;
    int total_frames_written;
};

typedef struct {
    char riff[4];
    uint32_t file_size;
    char avi[4];
    char list1[4];
    uint32_t list1_size;
    char hdrl[4];
    char avih[4];
    uint32_t avih_size;
    uint32_t micro_sec_per_frame;
    uint32_t max_bytes_per_sec;
    uint32_t padding_granularity;
    uint32_t flags;
    uint32_t total_frames;
    uint32_t initial_frames;
    uint32_t streams;
    uint32_t suggested_buffer_size;
    uint32_t video_width;
    uint32_t video_height;
    uint32_t reserved[4];
} AVIMainHeader;

typedef struct {
    char list[4];
    uint32_t list_size;
    char strl[4];
    char strh[4];
    uint32_t strh_size;
    char fcc_type[4];
    char fcc_handler[4];
    uint32_t flags;
    uint16_t priority;
    uint16_t language;
    uint32_t initial_frames;
    uint32_t scale;
    uint32_t rate;
    uint32_t start;
    uint32_t length;
    uint32_t suggested_buffer_size;
    uint32_t quality;
    uint32_t sample_size;
    int16_t rc_left;
    int16_t rc_top;
    int16_t rc_right;
    int16_t rc_bottom;
    char strf[4];
    uint32_t strf_size;
    uint32_t bi_size;
    int32_t bi_width;
    int32_t bi_height;
    int16_t bi_planes;
    int16_t bi_bit_count;
    uint32_t bi_compression;
    uint32_t bi_size_image;
    int32_t bi_x_pels_per_meter;
    int32_t bi_y_pels_per_meter;
    uint32_t bi_clr_used;
    uint32_t bi_clr_important;
} AVIStreamHeader;

static void write_uint32(FILE* f, uint32_t val) {
    uint8_t buf[4];
    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;
    buf[2] = (val >> 16) & 0xFF;
    buf[3] = (val >> 24) & 0xFF;
    fwrite(buf, 1, 4, f);
}

static void write_uint16(FILE* f, uint16_t val) {
    uint8_t buf[2];
    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;
    fwrite(buf, 1, 2, f);
}

static void write_int32(FILE* f, int32_t val) {
    uint8_t buf[4];
    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;
    buf[2] = (val >> 16) & 0xFF;
    buf[3] = (val >> 24) & 0xFF;
    fwrite(buf, 1, 4, f);
}

static void write_int16(FILE* f, int16_t val) {
    uint16_t uval = (uint16_t)val;
    uint8_t buf[2];
    buf[0] = uval & 0xFF;
    buf[1] = (uval >> 8) & 0xFF;
    fwrite(buf, 1, 2, f);
}

static void write_fourcc(FILE* f, const char* fourcc) {
    fwrite(fourcc, 1, 4, f);
}

static int32_t pad_to_even(int32_t size) __attribute__((unused));
static int32_t pad_to_even(int32_t size) {
    return (size + 1) & ~1;
}

VideoWriter* video_writer_create(const char* output_path, int width, int height, float fps) {
    if (!output_path || width <= 0 || height <= 0 || fps <= 0.0f) return NULL;

    VideoWriter* vw = (VideoWriter*)calloc(1, sizeof(VideoWriter));
    if (!vw) return NULL;

    strncpy(vw->output_path, output_path, MAX_PATH_LEN - 1);
    vw->width = width;
    vw->height = height;
    vw->fps = fps;
    vw->frame_count = 0;
    vw->frames = (uint8_t**)calloc(MAX_FRAMES_IN_BUFFER, sizeof(uint8_t*));
    vw->frame_sizes = (int*)calloc(MAX_FRAMES_IN_BUFFER, sizeof(int));
    vw->buffer_count = 0;
    vw->file = NULL;
    vw->incremental_mode = true;
    vw->total_frames_written = 0;

    int row_size = (width * 3 + 3) & ~3;
    int frame_size = row_size * height;

    for (int i = 0; i < MAX_FRAMES_IN_BUFFER; i++) {
        vw->frames[i] = (uint8_t*)malloc(frame_size);
        if (!vw->frames[i]) {
            video_writer_destroy(vw);
            return NULL;
        }
        memset(vw->frames[i], 0, frame_size);
    }

    if (vw->incremental_mode) {
        vw->file = fopen(vw->output_path, "wb");
        if (vw->file) {
            long header_pos = ftell(vw->file);
            (void)header_pos;
            write_fourcc(vw->file, "RIFF");
            write_uint32(vw->file, 0);
            write_fourcc(vw->file, "AVI ");
            write_fourcc(vw->file, "LIST");
            write_uint32(vw->file, 0);
            write_fourcc(vw->file, "hdrl");
            write_fourcc(vw->file, "avih");
            write_uint32(vw->file, 56);
            long avih_pos = ftell(vw->file);
            (void)avih_pos;
            uint8_t zero_header[56];
            memset(zero_header, 0, 56);
            fwrite(zero_header, 1, 56, vw->file);

            write_fourcc(vw->file, "LIST");
            write_uint32(vw->file, 0);
            write_fourcc(vw->file, "strl");
            write_fourcc(vw->file, "strh");
            write_uint32(vw->file, 56);
            uint8_t zero_strh[56];
            memset(zero_strh, 0, 56);
            fwrite(zero_strh, 1, 56, vw->file);

            write_fourcc(vw->file, "strf");
            write_uint32(vw->file, 40);
            write_uint32(vw->file, 40);
            write_int32(vw->file, vw->width);
            write_int32(vw->file, vw->height);
            write_uint16(vw->file, 1);
            write_uint16(vw->file, 24);
            write_uint32(vw->file, 0);
            write_uint32(vw->file, frame_size);
            write_int32(vw->file, 0);
            write_int32(vw->file, 0);
            write_uint32(vw->file, 0);
            write_uint32(vw->file, 0);

            long movi_start = ftell(vw->file);
            (void)movi_start;
            write_fourcc(vw->file, "LIST");
            write_uint32(vw->file, 0);
            write_fourcc(vw->file, "movi");

            fflush(vw->file);
            log_info("VideoWriter created (incremental): %dx%d @ %.1f FPS", width, height, fps);
        } else {
            log_warning("Cannot open output file for incremental write, using buffered mode");
            vw->incremental_mode = false;
        }
    }

    if (!vw->incremental_mode) {
        log_info("VideoWriter created: %dx%d @ %.1f FPS", width, height, fps);
    }
    return vw;
}

int video_writer_write_frame(VideoWriter* vw, const uint8_t* frame_data) {
    if (!vw || !frame_data) return -1;

    if (vw->incremental_mode && vw->file) {
        int row_size = (vw->width * 3 + 3) & ~3;
        int frame_size = row_size * vw->height;

        uint8_t* frame_buf = (uint8_t*)malloc(frame_size);
        if (!frame_buf) return -1;

        memset(frame_buf, 0, frame_size);

        for (int y = vw->height - 1; y >= 0; y--) {
            int src_row_start = (vw->height - 1 - y) * vw->width * 3;
            int dst_row_start = y * row_size;

            for (int x = 0; x < vw->width; x++) {
                int src_idx = src_row_start + x * 3;
                int dst_idx = dst_row_start + x * 3;
                frame_buf[dst_idx + 0] = frame_data[src_idx + 2];
                frame_buf[dst_idx + 1] = frame_data[src_idx + 1];
                frame_buf[dst_idx + 2] = frame_data[src_idx + 0];
            }
        }

        write_fourcc(vw->file, "00db");
        write_uint32(vw->file, frame_size);
        fwrite(frame_buf, 1, frame_size, vw->file);
        free(frame_buf);

        vw->frame_count++;
        vw->total_frames_written++;

        if (vw->frame_count % 30 == 0) {
            fflush(vw->file);
        }

        return 0;
    }

    if (vw->buffer_count >= MAX_FRAMES_IN_BUFFER) {
        log_warning("Video buffer full, auto-flushing to disk");
        video_writer_flush(vw);
    }

    if (vw->buffer_count >= MAX_FRAMES_IN_BUFFER) {
        log_error("Video buffer still full after flush");
        return -1;
    }

    int row_size = (vw->width * 3 + 3) & ~3;
    int pixels = vw->width * vw->height;
    (void)pixels;

    uint8_t* frame_buf = vw->frames[vw->buffer_count];
    memset(frame_buf, 0, row_size * vw->height);

    for (int y = vw->height - 1; y >= 0; y--) {
        int src_row_start = (vw->height - 1 - y) * vw->width * 3;
        int dst_row_start = y * row_size;

        for (int x = 0; x < vw->width; x++) {
            int src_idx = src_row_start + x * 3;
            int dst_idx = dst_row_start + x * 3;

            frame_buf[dst_idx + 0] = frame_data[src_idx + 2];
            frame_buf[dst_idx + 1] = frame_data[src_idx + 1];
            frame_buf[dst_idx + 2] = frame_data[src_idx + 0];
        }
    }

    vw->frame_sizes[vw->buffer_count] = row_size * vw->height;
    vw->buffer_count++;
    vw->frame_count++;

    return 0;
}

int video_writer_flush(VideoWriter* vw) {
    if (!vw) return -1;

    if (vw->incremental_mode && vw->file) {
        fflush(vw->file);
        log_debug("VideoWriter incremental flush: %d frames", vw->frame_count);
        return 0;
    }

    if (vw->buffer_count == 0) return 0;

    FILE* flush_file = fopen(vw->output_path, "wb");
    if (!flush_file) {
        log_error("Cannot create output video file: %s", vw->output_path);
        return -1;
    }

    int row_size = (vw->width * 3 + 3) & ~3;
    int frame_data_size = row_size * vw->height;
    int total_frames = vw->buffer_count;

    uint32_t micro_sec_per_frame = (uint32_t)(1000000.0f / vw->fps);
    uint32_t max_bytes_per_sec = frame_data_size * (uint32_t)vw->fps;

    write_fourcc(flush_file, "RIFF");
    uint32_t file_size_pos = ftell(flush_file);
    write_uint32(flush_file, 0);

    write_fourcc(flush_file, "AVI ");
    write_fourcc(flush_file, "LIST");
    write_uint32(flush_file, 216);
    write_fourcc(flush_file, "hdrl");

    write_fourcc(flush_file, "avih");
    write_uint32(flush_file, 56);
    write_uint32(flush_file, micro_sec_per_frame);
    write_uint32(flush_file, max_bytes_per_sec);
    write_uint32(flush_file, 0);
    write_uint32(flush_file, 0x10);
    write_uint32(flush_file, total_frames);
    write_uint32(flush_file, 0);
    write_uint32(flush_file, 1);
    write_uint32(flush_file, frame_data_size);
    write_uint32(flush_file, vw->width);
    write_uint32(flush_file, vw->height);
    for (int i = 0; i < 4; i++) write_uint32(flush_file, 0);

    write_fourcc(flush_file, "LIST");
    write_uint32(flush_file, 124);
    write_fourcc(flush_file, "strl");

    write_fourcc(flush_file, "strh");
    write_uint32(flush_file, 56);
    write_fourcc(flush_file, "vids");
    write_fourcc(flush_file, "DIB ");
    write_uint32(flush_file, 0);
    write_uint16(flush_file, 0);
    write_uint16(flush_file, 0);
    write_uint32(flush_file, 0);
    write_uint32(flush_file, 1000000);
    write_uint32(flush_file, (uint32_t)(vw->fps * 1000000));
    write_uint32(flush_file, 0);
    write_uint32(flush_file, total_frames);
    write_uint32(flush_file, frame_data_size);
    write_uint32(flush_file, 0xFFFFFFFF);
    write_uint32(flush_file, 0);
    write_int16(flush_file, 0);
    write_int16(flush_file, 0);
    write_int16(flush_file, vw->width);
    write_int16(flush_file, vw->height);

    write_fourcc(flush_file, "strf");
    write_uint32(flush_file, 40);
    write_uint32(flush_file, 40);
    write_int32(flush_file, vw->width);
    write_int32(flush_file, vw->height);
    write_uint16(flush_file, 1);
    write_uint16(flush_file, 24);
    write_uint32(flush_file, 0);
    write_uint32(flush_file, frame_data_size);
    write_int32(flush_file, 0);
    write_int32(flush_file, 0);
    write_uint32(flush_file, 0);
    write_uint32(flush_file, 0);

    write_fourcc(flush_file, "LIST");
    uint32_t movi_list_pos = ftell(flush_file);
    (void)movi_list_pos;
    write_uint32(flush_file, 0);
    write_fourcc(flush_file, "movi");

    for (int i = 0; i < total_frames; i++) {
        write_fourcc(flush_file, "00db");
        write_uint32(flush_file, frame_data_size);
        fwrite(vw->frames[i], 1, frame_data_size, flush_file);
    }

    long file_size = ftell(flush_file);

    fseek(flush_file, file_size_pos, SEEK_SET);
    write_uint32(flush_file, file_size - 8);

    fclose(flush_file);

    log_info("VideoWriter flushed: %d frames to %s", total_frames, vw->output_path);

    vw->buffer_count = 0;

    return 0;
}

void video_writer_destroy(VideoWriter* vw) {
    if (!vw) return;

    if (vw->incremental_mode && vw->file) {
        fflush(vw->file);

        long current_pos = ftell(vw->file);

        int row_size = (vw->width * 3 + 3) & ~3;
        int frame_data_size = row_size * vw->height;
        uint32_t micro_sec_per_frame = (uint32_t)(1000000.0f / vw->fps);
        uint32_t max_bytes_per_sec = frame_data_size * (uint32_t)vw->fps;

        fseek(vw->file, 0, SEEK_SET);
        write_fourcc(vw->file, "RIFF");
        write_uint32(vw->file, current_pos - 8);
        write_fourcc(vw->file, "AVI ");
        write_fourcc(vw->file, "LIST");
        write_uint32(vw->file, 216);
        write_fourcc(vw->file, "hdrl");
        write_fourcc(vw->file, "avih");
        write_uint32(vw->file, 56);
        write_uint32(vw->file, micro_sec_per_frame);
        write_uint32(vw->file, max_bytes_per_sec);
        write_uint32(vw->file, 0);
        write_uint32(vw->file, 0x10);
        write_uint32(vw->file, vw->frame_count);
        write_uint32(vw->file, 0);
        write_uint32(vw->file, 1);
        write_uint32(vw->file, frame_data_size);
        write_uint32(vw->file, vw->width);
        write_uint32(vw->file, vw->height);
        for (int i = 0; i < 4; i++) write_uint32(vw->file, 0);

        write_fourcc(vw->file, "LIST");
        write_uint32(vw->file, 124);
        write_fourcc(vw->file, "strl");
        write_fourcc(vw->file, "strh");
        write_uint32(vw->file, 56);
        write_fourcc(vw->file, "vids");
        write_fourcc(vw->file, "DIB ");
        write_uint32(vw->file, 0);
        write_uint16(vw->file, 0);
        write_uint16(vw->file, 0);
        write_uint32(vw->file, 0);
        write_uint32(vw->file, 1000000);
        write_uint32(vw->file, (uint32_t)(vw->fps * 1000000));
        write_uint32(vw->file, 0);
        write_uint32(vw->file, vw->frame_count);
        write_uint32(vw->file, frame_data_size);
        write_uint32(vw->file, 0xFFFFFFFF);
        write_uint32(vw->file, 0);
        write_int16(vw->file, 0);
        write_int16(vw->file, 0);
        write_int16(vw->file, vw->width);
        write_int16(vw->file, vw->height);

        fclose(vw->file);
        vw->file = NULL;

        log_info("VideoWriter finalized: %s (%d frames)", vw->output_path, vw->frame_count);
    } else if (vw->buffer_count > 0) {
        video_writer_flush(vw);
    }

    if (vw->frames) {
        for (int i = 0; i < MAX_FRAMES_IN_BUFFER; i++) {
            if (vw->frames[i]) free(vw->frames[i]);
        }
        free(vw->frames);
    }
    free(vw->frame_sizes);
    free(vw);
}
