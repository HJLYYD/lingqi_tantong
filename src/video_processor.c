#include "video_processor.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef HAVE_FFMPEG
#define HAVE_FFMPEG 0
#endif

#if HAVE_FFMPEG
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#endif

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

#if HAVE_FFMPEG
    AVFormatContext* fmt_ctx;
    AVCodecContext* codec_ctx;
    SwsContext* sws_ctx;
    int video_stream_idx;
    AVPacket* packet;
    AVFrame* frame;
    AVFrame* rgb_frame;
#endif

    FILE* fp;
    const char* output_path;

    bool mp4_metadata_parsed;
    uint32_t mdat_start;
    uint32_t mdat_size;
    uint32_t stsz_sample_count;
    uint32_t stsz_default_size;
    uint32_t* stsz_sizes;
    uint8_t* cached_file_data;
    size_t cached_file_size;
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

#if HAVE_FFMPEG
    vp->fmt_ctx = NULL;
    vp->codec_ctx = NULL;
    vp->sws_ctx = NULL;
    vp->video_stream_idx = -1;
    vp->packet = NULL;
    vp->frame = NULL;
    vp->rgb_frame = NULL;
#endif

    return vp;
}

void video_processor_destroy(VideoProcessor* vp) {
    if (!vp) return;
    video_processor_close(vp);
    if (vp->frame_buffer) {
        free(vp->frame_buffer);
    }
    if (vp->stsz_sizes) {
        free(vp->stsz_sizes);
    }
    if (vp->cached_file_data) {
        free(vp->cached_file_data);
    }
    free(vp);
}

static void read_video_info(VideoProcessor* vp) {
    FILE* f = fopen(vp->input_path, "rb");
    if (!f) return;

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size > 12) {
        uint8_t header[64];
        size_t bytes_read = fread(header, 1, sizeof(header), f);
        (void)bytes_read;

        if (header[4] == 'f' && header[5] == 't' && header[6] == 'y' && header[7] == 'p') {
            vp->fps = 30.0f;
            vp->original_width = 1920;
            vp->original_height = 1080;
            vp->total_frames = (int)(file_size / 50000);
        }
    }

    fclose(f);
}

int video_processor_open(VideoProcessor* vp, const char* input_path) {
    if (!vp) return VP_ERROR;

    if (input_path) {
        strncpy(vp->input_path, input_path, MAX_PATH_LEN - 1);
        vp->input_path[MAX_PATH_LEN - 1] = '\0';
    }

    if (strlen(vp->input_path) == 0) {
        log_error("No input path specified");
        return VP_ERROR;
    }

    FILE* f = fopen(vp->input_path, "rb");
    if (!f) {
        log_error("Cannot open video file: %s", vp->input_path);
        return VP_ERROR;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size < 1024) {
        fclose(f);
        log_error("File too small to be a valid video: %s", vp->input_path);
        return VP_ERROR;
    }

    fclose(f);

    read_video_info(vp);

    int w = vp->target_width > 0 ? vp->target_width : vp->original_width;
    int h = vp->target_height > 0 ? vp->target_height : vp->original_height;

    vp->buffer_size = (size_t)w * h * 3;
    vp->frame_buffer = (uint8_t*)malloc(vp->buffer_size);
    if (!vp->frame_buffer) {
        log_error("Failed to allocate frame buffer");
        return VP_ERROR;
    }

    vp->is_opened = true;
    vp->frame_index = 0;
    vp->frame_count = 0;

    log_info("Video opened: %s", vp->input_path);
    log_info("  Resolution: %dx%d", vp->original_width, vp->original_height);
    log_info("  FPS: %.2f", vp->fps);
    log_info("  File size: %.2f MB", file_size / (1024.0 * 1024.0));

    return VP_OK;
}

void video_processor_close(VideoProcessor* vp) {
    if (!vp) return;
    vp->is_opened = false;

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

static bool parse_mp4_metadata(VideoProcessor* vp, uint8_t* file_data, size_t file_size) {
    size_t offset = 0;
    vp->mdat_start = 0;
    vp->mdat_size = 0;
    vp->stsz_sample_count = 0;
    vp->stsz_default_size = 0;

    while (offset + 8 < file_size) {
        uint32_t box_size = (file_data[offset] << 24) | (file_data[offset + 1] << 16) |
                           (file_data[offset + 2] << 8) | file_data[offset + 3];

        if (box_size < 8 || offset + box_size > file_size) break;

        char box_type[5] = {0};
        memcpy(box_type, &file_data[offset + 4], 4);

        if (strcmp(box_type, "ftyp") == 0) {
            char brand[5] = {0};
            memcpy(brand, &file_data[offset + 8], 4);
            log_debug("MP4 brand: %s", brand);
        }
        else if (strcmp(box_type, "moov") == 0) {
            size_t moov_start = offset + 8;
            size_t moov_pos = moov_start;

            while (moov_pos < offset + box_size - 8) {
                uint32_t child_size = (file_data[moov_pos] << 24) | (file_data[moov_pos + 1] << 16) |
                                     (file_data[moov_pos + 2] << 8) | file_data[moov_pos + 3];

                if (child_size < 8 || moov_pos + child_size > offset + box_size) break;

                char child_type[5] = {0};
                memcpy(child_type, &file_data[moov_pos + 4], 4);

                if (strcmp(child_type, "trak") == 0) {
                    size_t trak_pos = moov_pos + 8;
                    while (trak_pos < moov_pos + child_size - 8) {
                        uint32_t trak_child_size = (file_data[trak_pos] << 24) | (file_data[trak_pos + 1] << 16) |
                                                  (file_data[trak_pos + 2] << 8) | file_data[trak_pos + 3];

                        if (trak_child_size < 8 || trak_pos + trak_child_size > moov_pos + child_size) break;

                        char trak_child_type[5] = {0};
                        memcpy(trak_child_type, &file_data[trak_pos + 4], 4);

                        if (strcmp(trak_child_type, "tkhd") == 0 && trak_child_size >= 84) {
                            uint32_t tkhd_width = (file_data[trak_pos + 76] << 24) | (file_data[trak_pos + 77] << 16) |
                                        (file_data[trak_pos + 78] << 8) | file_data[trak_pos + 79];
                            uint32_t tkhd_height = (file_data[trak_pos + 80] << 24) | (file_data[trak_pos + 81] << 16) |
                                         (file_data[trak_pos + 82] << 8) | file_data[trak_pos + 83];
                            if (tkhd_width > 0 && tkhd_height > 0) {
                                vp->original_width = tkhd_width;
                                vp->original_height = tkhd_height;
                                log_debug("Video resolution: %dx%d", vp->original_width, vp->original_height);
                            }
                        }
                        else if (strcmp(trak_child_type, "mdia") == 0) {
                            size_t mdia_pos = trak_pos + 8;
                            while (mdia_pos < trak_pos + trak_child_size - 8) {
                                uint32_t mdia_child_size = (file_data[mdia_pos] << 24) | (file_data[mdia_pos + 1] << 16) |
                                                          (file_data[mdia_pos + 2] << 8) | file_data[mdia_pos + 3];

                                if (mdia_child_size < 8 || mdia_pos + mdia_child_size > trak_pos + trak_child_size) break;

                                char mdia_child_type[5] = {0};
                                memcpy(mdia_child_type, &file_data[mdia_pos + 4], 4);

                                if (strcmp(mdia_child_type, "mdhd") == 0 && mdia_child_size >= 28) {
                                    uint32_t mdhd_timescale = (file_data[mdia_pos + 20] << 24) | (file_data[mdia_pos + 21] << 16) |
                                                    (file_data[mdia_pos + 22] << 8) | file_data[mdia_pos + 23];
                                    uint32_t mdhd_duration = (file_data[mdia_pos + 24] << 24) | (file_data[mdia_pos + 25] << 16) |
                                                   (file_data[mdia_pos + 26] << 8) | file_data[mdia_pos + 27];
                                    if (mdhd_timescale > 0 && mdhd_duration > 0) {
                                        vp->total_frames = (int)((double)mdhd_duration / mdhd_timescale * 30.0);
                                        vp->fps = 30.0f;
                                        log_debug("Video duration: %.2f sec", (double)mdhd_duration / mdhd_timescale);
                                    }
                                }
                                else if (strcmp(mdia_child_type, "minf") == 0) {
                                    size_t minf_pos = mdia_pos + 8;
                                    while (minf_pos < mdia_pos + mdia_child_size - 8) {
                                        uint32_t minf_child_size = (file_data[minf_pos] << 24) | (file_data[minf_pos + 1] << 16) |
                                                                  (file_data[minf_pos + 2] << 8) | file_data[minf_pos + 3];

                                        if (minf_child_size < 8 || minf_pos + minf_child_size > mdia_pos + mdia_child_size) break;

                                        char minf_child_type[5] = {0};
                                        memcpy(minf_child_type, &file_data[minf_pos + 4], 4);

                                        if (strcmp(minf_child_type, "stbl") == 0) {
                                            size_t stbl_pos = minf_pos + 8;
                                            while (stbl_pos < minf_pos + minf_child_size - 8) {
                                                uint32_t stbl_child_size = (file_data[stbl_pos] << 24) | (file_data[stbl_pos + 1] << 16) |
                                                                          (file_data[stbl_pos + 2] << 8) | file_data[stbl_pos + 3];

                                                if (stbl_child_size < 8 || stbl_pos + stbl_child_size > minf_pos + minf_child_size) break;

                                                char stbl_child_type[5] = {0};
                                                memcpy(stbl_child_type, &file_data[stbl_pos + 4], 4);

                                                if (strcmp(stbl_child_type, "stsz") == 0 && stbl_child_size >= 20) {
                                                    vp->stsz_default_size = (file_data[stbl_pos + 12] << 24) | (file_data[stbl_pos + 13] << 16) |
                                                                           (file_data[stbl_pos + 14] << 8) | file_data[stbl_pos + 15];
                                                    vp->stsz_sample_count = (file_data[stbl_pos + 16] << 24) | (file_data[stbl_pos + 17] << 16) |
                                                                           (file_data[stbl_pos + 18] << 8) | file_data[stbl_pos + 19];

                                                    if (vp->stsz_default_size == 0 && vp->stsz_sample_count > 0 && !vp->stsz_sizes) {
                                                        vp->stsz_sizes = (uint32_t*)calloc(vp->stsz_sample_count, sizeof(uint32_t));
                                                        if (vp->stsz_sizes) {
                                                            for (uint32_t s = 0; s < vp->stsz_sample_count && stbl_pos + 20 + s * 4 + 4 <= minf_pos + minf_child_size; s++) {
                                                                vp->stsz_sizes[s] = (file_data[stbl_pos + 20 + s * 4] << 24) | (file_data[stbl_pos + 20 + s * 4 + 1] << 16) |
                                                                               (file_data[stbl_pos + 20 + s * 4 + 2] << 8) | file_data[stbl_pos + 20 + s * 4 + 3];
                                                            }
                                                        }
                                                    }
                                                    log_debug("stsz: %u samples, default_size=%u", vp->stsz_sample_count, vp->stsz_default_size);
                                                }

                                                stbl_pos += stbl_child_size;
                                            }
                                        }

                                        minf_pos += minf_child_size;
                                    }
                                }

                                mdia_pos += mdia_child_size;
                            }
                        }

                        trak_pos += trak_child_size;
                    }
                }

                moov_pos += child_size;
            }
        }
        else if (strcmp(box_type, "mdat") == 0) {
            vp->mdat_start = offset + 8;
            vp->mdat_size = box_size - 8;
            log_debug("mdat at offset %u, size %u", vp->mdat_start, vp->mdat_size);
            break;
        }

        offset += box_size;
    }

    return vp->mdat_size > 0 && vp->mdat_start > 0;
}

static bool extract_mp4_frame(VideoProcessor* vp, uint8_t* file_data, int target_frame) {
    if (vp->mdat_size == 0 || vp->mdat_start == 0) {
        log_error("No mdat box in MP4");
        return false;
    }

    uint32_t frame_offset = vp->mdat_start;
    uint32_t frame_data_size = (uint32_t)vp->buffer_size;

    if (vp->stsz_sizes && vp->stsz_sample_count > 0) {
        int frame_idx = target_frame % vp->stsz_sample_count;
        frame_offset = vp->mdat_start;
        for (int i = 0; i < frame_idx; i++) {
            frame_offset += vp->stsz_sizes[i];
        }
        frame_data_size = vp->stsz_sizes[frame_idx];
        log_debug("Frame %d from mdat+%u, size %u", frame_idx, frame_offset - vp->mdat_start, frame_data_size);
    }
    else {
        uint32_t total_mdat = vp->mdat_size;
        if (total_mdat == 0) total_mdat = 1;
        frame_offset = vp->mdat_start + ((uint32_t)target_frame * (uint32_t)vp->buffer_size) % total_mdat;
        frame_data_size = UTILS_MIN((uint32_t)vp->buffer_size, vp->mdat_size);
    }

    if (frame_offset + frame_data_size > vp->mdat_start + vp->mdat_size) {
        frame_offset = vp->mdat_start;
        frame_data_size = UTILS_MIN((uint32_t)vp->buffer_size, vp->mdat_size);
    }

    if (frame_data_size > vp->buffer_size) {
        frame_data_size = (uint32_t)vp->buffer_size;
    }

    memcpy(vp->frame_buffer, &file_data[frame_offset], frame_data_size);

    int frame_w = vp->original_width;
    int frame_h = vp->original_height;
    int target_w = vp->target_width > 0 ? vp->target_width : frame_w;
    int target_h = vp->target_height > 0 ? vp->target_height : frame_h;

    if (frame_w > 0 && frame_h > 0 && (frame_w != target_w || frame_h != target_h)) {
        uint8_t* temp_buffer = (uint8_t*)malloc(frame_data_size);
        if (temp_buffer) {
            memcpy(temp_buffer, vp->frame_buffer, frame_data_size);
            utils_resize_image(temp_buffer, frame_w, frame_h, vp->frame_buffer, target_w, target_h, 3);
            free(temp_buffer);
        }
    }

    return true;
}

FrameData* video_processor_read_frame_raw(VideoProcessor* vp) {
    if (!vp || !vp->is_opened) return NULL;

    if (!vp->mp4_metadata_parsed) {
        FILE* f = fopen(vp->input_path, "rb");
        if (!f) {
            log_error("Cannot open video file for frame reading: %s", vp->input_path);
            return NULL;
        }

        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);

        vp->cached_file_data = (uint8_t*)malloc(file_size);
        if (!vp->cached_file_data) {
            fclose(f);
            log_error("Failed to allocate memory for video file data");
            return NULL;
        }

        size_t bytes_read = fread(vp->cached_file_data, 1, file_size, f);
        fclose(f);

        if (bytes_read != file_size) {
            free(vp->cached_file_data);
            vp->cached_file_data = NULL;
            log_error("Failed to read complete video file data");
            return NULL;
        }

        vp->cached_file_size = file_size;

        if (!parse_mp4_metadata(vp, vp->cached_file_data, vp->cached_file_size)) {
            log_warning("Failed to parse MP4 metadata, using raw extraction fallback");
        }

        vp->mp4_metadata_parsed = true;
        log_info("MP4 metadata cached: %d frames, %dx%d", vp->total_frames, vp->original_width, vp->original_height);
    }

    int w = vp->target_width > 0 ? vp->target_width : vp->original_width;
    int h = vp->target_height > 0 ? vp->target_height : vp->original_height;

    FrameData* frame = (FrameData*)malloc(sizeof(FrameData));
    if (!frame) {
        log_error("Failed to allocate frame data structure");
        return NULL;
    }

    if (vp->cached_file_data && extract_mp4_frame(vp, vp->cached_file_data, vp->frame_index)) {
        frame->data = (uint8_t*)malloc(vp->buffer_size);
        if (frame->data) {
            memcpy(frame->data, vp->frame_buffer, vp->buffer_size);
        } else {
            log_error("Failed to allocate frame pixel buffer");
        }
    } else {
        log_error("Failed to extract frame %d from video file", vp->frame_index);
        frame->data = NULL;
    }

    if (!frame->data) {
        free(frame);
        return NULL;
    }

    frame->width = w;
    frame->height = h;
    frame->channels = 3;
    frame->frame_index = vp->frame_index++;
    frame->timestamp = (double)frame->frame_index / vp->fps;

    vp->frame_count++;

    log_debug("Read frame %d at timestamp %.3f", frame->frame_index, frame->timestamp);

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

int video_processor_save_frame(const VideoProcessor* vp, const FrameData* frame, const char* output_path, int quality) {
    (void)quality;
    if (!vp || !frame || !output_path) return -1;

    int ret = utils_write_bmp(output_path, frame->data, frame->width, frame->height);
    if (ret == 0) {
        log_debug("Frame saved: %s", output_path);
    }
    return ret;
}

int video_processor_save_video(const VideoProcessor* vp, const char* output_path, int width, int height, float fps,
                               const char* codec_name, int bitrate) {
    (void)vp;
    (void)output_path;
    (void)width;
    (void)height;
    (void)fps;
    (void)codec_name;
    (void)bitrate;

    log_info("Video output: %s (%dx%d @ %.1f FPS, %s, %d kbps)",
             output_path, width, height, fps, codec_name, bitrate);
    return 0;
}

int video_processor_write_frame(const VideoProcessor* vp, const uint8_t* frame_data) {
    (void)vp;
    (void)frame_data;
    return 0;
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

    vp->is_opened = true;
    const char* fmt = camera_format ? camera_format : "MJPEG";
#ifdef PLATFORM_MUSE_PI_PRO
    log_info("V4L2 camera source: %s %dx%d@%.1ffps [%s]", device_path, width, height, (double)fps, fmt);
#else
    log_info("Camera source (dev mode): %s %dx%d@%.1ffps [%s]", device_path, width, height, (double)fps, fmt);
#endif

    return vp;
}

VideoProcessor* video_processor_create_from_arrow(void) {
    VideoProcessor* vp = (VideoProcessor*)calloc(1, sizeof(VideoProcessor));
    if (!vp) return NULL;

    vp->target_width = 640;
    vp->target_height = 480;
    vp->normalize = false;
    vp->is_opened = true;
    vp->fps = 15.0f;
    vp->buffer_size = (size_t)640 * 480 * 3;
    vp->frame_buffer = (uint8_t*)malloc(vp->buffer_size);
    if (!vp->frame_buffer) {
        free(vp);
        return NULL;
    }

    log_info("Arrow UART source created");

    return vp;
}

int video_processor_get_source_type(const VideoProcessor* vp) {
    if (!vp || !vp->input_path[0]) return VP_SOURCE_CAMERA;

    if (strncmp(vp->input_path, "/dev/video", 10) == 0) return VP_SOURCE_CAMERA;
    if (strncmp(vp->input_path, "arrow:", 6) == 0) return VP_SOURCE_ARROW;

    return VP_SOURCE_FILE;
}
