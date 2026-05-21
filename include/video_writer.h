#ifndef VIDEO_WRITER_H
#define VIDEO_WRITER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VideoWriter VideoWriter;

VideoWriter* video_writer_create(const char* output_path, int width, int height, float fps);
int video_writer_write_frame(VideoWriter* vw, const uint8_t* frame_data);
int video_writer_flush(VideoWriter* vw);
void video_writer_destroy(VideoWriter* vw);

#ifdef __cplusplus
}
#endif

#endif
