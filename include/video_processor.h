#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VP_OK       0
#define VP_ERROR   -1
#define VP_EOF     -2

#define VP_SOURCE_FILE      0
#define VP_SOURCE_CAMERA    1
#define VP_SOURCE_ARROW     2

typedef struct VideoProcessor VideoProcessor;

VideoProcessor* video_processor_create(const char* input_path, int target_width, int target_height, bool normalize);
void video_processor_destroy(VideoProcessor* vp);

int  video_processor_open(VideoProcessor* vp, const char* input_path);
void video_processor_close(VideoProcessor* vp);
bool video_processor_is_opened(const VideoProcessor* vp);

FrameData* video_processor_read_frame(VideoProcessor* vp);
FrameData* video_processor_read_frame_raw(VideoProcessor* vp);
void frame_data_destroy(FrameData* frame);

float video_processor_get_fps(const VideoProcessor* vp);
int   video_processor_get_frame_count(const VideoProcessor* vp);
int   video_processor_get_width(const VideoProcessor* vp);
int   video_processor_get_height(const VideoProcessor* vp);

VideoProcessor* video_processor_create_from_camera(const char* device_path, int width, int height, float fps, const char* camera_format);

int video_processor_get_source_type(const VideoProcessor* vp);

#ifdef __cplusplus
}
#endif

#endif
