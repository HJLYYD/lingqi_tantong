#ifndef AR_RENDERER_H
#define AR_RENDERER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AR_MAX_MARKERS      64

typedef struct {
    float position[2];
    float size[2];
    char label[MAX_STRING_LEN];
    bool is_alert;
    uint8_t color[3];
} ARMarker;

typedef struct {
    int render_width;
    int render_height;
    bool enable_depth_test;
} ARRenderer;

ARRenderer* ar_renderer_create(int render_w, int render_h, bool depth_test);
void ar_renderer_destroy(ARRenderer* renderer);

int ar_renderer_render_frame(ARRenderer* renderer,
                              const uint8_t* video_frame, int width, int height,
                              const ARMarker* markers, int num_markers,
                              uint8_t* output_frame);

#ifdef __cplusplus
}
#endif

#endif
