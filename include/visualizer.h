#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VIS_MAX_COLORS          16
#define VIS_INFO_BAR_HEIGHT     60

typedef struct {
    bool show_info_bar;
    bool corner_markers;
    bool crosshair;
    bool skeleton_aware_bbox;
} Visualizer;

Visualizer* visualizer_create(bool show_info, bool corners, bool crosshair, bool skeleton_bbox);
void visualizer_destroy(Visualizer* vis);

int visualizer_render_detection_view(Visualizer* vis,
                                      const uint8_t* input_frame, int width, int height,
                                      const TrackedObject* objects, int num_objects,
                                      int frame_num, float fps,
                                      uint8_t* output_frame);

int visualizer_render_trajectory_view(Visualizer* vis,
                                       const TrackedObject* objects, int num_objects,
                                       int size_w, int size_h,
                                       float max_range,
                                       uint8_t* output_frame);

#ifdef __cplusplus
}
#endif

#endif
