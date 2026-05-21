#include "ar_renderer.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ARRenderer* ar_renderer_create(int render_w, int render_h, bool depth_test) {
    ARRenderer* renderer = (ARRenderer*)calloc(1, sizeof(ARRenderer));
    if (!renderer) return NULL;

    renderer->render_width = render_w;
    renderer->render_height = render_h;
    renderer->enable_depth_test = depth_test;

    return renderer;
}

void ar_renderer_destroy(ARRenderer* renderer) {
    free(renderer);
}

static void draw_ar_marker(uint8_t* img, int width, int height,
                           int cx, int cy, int w, int h,
                           const uint8_t color[3], const char* label) {
    int x1 = cx - w / 2;
    int y1 = cy - h / 2;
    int x2 = cx + w / 2;
    int y2 = cy + h / 2;

    x1 = UTILS_CLAMP(x1, 0, width - 1);
    y1 = UTILS_CLAMP(y1, 0, height - 1);
    x2 = UTILS_CLAMP(x2, 0, width - 1);
    y2 = UTILS_CLAMP(y2, 0, height - 1);

    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            if (x == x1 || x == x2 || y == y1 || y == y2) {
                int idx = (y * width + x) * 3;
                img[idx + 0] = color[2];
                img[idx + 1] = color[1];
                img[idx + 2] = color[0];
            }
        }
    }

    if (label && label[0]) {
        int text_x = UTILS_CLAMP(x1 + 2, 0, width - 1);
        int text_y = UTILS_CLAMP(y1 - 4, 0, height - 1);
        int label_len = (int)strlen(label);
        for (int i = 0; i < label_len && (text_x + i * 6) < (width - 1); i++) {
            int px = text_x + i * 6;
            if (px < 0 || px >= width) continue;
            for (int dy = -3; dy <= 3 && (text_y + dy) >= 0 && (text_y + dy) < height; dy++) {
                int idx = ((text_y + dy) * width + px) * 3;
                img[idx + 0] = 255;
                img[idx + 1] = 255;
                img[idx + 2] = 255;
            }
        }
    }
}

int ar_renderer_render_frame(ARRenderer* renderer,
                              const uint8_t* video_frame, int width, int height,
                              const ARMarker* markers, int num_markers,
                              uint8_t* output_frame) {
    if (!renderer || !video_frame || !output_frame) return -1;

    memcpy(output_frame, video_frame, width * height * 3);

    for (int i = 0; i < num_markers; i++) {
        const ARMarker* marker = &markers[i];
        int cx = (int)marker->position[0];
        int cy = (int)marker->position[1];
        int w = (int)marker->size[0];
        int h = (int)marker->size[1];

        uint8_t color[3];
        if (marker->is_alert) {
            color[0] = 0; color[1] = 0; color[2] = 255;
        } else {
            color[0] = 0; color[1] = 255; color[2] = 0;
        }

        draw_ar_marker(output_frame, width, height, cx, cy, w, h, color, marker->label);
    }

    return 0;
}

int ar_renderer_compensate_motion(ARRenderer* renderer,
                                   const uint8_t* frame, int width, int height,
                                   const float euler[3],
                                   uint8_t* output_frame) {
    (void)renderer;
    if (!frame || !output_frame || !euler) return -1;

    float pitch_rad = euler[0] * ((float)M_PI / 180.0f);
    float roll_rad = euler[1] * ((float)M_PI / 180.0f);
    float yaw_rad = euler[2] * ((float)M_PI / 180.0f);

    float total_angle = fabsf(pitch_rad) + fabsf(roll_rad) + fabsf(yaw_rad);
    if (total_angle < 0.0001f) {
        memcpy(output_frame, frame, width * height * 3);
        return 0;
    }

    memcpy(output_frame, frame, width * height * 3);

    int center_x = width / 2;
    int center_y = height / 2;
    float focal = (float)width * 0.5f;

    float cp = cosf(pitch_rad), sp = sinf(pitch_rad);
    float cr = cosf(roll_rad), sr = sinf(roll_rad);
    float cy = cosf(yaw_rad), sy = sinf(yaw_rad);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = (float)(x - center_x);
            float dy = (float)(y - center_y);

            float nx = dx / focal;
            float ny = dy / focal;
            float nz = 1.0f;

            float rx = nx * cp * cy + ny * (sr * sp * cy - cr * sy) + nz * (cr * sp * cy + sr * sy);
            float ry = nx * cp * sy + ny * (sr * sp * sy + cr * cy) + nz * (cr * sp * sy - sr * cy);
            float rz = nx * (-sp) + ny * (sr * cp) + nz * (cr * cp);

            if (rz <= 0.001f) {
                int dst_idx = (y * width + x) * 3;
                output_frame[dst_idx + 0] = 0;
                output_frame[dst_idx + 1] = 0;
                output_frame[dst_idx + 2] = 0;
                continue;
            }

            int src_x = center_x + (int)(rx / rz * focal);
            int src_y = center_y + (int)(ry / rz * focal);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                int dst_idx = (y * width + x) * 3;
                int src_idx = (src_y * width + src_x) * 3;
                output_frame[dst_idx + 0] = frame[src_idx + 0];
                output_frame[dst_idx + 1] = frame[src_idx + 1];
                output_frame[dst_idx + 2] = frame[src_idx + 2];
            } else {
                int dst_idx = (y * width + x) * 3;
                output_frame[dst_idx + 0] = 0;
                output_frame[dst_idx + 1] = 0;
                output_frame[dst_idx + 2] = 0;
            }
        }
    }

    return 0;
}

int ar_renderer_resize_to_render_size(ARRenderer* renderer,
                                       const uint8_t* frame, int width, int height,
                                       uint8_t* output_frame) {
    if (!renderer || !frame || !output_frame) return -1;

    if (width == renderer->render_width && height == renderer->render_height) {
        memcpy(output_frame, frame, width * height * 3);
        return 0;
    }

    utils_resize_image(frame, width, height, output_frame, renderer->render_width, renderer->render_height, 3);
    return 0;
}
