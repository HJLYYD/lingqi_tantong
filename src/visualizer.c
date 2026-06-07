#include "visualizer.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static const uint8_t COLORS[VIS_MAX_COLORS][3] = {
    /* Single green palette — all person detections use the same color
     * because the INT8-quantized YOLOv8 model cannot distinguish classes.
     * Every detection is forced to class_id=0 (person).  Using one color
     * avoids the "rainbow chaos" visual and clearly communicates that
     * all boxes are person detections.  Different people are identified
     * by the "Person#N" label text, not by color. */
    {0, 255, 0}, {0, 240, 0}, {0, 225, 0}, {0, 210, 0},
    {0, 195, 0}, {0, 180, 0}, {0, 165, 0}, {0, 150, 0},
    {30, 255, 30}, {30, 240, 30}, {30, 225, 30}, {30, 210, 30},
    {60, 255, 60}, {60, 240, 60}, {60, 225, 60}, {60, 210, 60}
};

static const uint8_t SKELETON_COLORS[8][3] = {
    {255, 255, 0}, {255, 0, 255}, {0, 255, 0}, {0, 255, 0},
    {0, 255, 255}, {255, 0, 0}, {0, 128, 255}, {0, 128, 255}
};

static const int SKELETON_CONNECTIONS[][3] = {
    {0, 1, 0}, {0, 2, 0}, {1, 3, 0}, {2, 4, 0},
    {5, 6, 1}, {5, 7, 2}, {7, 9, 2}, {6, 8, 3}, {8, 10, 3},
    {5, 11, 4}, {6, 12, 4}, {11, 12, 5}, {11, 13, 6}, {13, 15, 6},
    {12, 14, 7}, {14, 16, 7}
};

static const uint8_t KEYPOINT_COLORS[17][3] = {
    {255, 255, 0}, {255, 200, 0}, {255, 200, 0}, {255, 150, 0}, {255, 150, 0},
    {0, 255, 0}, {0, 255, 0}, {0, 200, 255}, {0, 200, 255}, {0, 100, 255},
    {0, 100, 255}, {255, 0, 255}, {255, 0, 255}, {200, 0, 255}, {200, 0, 255},
    {128, 0, 128}, {128, 0, 128}
};

Visualizer* visualizer_create(bool show_info, bool corners, bool crosshair, bool skeleton_bbox) {
    Visualizer* vis = (Visualizer*)calloc(1, sizeof(Visualizer));
    if (!vis) return NULL;

    vis->show_info_bar = show_info;
    vis->corner_markers = corners;
    vis->crosshair = crosshair;
    vis->skeleton_aware_bbox = skeleton_bbox;

    return vis;
}

void visualizer_destroy(Visualizer* vis) {
    free(vis);
}

static void draw_rect(uint8_t* img, int width, int height, int x1, int y1, int x2, int y2,
                      const uint8_t color[3], int thickness) {
    for (int t = 0; t < thickness; t++) {
        for (int x = x1 - t; x <= x2 + t; x++) {
            if (x >= 0 && x < width) {
                if (y1 - t >= 0 && y1 - t < height) {
                    int idx = ((y1 - t) * width + x) * 3;
                    img[idx + 0] = color[2];
                    img[idx + 1] = color[1];
                    img[idx + 2] = color[0];
                }
                if (y2 + t >= 0 && y2 + t < height) {
                    int idx = ((y2 + t) * width + x) * 3;
                    img[idx + 0] = color[2];
                    img[idx + 1] = color[1];
                    img[idx + 2] = color[0];
                }
            }
        }
        for (int y = y1 - t; y <= y2 + t; y++) {
            if (y >= 0 && y < height) {
                if (x1 - t >= 0 && x1 - t < width) {
                    int idx = (y * width + (x1 - t)) * 3;
                    img[idx + 0] = color[2];
                    img[idx + 1] = color[1];
                    img[idx + 2] = color[0];
                }
                if (x2 + t >= 0 && x2 + t < width) {
                    int idx = (y * width + (x2 + t)) * 3;
                    img[idx + 0] = color[2];
                    img[idx + 1] = color[1];
                    img[idx + 2] = color[0];
                }
            }
        }
    }
}

static void draw_line(uint8_t* img, int width, int height,
                      int x1, int y1, int x2, int y2,
                      const uint8_t color[3], int thickness) {
    int dx = UTILS_ABS(x2 - x1);
    int dy = UTILS_ABS(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (1) {
        for (int tx = -thickness / 2; tx <= thickness / 2; tx++) {
            for (int ty = -thickness / 2; ty <= thickness / 2; ty++) {
                int px = x1 + tx;
                int py = y1 + ty;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 3;
                    img[idx + 0] = color[2];
                    img[idx + 1] = color[1];
                    img[idx + 2] = color[0];
                }
            }
        }

        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 < dx) { err += dx; y1 += sy; }
    }
}

static void draw_circle(uint8_t* img, int width, int height, int cx, int cy, int radius,
                        const uint8_t color[3], bool fill) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int dist = x * x + y * y;
            if (dist <= radius * radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    if (fill || dist >= (radius - 2) * (radius - 2)) {
                        int idx = (py * width + px) * 3;
                        img[idx + 0] = color[2];
                        img[idx + 1] = color[1];
                        img[idx + 2] = color[0];
                    }
                }
            }
        }
    }
}

static const uint8_t FONT_5X7[96][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    {0x20,0x20,0x20,0x20,0x20,0x00,0x20},
    {0x50,0x50,0x00,0x00,0x00,0x00,0x00},
    {0x50,0xF8,0x50,0x50,0xF8,0x50,0x50},
    {0x70,0xA0,0x70,0x28,0xF0,0x20,0x20},
    {0xC0,0xC8,0x10,0x20,0x40,0x98,0x18},
    {0x40,0xA0,0x40,0xA0,0xA0,0x90,0x60},
    {0x30,0x20,0x40,0x00,0x00,0x00,0x00},
    {0x10,0x20,0x40,0x40,0x40,0x20,0x10},
    {0x40,0x20,0x10,0x10,0x10,0x20,0x40},
    {0x00,0x20,0xA8,0x70,0xA8,0x20,0x00},
    {0x00,0x20,0x20,0xF8,0x20,0x20,0x00},
    {0x00,0x00,0x00,0x00,0x30,0x20,0x40},
    {0x00,0x00,0x00,0xF8,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x30,0x30},
    {0x00,0x08,0x10,0x20,0x40,0x80,0x00},
    {0x70,0x88,0x98,0xA8,0xC8,0x88,0x70},
    {0x20,0x60,0x20,0x20,0x20,0x20,0x70},
    {0x70,0x88,0x08,0x30,0x40,0x80,0xF8},
    {0xF8,0x08,0x10,0x30,0x08,0x88,0x70},
    {0x10,0x30,0x50,0x90,0xF8,0x10,0x10},
    {0xF8,0x80,0xF0,0x08,0x08,0x88,0x70},
    {0x30,0x40,0x80,0xF0,0x88,0x88,0x70},
    {0xF8,0x08,0x10,0x20,0x40,0x40,0x40},
    {0x70,0x88,0x88,0x70,0x88,0x88,0x70},
    {0x70,0x88,0x88,0x78,0x08,0x10,0x60},
    {0x00,0x30,0x30,0x00,0x30,0x30,0x00},
    {0x00,0x30,0x30,0x00,0x30,0x10,0x20},
    {0x08,0x10,0x20,0x40,0x20,0x10,0x08},
    {0x00,0x00,0xF8,0x00,0xF8,0x00,0x00},
    {0x40,0x20,0x10,0x08,0x10,0x20,0x40},
    {0x70,0x88,0x10,0x20,0x20,0x00,0x20},
    {0x70,0x88,0xB8,0xA8,0xB8,0x80,0x70},
    {0x20,0x50,0x88,0x88,0xF8,0x88,0x88},
    {0xF0,0x88,0x88,0xF0,0x88,0x88,0xF0},
    {0x70,0x88,0x80,0x80,0x80,0x88,0x70},
    {0xF0,0x88,0x88,0x88,0x88,0x88,0xF0},
    {0xF8,0x80,0x80,0xF0,0x80,0x80,0xF8},
    {0xF8,0x80,0x80,0xF0,0x80,0x80,0x80},
    {0x70,0x88,0x80,0xB8,0x88,0x88,0x70},
    {0x88,0x88,0x88,0xF8,0x88,0x88,0x88},
    {0x70,0x20,0x20,0x20,0x20,0x20,0x70},
    {0x38,0x10,0x10,0x10,0x10,0x90,0x60},
    {0x88,0x90,0xA0,0xC0,0xA0,0x90,0x88},
    {0x80,0x80,0x80,0x80,0x80,0x80,0xF8},
    {0x88,0xD8,0xA8,0xA8,0x88,0x88,0x88},
    {0x88,0xC8,0xA8,0x98,0x88,0x88,0x88},
    {0x70,0x88,0x88,0x88,0x88,0x88,0x70},
    {0xF0,0x88,0x88,0xF0,0x80,0x80,0x80},
    {0x70,0x88,0x88,0x88,0xA8,0x90,0x68},
    {0xF0,0x88,0x88,0xF0,0xA0,0x90,0x88},
    {0x70,0x88,0x80,0x70,0x08,0x88,0x70},
    {0xF8,0x20,0x20,0x20,0x20,0x20,0x20},
    {0x88,0x88,0x88,0x88,0x88,0x88,0x70},
    {0x88,0x88,0x88,0x88,0x88,0x50,0x20},
    {0x88,0x88,0x88,0xA8,0xA8,0xD8,0x88},
    {0x88,0x88,0x50,0x20,0x50,0x88,0x88},
    {0x88,0x88,0x50,0x20,0x20,0x20,0x20},
    {0xF8,0x08,0x10,0x20,0x40,0x80,0xF8},
    {0x70,0x40,0x40,0x40,0x40,0x40,0x70},
    {0x00,0x80,0x40,0x20,0x10,0x08,0x00},
    {0x70,0x10,0x10,0x10,0x10,0x10,0x70},
    {0x20,0x50,0x88,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x00,0x00,0x00,0x00,0xFC},
    {0x40,0x20,0x10,0x00,0x00,0x00,0x00},
    {0x00,0x00,0x70,0x08,0x78,0x88,0x78},
    {0x80,0x80,0xF0,0x88,0x88,0x88,0xF0},
    {0x00,0x00,0x70,0x88,0x80,0x88,0x70},
    {0x08,0x08,0x78,0x88,0x88,0x88,0x78},
    {0x00,0x00,0x70,0x88,0xF8,0x80,0x70},
    {0x30,0x48,0x40,0xF0,0x40,0x40,0x40},
    {0x00,0x78,0x88,0x78,0x08,0x88,0x70},
    {0x80,0x80,0xF0,0x88,0x88,0x88,0x88},
    {0x20,0x00,0x60,0x20,0x20,0x20,0x70},
    {0x10,0x00,0x30,0x10,0x10,0x90,0x60},
    {0x80,0x80,0x90,0xA0,0xC0,0xA0,0x90},
    {0x60,0x20,0x20,0x20,0x20,0x20,0x70},
    {0x00,0x00,0xD0,0xA8,0xA8,0xA8,0xA8},
    {0x00,0x00,0xB0,0xC8,0x88,0x88,0x88},
    {0x00,0x00,0x70,0x88,0x88,0x88,0x70},
    {0x00,0x00,0xF0,0x88,0xF0,0x80,0x80},
    {0x00,0x00,0x78,0x88,0x78,0x08,0x08},
    {0x00,0x00,0xB0,0xC8,0x80,0x80,0x80},
    {0x00,0x00,0x70,0x80,0x70,0x08,0xF0},
    {0x40,0x40,0xF0,0x40,0x40,0x48,0x30},
    {0x00,0x00,0x88,0x88,0x88,0x98,0x68},
    {0x00,0x00,0x88,0x88,0x88,0x50,0x20},
    {0x00,0x00,0x88,0xA8,0xA8,0xA8,0x50},
    {0x00,0x00,0x88,0x50,0x20,0x50,0x88},
    {0x00,0x00,0x88,0x88,0x78,0x08,0x70},
    {0x00,0x00,0xF8,0x10,0x20,0x40,0xF8},
    {0x10,0x20,0x20,0x40,0x20,0x20,0x10},
    {0x20,0x20,0x20,0x20,0x20,0x20,0x20},
    {0x40,0x20,0x20,0x10,0x20,0x20,0x40},
    {0x00,0x00,0x00,0x68,0xB0,0x00,0x00},
};

static void draw_text(uint8_t* img, int width, int height, int x, int y,
                      const char* text, const uint8_t color[3]);
static void draw_text(uint8_t* img, int width, int height, int x, int y,
                      const char* text, const uint8_t color[3]) {
    if (!text || !img || width <= 0 || height <= 0) return;

    int font_w = 5;
    int font_h = 7;
    int spacing = 1;
    int px = x;
    int py = y;

    while (*text && px + font_w < width) {
        int idx = (int)((unsigned char)*text) - 32;
        if (idx < 0) idx = 0;
        if (idx >= 96) idx = 95;

        for (int row = 0; row < font_h; row++) {
            uint8_t row_data = FONT_5X7[idx][row];
            for (int col = 0; col < font_w; col++) {
                if ((row_data >> (font_w - 1 - col)) & 1) {
                    int dx = px + col;
                    int dy = py + row;
                    if (dx >= 0 && dx < width && dy >= 0 && dy < height) {
                        int pixel_idx = (dy * width + dx) * 3;
                        img[pixel_idx + 0] = color[0];
                        img[pixel_idx + 1] = color[1];
                        img[pixel_idx + 2] = color[2];
                    }
                }
            }
        }

        px += font_w + spacing;
        text++;
    }
}

static void draw_label(uint8_t* img, int width, int height, const TrackedObject* obj,
                       int x1, int y1, const uint8_t color[3]) {
    char label[MAX_LABEL_LEN];
    const char* class_name = obj->detection.class_name[0] != '\0'
                             ? obj->detection.class_name : "person";
    snprintf(label, sizeof(label), "%s#%d %.0f%% | %.1fm",
             class_name, obj->track_id,
             obj->detection.confidence * 100.0f,
             spatial_distance_from_origin(&obj->spatial_pos));

    if (obj->has_height) {
        char height_str[32];
        snprintf(height_str, sizeof(height_str), " | H:%.2fm", obj->height_meters);
        strncat(label, height_str, sizeof(label) - strlen(label) - 1);
    }

    if (obj->has_face && obj->face.identity[0] != '\0' && strcmp(obj->face.identity, "unknown") != 0) {
        strncat(label, " | ", sizeof(label) - strlen(label) - 1);
        strncat(label, obj->face.identity, sizeof(label) - strlen(label) - 1);
    }

    if (obj->is_occluded) {
        char occl_str[32];
        snprintf(occl_str, sizeof(occl_str), " [OCCL:%d]", obj->occluded_frames);
        strncat(label, occl_str, sizeof(label) - strlen(label) - 1);
    }

    int label_w = (int)strlen(label) * 8 + 6;
    int label_h = 18;
    int label_x = x1;
    int label_y = y1 - label_h;
    if (label_y < 0) label_y = y1 + 20;

    for (int dy = 0; dy < label_h; dy++) {
        for (int dx = 0; dx < label_w; dx++) {
            int px = label_x + dx;
            int py = label_y + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                int idx = (py * width + px) * 3;
                img[idx + 0] = color[2];
                img[idx + 1] = color[1];
                img[idx + 2] = color[0];
            }
        }
    }

    /* Draw text on top of label background in white for contrast */
    {
        uint8_t white[3] = {255, 255, 255};
        draw_text(img, width, height, label_x + 3, label_y + 3, label, white);
    }
}

static void draw_skeleton(uint8_t* img, int width, int height, const PoseEstimation* pose) {
    if (!pose) return;

    int visible_count = 0;
    for (int i = 0; i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence > 0.3f) visible_count++;
    }
    float occlusion_level = pose->num_keypoints > 0 ? 1.0f - (float)visible_count / pose->num_keypoints : 1.0f;
    int thickness = occlusion_level < 0.3f ? 3 : 2;

    for (int c = 0; c < 16; c++) {
        int i = SKELETON_CONNECTIONS[c][0];
        int j = SKELETON_CONNECTIONS[c][1];
        int color_idx = SKELETON_CONNECTIONS[c][2];

        if (i < pose->num_keypoints && j < pose->num_keypoints) {
            const Keypoint* kp1 = &pose->keypoints[i];
            const Keypoint* kp2 = &pose->keypoints[j];

            if (kp1->confidence > 0.3f && kp2->confidence > 0.3f) {
                draw_line(img, width, height,
                          (int)kp1->x, (int)kp1->y,
                          (int)kp2->x, (int)kp2->y,
                          SKELETON_COLORS[color_idx], thickness);
            } else if (kp1->confidence > 0.1f || kp2->confidence > 0.1f) {
                uint8_t dim_color[3];
                for (int k = 0; k < 3; k++) dim_color[k] = (uint8_t)(SKELETON_COLORS[color_idx][k] * 0.3f);
                draw_line(img, width, height,
                          (int)kp1->x, (int)kp1->y,
                          (int)kp2->x, (int)kp2->y,
                          dim_color, 1);
            }
        }
    }

    for (int i = 0; i < pose->num_keypoints; i++) {
        const Keypoint* kp = &pose->keypoints[i];
        if (kp->confidence > 0.3f) {
            int radius = occlusion_level < 0.5f ? 6 : 4;
            draw_circle(img, width, height, (int)kp->x, (int)kp->y, radius, KEYPOINT_COLORS[i], true);
            uint8_t white[3] = {255, 255, 255};
            draw_circle(img, width, height, (int)kp->x, (int)kp->y, radius + 2, white, false);
        } else if (kp->confidence > 0.1f) {
            uint8_t gray[3] = {100, 100, 100};
            draw_circle(img, width, height, (int)kp->x, (int)kp->y, 4, gray, false);
        }
    }
}

int visualizer_render_detection_view(Visualizer* vis,
                                      const uint8_t* input_frame, int width, int height,
                                      const TrackedObject* objects, int num_objects,
                                      int frame_num, float fps,
                                      uint8_t* output_frame) {
    (void)frame_num;
    (void)fps;
    if (!vis || !input_frame || !output_frame) return -1;

    memcpy(output_frame, input_frame, width * height * 3);

    for (int i = 0; i < num_objects; i++) {
        const TrackedObject* obj = &objects[i];
        const uint8_t* color = COLORS[obj->track_id % VIS_MAX_COLORS];

        int x1 = (int)obj->detection.bbox.x_min;
        int y1 = (int)obj->detection.bbox.y_min;
        int x2 = (int)obj->detection.bbox.x_max;
        int y2 = (int)obj->detection.bbox.y_max;

        draw_rect(output_frame, width, height, x1, y1, x2, y2, color, 2);
        draw_label(output_frame, width, height, obj, x1, y1, color);

        if (obj->has_pose) {
            draw_skeleton(output_frame, width, height, &obj->pose);
        }
    }

    if (vis->show_info_bar) {
        for (int y = 0; y < VIS_INFO_BAR_HEIGHT && y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                output_frame[idx + 0] = 25;
                output_frame[idx + 1] = 25;
                output_frame[idx + 2] = 25;
            }
        }
    }

    return 0;
}

int visualizer_render_trajectory_view(Visualizer* vis,
                                       const TrackedObject* objects, int num_objects,
                                       int size_w, int size_h,
                                       float max_range,
                                       uint8_t* output_frame) {
    (void)vis;
    if (!output_frame) return -1;

    memset(output_frame, 0, size_w * size_h * 3);

    for (int y = 0; y < size_h; y++) {
        for (int x = 0; x < size_w; x++) {
            int idx = (y * size_w + x) * 3;
            output_frame[idx + 0] = 20;
            output_frame[idx + 1] = 15;
            output_frame[idx + 2] = 10;
        }
    }

    float scale = size_w / (2.0f * max_range);
    int origin_x = size_w / 2;
    int origin_y = size_h / 2;

    uint8_t grid_color[3] = {50, 40, 30};
    for (int i = -10; i <= 10; i++) {
        if (i == 0) continue;
        int x = origin_x + (int)(i * 2 * scale);
        int y = origin_y - (int)(i * 2 * scale);
        if (x >= 0 && x < size_w) {
            for (int py = 0; py < size_h; py++) {
                int idx = (py * size_w + x) * 3;
                output_frame[idx + 0] = grid_color[2];
                output_frame[idx + 1] = grid_color[1];
                output_frame[idx + 2] = grid_color[0];
            }
        }
        if (y >= 0 && y < size_h) {
            for (int px = 0; px < size_w; px++) {
                int idx = (y * size_w + px) * 3;
                output_frame[idx + 0] = grid_color[2];
                output_frame[idx + 1] = grid_color[1];
                output_frame[idx + 2] = grid_color[0];
            }
        }
    }

    uint8_t axis_color[3] = {255, 255, 0};
    draw_circle(output_frame, size_w, size_h, origin_x, origin_y, 8, axis_color, true);
    draw_circle(output_frame, size_w, size_h, origin_x, origin_y, 12, axis_color, false);

    for (int i = 0; i < num_objects; i++) {
        const TrackedObject* obj = &objects[i];
        const uint8_t* color = COLORS[obj->track_id % VIS_MAX_COLORS];

        if (obj->trajectory.count > 1) {
            for (int j = 1; j < obj->trajectory.count; j++) {
                const SpatialPosition* p1 = &obj->trajectory.positions[j - 1];
                const SpatialPosition* p2 = &obj->trajectory.positions[j];

                int sx1 = origin_x + (int)(p1->x * scale);
                int sy1 = origin_y - (int)(p1->z * scale);
                int sx2 = origin_x + (int)(p2->x * scale);
                int sy2 = origin_y - (int)(p2->z * scale);

                float alpha = 0.3f + 0.7f * ((float)j / obj->trajectory.count);
                uint8_t line_color[3];
                for (int c = 0; c < 3; c++) line_color[c] = (uint8_t)(color[c] * alpha);

                draw_line(output_frame, size_w, size_h, sx1, sy1, sx2, sy2, line_color, 2);
            }
        }

        if (obj->trajectory.count > 0) {
            const SpatialPosition* last = &obj->trajectory.positions[obj->trajectory.count - 1];
            int lx = origin_x + (int)(last->x * scale);
            int ly = origin_y - (int)(last->z * scale);

            draw_circle(output_frame, size_w, size_h, lx, ly, 10, color, true);
            uint8_t white[3] = {255, 255, 255};
            draw_circle(output_frame, size_w, size_h, lx, ly, 12, white, false);
        }
    }

    return 0;
}
