#ifndef UTILS_H
#define UTILS_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UTILS_CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define UTILS_MIN(a, b) ((a) < (b) ? (a) : (b))
#define UTILS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define UTILS_ABS(x) ((x) < 0 ? -(x) : (x))
#define UTILS_SQR(x) ((x) * (x))

static inline float utils_fast_exp(float x) {
    union { uint32_t i; float f; } v;
    v.i = (uint32_t)(12102203.161561485f * x + 1065353216.0f);
    v.i = (v.i & 0x7f800000) == 0 ? 0 : v.i;
    return v.f;
}

static inline float utils_sigmoid(float x) {
    return 1.0f / (1.0f + utils_fast_exp(-x));
}

static inline float utils_fast_sqrt(float x) {
    union { float f; uint32_t i; } v = { x };
    v.i = 0x5f375a86 - (v.i >> 1);
    v.f *= 1.5f - 0.5f * x * v.f * v.f;
    return v.f * x;
}

void utils_rgb_to_bgr(const uint8_t* restrict rgb, uint8_t* restrict bgr, int width, int height);
void utils_bgr_to_rgb(const uint8_t* restrict bgr, uint8_t* restrict rgb, int width, int height);
void utils_resize_image(const uint8_t* restrict src, int src_w, int src_h,
                        uint8_t* restrict dst, int dst_w, int dst_h,
                        int channels);
void utils_crop_image(const uint8_t* restrict src, int full_w, int full_h,
                      int roi_x, int roi_y, int roi_w, int roi_h,
                      uint8_t* restrict dst);
void utils_letterbox(const uint8_t* restrict src, int src_w, int src_h,
                     uint8_t* restrict dst, int dst_w, int dst_h,
                     int channels, float* out_scale, int* out_pad_x, int* out_pad_y);

void utils_normalize_chw(const uint8_t* restrict src, int width, int height,
                         float* restrict dst, float scale, float mean, float std);

float utils_median_float(float* arr, int len);
void utils_sort_float_desc(float* arr, int* indices, int len);
void utils_sort_detections_by_confidence(Detection* dets, int len);

int utils_matrix_inverse_4x4(const float src[4][4], float dst[4][4]);
void utils_matrix_multiply_abt(const float a[7][7], const float b[7][7], float out[7][7]);

int64_t utils_get_time_ms(void);
void utils_sleep_ms(int ms);

int soft_jpeg_decode_to_rgb(const uint8_t* jpeg_data, size_t jpeg_len,
                            uint8_t* rgb_out, int out_w, int out_h);

int utils_write_bmp(const char* path, const uint8_t* data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
