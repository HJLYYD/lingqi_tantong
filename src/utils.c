#define _POSIX_C_SOURCE 199309L

#include "utils.h"
#include "logger.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#ifdef HAS_LIBJPEG_TURBO
#include <turbojpeg.h>
#endif


#ifdef HAS_RVV_OPT
#if defined(__GNUC__) && !defined(__clang__)
#include <riscv_vector.h>
#endif
#endif

#ifdef HAS_OPENMP
#ifdef __clang__
/* Clang + riscv64: libomp not packaged on Bianbu, we link libgomp via
 * -fopenmp=libgomp.  GCC's omp.h uses __attribute__((__malloc__(omp_free)))
 * which Clang rejects.  Declare only the functions we actually use. */
void omp_set_num_threads(int n);
int  omp_get_num_threads(void);
int  omp_get_thread_num(void);
int  omp_get_max_threads(void);
#else
#include <omp.h>
#endif
#endif

void utils_rgb_to_bgr(const uint8_t* restrict rgb, uint8_t* restrict bgr, int width, int height) {
    int pixels = width * height;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int i = 0; i < pixels; i++) {
        bgr[i * 3 + 0] = rgb[i * 3 + 2];
        bgr[i * 3 + 1] = rgb[i * 3 + 1];
        bgr[i * 3 + 2] = rgb[i * 3 + 0];
    }
}

void utils_bgr_to_rgb(const uint8_t* restrict bgr, uint8_t* restrict rgb, int width, int height) {
    int pixels = width * height;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int i = 0; i < pixels; i++) {
        rgb[i * 3 + 0] = bgr[i * 3 + 2];
        rgb[i * 3 + 1] = bgr[i * 3 + 1];
        rgb[i * 3 + 2] = bgr[i * 3 + 0];
    }
}

/*
 * Crop a rectangular region from a full-frame image (with stride=full_w*3).
 * Copies roi_w × roi_h pixels from (roi_x, roi_y) in the source frame to a
 * contiguous output buffer (stride=roi_w*3).  Channels is always 3 (RGB).
 */
void utils_crop_image(const uint8_t* restrict src, int full_w, int full_h,
                      int roi_x, int roi_y, int roi_w, int roi_h,
                      uint8_t* restrict dst) {
    if (!src || !dst || roi_w <= 0 || roi_h <= 0) return;
    for (int y = 0; y < roi_h; y++) {
        int src_y = roi_y + y;
        if (src_y < 0 || src_y >= full_h) {
            memset(dst + (size_t)y * roi_w * 3, 0, (size_t)roi_w * 3);
            continue;
        }
        int src_start_x = roi_x;
        int copy_w = roi_w;
        int dst_offset = 0;
        if (src_start_x < 0) {
            dst_offset = -src_start_x * 3;
            copy_w += src_start_x;
            src_start_x = 0;
        }
        if (src_start_x + copy_w > full_w) {
            copy_w = full_w - src_start_x;
        }
        if (copy_w > 0) {
            memcpy(dst + (size_t)y * roi_w * 3 + dst_offset,
                   src + ((size_t)src_y * full_w + src_start_x) * 3,
                   (size_t)copy_w * 3);
        }
        /* Zero any remaining right-side padding */
        if ((size_t)(dst_offset + copy_w * 3) < (size_t)roi_w * 3) {
            memset(dst + (size_t)y * roi_w * 3 + dst_offset + copy_w * 3, 0,
                   (size_t)roi_w * 3 - dst_offset - (size_t)copy_w * 3);
        }
    }
}

void utils_resize_image(const uint8_t* restrict src, int src_w, int src_h,
                        uint8_t* restrict dst, int dst_w, int dst_h,
                        int channels) {
    float x_ratio = (float)src_w / dst_w;
    float y_ratio = (float)src_h / dst_h;

#if defined(HAS_RVV_OPT) && defined(__GNUC__) && !defined(__clang__)
    /*
     * RVV-accelerated nearest-neighbor resize for the 3-channel hot path.
     * Processes 8 pixels at a time using LMUL=8 (vlen=256 → 32 bytes).
     * Falls back to scalar for edge cases (non-3-channel, small images).
     */
    if (channels == 3 && dst_w >= 8) {
        /* Pre-compute source x indices for each destination column */
        int* src_x_tab = (int*)malloc((size_t)dst_w * sizeof(int));
        if (src_x_tab) {
            for (int x = 0; x < dst_w; x++) {
                int sx = (int)(x * x_ratio);
                src_x_tab[x] = UTILS_MIN(sx, src_w - 1);
            }

#ifdef HAS_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int y = 0; y < dst_h; y++) {
                int src_y = (int)(y * y_ratio);
                src_y = UTILS_MIN(src_y, src_h - 1);
                uint8_t* dst_row = dst + (size_t)y * dst_w * 3;

                size_t vl;
                int x = 0;
                /* Process 8 pixels per RVV iteration (24 bytes per pixel-group × 8 = 192 bytes with LMUL=8) */
                for (; x + 7 < dst_w; x += 8) {
                    vl = __riscv_vsetvl_e8m4(24); /* 8 pixels × 3 channels */
                    /* Gather source pixels */
                    uint8_t gathered[24];
                    for (int k = 0; k < 8; k++) {
                        int sx = src_x_tab[x + k];
                        int si = (src_y * src_w + sx) * 3;
                        gathered[k * 3 + 0] = src[si + 0];
                        gathered[k * 3 + 1] = src[si + 1];
                        gathered[k * 3 + 2] = src[si + 2];
                    }
                    vint8m4_t vec = __riscv_vle8_v_i8m4((int8_t*)gathered, vl);
                    __riscv_vse8_v_i8m4((int8_t*)(dst_row + x * 3), vec, vl);
                }
                /* Tail: scalar fallback */
                for (; x < dst_w; x++) {
                    int sx = src_x_tab[x];
                    int si = (src_y * src_w + sx) * 3;
                    int di = x * 3;
                    dst_row[di + 0] = src[si + 0];
                    dst_row[di + 1] = src[si + 1];
                    dst_row[di + 2] = src[si + 2];
                }
            }
            free(src_x_tab);
            return;
        }
    }
#endif
    /* Fallback scalar path */
#ifdef HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            src_x = UTILS_MIN(src_x, src_w - 1);
            src_y = UTILS_MIN(src_y, src_h - 1);

            int dst_idx = (y * dst_w + x) * channels;
            int src_idx = (src_y * src_w + src_x) * channels;

            for (int c = 0; c < channels; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

void utils_letterbox(const uint8_t* restrict src, int src_w, int src_h,
                     uint8_t* restrict dst, int dst_w, int dst_h,
                     int channels, float* out_scale, int* out_pad_x, int* out_pad_y) {
    float scale = UTILS_MIN((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    memset(dst, 114, (size_t)dst_w * dst_h * channels);

#ifdef HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = (int)(x / scale);
            int src_y = (int)(y / scale);
            src_x = UTILS_MIN(src_x, src_w - 1);
            src_y = UTILS_MIN(src_y, src_h - 1);

            int dst_idx = ((pad_y + y) * dst_w + (pad_x + x)) * channels;
            int src_idx = (src_y * src_w + src_x) * channels;

            for (int c = 0; c < channels; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    if (out_scale) *out_scale = scale;
    if (out_pad_x) *out_pad_x = pad_x;
    if (out_pad_y) *out_pad_y = pad_y;
}

void utils_normalize_chw(const uint8_t* restrict src, int width, int height,
                         float* restrict dst, float scale, float mean, float std) {
    int pixels = width * height;
#ifdef HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
            for (int x = 0; x < width; x++) {
                int src_idx = (y * width + x) * 3 + c;
                int dst_idx = c * pixels + y * width + x;
                dst[dst_idx] = ((src[src_idx] * scale) - mean) / std;
            }
        }
    }
}

static int compare_float_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}

void utils_sort_float_desc(float* arr, int* indices, int len) {
    typedef struct { float val; int idx; } Pair;
    Pair* pairs = (Pair*)malloc(len * sizeof(Pair));
    for (int i = 0; i < len; i++) {
        pairs[i].val = arr[i];
        pairs[i].idx = i;
    }
    qsort(pairs, len, sizeof(Pair), compare_float_desc);
    for (int i = 0; i < len; i++) {
        arr[i] = pairs[i].val;
        indices[i] = pairs[i].idx;
    }
    free(pairs);
}

static int compare_detection_desc(const void* a, const void* b) {
    const Detection* da = (const Detection*)a;
    const Detection* db = (const Detection*)b;
    return (da->confidence < db->confidence) - (da->confidence > db->confidence);
}

void utils_sort_detections_by_confidence(Detection* dets, int len) {
    qsort(dets, len, sizeof(Detection), compare_detection_desc);
}

int utils_matrix_inverse_4x4(const float src[4][4], float dst[4][4]) {
    float a[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            a[i][j] = src[i][j];
            a[i][j + 4] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int col = 0; col < 4; col++) {
        int pivot = col;
        float max_val = fabsf(a[col][col]);
        for (int row = col + 1; row < 4; row++) {
            if (fabsf(a[row][col]) > max_val) {
                max_val = fabsf(a[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-10f) return -1;
        if (pivot != col) {
            for (int j = 0; j < 8; j++) {
                float tmp = a[col][j];
                a[col][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
        }

        float pivot_val = a[col][col];
        for (int j = 0; j < 8; j++) {
            a[col][j] /= pivot_val;
        }

        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            float factor = a[row][col];
            for (int j = 0; j < 8; j++) {
                a[row][j] -= factor * a[col][j];
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dst[i][j] = a[i][j + 4];
        }
    }
    return 0;
}

void utils_matrix_multiply_abt(const float a[7][7], const float b[7][7], float out[7][7]) {
    float temp[7][7] = {{0}};
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            out[i][j] = 0.0f;
            for (int k = 0; k < 7; k++) {
                out[i][j] += temp[i][k] * b[j][k];
            }
        }
    }
}

/**
 * Compute out = F * P * F^T  — correct Kalman covariance propagation.
 *
 * 方差传播规则: 若 x ~ N(μ, P)，则 Fx ~ N(Fμ, FPF^T)
 * 参考: Kalman (1960), Bar-Shalom et al. (2001)
 *
 * @param F  State transition matrix [7][7]
 * @param P  Current covariance matrix [7][7] (must be symmetric)
 * @param out Result matrix [7][7] = F * P * F^T
 */
void utils_matrix_multiply_fpfT(const float f[7][7], const float p[7][7], float out[7][7]) {
    float temp[7][7] = {{0}};
    /* temp = F * P */
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                temp[i][j] += f[i][k] * p[k][j];
            }
        }
    }
    /* out = temp * F^T  (F^T[k][j] = F[j][k]) */
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            out[i][j] = 0.0f;
            for (int k = 0; k < 7; k++) {
                out[i][j] += temp[i][k] * f[j][k];  /* f[j][k] = F^T[k][j] */
            }
        }
    }
}

float utils_median_float(float* arr, int len) {
    if (len <= 0 || !arr) return 0.0f;
    float* copy = (float*)malloc(len * sizeof(float));
    if (!copy) return arr[len / 2];  /* fallback: return midpoint of original */
    memcpy(copy, arr, len * sizeof(float));  /* BUGFIX: was sorting uninitialized memory */
    qsort(copy, len, sizeof(float), compare_float_desc);
    float median = (len % 2 == 0) ? (copy[len/2 - 1] + copy[len/2]) * 0.5f : copy[len/2];
    free(copy);
    return median;
}

int64_t utils_get_time_ms(void) {
#ifdef _WIN32
    return GetTickCount64();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

void utils_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

int soft_jpeg_decode_to_rgb(const uint8_t* jpeg_data, size_t jpeg_len,
                            uint8_t* rgb_out, int out_w, int out_h) {
    if (!jpeg_data || !rgb_out || out_w <= 0 || out_h <= 0) {
        return -1;
    }
    if (jpeg_len < 4 || jpeg_data[0] != 0xFF || jpeg_data[1] != 0xD8) {
        log_warning("JPEG decode: invalid SOI marker (len=%zu, b0=%02X b1=%02X)",
                    jpeg_len, jpeg_len > 0 ? jpeg_data[0] : 0,
                    jpeg_len > 1 ? jpeg_data[1] : 0);
        return -2;
    }

#ifdef HAS_LIBJPEG_TURBO
    tjhandle handle = tjInitDecompress();
    if (!handle) {
        log_error("tjInitDecompress failed: %s", tjGetErrorStr());
        return -3;
    }

    int src_w = 0, src_h = 0, src_subsamp = 0, src_colorspace = 0;
    if (tjDecompressHeader3(handle, jpeg_data, (unsigned long)jpeg_len,
                            &src_w, &src_h, &src_subsamp, &src_colorspace) != 0) {
        log_error("tjDecompressHeader3 failed: %s", tjGetErrorStr2(handle));
        tjDestroy(handle);
        return -4;
    }

    if (src_w == out_w && src_h == out_h) {
        int ret = tjDecompress2(handle, jpeg_data, (unsigned long)jpeg_len,
                                rgb_out, out_w, 0 /* pitch=0 → tight */,
                                out_h, TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);
        if (ret != 0) {
            log_error("tjDecompress2 failed: %s", tjGetErrorStr2(handle));
            tjDestroy(handle);
            return -5;
        }
        tjDestroy(handle);
        return 0;
    }

    uint8_t* tmp_rgb = (uint8_t*)malloc((size_t)src_w * src_h * 3);
    if (!tmp_rgb) {
        tjDestroy(handle);
        log_error("JPEG decode: failed to allocate %dx%d intermediate buffer",
                  src_w, src_h);
        return -6;
    }

    if (tjDecompress2(handle, jpeg_data, (unsigned long)jpeg_len,
                      tmp_rgb, src_w, 0, src_h,
                      TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) != 0) {
        log_error("tjDecompress2 (scaled) failed: %s", tjGetErrorStr2(handle));
        free(tmp_rgb);
        tjDestroy(handle);
        return -7;
    }
    tjDestroy(handle);

    utils_resize_image(tmp_rgb, src_w, src_h, rgb_out, out_w, out_h, 3);
    free(tmp_rgb);
    return 0;
#else
    log_error("JPEG decode requested but HAS_LIBJPEG_TURBO not defined at build time");
    return -8;
#endif
}

int utils_write_bmp(const char* path, const uint8_t* data, int width, int height) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int row_size = (width * 3 + 3) & ~3;
    int pixel_data_size = row_size * height;
    int file_size = 54 + pixel_data_size;

    uint8_t header[54] = {0};
    header[0] = 'B'; header[1] = 'M';
    header[2] = (uint8_t)(file_size & 0xFF);
    header[3] = (uint8_t)((file_size >> 8) & 0xFF);
    header[4] = (uint8_t)((file_size >> 16) & 0xFF);
    header[5] = (uint8_t)((file_size >> 24) & 0xFF);
    header[10] = 54;
    header[14] = 40;
    header[18] = (uint8_t)(width & 0xFF);
    header[19] = (uint8_t)((width >> 8) & 0xFF);
    header[20] = (uint8_t)((width >> 16) & 0xFF);
    header[21] = (uint8_t)((width >> 24) & 0xFF);
    header[22] = (uint8_t)(height & 0xFF);
    header[23] = (uint8_t)((height >> 8) & 0xFF);
    header[24] = (uint8_t)((height >> 16) & 0xFF);
    header[25] = (uint8_t)((height >> 24) & 0xFF);
    header[26] = 1;
    header[28] = 24;

    fwrite(header, 1, 54, f);

    uint8_t* row = (uint8_t*)malloc(row_size);
    if (!row) {
        fclose(f);
        return -1;
    }
    memset(row, 0, row_size);

    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * 3;
            int dst_idx = x * 3;
            row[dst_idx + 0] = data[src_idx + 2];
            row[dst_idx + 1] = data[src_idx + 1];
            row[dst_idx + 2] = data[src_idx + 0];
        }
        fwrite(row, 1, row_size, f);
    }

    free(row);
    fclose(f);
    return 0;
}
