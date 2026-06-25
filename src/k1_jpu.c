/*
 * K1 JPU — 硬件 JPEG 解码 (v3 — 使用 libjpu.so 真实 API)
 *
 * API 逆向自:
 *   $ nm -D /usr/lib/libjpu.so
 *   $ /usr/bin/jpu_dec_test --help
 *
 * 头文件声明见 include/k1_jpu_api.h (手写, 包内未附带 .h 文件)
 *
 * 使用方式:
 *   1. sudo apt install k1x-jpu     (Bianbu, 已预装在 /usr/lib/libjpu.so)
 *   2. cmake ..                      (CMakeLists.txt 自动检测 /usr/lib/libjpu.so)
 *   3. 程序运行时自动启用 JPU 硬解码
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "k1_jpu.h"
#include "k1_jpu_api.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef HAS_K1_JPU_LIB
#include <dlfcn.h>

/*
 * libspacemit_mpp.so ALSO exports AsrJpuDecOpen (a different implementation).
 * Direct-linking resolves to the wrong one → segfault.  We must dlopen
 * libjpu.so explicitly and bind its symbols via dlsym to get the right
 * JPU implementation.  RTLD_LOCAL keeps libjpu.so symbols from clashing
 * with the global namespace.
 *
 * LogMsg stub: libjpu.so depends on this but k1x-jpu doesn't ship it.
 * The stub is exported via --export-dynamic and visible to dlopen'd libs.
 */
__attribute__((visibility("default"), used))
void LogMsg(int level, const char* fmt, ...) {
    (void)level;
    (void)fmt;
}

/* ── Function ptrs bound from libjpu.so (NOT the global / mpp.so versions) ── */
static int  (*jpudec_Open)(JpuDecHandle*) = NULL;
static int  (*jpudec_GetInitialInfo)(JpuDecHandle, JpuDecInitialInfo*) = NULL;
static int  (*jpudec_SetParam)(JpuDecHandle, const JpuDecParam*) = NULL;
static int  (*jpudec_Decode)(JpuDecHandle, const uint8_t*, int, uint8_t*, int*) = NULL;
static void (*jpudec_Close)(JpuDecHandle) = NULL;

static void* g_jpu_so          = NULL;  /* dlopen handle for libjpu.so */
static JpuDecHandle g_jpu_h    = NULL;
static JpuDecHandle g_jpu_h_ex = NULL;
static bool g_jpu_probed       = false;
static bool g_jpu_available    = false;
static int  g_jpu_out_w        = 0;
static int  g_jpu_out_h        = 0;
static uint8_t* g_jpu_yuv_buf  = NULL;
static size_t g_jpu_yuv_cap    = 0;

/* ── 简单 YUV420 → RGB (最近邻色度上采样) ──
 * ivdep: 每像素独立, 无跨迭代依赖, 允许 GCC 自动向量化 */
static void yuv420_to_rgb(const uint8_t* restrict yuv, int w, int h, uint8_t* restrict rgb) {
    int wh = w * h;
    const uint8_t* restrict Y  = yuv;
    const uint8_t* restrict U  = yuv + wh;
    const uint8_t* restrict V  = yuv + wh + wh / 4;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int yi = y * w + x;
            int uvi = (y / 2) * (w / 2) + (x / 2);
            int yv = Y[yi];
            int uv_u = U[uvi] - 128;
            int uv_v = V[uvi] - 128;

            int r = (298 * yv + 409 * uv_v + 128) >> 8;
            int g = (298 * yv - 100 * uv_u - 208 * uv_v + 128) >> 8;
            int b = (298 * yv + 516 * uv_u + 128) >> 8;

            int di = yi * 3;
            rgb[di + 0] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            rgb[di + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
            rgb[di + 2] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
        }
    }
}

/* ── dlopen libjpu.so, bind the CORRECT AsrJpuDec* symbols ── */
static bool jpu_load_library(void) {
    if (g_jpu_so) return true;

    const char* paths[] = {
        "/usr/lib/libjpu.so",
        "/lib/libjpu.so",
        "libjpu.so",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        /* RTLD_NOW: LogMsg must resolve immediately (our stub is exported).
         * RTLD_LOCAL: keep libjpu.so symbols out of global scope to avoid
         *             clashing with libspacemit_mpp.so's AsrJpuDecOpen. */
        g_jpu_so = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_jpu_so) {
            log_info("[K1-JPU] dlopen %s", paths[i]);
            break;
        }
    }
    if (!g_jpu_so) {
        log_warning("[K1-JPU] dlopen libjpu.so failed: %s", dlerror());
        return false;
    }

    /* Bind symbols from THIS handle — guaranteed to be libjpu.so's versions */
    #define BIND(dst, name) do { \
        dst = dlsym(g_jpu_so, name); \
        if (!dst) { \
            log_error("[K1-JPU] dlsym(%s) failed: %s", name, dlerror()); \
            dlclose(g_jpu_so); g_jpu_so = NULL; \
            return false; \
        } \
    } while(0)

    BIND(jpudec_Open,           "AsrJpuDecOpen");
    BIND(jpudec_GetInitialInfo, "AsrJpuDecGetInitialInfo");
    BIND(jpudec_SetParam,       "AsrJpuDecSetParam");
    BIND(jpudec_Decode,         "AsrJpuDecStartOneFrame");
    BIND(jpudec_Close,          "AsrJpuDecClose");
    #undef BIND

    return true;
}

bool k1_jpu_is_available(void) {
    if (g_jpu_probed) return g_jpu_available;
    g_jpu_probed = true;

    if (!jpu_load_library()) {
        log_warning("[K1-JPU] libjpu.so failed to load — JPEG decode will use software");
        return false;
    }

    /* ── Quick hardware probe ──
     * jpudec_Open is bound via dlsym from libjpu.so specifically,
     * avoiding the conflicting AsrJpuDecOpen in libspacemit_mpp.so. */
    JpuDecHandle h = NULL;
    int rc = jpudec_Open(&h);
    if (rc != 0 || !h) {
        log_warning("[K1-JPU] Hardware probe: jpudec_Open rc=%d h=%p — "
                    "will retry on first real decode",
                    rc, (void*)h);
        g_jpu_available = true;
        return true;
    }
    jpudec_Close(h);

    g_jpu_available = true;
    log_info("[K1-JPU] Hardware JPEG decoder ready (libjpu.so directly linked)");
    return true;
}

int k1_jpu_decode_to_rgb(const uint8_t* jpeg_data, size_t jpeg_len,
                         uint8_t* rgb_out, int out_w, int out_h) {
    if (!jpeg_data || !rgb_out || out_w <= 0 || out_h <= 0) return -1;
    if (!g_jpu_available) return -2;

    /* ── 按需打开解码器 (每个分辨率打开一次) ── */
    if (!g_jpu_h) {
        static int open_fail_cnt = 0;
        if (open_fail_cnt >= 3) {
            /* Persistent failure — disable JPU for this session */
            static bool disabled = false;
            if (!disabled) {
                disabled = true;
                log_warning("[K1-JPU] DecOpen failed 3+ times — disabling JPU, using software permanently");
            }
            return -2;  /* triggers caller to fall back silently */
        }
        int rc = jpudec_Open(&g_jpu_h);
        if (rc != 0 || !g_jpu_h) {
            open_fail_cnt++;
            if (open_fail_cnt <= 3) {
                log_error("[K1-JPU] DecOpen failed (rc=%d)", rc);
            }
            return -3;
        }
        /* Success — reset counter */
        open_fail_cnt = 0;

        /* 设置默认参数: YUV420 planar 输出, 不旋转不缩放 */
        JpuDecParam param;
        memset(&param, 0, sizeof(param));
        param.subsample     = 420;   /* YUV420 输出 */
        param.ordering      = 0;     /* planar */
        param.stream_endian = 1;     /* JPEG 标准是大端 */
        param.frame_endian  = 0;     /* little-endian (RISC-V) */
        param.rotation      = 0;
        param.scale_h       = 0;
        param.scale_v       = 0;
        jpudec_SetParam(g_jpu_h, &param);

        g_jpu_out_w = out_w;
        g_jpu_out_h = out_h;
    }

    /* ── 按需扩展 YUV 中间缓冲 ──
     * YUV420 planar: Y=W*H, U=W*H/4, V=W*H/4 → 1.5×W×H */
    size_t yuv_need = (size_t)out_w * out_h * 3 / 2;
    if (yuv_need > g_jpu_yuv_cap) {
        free(g_jpu_yuv_buf);
        g_jpu_yuv_cap = (yuv_need + 4096 + 63) & ~(size_t)63;
        g_jpu_yuv_buf = (uint8_t*)aligned_alloc(64, g_jpu_yuv_cap);
        if (!g_jpu_yuv_buf) {
            g_jpu_yuv_cap = 0;
            return -4;
        }
    }

    /* ── 硬件解码 ── */
    int out_size = 0;
    int rc = jpudec_Decode(g_jpu_h, jpeg_data, (int)jpeg_len,
                                    g_jpu_yuv_buf, &out_size);
    if (rc != 0) {
        static int fail_cnt = 0;
        if (++fail_cnt <= 3) {
            log_warning("[K1-JPU] DecStartOneFrame failed (rc=%d, jpeg_len=%zu)",
                        rc, jpeg_len);
        }
        return -5;
    }

    /* First successful HW decode — confirm JPU is active */
    static int success_cnt = 0;
    if (success_cnt == 0) {
        success_cnt = 1;
        log_info("[K1-JPU] Hardware JPEG decode ACTIVE — %dx%d YUV420, %d bytes output",
                 out_w, out_h, out_size);
    }

    /* ── YUV420 → RGB 转换 ── */
    yuv420_to_rgb(g_jpu_yuv_buf, out_w, out_h, rgb_out);
    return 0;
}

int k1_jpu_decode_to_rgb_ex(const uint8_t* jpeg_data, size_t jpeg_len,
                            uint8_t* rgb_out, int out_w, int out_h,
                            int rotation, int scale_h, int scale_v) {
    if (!jpeg_data || !rgb_out || out_w <= 0 || out_h <= 0) return -1;
    if (!g_jpu_available) return -2;

    /* Validate rotation: only 0/90/180/270 */
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
        rotation = 0;
    }
    /* Validate scale: 0=none, 1=1/2, 2=1/4, 3=1/8 */
    if (scale_h < 0 || scale_h > 3) scale_h = 0;
    if (scale_v < 0 || scale_v > 3) scale_v = 0;

    /* Use independent handle — does not interfere with k1_jpu_decode_to_rgb() */
    if (!g_jpu_h_ex) {
        int rc = jpudec_Open(&g_jpu_h_ex);
        if (rc != 0 || !g_jpu_h_ex) {
            log_error("[K1-JPU] DecOpen failed for _ex (rc=%d)", rc);
            return -3;
        }

        JpuDecParam param;
        memset(&param, 0, sizeof(param));
        param.subsample     = 420;
        param.ordering      = 0;
        param.stream_endian = 1;
        param.frame_endian  = 0;
        param.rotation      = rotation;
        param.scale_h       = scale_h;
        param.scale_v       = scale_v;
        jpudec_SetParam(g_jpu_h_ex, &param);
    }

    /* Allocate YUV intermediate buffer (shared with base decoder) */
    size_t yuv_need = (size_t)out_w * out_h * 3 / 2;
    if (yuv_need > g_jpu_yuv_cap) {
        free(g_jpu_yuv_buf);
        g_jpu_yuv_cap = (yuv_need + 4096 + 63) & ~(size_t)63;
        g_jpu_yuv_buf = (uint8_t*)aligned_alloc(64, g_jpu_yuv_cap);
        if (!g_jpu_yuv_buf) {
            g_jpu_yuv_cap = 0;
            return -4;
        }
    }

    int out_size = 0;
    int rc = jpudec_Decode(g_jpu_h_ex, jpeg_data, (int)jpeg_len,
                                   g_jpu_yuv_buf, &out_size);
    if (rc != 0) {
        static int fail_cnt = 0;
        if (++fail_cnt <= 3) {
            log_warning("[K1-JPU] DecStartOneFrame failed (rc=%d, jpeg_len=%zu)",
                        rc, jpeg_len);
        }
        return -5;
    }

    yuv420_to_rgb(g_jpu_yuv_buf, out_w, out_h, rgb_out);
    return 0;
}

#else
/* ── libjpu.so 不可用 ── */
bool k1_jpu_is_available(void) {
    return false;
}

int k1_jpu_decode_to_rgb(const uint8_t* jpeg_data, size_t jpeg_len,
                         uint8_t* rgb_out, int out_w, int out_h) {
    (void)jpeg_data; (void)jpeg_len; (void)rgb_out; (void)out_w; (void)out_h;
    return -1;
}

int k1_jpu_decode_to_rgb_ex(const uint8_t* jpeg_data, size_t jpeg_len,
                            uint8_t* rgb_out, int out_w, int out_h,
                            int rotation, int scale_h, int scale_v) {
    (void)jpeg_data; (void)jpeg_len; (void)rgb_out; (void)out_w; (void)out_h;
    (void)rotation; (void)scale_h; (void)scale_v;
    return -1;
}
#endif /* HAS_K1_JPU_LIB */
