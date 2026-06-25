/*
 * K1 JPU Hardware Decoder API (逆向自 libjpu.so 符号表 + jpu_dec_test 用法)
 *
 * libjpu.so 来自 k1x-jpu 包 (Bianbu apt install k1x-jpu)
 * 头文件未随包安装 — 以下声明从 nm -D /usr/lib/libjpu.so 和
 * /usr/bin/jpu_dec_test 的命令行参数逆向推断。
 *
 * 解码流程:
 *   1. AsrJpuDecOpen(&handle)           — 打开 JPU 设备
 *   2. AsrJpuDecGetInitialInfo(h, &info) — 解析 JPEG 头 (宽/高/子采样)
 *   3. AsrJpuDecSetParam(h, &param)     — 设置输出格式/旋转/缩放
 *   4. AsrJpuDecStartOneFrame(h, bs, sz, out, &out_sz) — 解码一帧
 *   5. AsrJpuDecClose(h)                — 关闭设备
 */

#ifndef K1_JPU_API_H
#define K1_JPU_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ── */
typedef void* JpuDecHandle;

/* ── JPEG 初始信息 (由 GetInitialInfo 填充) ── */
typedef struct {
    int width;
    int height;
    int subsample;          /* 444=0, 422=1, 420=2, 440=3, 400=4, 411=5 */
    int num_components;     /* 1=灰度, 3=YCbCr */
    int restart_interval;   /* 0=无重启标记 */
} JpuDecInitialInfo;

/* ── 解码参数 ── */
typedef struct {
    int rotation;           /* 0, 90, 180, 270 */
    int mirror;             /* 0=none, 1=vertical, 2=horizontal, 3=both */
    int scale_h;            /* 0=none, 1=1/2, 2=1/4, 3=1/8 */
    int scale_v;            /* 0=none, 1=1/2, 2=1/4, 3=1/8 */
    int subsample;          /* 输出子采样: 444, 422, 420 (0=保持原始) */
    int ordering;           /* 输出排序: 0=planar, 1=NV12, 2=NV21, 3=YUYV, 4=YVYU, 5=UYVY, 6=VYUY, 7=AYUV */
    int stream_endian;      /* 0=little, 1=big (most JPEGs are big-endian) */
    int frame_endian;       /* 像素字节序 */
    int pixel_j;            /* 16-bit像素对齐: 0=MSB, 1=LSB little-endian */
    int roi_x, roi_y, roi_w, roi_h;  /* ROI 区域 (0=全图) */
} JpuDecParam;

/* ── 解码器 API ── */
int  AsrJpuDecOpen(JpuDecHandle* handle);
int  AsrJpuDecGetInitialInfo(JpuDecHandle handle, JpuDecInitialInfo* info);
int  AsrJpuDecSetParam(JpuDecHandle handle, const JpuDecParam* param);
int  AsrJpuDecStartOneFrame(JpuDecHandle handle,
                            const uint8_t* bs_buf, int bs_size,
                            uint8_t* out_buf, int* out_size);
void AsrJpuDecClose(JpuDecHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* K1_JPU_API_H */
