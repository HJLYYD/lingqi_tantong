/*
 * K1 JPU — 硬件 JPEG 解码加速器接口
 *
 * K1 芯片集成硬件 JPEG 解码单元 (JPU)，通过 /dev/jpu 设备节点访问。
 * 当 JPU SDK (k1x-jpu) 可用时，用于替换 libjpeg-turbo 软解码，
 * 释放 CPU 资源给推理任务。
 *
 * 使用方式:
 *   1. 安装 k1x-jpu SDK 并设置 -DK1_JPU_DIR=/path/to/k1x-jpu
 *   2. CMake 自动检测并链接 libjpu，定义 HAS_K1_JPU
 *   3. soft_jpeg_decode_to_rgb() 优先使用硬件解码
 */

#ifndef K1_JPU_H
#define K1_JPU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 检查 JPU 硬件是否可用 (设备节点 + 库符号都存在) */
bool k1_jpu_is_available(void);

/**
 * JPU 硬件 JPEG 解码为 RGB
 *
 * @param jpeg_data   JPEG 原始数据
 * @param jpeg_len    JPEG 数据长度 (字节)
 * @param rgb_out     输出 RGB 缓冲区 (调用者分配, 至少 out_w*out_h*3 字节)
 * @param out_w       目标宽度
 * @param out_h       目标高度
 * @return 0 = 成功, < 0 = 错误码
 */
int k1_jpu_decode_to_rgb(const uint8_t* jpeg_data, size_t jpeg_len,
                         uint8_t* rgb_out, int out_w, int out_h);

/**
 * JPU 硬件 JPEG 解码为 RGB（扩展版，支持硬件旋转和缩放）
 *
 * JPU 硬件原生支持旋转/缩放/镜像，在解码阶段零成本完成。
 *
 * @param rotation    旋转角度: 0=none, 90, 180, 270
 * @param scale_h     水平缩放: 0=none, 1=1/2, 2=1/4, 3=1/8
 * @param scale_v     垂直缩放: 0=none, 1=1/2, 2=1/4, 3=1/8
 * @return 0 = 成功, < 0 = 错误码
 */
int k1_jpu_decode_to_rgb_ex(const uint8_t* jpeg_data, size_t jpeg_len,
                            uint8_t* rgb_out, int out_w, int out_h,
                            int rotation, int scale_h, int scale_v);

#ifdef __cplusplus
}
#endif

#endif /* K1_JPU_H */
