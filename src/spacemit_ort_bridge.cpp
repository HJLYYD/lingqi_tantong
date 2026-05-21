#include "spacemit_ort_bridge.h"

#ifdef HAS_SPACEMIT_EP
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

int spacemit_ort_session_options_init(OrtSessionOptions* c_session_opts) {
    if (!c_session_opts) return -1;

    Ort::SessionOptions opts(c_session_opts);
    SessionOptionsSpaceMITEnvInit(opts);
    opts.release();
    return 0;
}
#else
int spacemit_ort_session_options_init(OrtSessionOptions* c_session_opts) {
    (void)c_session_opts;
    /*
     * 当前为 stub 实现 (HAS_SPACEMIT_EP 未定义)。
     * 要启用 K1 X60 RISC-V AI 指令加速:
     *   1. 编译 libonnxruntime.so 时链接 libspacemit_ep.so
     *   2. CMake 构建时确保 spacemit_ort_env.h 在 include 路径中
     *   3. 确认 /dev/tcm 设备节点存在且可访问
     * 参考: 进迭时空 K1 官方文档 — ONNX Runtime + SpacemiT EP
     */
    return -1;
}
#endif