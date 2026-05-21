# 灵柒·探瞳 构建与部署指南

> 本文档描述如何在各平台上构建、测试和部署 `lingqi_tantong_c` 项目。

---

## 1. 快速构建

### 1.1 依赖清单

| 依赖 | 必须 | 版本要求 | 用途 |
|------|------|---------|------|
| C 编译器 | ✅ | GCC≥8 / Clang≥10 / MSVC 2022 | 编译 |
| CMake | 推荐 | ≥3.16 | 构建系统 |
| ONNX Runtime | 可选 | SpacemiT ORT 2.0.2 | 真实 AI 推理（无则回退启发式） |
| Python 3 | 可选 | ≥3.8 | build.py 跨平台构建脚本 |

### 1.2 一行构建命令

```bash
# Linux / WSL
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# macOS
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)

# Windows (PowerShell, MSVC 工具链)
mkdir build; cd build; cmake .. -G "Visual Studio 17 2022" -A x64; cmake --build . --config Release
```

---

## 2. 构建方法详解

项目提供 2 种构建方式，推荐优先级：**CMake > build.py**

### 2.1 CMake 构建（推荐）

```bash
# ===== Release 构建 =====
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# ===== Debug 构建 =====
mkdir build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug

# ===== 启用 ONNX Runtime =====
mkdir build_onnx && cd build_onnx
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 \
         -DUSE_ONNX_RUNTIME=ON
cmake --build . --config Release
```

**CMake 构建选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` / `Debug` / `RelWithDebInfo` |
| `ONNX_RUNTIME_DIR` | *(empty)* | SpacemiT ONNX Runtime 安装路径（含 `include/` 和 `lib/`） |
| `USE_ONNX_RUNTIME` | `ON` | 是否启用 ONNX Runtime 推理（关闭则用启发式回退） |
| `USE_SPACENGINE_AI` | `OFF` | 是否启用 SpacemiT RISC-V AI 指令加速 |
| `USE_OPENMP` | `ON` | 是否启用 OpenMP 多核并行 |
| `ENABLE_RVV_OPT` | `OFF` | 是否启用 RISC-V Vector 手写向量化优化 |
| `ENABLE_K1_PIPELINE` | `ON` | 启用 K1 双 Cluster 流水线并行 |
| `ENABLE_K1_TCM` | `ON` | 启用 K1 TCM 紧耦合内存（AI 权重预加载） |
| `ENABLE_K1_VPU` | `ON` | 启用 K1 VPU 硬件视频/JPEG 加速 |
| `ENABLE_K1_JPU` | `ON` | 启用 K1 JPU 硬件 JPEG 解码 |
| `BIANBU_SYSROOT` | *(empty)* | Bianbu OS sysroot 路径（交叉编译时必需） |
| `SPACENGINE_DIR` | *(empty)* | SpacemiT Spacengine SDK 路径 |
| `K1_MPP_DIR` | *(empty)* | K1 MPP 媒体处理平台 SDK 路径 |
| `K1_JPU_DIR` | *(empty)* | K1 JPU 硬件 JPEG 解码库路径 |

### 2.2 Python build.py 构建

无需 CMake，直接调用编译器：

```bash
python build.py build                 # Release 构建
python build.py build --debug         # Debug 构建
python build.py test                  # 运行测试
python build.py all                   # 构建+测试
python build.py clean                 # 清理构建产物
```

**说明：** `build.py` 检测平台后自动选择 GCC/Clang/MSVC 编译器。

---
 
## 3. 运行与测试

### 3.1 构建产物

```
build/
├── lingqi_tantong(.exe)    # 主程序
└── test_basic(.exe)        # 单元测试
```

### 3.2 运行命令

```bash
# 最小运行
./lingqi_tantong --video_path test_video.mp4

# 完整参数
./lingqi_tantong \
    --video_path test_video.mp4 \
    --output_path output/results \
    --config configs/default.yaml \
    --max_frames 500 \
    --save_frame_interval 1

# 查看帮助
./lingqi_tantong --help
```

### 3.3 运行测试

```bash
./test_basic
```

预期输出：
```
===== LingQi TanTong Basic Tests =====
PASSED: test_bounding_box_iou
PASSED: test_distance_calculation
...
===== All 12 tests PASSED =====
```

### 3.4 WSL 一键脚本

在 Windows WSL 环境：

```bash
bash run_wsl.sh
```

脚本自动完成：构建 → 测试 → 运行 → 结果检查。

---

## 4. RISC-V 交叉编译（目标平台：SpacemiT X60）

### 4.1 工具链安装

```bash
# 方式1: 官方 Bianbu SDK (推荐)
wget https://archive.bianbu.space/spacelings/bianbu-sdk.tar.gz
tar -xzf bianbu-sdk.tar.gz -C /opt/bianbu-sdk

# 方式2: 通用 RISC-V 工具链
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

### 4.2 CMake 交叉编译配置

```bash
mkdir build_riscv && cd build_riscv
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBIANBU_SYSROOT=/opt/bianbu-sdk/sysroot
make -j$(nproc)
```

### 4.3 RISC-V 工具链 CMake 文件

项目已提供 `cmake/riscv64-toolchain.cmake`，核心配置如下：

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(TOOLCHAIN_PREFIX /opt/bianbu-sdk/bin/riscv64-unknown-linux-gnu-)
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

set(CMAKE_FIND_ROOT_PATH /opt/bianbu-sdk/sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# SpacemiT X60 支持 RVV 0.7.1 / 1.0
set(CMAKE_C_FLAGS "-march=rv64gcv -O3")
```

### 4.4 RVV 向量化编译标志

```c
// 在代码中使用 RVV intrinsics（规划中）
// #include <riscv_vector.h>  // X60 使用 RVV 1.0 或 0.7.1
```

编译时启用：
```bash
cmake .. -DENABLE_RVV_OPT=ON
```

---

## 5. 部署到 Bianbu Linux

### 5.1 文件清单

部署到 K1 Muse Pi Pro 需要的文件：

```
target_root/
├── lingqi_tantong               # 可执行文件 (RISC-V)
├── configs/default.yaml         # 配置文件
├── models/                      # ONNX 模型文件
│   ├── Human Recognition/yolov8n.onnx
│   ├── Face Recognition/scrfd_10g_bnkps.onnx
│   ├── Face Recognition/glintr100.onnx
│   └── Action Prediction/.../yolov8n-pose.onnx
└── libs/                        # 依赖库 (ONNX Runtime RISC-V 等)
```

### 5.2 传输与运行

```bash
# SCP 传输至设备
scp lingqi_tantong root@k1-muse-pi:/opt/lingqi/
scp configs/default.yaml root@k1-muse-pi:/opt/lingqi/configs/
scp -r models/ root@k1-muse-pi:/opt/lingqi/models/

# SSH 运行
ssh root@k1-muse-pi
cd /opt/lingqi
./lingqi_tantong --video_path /data/video/input.mp4 --config configs/default.yaml
```

### 5.3 systemd 服务（可选）

```ini
# /etc/systemd/system/lingqi.service
[Unit]
Description=LingQi TanTong Detection Service
After=network.target

[Service]
Type=simple
ExecStart=/opt/lingqi/lingqi_tantong --video_path /dev/video0 --config /opt/lingqi/configs/default.yaml
Restart=on-failure
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```

---

## 6. 常见问题

### 6.1 编译错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `fatal error: onnxruntime_c_api.h: No such file` | ONNX Runtime 未安装或路径不对 | 安装并指定 `-DONNX_RUNTIME_DIR` |
| `undefined reference to '__atomic_*'` | 链接时缺少 `-latomic` | CMakeLists 已处理，RISC-V 需确认 |
| `error: unrecognized command line option '-march=native'` | 无 x86 编译器 | 使用交叉编译工具链 |
| `MSVC: C1010 unexpected end of file` | 预编译头问题 | 清理 build 目录重新 cmake |

### 6.2 运行时错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Failed to load model: models/...` | 模型文件缺失或路径错误 | 确认 models/ 目录完整 |
| `imgdecode failed` | 视频文件格式不支持 | 确认输入为 MP4/H.264 |
| `Segmentation fault` | 内存越界 | 使用 Address Sanitizer 编译定位 |

---

## 7. 性能基准

### 7.1 参考性能（x86-64, Intel i7-12700H）

| 场景 | FPS | 延迟 (ms) | 内存 (MB) |
|------|-----|-----------|-----------|
| 无 ONNX (启发式回退) | 85-120 | 8-12 | 45 |
| ONNX CPU (fp32) | 18-25 | 40-55 | 180 |
| ONNX CPU (int8) | 35-50 | 20-28 | 120 |

### 7.2 目标性能（RISC-V SpacemiT X60）

| 场景 | 目标 FPS | 目标延迟 (ms) | 目标内存 (MB) |
|------|----------|---------------|---------------|
| RISC-V AI 指令加速 INT8 | 30 | 33 | <800 |
| CPU 回退模式 | 10 | 100 | <200 |

---

## 相关文档

| 文档 | 路径 |
|------|------|
| 项目 README | `README.md` |
| 架构文档 | `docs/ARCHITECTURE.md` |
| 未实现模块 | `docs/IMPLEMENTATION_GAPS.md` |
| 配置文件 | `configs/default.yaml` |