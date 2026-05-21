# 灵柒·探瞳 (LingQi TanTong) 智能探测箭系统 — C 语言实现

> **版本**: v1.1.0 — Muse Pi Pro (SpacemiT K1 X60) 完全适配  
> **目标平台**: 进迭时空 SpacemiT K1 (X60) Muse Pi Pro — RISC-V 64  
> **开发平台**: Linux (交叉编译) / K1 原生 (native build)  
> **语言标准**: C11 + C++17 (EP 桥接)  
> **构建系统**: CMake ≥3.16 + Python build.py  

---

## 项目概述

灵柒·探瞳是一款基于国产 RISC-V 芯片（进迭时空 K1 X60）的智能探测箭系统。

本仓库为射手端核心算法的 **C 语言高性能实现**，专为 Muse Pi Pro 适配：

- **Arrow UART 实时视频链路** — ESP32-P4 箭矢端 → K1 射手端 JPEG+IMU 帧分发  
- **SpacemiT Execution Provider 加速推理** — 256-bit RVV 1.0 + IME 矩阵扩展，最高 2.0 TOPS  
- **OpenMP 多核并行** — K1 8 核 RISC-V 充分利用  
- **RISC-V Vector 1.0** — 256-bit SIMD 向量化图像预处理和矩阵运算  

### 系统架构

```
                           ┌── Arrow UART (4线) ──┐
                           │  JPEG + IMU_Pose     │
                           ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SystemController                             │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │   Data Layer    │  │ Biz Logic      │  │  Presentation    │  │
│  │                 │  │                │  │                  │  │
│  │ VideoProcessor  │  │ InferencePipe  │  │  Visualizer      │  │
│  │ ArrowReceiver   │  │ ObjectTracker  │  │  ARRenderer      │  │
│  │ IMUHandler      │  │ SpatialEngine  │  │  VideoWriter     │  │
│  │ ModelStore      │  │ ORT + SpacemiT │  │                  │  │
│  └─────────────────┘  └────────────────┘  └──────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Logger │ ConfigManager │ ResultManager │ Mahony │ KCP-Lite │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 硬件加速说明

### ⚠️ 概念澄清

| 常见误解 | 事实 |
|----------|------|
| K1 X60 有独立 NPU 芯片 | **K1 无独立 NPU**。CPU 核通过 IME 扩展指令在核内完成矩阵运算 |
| 推理加速需要 Spacengine SDK | SpacemiT EP 通过 `SessionOptionsSpaceMITEnvInit()` 注册自动启用 |
| 需要单独安装 NPU 驱动 | 只需 `libonnxruntime.so` + `libspacemit_ep.so` + `/dev/tcm` 权限 |

### SpacemiT K1 X60 加速路径

```
应用程序 (C/C++)
    │
    ▼
ONNX Runtime C/C++ API
    │
    ├── SessionOptionsSpaceMITEnvInit()  ← 注册 SpacemiT EP
    │
    ▼
libonnxruntime.so   ──动态加载──▶  libspacemit_ep.so
    │                                    │
    ├── libonnxruntime_mlas.so           ├── 256-bit RVV 1.0 向量指令
    │   (RISC-V 优化数学库)              ├── IME 矩阵乘法指令 (16条)
    │                                    └── IME 滑窗指令
    ▼
K1 X60 CPU 核 (X60™ 8核)
  ├── RVV 1.0 向量寄存器 (VLEN=256)
  ├── IME 自定义扩展指令集
  └── 2.0 TOPS AI 算力 (核内融合)
```

**IME (Intelligent Matrix Extension)** 是进迭时空 X60 核独有的 16 条自定义 AI 指令，包括矩阵乘法和滑窗操作。配合 256-bit RVV 1.0，K1 不依赖独立 NPU 即可达到 2.0 TOPS。

---

## 功能特性

| 模块 | 功能 | 状态 |
|------|------|------|
| **目标检测** | YOLOv8n 行人检测 + SpacemiT ORT / 启发式回退 | ✅ |
| **姿态估计** | YOLOv8-Pose 17 关键点 COCO 骨架 | ✅ |
| **人脸识别** | SCRFD 人脸检测 + ArcFace 特征提取 | ✅ |
| **多目标跟踪** | ByteTrack + 7状态 Kalman + EMA 平滑 | ✅ |
| **3D 空间定位** | 针孔相机模型 + 坐标系初始化 + 深度估计 | ✅ |
| **可视化** | 边界框、骨架线、轨迹图、AR 标记 | ✅ |
| **配置管理** | YAML 配置文件 + 可编程默认值 | ✅ |
| **日志系统** | 线程安全、分级日志、文件/控制台双输出 | ✅ |
| **结果管理** | Session 追踪、JSON/CSV 报告导出 | ✅ |
| **Arrow 协议** | ESP32-P4 ↔ K1 双 UART 实时帧链路（3 Mbps） | ✅ |
| **SpacemiT EP 推理** | RVV 1.0 + IME 硬件加速（2.0 TOPS） | ✅ |
| **实时管道** | `process_realtime` — Arrow → IMU → 推理 → 跟踪闭环 | ✅ |
| **OpenMP 并行** | K1 8 核多线程推理 + 图像处理 | ✅ |
| **Mahony 滤波** | 9-DOF IMU 互补滤波姿态解算 | ✅ |
| **KCP-Lite** | 自定义可靠 UDP 传输协议 | ✅ |
| **模型量化 (IME 适配)** | YOLO/SCRFD/ArcFace ONNX INT8 量化 | 🔜 高优先级 |
| **RVV 手写向量化** | letterbox / NMS / 矩阵乘法手写 RVV intrinsic | 🔜 规划中 |
| **VINS-Mono SLAM** | 视觉-惯性紧耦合里程计 | 🔜 规划中 |
| **ATW 补偿** | 异步时间扭曲 AR 延迟补偿 | 🔜 规划中 |
| **MiDaS 深度** | 单目深度估计模型 | 🔜 规划中 |

---

## 目录结构

```
lingqi_tantong_c/
├── include/                          # 头文件（28 个）
│   ├── core_types.h                  #   核心类型 + 内联工具函数
│   ├── config_manager.h              #   配置管理器
│   ├── logger.h                      #   日志系统
│   ├── utils.h                       #   工具函数（数学、排序、图像）
│   ├── model_store.h                 #   模型文件管理
│   ├── video_processor.h             #   视频输入（文件/摄像头/Arrow）
│   ├── video_writer.h                #   AVI 视频输出
│   ├── imu_handler.h                 #   IMU 传感器数据处理
│   ├── arrow_receiver.h              #   Arrow 协议 UART 接收器（双链路）
│   ├── kcp_lite.h                    #   KCP 可靠传输协议（轻量版）
│   ├── mahony_filter.h               #   Mahony 互补滤波（9-DOF 姿态解算）
│   ├── ai_accel_adapter.h            #   RISC-V AI 指令加速适配层（Spacengine + stub）
│   ├── ort_common.h                  #   ONNX Runtime 共享模块（统一入口）
│   ├── spacemit_ort_bridge.h         #   SpacemiT EP C↔C++ 桥接头
│   ├── yolov8_detector.h             #   YOLOv8 目标检测
│   ├── yolov8_pose_estimator.h       #   YOLOv8 姿态估计
│   ├── scrfd_detector.h              #   SCRFD 人脸检测
│   ├── arcface_recognizer.h          #   ArcFace 人脸识别
│   ├── inference_pipeline.h          #   统一 AI 推理流水线
│   ├── tracking_manager.h            #   多目标跟踪 (ByteTrack + Kalman)
│   ├── spatial_engine.h              #   3D 空间定位 + 轨迹管理
│   ├── visualizer.h                  #   可视化渲染
│   ├── ar_renderer.h                 #   AR 渲染引擎
│   ├── result_manager.h              #   结果管理 + 报告导出
│   ├── system_controller.h           #   系统主控制器（离线+实时）
│   ├── model_export.h                #   3D 模型导出
│   └── benchmark.h                   #   性能基准测试
├── src/                              # 源文件（29 个）
│   ├── main.c                        #   入口（离线 + 实时模式）
│   ├── system_controller.c           #   系统控制器（离线 + 实时管道）
│   ├── inference_pipeline.c          #   AI 推理流水线（级联检测 + 过滤）
│   ├── tracking_manager.c            #   多目标跟踪（3 阶段匹配）
│   ├── spatial_engine.c              #   空间定位（IMU 注入 + 深度修正）
│   ├── yolov8_detector.c             #   YOLOv8（SpacemiT ORT + 启发式回退）
│   ├── yolov8_pose_estimator.c       #   YOLOv8-Pose 姿态估计
│   ├── scrfd_detector.c              #   SCRFD 人脸检测
│   ├── arcface_recognizer.c          #   ArcFace 识别（分块特征 + 余弦相似度）
│   ├── visualizer.c                  #   可视化渲染
│   ├── ar_renderer.c                 #   AR 渲染（运动补偿 + 标记叠加）
│   ├── video_processor.c             #   视频输入（文件/摄像头/Arrow）
│   ├── video_writer.c                #   AVI 视频输出
│   ├── imu_handler.c                 #   IMU 处理（验证/解析/平滑 + 外部姿态注入）
│   ├── arrow_receiver.c              #   Arrow UART 接收器（双链路 + JPEG+IMU 帧解析）
│   ├── ai_accel_adapter.c            #   AI 加速适配层（Spacengine / stub）
│   ├── kcp_lite.c                    #   KCP 可靠传输协议
│   ├── mahony_filter.c               #   Mahony 互补滤波
│   ├── ort_common.c                  #   ONNX Runtime 共享模块（全局单例 + EP 注册）
│   ├── spacemit_ort_bridge.cpp       #   SpacemiT EP C++ 桥接（调用官方 SDK）
│   ├── config_manager.c              #   配置管理（YAML 解析）
│   ├── logger.c                      #   日志系统（线程安全）
│   ├── utils.c                       #   工具函数
│   ├── result_manager.c              #   结果管理（Session/JSON/CSV）
│   ├── model_store.c                 #   模型文件管理
│   ├── model_export.c                #   3D 模型导出（OBJ）
│   ├── core_types.c                  #   核心类型辅助函数
│   └── benchmark.c                   #   性能基准测试
├── cmake/                            # CMake 工具链
│   ├── riscv64-toolchain.cmake       #   RISC-V 交叉编译（Bianbu sysroot）
│   └── esp32p4-toolchain.cmake       #   ESP32-P4 交叉编译
├── configs/
│   └── default.yaml                  #   默认配置（EP、Arrow、IMU、推理参数）
├── docs/
│   ├── ARCHITECTURE.md               #   架构设计文档
│   ├── BUILD_GUIDE.md                #   构建与部署指南
│   ├── IMPLEMENTATION_GAPS.md        #   未实现模块清单
│   ├── IMPROVEMENT_PLAN.md           #   Muse Pi Pro 改造计划
│   └── 改进.md                       #   改进细则
├── models/                           # ONNX 模型文件
│   ├── Human Recognition/
│   │   └── yolov8n.onnx              #   ⚠ FP32 → 待量化 INT8
│   ├── Face Recognition/
│   │   ├── scrfd_10g_bnkps.onnx      #   ⚠ FP32 → 待量化 INT8
│   │   ├── glintr100.onnx            #   ⚠ FP32 → 待量化 INT8
│   │   ├── 1k3d68.onnx               #   ⚠ FP32 → 待量化 INT8
│   │   ├── 2d106det.onnx
│   │   └── genderage.onnx
│   └── Action Prediction/
│       └── Skeleton Recognition/
│           └── yolov8n-pose.onnx     #   ⚠ FP32 → 待量化 INT8
├── tests/
│   └── test_basic.c                  #   单元测试（12 项）
├── output/                           # 输出目录
├── CMakeLists.txt                    # CMake 构建配置（RISC-V 优先 + EP 检测）
├── build.py                          # Python 构建/部署脚本
├── run_wsl.sh                        # WSL 一键运行脚本
├── verify_port.py                    # 端口验证工具
└── README.md                         # 本文件
```

---

## 核心数据结构

全部定义在 [include/core_types.h](include/core_types.h)：

| 结构体 | 用途 | 关键字段 |
|--------|------|----------|
| `BoundingBox` | 2D 边界框 | x, y, w, h, area, center_x, center_y |
| `Detection` | 检测结果 | bbox, confidence, class_id, class_name |
| `Keypoint` | 人体关键点 | x, y, confidence, name[16] |
| `PoseEstimation` | 姿态估计 | keypoints[17], bbox, has_bbox, confidence |
| `FaceIdentity` | 人脸识别结果 | bbox, identity[64], similarity, feature_vector[512] |
| `SpatialPosition` | 3D 空间位置 | x, y, z, depth_confidence, is_valid, world_x, world_z |
| `TrackedObject` | 跟踪目标 | track_id, detection, spatial_pos, pose, face, trajectory, velocity |
| `IMUData` | IMU 传感器数据 | timestamp, accel[3], gyro[3] |
| `ArrowSourceFrame` | Arrow 源帧 | jpeg_data (65KB), imu_timestamp, header/flags |
| `PipelineMode` | 运行模式枚举 | OFFLINE / REALTIME / BENCHMARK |
| `ArrowReceiveState` | Arrow 状态枚举 | IDLE / SYNCING / LOCKED / ERROR |
| `AIAcclTensor` | AI 加速张量 | data, dims[4], data_type, size |

---

## 处理流水线

### 离线模式

```
Frame Read → AI Inference → Coordinate Init → Spatial Calc →
  Tracking → [Pose/Face Assoc] → Trajectory → Visualization → Output
```

### 实时模式（Muse Pi Pro）

```
Arrow UART (A+C) → Frame Decode → IMU Pose Injection → Spatial Correction →
  SpacemiT EP Inference (RVV+IME) → Tracking → Trajectory → Visualization → Display
                                    ↑
              ESP32-P4 IMU ─────────┘ (cos(pitch) depth correction)
```

1. **Arrow UART 双链路接收**: 3 Mbps 双 UART 同时接收 JPEG 帧和 IMU 姿态数据  
2. **IMU 注入**: 外部 ESP32-P4 IMU 姿态直接注入 spatial_engine，cos(pitch) 深度修正  
3. **SpacemiT EP 推理**: `SessionOptionsSpaceMITEnvInit` 注册 → RVV 1.0 + IME 硬件加速推理  
4. **空间定位**: 针孔相机模型 2D→3D + 人体高度估计 + IMU 补偿  
5. **多目标跟踪**: ByteTrack 三阶段匹配 + 7 状态 Kalman + EMA  
6. **可视化**: 边界框 + 骨架 + ID + 3D 轨迹 + AR 标记叠加  

### SpacemiT EP 调用链

```
ort_global_init()                           → CreateEnv("LingQiTanTong")
ort_create_session(path, 4)                 → CreateSessionOptions
                                               ├── SetGraphOptimizationLevel(ORT_ENABLE_ALL)
                                               ├── SetIntraOpNumThreads(4)
                                               ├── spacemit_ort_session_options_init()  ← C++ bridge 调用
                                               │      │
                                               │      └── SessionOptionsSpaceMITEnvInit(opts)
                                               │            (注册 libspacemit_ep.so)
                                               └── CreateSession(model_path)
                                                    │
                                                    └── 自动加载 libspacemit_ep.so
                                                         → RVV 1.0 + IME 加速推理
```

---

## 快速开始

### 前置要求

| 依赖 | 说明 |
|------|------|
| GCC 13.2+ (riscv64) | K1 原生 / 交叉编译 `riscv64-linux-gnu-gcc` |
| CMake ≥3.16 | 构建系统 |
| SpacemiT ONNX Runtime 2.0.2 | [下载](https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.2.tar.gz) — 推荐使用此版本 |
| `/dev/tcm` 权限 | IME 硬件加速资源（需要 `sudo chmod 777 /dev/tcm`） |
| ONNX 模型 | FP32 可运行，但建议量化 INT8 以获得 IME 加速 |

### K1 原生构建（推荐）

```bash
# 1. 安装 SpacemiT ORT 库和头文件
sudo cp spacemit-ort.riscv64.2.0.2/lib/* /usr/lib/
sudo ldconfig
sudo cp -r spacemit-ort.riscv64.2.0.2/include/* /usr/local/include/

# 2. 构建
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ONNX_RUNTIME=ON \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2
make -j$(nproc)

# 3. 给予 IME 硬件访问权限
sudo chmod 777 /dev/tcm

# 4. 设置 LD_LIBRARY_PATH 并运行
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
./lingqi_tantong --video_path test.mp4

# 5. （推荐）先用 perf_test 测试单模型性能
./spacemit-ort.riscv64.2.0.2/bin/onnxruntime_perf_test \
  -e spacemit -r 100 models/Human\ Recognition/yolov8n.onnx
```

### 交叉编译（x86 Linux → RISC-V）

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
  -DBIANBU_SYSROOT=/opt/bianbu-sysroot \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
# 部署到 K1
scp build/lingqi_tantong k1@192.168.1.x:/home/k1/
```

### Python 构建脚本

```bash
python build.py riscv64              # 交叉编译 RISC-V
python build.py host                 # 本地开发测试（host mode, 无 EP）
python build.py test                 # 编译并运行测试（QEMU）
python build.py clean                # 清理编译产物
python build.py deploy --deploy-host 192.168.1.100   # 自动 scp 到 K1
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video_path <path>` | 离线模式输入视频 | 无 |
| `--realtime` | 启用实时管道模式 | 关闭 |
| `--uart-A <path>` | Arrow UART 主链路 | `/dev/ttyS0` |
| `--uart-C <path>` | Arrow UART 辅链路 | `/dev/ttyS1` |
| `--baudrate <rate>` | UART 波特率 | `3000000` |
| `--camera <path>` | 摄像头设备 | `/dev/video0` |
| `--output_path <path>` | 输出目录 | `output/reports` |
| `--max_frames <N>` | 最大帧数（0=全部） | `0` |
| `--save_frame_interval <N>` | 帧保存间隔 | `10` |
| `--config <path>` | 配置文件路径 | `configs/default.yaml` |
| `--help` | 显示帮助 | — |

---

## 加速效果对比

### 预期性能（基于 K1 X60 + SpacemiT EP 2.0.2）

| 模型 | CPU-only (FP32) | EP (FP32) | EP (INT8 量化) |
|------|----------------|-----------|---------------|
| YOLOv8n (640×640) | ~500ms | ~200ms | **~60ms** |
| YOLOv8-Pose (640×640) | ~800ms | ~350ms | **~100ms** |
| SCRFD (640×640) | ~300ms | ~120ms | **~40ms** |
| ArcFace (112×112) | ~100ms | ~40ms | **~15ms** |

> ⚠ 以上为基于同类 RISC-V 平台的**预估数据**，实际性能需在 K1 上使用 `onnxruntime_perf_test -e spacemit` 实测。FP32→INT8 量化后性能可提升 **3-5 倍**（IME 矩阵指令对 INT8 有特殊优化）。

---

## CMake 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Release` | Release / Debug |
| `CMAKE_TOOLCHAIN_FILE` | — | 交叉编译工具链文件 |
| `BIANBU_SYSROOT` | — | Bianbu sysroot 路径 |
| `ONNX_RUNTIME_DIR` | — | SpacemiT ONNX Runtime 2.0.2 路径 |
| `USE_ONNX_RUNTIME` | `ON` | 启用 ONNX Runtime（含 EP） |
| `USE_SPACENGINE_AI` | `OFF` | 启用 Spacengine RISC-V AI 指令加速 |
| `USE_OPENMP` | `ON` | 启用 OpenMP 多核并行 |
| `ENABLE_RVV_OPT` | `OFF` | 启用 RVV 手写向量化 intrinsic |
| `ENABLE_K1_PIPELINE` | `ON` | 启用 K1 双 Cluster 流水线并行 |
| `ENABLE_K1_TCM` | `ON` | 启用 K1 TCM 紧耦合内存 |
| `ENABLE_K1_VPU` | `ON` | 启用 K1 VPU 硬件视频加速 |
| `ENABLE_K1_JPU` | `ON` | 启用 K1 JPU 硬件 JPEG 解码 |
| `SPACENGINE_DIR` | — | Spacengine SDK 路径 |
| `K1_MPP_DIR` | — | K1 MPP 媒体处理 SDK 路径 |
| `K1_JPU_DIR` | — | K1 JPU 硬件解码库路径 |

---

## 性能调优

### 1. 模型量化（🔴 最高优先级）

SpacemiT EP 的 IME 指令对 **INT8 量化模型** 有 3-5x 加速。量化步骤：

```python
# 在 K1 上使用 spacemit-ort Python 包
python3 -m onnxruntime.quantization.preprocess --input yolov8n.onnx --output yolov8n.q.onnx

# 或在 x86 上使用 ONNX Runtime 量化工具
# pip install onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("yolov8n.onnx", "yolov8n.q.onnx", weight_type=QuantType.QInt8)
```

### 2. 线程调优

```yaml
# configs/default.yaml
performance:
  num_threads: 4          # K1 推荐 4（8核中有4个给推理）
  inter_op_threads: 2
  cpu_affinity: false     # K1 核心绑定需内核支持
```

### 3. 运行环境

```bash
# 必需：IME 硬件访问
sudo chmod 777 /dev/tcm

# 必需：EP 动态库路径
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# 推荐：关闭 CPU 频率调节
sudo cpufreq-set -g performance
```

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计详解 |
| [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | 构建与部署指南 |
| [docs/IMPLEMENTATION_GAPS.md](docs/IMPLEMENTATION_GAPS.md) | 未实现模块与差距分析 |
| [docs/IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) | Muse Pi Pro 改造计划 |

### 外部参考

| 资源 | 链接 |
|------|------|
| SpacemiT ORT 下载 | https://archive.spacemit.com/spacemit-ai/onnxruntime/ |
| Bianbu C++ 推理示例 | https://bianbu.spacemit.com/en/brdk/Model_deployment/4.3_CPP_Inference_Example/ |
| SpacemiT EP 使用指南 | https://docs.bit-brick.com/docs/k1/ml/model-deploy |
| 社区论坛 | https://forum.spacemit.com/c/ai/18 |

---

## 开发路线图

### v1.1 ✅ 当前版本
- [x] SpacemiT EP 集成（RVV 1.0 + IME 加速）
- [x] C↔C++ 桥接 (`spacemit_ort_bridge.cpp`)
- [x] Arrow UART 双链路实时管道
- [x] 实时管道主循环 `process_realtime`
- [x] OpenMP 8 核多线程并行
- [x] RISC-V MLAS 优化数学库链接
- [x] 去除 x86 支持，Muse Pi Pro 唯一目标
- [x] 4 个 detector **统一**使用 `ort_common`（消除 ~100 行重复代码）

### v1.2 🔜 下一步
- [ ] **模型 INT8 量化**（YOLOv8n / YOLOv8-Pose / SCRFD / ArcFace）
- [ ] `onnxruntime_perf_test` 基准测试报告
- [ ] EP vs CPU-only 实测对比数据
- [ ] RVV 手写向量化（letterbox、NMS、矩阵乘法）
- [ ] KCP-Lite + Mahony 箭矢端集成验证
- [ ] MiDaS 深度估计 ONNX 推理 + 量化

### v2.0 🔜 中长期
- [ ] VINS-Mono 视觉-惯性里程计
- [ ] ATW 异步时间扭曲 AR 补偿
- [ ] IME INT8 全模型性能达标（目标：YOLOv8n <50ms）
- [ ] TinierHAR 动作意图预测
- [ ] ICP 点云配准（箭矢-射手空间对齐）