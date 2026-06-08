# LingQi TanTong Intelligent Detection Arrow System — High-Performance C Implementation

> **Version**: v2.0 — Muse Pi Pro (SpacemiT K1 X60) Fully Adapted
> **Target Platform**: SpacemiT K1 (X60) Muse Pi Pro — RISC-V 64 (Bianbu Linux)
> **Development Platform**: Linux (Cross-compilation) / K1 Native Build
> **Language Standard**: C11 + C++17 (Execution Provider Bridge)
> **Build System**: CMake ≥3.16 

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Literature Review](#2-literature-review)
3. [Methodology](#3-methodology)
4. [System Architecture](#4-system-architecture)
5. [Technical Implementation](#5-technical-implementation)
6. [Usage Guide](#6-usage-guide)
7. [Configuration Options](#7-configuration-options)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [Performance Metrics](#9-performance-metrics)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [References](#11-references)

---

## 1. Project Overview

### 1.1 Project Background and Research Motivation

In the field of modern tactical reconnaissance and intelligent perception, rapid detection of unknown spaces and real-time information backhaul constitute core technical challenges. Traditional detection methods (such as static surveillance cameras and handheld detectors) have inherent limitations in dynamic, unstructured environments, including large perception blind spots, high response latency, and insufficient spatial positioning accuracy. The **LingQi TanTong** project aims to build an intelligent detection arrow system based on dual-device heterogeneous computing, achieving real-time 3D perception and intelligent target analysis of complex indoor and outdoor environments through deep integration of visual perception, inertial navigation, deep learning inference, and augmented reality visualization.

The core innovation of this system lies in adopting the SpacemiT K1 X60 RISC-V chip as the primary computing platform, combined with the ESP32-P4 microcontroller as the front-end perception node, to build a low-cost, low-power, high-real-time embedded AI vision system. This design philosophy aligns with the macro trend of embedded artificial intelligence migrating toward edge computing [Shi et al., 2016], and validates the feasibility of domestic autonomous chips in intelligent perception by fully leveraging the extensibility of the RISC-V open instruction set architecture.

### 1.2 System Feature Overview

The LingQi TanTong system consists of two physical computing nodes:

| Computing Node | Hardware Platform | Core Responsibilities |
|----------|----------|----------|
| **Arrow End** | ESP32-P4 + OV5640 + GY-87 | Image capture (MIPI CSI), IMU attitude estimation (9-DOF Madgwick AHRS), data packaging and transmission (UART 3 Mbps) |
| **Shooter End** | SpacemiT K1 Muse Pi Pro | AI inference acceleration (RVV 1.0 + IME), multi-object tracking (ByteTrack + Hungarian + cascade matching), 3D spatial localization (pinhole camera model + IMU correction), AR visualization rendering |

The system provides the following core features:

1. **Person Detection (Dual-Model Cascade)**: YOLOv8-Pose (PRIMARY person detector with 17 COCO keypoints) + YOLO11n (SECONDARY person detector), hardware-accelerated via ONNX Runtime + SpacemiT Execution Provider
2. **Pose Estimation**: YOLOv8-Pose 17-keypoint COCO human skeleton estimation with OKS-based NMS
3. **Face Recognition**: YOLOv5-Face lightweight face detection cascaded with ArcFace deep feature extraction (128-dimensional embedding vector)
4. **Action Recognition**: ST-GCN (Spatial-Temporal Graph Convolutional Network) for skeleton-based action recognition from pose keypoints
5. **Multi-Object Tracking**: ByteTrack-inspired cascade matching + Hungarian algorithm (Kuhn-Munkres) + 7-state Kalman filtering + EMA smoothing (α=0.30)
6. **3D Spatial Localization**: Pinhole camera model-driven monocular depth estimation, fused with IMU pitch angle correction and world coordinate system initialization
7. **AR Visualization**: Bounding box rendering, skeleton line overlay, trajectory top-down view rendering
8. **Real-time Data Link**: Arrow custom protocol implementing ESP32-P4 → K1 3 Mbps dual UART real-time frame transmission
9. **K1 Hardware Acceleration**: RISC-V Vector 1.0 (256-bit VLEN) vectorization + IME matrix extension instructions (2.0 TOPS)
10. **Result Management**: Session-level tracking, JSON/CSV multi-format analysis report auto-export

### 1.3 Technical Specifications

| Metric | Target Value | Current Status |
|------|--------|----------|
| Object Detection FPS (K1 EP INT8) | ≥25 | Estimated achievable, pending real-world testing |
| Object Detection mAP50 (YOLO11n) | ≥50% | Model integrated |
| Face Detection AP (YOLOv5-Face) | ≥90% | Model integrated |
| Action Recognition Accuracy (ST-GCN) | ≥70% | Model integrated |
| Tracking ID Switch Rate | <5% | Implemented |
| Spatial Localization Error (<10m) | <20% | Implemented |
| End-to-end Latency (Arrow→Display) | <200ms | Implemented |
| System Memory Usage | <800 MB | Constraint met |

### 1.4 Project Significance and Contributions

The academic and technical contributions of this project can be summarized in the following four aspects:

1. **Heterogeneous Embedded AI System Paradigm**: Proposed and validated a dual-device heterogeneous computing architecture of "low-power front-end perception + RISC-V back-end inference," providing a reusable reference paradigm for embedded AI system design.

2. **RISC-V AI Acceleration Practice**: Systematically explored the application of SpacemiT K1 chip's RVV 1.0 + IME instruction set in computer vision inference acceleration, accumulating valuable practical experience for this chip ecosystem.

3. **Multi-sensor Fusion Localization**: Implemented a spatial localization scheme fusing two sources — vision (monocular depth estimation) and inertial (IMU attitude correction) — effectively improving the robustness of monocular visual depth estimation.

4. **Full C Language Embedded Implementation**: The entire system is written in C11 standard with a C++17 bridge for SpacemiT EP, without relying on any heavyweight external frameworks, and can be ported to various resource-constrained platforms.

---

## 2. Literature Review

### 2.1 Evolution of Object Detection Algorithms

Object detection, as one of the core tasks in computer vision, has undergone a profound transformation from traditional handcrafted feature methods to deep learning end-to-end paradigms. Early Viola-Jones detectors [Viola & Jones, 2001] and HOG + SVM [Dalal & Triggs, 2005] relied on sliding windows and manually designed features, with limited accuracy and generalization capability. The R-CNN series [Girshick et al., 2014; Girshick, 2015; Ren et al., 2015] pioneered the region-proposal-based two-stage detection paradigm, known for high accuracy but insufficient real-time performance.

The emergence of single-stage detectors — YOLO [Redmon et al., 2016], SSD [Liu et al., 2016] — transformed the detection problem into a regression problem, achieving end-to-end real-time inference. YOLOv8 [Jocher et al., 2023], as the latest iteration of the Ultralytics YOLO series, introduced an anchor-free detection head, C2f cross-stage partial connection module, and Decoupled Head architecture, achieving 53.6% mAP50 on the COCO dataset while the model size is only 3.2M parameters, making it highly suitable for embedded deployment.

This project selected YOLOv8n (nano version) as the pedestrian detection backbone model, primarily based on the following considerations: (1) Only 3.2M parameters, allowing weight preloading in K1's TCM tightly-coupled memory (512KB); (2) Compatible with SpacemiT Execution Provider after ONNX format export; (3) Output format is standard (cx, cy, w, h) tensors, convenient for interfacing with the ByteTrack tracker.

### 2.2 Face Detection and Recognition

For face detection, SCRFD [Guo et al., 2021] proposed Sample Redistribution and Computation Redistribution strategies, achieving competitive detection accuracy on the WIDER FACE dataset with an extremely small model of 0.8M parameters (SCRFD-0.5G). This project adopts the YOLOv5-Face version (~0.8M parameters), balancing accuracy and inference efficiency.

For face recognition, ArcFace [Deng et al., 2019] significantly improved the inter-class separability and intra-class compactness of face feature embeddings by adding an Additive Angular Margin Loss in angular space. This project uses an ArcFace model based on the MobileFaceNet-cuted architecture, outputting 128-dimensional feature vectors for identity matching via cosine similarity.

### 2.3 Multi-Object Tracking Methods

The mainstream paradigms of Multiple Object Tracking (MOT) are divided into Tracking-by-Detection and Joint Detection and Tracking. SORT [Bewley et al., 2016] constructed a simple and efficient tracking framework using Kalman filter prediction and Hungarian algorithm association. DeepSORT [Wojke et al., 2017] introduced appearance features to improve tracking continuity in occlusion scenarios. ByteTrack [Zhang et al., 2022] innovatively proposed a strategy of using low-confidence detection boxes for secondary matching, achieving SOTA performance on MOT17/MOT20 datasets.

This project draws on ByteTrack's three-stage data association strategy, using a 7-state Kalman filter (state vector: [cx, cy, area, aspect_ratio, vx, vy, vs]) for motion prediction, and EMA (α=0.25) smoothing of spatial coordinates to suppress high-frequency jitter.

### 2.4 Inertial Navigation and Attitude Estimation

The Mahony complementary filter [Mahony et al., 2008] is a classic explicit complementary filtering algorithm that fuses accelerometer (low-frequency gravity reference) and gyroscope (high-frequency angular velocity) data through a PI controller, with extremely low computational cost (~200 floating-point operations per update), suitable for real-time operation on microcontrollers. The Madgwick filter [Madgwick et al., 2011] uses gradient descent optimization instead of PI control, simultaneously fusing accelerometer, gyroscope, and magnetometer (9-DOF) data to provide drift-free yaw angle estimation. This project uses the Madgwick algorithm for 9-DOF fusion on the Arrow End (ESP32-P4), and the Mahony algorithm as an alternative verification scheme on the Shooter End (K1).

### 2.5 RISC-V Vector Extension and AI Acceleration

The RISC-V Vector Extension (RVV) [RISC-V International, 2021] is a variable-length SIMD instruction set architecture supporting flexible vector lengths from VLEN=32 to VLEN=65536. The SpacemiT X60 core implements the RVV 1.0 standard with vector register width VLEN=256 bits; a single vector instruction can simultaneously process 8 float32 or 16 int16 operands. Additionally, the X60 core's proprietary IME (Intelligent Matrix Extension) instruction set contains 16 custom AI acceleration instructions (matrix multiplication and sliding window operations), achieving 2.0 TOPS of in-core AI compute when working in concert with RVV 1.0 [SpacemiT, 2024].

### 2.6 Embedded Deep Inference Framework

ONNX Runtime [Microsoft, 2023] is a cross-platform deep learning inference acceleration engine supporting multiple hardware Execution Providers (EP). SpacemiT provides a customized EP for the RISC-V platform (libspacemit_ep.so), registered through the `SessionOptionsSpaceMITEnvInit()` API, which automatically maps operators such as convolutions and matrix multiplications in the computation graph to RVV 1.0 + IME instruction execution [SpacemiT, 2024].

---

## 3. Methodology

### 3.1 Research Paradigm and Design Philosophy

The research and development of this project follows a four-stage spiral iterative methodology of "prototype validation → platform migration → hardware acceleration → system optimization":

```
Python Prototype (Algorithm Validation)
    ↓
C Language Porting (Embedded Adaptation)
    ↓
RISC-V Cross-compilation (Platform Migration)
    ↓
SpacemiT EP Integration (Hardware Acceleration)
    ↓
RVV/IME Hand-written Optimization (Ultimate Performance)
```

At the design philosophy level, the system adheres to the following core principles:

1. **Minimal Dependency Principle**: Core algorithm libraries have zero external dependencies (except the standard C library). ONNX Runtime is optionally integrated through `#ifdef HAS_ONNX_RUNTIME` conditional compilation, ensuring buildability and runnability in any POSIX-compatible environment.

2. **ONNX Runtime Required**: All AI inference modules require ONNX Runtime with either SpacemiT EP (RVV 1.0 + IME hardware acceleration) or CPU EP (real ONNX inference, no hardware acceleration). There is no heuristic fallback — the system depends on ONNX Runtime for inference.

3. **Configuration-Driven Architecture**: All tunable parameters are managed uniformly through YAML-style configuration files, supporting runtime dynamic loading. Detection thresholds, tracking parameters, and visualization options can be adjusted without recompilation.

4. **Modular Composition Pattern**: The system adopts a strict `create → process → destroy` lifecycle management pattern. Each module encapsulates internal state through structs, eliminates global variables, and ensures thread safety and memory safety.

### 3.2 Development Environment and Toolchain

| Tool/Component | Version/Specification | Purpose |
|-----------|----------|------|
| C Compiler | GCC 13.2+ (riscv64-linux-gnu-gcc) | K1 native/cross-compilation |
| C++ Compiler | G++ 13.2+ (riscv64-linux-gnu-g++) | SpacemiT EP C++ bridge layer |
| CMake | ≥3.16 | Cross-platform build management |
| Python 3 | ≥3.8 | Development/debugging utilities |
| ESP-IDF | v5.1+ | ESP32-P4 firmware development framework |
| ONNX Runtime | SpacemiT ORT 2.0.2 | RISC-V AI inference acceleration |
| SpacemiT EP | libspacemit_ep.so | RVV 1.0 + IME hardware acceleration |
| libyaml | — | YAML config parsing (built-in minimal parser) |

### 3.3 Development Process and Quality Control

The project adopts a quality assurance system driven by cognitive reviews. During the v1.0 → v1.1 iteration, systematic code review (covering all 28 source files and 28 header files) identified and fixed 41 issues, including 7 CRITICAL-level defects and 8 HIGH-level issues. Key fixes include:

- **YOLOv8 ONNX Output Format Correction**: (x1,y1,x2,y2) → (cx,cy,w,h) format correction in pose estimator
- **K1 No Independent NPU Cognitive Correction**: Comprehensively cleaned up all improper "NPU" references in the project, replacing them with "RISC-V AI instruction acceleration"
- **Global Semantic Renaming**: `use_gpu` → `use_onnx` (17 files, 31 occurrences), accurately reflecting inference path semantics
- **Realtime JPEG Decode Completion**: Changed from gray fill placeholder to `soft_jpeg_decode_to_rgb()` real decoding
- **Array Bounds Protection**: Added `MAX_DETECTIONS_PER_FRAME` boundary checks when writing inference results

For detailed change records, see [docs/CODE_CHANGE_LOG.md](docs/CODE_CHANGE_LOG.md).

---

## 4. System Architecture

### 4.1 Overall Architecture Overview

The LingQi TanTong system adopts a **dual-device heterogeneous computing architecture**, consisting of the Arrow End (ESP32-P4 front-end perception node) and the Shooter End (K1 Muse Pi Pro primary computing node) interconnected via UART serial bus, forming a complete perception-transmission-computation-display data pathway.

```
┌────────────────────────────── Arrow End ──────────────────────────┐
│                        ESP32-P4 (RISC-V 400MHz ×2)                          │
│                                                                             │
│  ┌──────────┐   MIPI CSI     ┌──────────────────┐                          │
│  │  OV5640  │───────────────▶│  ISP + JPEG Enc  │                          │
│  │  Camera  │                │  (HW Pipeline)   │                          │
│  └──────────┘                └────────┬─────────┘                          │
│                                       │ JPEG frames                         │
│  ┌──────────┐   I²C(400kHz)  ┌───────▼─────────┐                          │
│  │  GY-87   │───────────────▶│  Madgwick AHRS  │                          │
│  │  IMU     │  MPU6050+      │  9-DOF Fusion   │                          │
│  │          │  QMC5883L+     │  → Quaternion   │                          │
│  │          │  BMP180        │  → Altitude     │                          │
│  └──────────┘                └────────┬─────────┘                          │
│                                       │                                     │
│                              ┌────────▼─────────┐                          │
│                              │  Protocol Packer │                          │
│                              │  Arrow Protocol  │                          │
│                              └────────┬─────────┘                          │
│                                       │                                     │
│                         UART (3 Mbps, 8N1, DMA)                            │
└───────────────────────────────────────┼─────────────────────────────────────┘
                                        │
           ═══════════════════════════════╪══════════════════════════════════
                                        │
┌────────────────────────────── Shooter End ────────────────────────┐
│                Muse Pi Pro — SpacemiT K1 X60 (8-core, RVV 1.0 + IME)          │
│                                        │                                    │
│  ┌─────────────────────────────────────▼──────────────────────────────────┐ │
│  │                     SystemController (Main Scheduler)                    │ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────────┐  │ │
│  │  │  Data Layer  │  │ Business Logic │  │   Presentation Layer     │  │ │
│  │  │              │  │                │  │                          │  │ │
│  │  │VideoProcessor│  │InferencePipeline│  │  Visualizer (Box/Skel/Trj)│ │ │
│  │  │ArrowReceiver │  │ ObjectTracker  │  │  ARRenderer (Motion Comp)│ │ │
│  │  │ IMUHandler   │  │ SpatialEngine  │  │  VideoWriter (AVI Export)│ │ │
│  │  │ ModelStore   │  │ ORT+SpacemiTEP │  │                          │  │ │
│  │  └──────────────┘  └────────────────┘  └──────────────────────────┘  │ │
│  │                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Support: Logger │ ConfigManager │ ResultManager │ Mahony │ KCP │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  HDMI 1080p60 / Framebuffer / SDL Display Output                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Shooter End Six-Layer Architecture

The Shooter End (K1) software system adopts a strict layered architecture design, divided into six layers from top to bottom:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Application Layer                              │
│  ┌─────────────────────┐                                             │
│  │      main.c         │  CLI parsing → Module creation → Main loop → Cleanup      │
│  └─────────┬───────────┘                                             │
├────────────┼─────────────────────────────────────────────────────────┤
│                         Controller Layer                              │
│  ┌─────────▼───────────┐                                             │
│  │ system_controller.c │  10-step frame processing pipeline (offline/realtime dual mode)         │
│  │                     │  K1 Pipeline: Capture → Infer → Post three threads  │
│  └─────────────────────┘                                             │
├──────────────────────────────────────────────────────────────────────┤
│                      Business Logic Layer                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐      │
│  │InferencePipeline │ │ TrackingManager  │ │  SpatialEngine   │      │
│  │                  │ │                  │ │                  │      │
│  │ YOLOv8-Pose(PRIMARY) │ │ Cascade+Hungarian │ │ Pinhole camera model    │      │
│  │ YOLO11n(SECONDARY)  │ │ + 7-state Kalman │ │ + IMU correction      │      │
│  │ YOLOv5-Face ─► ArcFace │ │ + EMA smoothing  │ │ + Trajectory management      │      │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘      │
├──────────────────────────────────────────────────────────────────────┤
│                       Data Processing Layer                           │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │VideoProcessor │ │  IMUHandler   │ │  ModelStore   │               │
│  │               │ │               │ │               │               │
│  │ MP4/Camera    │ │ Sliding window smoothing  │ │ Model file management  │               │
│  │ Arrow input     │ │ External pose injection  │ │ Path resolution      │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                       Presentation Layer                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │  Visualizer   │ │  ARRenderer   │ │  VideoWriter  │               │
│  │               │ │               │ │               │               │
│  │ Bounding box rendering    │ │ Motion compensation      │ │ AVI encoding output  │               │
│  │ Skeleton/Trajectory     │ │ Marker overlay      │ │               │               │
│  │ 5×7 bitmap    │ │ (CPU currently)     │ │               │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                         Support Layer                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │ConfigManager  │ │    Logger     │ │ResultManager  │               │
│  │               │ │               │ │               │               │
│  │ YAML key-value parsing  │ │ Leveled logging      │ │ Session tracking   │               │
│  │ Programmable defaults  │ │ Thread-safe      │ │ JSON/CSV reports  │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Core Data Flow

#### 4.3.1 Frame Data Flow (Single-Frame Processing Pipeline)

```
FrameData (uint8*, width, height, channels, timestamp)
    │
    ├──▶ YOLOv8-Pose ──▶ Detection[] (bbox, conf, class_id, class_name)
    │                        │
    │                        ├── Filtering (confidence/area/aspect ratio/NMS)
    │                        │
    │                        └──▶ YOLOv8-Pose (per-target cropping) ──▶ PoseEstimation[]
    │
    ├──▶ YOLOv5-Face  ──▶ FaceIdentity[] ──▶ ArcFace ──▶ Backfill feature_vector[512]
    │
    └──▶ (depth_map) ──▶ SpatialEngine (optional external depth map)
```

#### 4.3.2 Target Association Graph (Data Aggregation and Fusion)

```
                    Frame
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        Detection   Pose    FaceIdentity
            │         │         │
            └────┬────┘    IoU association to
                 │         PoseEstimation
            IoU matching          │
                 │    IoU association to FaceIdentity
                 ▼              │
           TrackedObject ◄──────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
  spatial_pos  pose   face_identity
        │
        ▼
  trajectory[] (historical SpatialPosition[])
```

### 4.4 Inter-Module Data Transfer Contract

| Producer | Data Product | Consumer | Transfer Method |
|--------|----------|--------|----------|
| VideoProcessor | FrameData* | SystemController | Function return value (heap allocated) |
| ArrowReceiver | ArrowSourceFrame | SystemController | Ring buffer + get_latest |
| InferencePipeline | InferenceResult (value type) | SystemController | Stack return value |
| SpatialEngine | SpatialResult | SystemController | Value return |
| TrackingManager | TrackingResult | SystemController | Stack return value |
| IMUHandler | IMUExternalPose | SpatialEngine | Function call parameter passing |
| SystemController | vis_buffer (uint8_t*) | VideoWriter | Function call parameter passing |

### 4.5 Memory Management Strategy

| Strategy | Description | Implementation |
|------|------|----------|
| **Stack First** | Small structs (BoundingBox, Detection, SpatialPosition) passed by value | C value semantics |
| **Fixed Upper Bounds** | All array capacities hard-constrained via macros | `MAX_DETECTIONS=100`, `MAX_TRACKED=100` |
| **Explicit Lifecycle** | Each module follows `create → process → destroy` | No implicit memory leak paths |
| **Zero Global Variables** | All runtime state encapsulated in module structs | Thread safety guarantee |
| **Ring Buffer** | Arrow UART data received through 64KB ring buffer | Zero-copy, no dynamic allocation |
| **Frame Data Heap Allocation** | FrameData malloc'd in video_processor, freed in controller | Clear ownership transfer |

---

## 5. Technical Implementation

### 5.1 System Initialization and Dependency Management

The system startup process follows a strict "configuration first, dependency injection, bottom-up" initialization order. The entry file [main.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/main.c) is responsible for command-line parsing and environment initialization, then delegates to [system_controller.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/system_controller.c) to complete all sub-module creation and connection.

#### 5.1.1 Initialization Sequence (`system_controller_create`)

```c
SystemController* system_controller_create(const char* config_path) {
    SystemController* sc = calloc(1, sizeof(SystemController));

    // Step 1: Configuration manager (initialized first, subsequent modules depend on its parameters)
    sc->config = config_manager_create(config_path);

    // Step 2: Model file manager (validate model directory integrity)
    sc->model_store = model_store_create("models",
        config_get_bool(sc->config, "system.use_onnx", false));

    // Step 3: IMU handler (window_size=10, min_interval=0.01s, max_gap=0.1s)
    sc->imu_handler = imu_handler_create(10, 0.01f, 0.1f);

    // Step 4: Inference pipeline (cascading 4 AI models)
    sc->inference_pipeline = inference_pipeline_create(
        config_get_bool(sc->config, "system.use_onnx", false));
    inference_pipeline_load_models(sc->inference_pipeline, "models");

    // Step 5: Tracking manager
    sc->tracking_manager = object_tracker_create(
        config_get_int(sc->config, "tracking.max_lost", 30),
        config_get_float(sc->config, "tracking.min_iou", 0.3f),
        config_get_float(sc->config, "tracking.max_distance", 5.0f),
        config_get_int(sc->config, "tracking.max_track_history", 300));

    // Step 6: Spatial engine
    float focal = config_get_float(sc->config, "spatial.fx", 500.0f);
    float avg_height = config_get_float(sc->config, "spatial.avg_human_height", 1.7f);
    sc->spatial_engine = spatial_engine_create(NULL, NULL, focal, avg_height);

    // Step 7: Visualizer / AR renderer / Result manager
    sc->visualizer = visualizer_create(/* ... */);
    sc->ar_renderer = ar_renderer_create(render_w, render_h, true);
    sc->result_manager = result_manager_create("output");
    return sc;
}
```

**Initialization Dependency Topology**:

```
ConfigManager  ───────────── Initialized first
    │
    ├── ModelStore ───────── Dependency: use_onnx flag in config
    ├── InferencePipeline ── Dependency: model directory, use_onnx
    ├── TrackingManager ──── Dependency: max_lost, min_iou, etc. in config
    ├── SpatialEngine ────── Dependency: focal_length, avg_height
    ├── IMUHandler ───────── Independent initialization (uses config defaults)
    ├── Visualizer ───────── Dependency: visualization config items
    ├── ARRenderer ───────── Independent initialization
    └── ResultManager ────── Dependency: output directory config
```

#### 5.1.2 Video Processor Lazy Creation

`VideoProcessor` is not created along with `system_controller_create`, but is dynamically created when `system_controller_process_video` is called, based on the `video_path` parameter:

```c
VideoProcessor* processor = video_processor_create(video_path, 0, 0, false);
if (!processor || video_processor_open(processor, video_path) != VP_OK) {
    // Error handling: log to ResultManager
}
```

Benefits of this lazy allocation strategy: (1) The entire system_controller can complete initialization without a video source; (2) Realtime mode and offline mode share the same system_controller instance; (3) Easy to switch video input sources at runtime.

### 5.2 System Controller Core Scheduling

[system_controller.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/system_controller.c) is the core scheduling module of the entire system, responsible for orchestrating the complete processing pipeline for each frame.

#### 5.2.1 Offline Mode Processing Loop

Offline mode reads video files frame by frame and executes the complete AI inference → tracking → spatial localization → visualization pipeline:

```
for each frame in video:
    1. video_processor_read_frame_raw()     → FrameData*
    2. inference_pipeline_process_frame()    → InferenceResult (containing detections/poses/faces)
    3. if first frame and person detected: spatial_engine_init_coord_system()
    4. for each detection: spatial_engine_calculate_position() → positions[]
    5. object_tracker_update(detections, positions) → TrackingResult
    6. associate_poses_with_objects()        → IoU match poses to targets
    7. associate_faces_with_objects()        → IoU match faces to targets
    8. for each tracked: spatial_engine_update_trajectory()
    9. for each tracked: spatial_engine_get_velocity()
    10. imu_handler → spatial_engine_set_camera_pose()
    11. visualizer_render_detection_view()    → vis_buffer
    12. video_writer_write_frame()           → Output AVI
    13. FPS statistics + log output
    14. frame_data_destroy()                 → Release frame memory
```

#### 5.2.2 Realtime Mode Arrow Pipeline

Realtime mode shares the same inference-tracking-localization core pipeline as offline mode, with the difference being that the data source comes from the Arrow UART receiver rather than a video file:

```
while (running):
    1. arrow_receiver_update()               → Update ring buffer
    2. arrow_receiver_get_latest_frame()     → ArrowSourceFrame (JPEG+IMU)
    3. if has_frame:
         soft_jpeg_decode_to_rgb()           → JPEG → RGB
         if has_pose: imu_handler_set_external_pose() → IMU quaternion injection
    4. else: continue (wait when no frame)
    5-14. Same as offline mode steps 2-12
```

#### 5.2.3 K1 Dual-Cluster Pipeline Parallelism

The K1 X60 processor has 8 cores divided into two clusters: Cluster0 (cores 0-3, 512KB TCM) dedicated to AI inference and image post-processing, and Cluster1 (cores 4-7) dedicated to I/O acquisition. The system enables a three-thread pipeline through the `HAS_K1_PIPELINE` conditional compilation macro:

```
Thread 1: Capture  (CPU4, Cluster1)
  └── arrow_receiver → JPEG decode → RingBuffer[slot].rgb_data

Thread 2: Inference (CPU1, Cluster0)
  └── RingBuffer[slot].rgb → inference_pipeline → RingBuffer[slot].inference

Thread 3: PostProcess (CPU0, Cluster0)
  └── RingBuffer[slot].inference → spatial → tracking → visualizer → release
```

**Ring Buffer Design**:

```c
#define K1_RING_SIZE 4  // 4 slots

typedef struct {
    uint8_t* rgb_data;           // Image data buffer
    InferenceResult inference;    // Inference results
    TrackingResult tracking;      // Tracking results
    bool has_frame;              // Status flag
    bool has_inference;
    bool has_tracking;
    pthread_mutex_t mutex;       // Slot mutex
    pthread_cond_t avail_cond;   // Condition variable
} K1PipelineSlot;
```

Slot acquisition and release adopt a **producer-consumer model**, implementing busy-wait-free synchronization between threads via `pthread_mutex_lock` + `pthread_cond_wait`. The `active_slots` counter limits the maximum number of concurrent slots, preventing memory bloat when production speed far exceeds consumption speed.

### 5.3 Inference Pipeline: Cascaded AI Model Execution

[inference_pipeline.c](src/inference_pipeline.c) implements cascaded multi-model AI inference, executing sequentially in the order YOLOv8-Pose (PRIMARY) → YOLO11n (SECONDARY) → YOLOv5-Face → ArcFace → ST-GCN.

#### 5.3.1 Model Loading

```c
int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config) {
    // 1. YOLOv8-Pose person detection + keypoints (PRIMARY, REQUIRED)
    snprintf(path_buf, sizeof(path_buf), "%s/Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx", model_dir);
    pipeline->pose_estimator = yolov8_pose_estimator_create(path_buf, 640, 640, 0.10f, 0.40f);
    // If loading fails, entire pipeline unavailable → return -1

    // 2. YOLO11n person detection (SECONDARY, optional — pose-only mode if missing)
    pipeline->detector = yolov8_detector_create(/* ... */);

    // 3. YOLOv5-Face face detection (optional)
    pipeline->face_detector = yolov5_face_detector_create(/* ... */);

    // 4. ArcFace face recognition (optional)
    pipeline->face_recognizer = arcface_recognizer_create(/* ... */);

    // 5. ST-GCN action recognition (optional)
    pipeline->action_recognizer = stgcn_action_recognizer_create(/* ... */);

    return loaded_count;
}
```

**Design Decision**: Only the YOLOv8-Pose model is required (returns -1 indicating system unavailability); the remaining four models are optional (load failure only produces a warning log). This ensures the system can still provide core person detection + tracking functionality when some model files are missing.

#### 5.3.2 Cascade State Machine

The inference pipeline adaptively switches between full-resolution search and reduced-frequency tracking to save inference time:

```
PIPELINE_CASCADE_SEARCHING: No confirmed tracks.
  → Run YOLOv8-Pose (PRIMARY) + YOLO11n (SECONDARY) at 640×640.
  → Apply keypoint anatomical + partial-body validation.
  → Maximum recall, higher latency.

PIPELINE_CASCADE_TRACKING: ≥1 confirmed track.
  → Run YOLOv8-Pose every frame (PRIMARY).
  → Run YOLO11n every ~3 frames (reduced frequency).
  → New people entering are still detected within ~1 second.

PIPELINE_CASCADE_VALIDATING: Periodic full-res check (1 frame).
  → Same as SEARCHING: both models at 640×640.
  → Triggered by: validation_interval timer OR multi-person detection trigger.
```

#### 5.3.3 Detection Result Filtering Pipeline

From model raw output to valid detection results, a multi-level filter is applied:

```
Raw proposals (up to 6000)
    │
    ▼ Level 1: DFL peakiness confidence filtering
    │  Discard proposals with dfl_conf < threshold
    │  (INT8 cls head is broken; DFL peakiness is the real signal)
    ▼
    │
    ▼ Level 2: Geometric plausibility filtering
    │  Discard boxes with area_ratio < 0.005 or > 0.40
    │  Discard boxes with height_ratio < 0.04 or > 0.75
    │  Discard boxes with aspect_ratio < 0.30 or > 1.80 (non-human proportions)
    │  Discard boxes completely outside the frame
    │  Check body aspect ratio (height/width must be human-like)
    ▼
    │
    ▼ Level 3: Keypoint-based anatomical validation (three-tier)
    │  Tier 1: Full-body (paired shoulders/hips/knees required)
    │  Tier 2: Upper-body fallback (keypoints 0-10, ≥4 valid)
    │  Tier 3: Side-body fallback (one-side chain, ≥3 valid)
    │  Partial-body detections marked but NOT rejected
    ▼
    │
    ▼ Level 4: Dual-model consensus filter
    │  When both models run, each detection must either:
    │    (a) pass keypoint anatomical validation, OR
    │    (b) be confirmed by BOTH models (IoU ≥ 0.35)
    │  Solo detections without consensus are discarded
    ▼
    │
    ▼ Level 5: Top-K + NMS
    │  Sort by confidence descending
    │  Top-K (50) globally across all stride groups
    │  NMS per-stride (IoU 0.30) + final NMS (IoU 0.35)
    ▼
Final detection results (max 20)
```

#### 5.3.4 YOLOv8-Pose Full-Frame Estimation

YOLOv8-Pose performs inference on the entire frame (not per-target cropping), outputting up to 20 pose estimations per frame with COCO-17 keypoints decoded from xquant-split DFL regression + keypoint branches.

#### 5.3.4 Face Detection and Recognition Cascade

YOLOv5-Face face detection → ArcFace feature extraction cascade flow:

```c
static int detect_faces(YOLOv5FaceDetector* face_detector, ArcFaceRecognizer* face_recognizer,
                        const uint8_t* frame, int width, int height,
                        FaceIdentity* out_faces, int max_faces) {
    // Step 1: Full-frame face detection (YOLOv5-Face)
    int num_detected = yolov5_face_detector_detect(face_detector, frame, width, height,
                                                    detected_faces, 20);

    // Step 2: Per-face cropping + ArcFace feature extraction
    for each detected_face:
        face_crop = yolov5_face_detector_crop_face(face_detector, frame, ..., 112, 112)
        identity = arcface_recognizer_recognize(face_recognizer, face_crop, 112, 112)
        // Backfill bbox, confidence, keypoints
        out_faces[num_faces++] = identity
}
```

### 5.4 ByteTrack Multi-Object Tracking: Cascade Matching + Hungarian + Kalman Filtering

[tracking_manager.c](src/tracking_manager.c) implements a multi-object tracker following the ByteTrack paradigm with cascade matching, Hungarian algorithm, appearance features, and occlusion handling.

#### 5.4.1 7-State Kalman Filter

**State Vector**: $\mathbf{x} = [cx, cy, area, aspect\_ratio, \dot{cx}, \dot{cy}, \dot{area}]$

Where $cx, cy$ are the bbox center coordinates, $area$ is the bbox area, $aspect\_ratio$ is width/height, and $\dot{cx}, \dot{cy}, \dot{area}$ are their velocities.

**State Transition Matrix** (constant velocity motion model):
$\mathbf{F} = \begin{bmatrix} 1 & 0 & 0 & 0 & dt & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & dt & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & dt \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$

**Initialization**:
```c
static void kalman_init(KalmanBoxTracker* kf, const BoundingBox* bbox, int track_id, float dt) {
    // State initialization
    kf->state[0] = bbox_center_x(bbox);      // cx
    kf->state[1] = bbox_center_y(bbox);      // cy
    kf->state[2] = bbox_area(bbox);           // area
    kf->state[3] = bbox_width / bbox_height;  // aspect ratio
    kf->state[4..6] = 0.0f;                   // velocities initialized to zero

    // Process noise: q_pos = 0.02 (position), q_vel = 0.04 (velocity)
    // Measurement noise: diagonal 0.15
}
```

**Prediction Step** ($\hat{\mathbf{x}}_{k+1} = \mathbf{F} \mathbf{x}_k$, $\mathbf{P}_{k+1} = \mathbf{F} \mathbf{P}_k \mathbf{F}^T + \mathbf{Q}$):
```c
static void kalman_predict(KalmanBoxTracker* kf) {
    // State prediction: x_new = F × x
    // Covariance prediction: P_new = F × P × F^T + Q
}
```

**Update Step** (standard Kalman update: $\mathbf{K} = \mathbf{P} \mathbf{H}^T (\mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R})^{-1}$, $\mathbf{x} = \mathbf{x} + \mathbf{K} (\mathbf{z} - \mathbf{H}\mathbf{x})$):
```c
static void kalman_update(KalmanBoxTracker* kf, const BoundingBox* bbox) {
    // Measurement vector z = [cx, cy, area, aspect_ratio]
    // Innovation y = z - H × x
    // Innovation covariance S = H × P × H^T + R
    // Kalman gain K = P × H^T × S^(-1)  (with diagonal fallback)
    // State update x = x + K × y
    // Covariance update P = (I - K×H) × P
}
```

#### 5.4.2 Cascade Matching with Hungarian Algorithm

The tracker uses cascade matching (grouping tracks by time-since-update, matching freshest first) with the Kuhn-Munkres Hungarian algorithm for optimal global assignment, replacing greedy IoU matching. The cost matrix combines IoU (65% weight) and appearance feature cosine distance (35% weight).

#### 5.4.3 EMA Spatial Coordinate Smoothing

For each successfully matched tracked object, Exponential Moving Average (α=0.30) smoothing is applied to spatial coordinates to reduce inter-frame jitter:
```c
smoothed.x = 0.30 * position->x + 0.70 * last->x;
smoothed.y = 0.30 * position->y + 0.70 * last->y;
smoothed.z = 0.30 * position->z + 0.70 * last->z;
```

### 5.5 3D Spatial Localization: Pinhole Camera Model

[spatial_engine.c](src/spatial_engine.c) implements monocular visual depth estimation and 3D coordinate computation based on the pinhole camera model.

#### 5.5.1 Mathematical Model

**Depth Estimation (based on average human height prior)**:

$$Z = \frac{f_y \cdot H_{avg}}{h_{bbox}}$$

Where $f_y$ is the camera focal length (pixels, default 960), $H_{avg} = 1.70\text{m}$ is the assumed average human height, and $h_{bbox}$ is the detection box height (pixels). Depth is clamped to [0.5m, 50m].

**3D Coordinate Back-projection (pixels → camera coordinate system)**:

$$X_{cam} = \frac{(u - c_x) \cdot Z}{f_x}$$
$$Y_{cam} = \frac{(v - c_y) \cdot Z}{f_y}$$
$$Z_{cam} = Z$$

Where $(u, v)$ is the detection box center pixel coordinates, and $(c_x, c_y)$ is the camera optical center (default: 960, 540 for 1920×1080).

**IMU Pitch Angle Depth Correction**:

$$Z_{corrected} = Z \cdot \cos(\theta_{pitch})$$

When the camera tilts downward (pitch angle $\theta_{pitch} > 0$), the true depth is greater than the vertical direction estimate.

#### 5.5.3 Coordinate System Initialization

The first valid detection target in the first frame is selected as the world coordinate origin:

```c
void spatial_engine_initialize_coordinate_system(SpatialLocalizationEngine* engine,
                                                  int frame_height, int frame_width,
                                                  const Detection* first_detection) {
    float bbox_h = bbox_height(&first_detection->bbox);
    float focal = engine->camera_matrix[0][0];
    float ref_depth = (engine->avg_person_height * focal) / bbox_h;

    // Calculate reference target's 3D coordinates
    float ref_x = (bbox_center_x - first_frame_center_x) * ref_depth / focal;
    float ref_y = (bbox_center_y - first_frame_center_y) * ref_depth / fy;
    float ref_z = ref_depth;

    engine->world_origin = {ref_x, ref_y, ref_z};
    engine->world_initialized = true;
}
```

**Design Rationale**: Setting the first target in the first frame as the world origin means all subsequent targets' coordinates are relative to this reference point. This **relative positioning** paradigm eliminates the need for absolute GPS coordinates, making it suitable for indoor GPS-denied scenarios.

#### 5.5.4 Trajectory Management and Velocity Estimation

Each tracked object maintains a trajectory buffer (up to 300 points). Velocity estimation is based on displacement over the last two frames:

```c
bool spatial_engine_get_velocity(SpatialLocalizationEngine* engine, int track_id,
                                  float dt, float out_velocity[3]) {
    // Need at least 2 trajectory points
    if (trajectory.count < 2) return false;

    const SpatialPosition* last = &trajectory[count - 1];
    const SpatialPosition* prev = &trajectory[count - 2];

    out_velocity[0] = (last->x - prev->x) / dt;
    out_velocity[1] = (last->y - prev->y) / dt;
    out_velocity[2] = (last->z - prev->z) / dt;

    // Velocity plausibility check (over 50m/s treated as error)
    float speed = sqrt(vx² + vy² + vz²);
    return speed <= 50.0f;
}
```

### 5.6 Mahony Complementary Filter: 9-DOF IMU Attitude Estimation

[mahony_filter.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/mahony_filter.c) implements the classic Mahony complementary filter, fusing accelerometer, gyroscope, and magnetometer (9-DOF) data to estimate device attitude.

#### 5.6.1 Algorithm Flow

```
Input: Gyroscope(gx,gy,gz), Accelerometer(ax,ay,az), Magnetometer(mx,my,mz), Time step dt
Output: Attitude quaternion [q0,q1,q2,q3], Euler angles [pitch, roll, yaw]

Step 1: Accelerometer normalization → Obtain gravity direction reference
Step 2: Magnetometer normalization → Obtain geomagnetic direction reference
Step 3: Calculate "theoretical gravity direction" and "theoretical geomagnetic direction" based on current quaternion
Step 4: Calculate cross-product error between measurements and theoretical values
    ex = (ay*vz - az*vy) + (my*wz - mz*wy)
    ey = (az*vx - ax*vz) + (mz*wx - mx*wz)
    ez = (ax*vy - ay*vx) + (mx*wy - my*wx)
Step 5: PI controller integral error elimination for gyroscope drift
    integral_fbx += ex * dt * 0.5
    gx_corrected = gx - kp*ex - ki*integral_fbx
    (gy, gz similarly)
Step 6: Quaternion update (first-order Runge-Kutta integration)
    q0 += 0.5*(-q1*gx - q2*gy - q3*gz)*dt
    q1 += 0.5*( q0*gx + q2*gz - q3*gy)*dt
    q2 += 0.5*( q0*gy - q1*gz + q3*gx)*dt
    q3 += 0.5*( q0*gz + q1*gy - q2*gx)*dt
Step 7: Quaternion normalization
Step 8: Quaternion → Euler angle conversion
    pitch = asin(-2*(q1*q3 - q0*q2))
    roll  = atan2(2*(q0*q1+q2*q3), q0²-q1²-q2²+q3²)
    yaw   = atan2(2*(q1*q2+q0*q3), q0²+q1²-q2²-q3²)
```

#### 5.6.2 Key Parameters

| Parameter | Default Value | Physical Meaning |
|------|--------|----------|
| `kp` | 0.5 | Proportional gain, controls accelerometer correction response speed |
| `ki` | 0.01 | Integral gain, eliminates long-term gyroscope drift |
| `sample_freq` | 200.0 Hz | IMU sampling frequency |
| `dt` | 0.005s | Time step (= 1/200Hz) |

**Parameter Selection Principles**:
- Larger `kp` leads to faster attitude convergence but greater sensitivity to accelerometer noise (high-frequency jitter)
- `ki` eliminates constant gyroscope drift; too large a value causes overshoot oscillation
- Default values (0.5, 0.01) are empirical values tested stable on the GY-87 module

### 5.7 Arrow Communication Protocol: Arrow End → Shooter End Real-time Data Link

#### 5.7.1 Arrow End Frame Format (ESP32-P4 → K1 UART)

The Arrow custom protocol defines the following frame format for ESP32-P4 → K1 UART communication:

```
┌──────────┬──────┬──────┬──────────┬────────┬──────────────┬──────┬──────────┐
│  Magic   │ Type │ Seq# │ Timestamp│ Length │   Payload    │ CRC16│  Magic   │
│ 2B(Head) │ 1B   │ 2B   │ 4B (ms)  │ 2B     │   N bytes    │ 2B   │ 2B(Tail) │
│ 0xA5 0x5A│      │      │          │        │              │      │ 0x5A 0xA5│
└──────────┴──────┴──────┴──────────┴────────┴──────────────┴──────┴──────────┘

Type definitions:
  0x01 — JPEG video frame (payload: JPEG compressed data)
  0x02 — IMU attitude quaternion + altitude (payload: qw,qx,qy,qz,altitude_m,temperature_c)
  0x03 — IMU raw data (payload: accel[3], gyro[3], mag[3])
  0x04 — Heartbeat/status frame (payload: battery%, FPS, error_code)
  0x05 — Control command response
```

**Dual Magic Header-Tail Design**: JPEG data streams may randomly contain `0xA5 0x5A` byte sequences. Using only a header Magic is insufficient to reliably distinguish frame boundaries. The triple verification mechanism of header Magic + tail Magic + CRC16 reduces frame synchronization mismatch probability to $\frac{1}{2^{16}} \times \frac{1}{2^{16}} \approx 2.3 \times 10^{-10}$.

#### 5.7.2 Shooter End Receiver State Machine

Frame synchronization state machine in [arrow_receiver.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/arrow_receiver.c):

```
STATE_IDLE ──(received 0xA5 0x5A)──▶ STATE_HEADER ──(received 9-byte header)──▶ STATE_PAYLOAD
     ▲                                                                   │
     │                              STATE_COMPLETE ◄── STATE_CRC ◄───────┘
     │                                    │
     └────────────(CRC check failed)────────────┘
```

**UART Configuration Parameters**:
- Baud rate: 3,000,000 bps (3 Mbps)
- Data bits: 8, Stop bits: 1, Parity: None (8N1)
- Dual-link support: Primary UART (type=0x01 JPEG) + Secondary UART (type=0x02 IMU)

### 5.8 KCP-Lite Reliable Transport Protocol

[kcp_lite.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/kcp_lite.c) (576-line complete implementation) is a lightweight reliable UDP transport protocol for embedded systems, referencing the core mechanisms of the KCP protocol.

**Protocol Features**:

| Feature | Parameter | Description |
|------|------|------|
| MTU | 1100 bytes | Adapted for wired/wireless links |
| Forward Error Correction (FEC) | RS(4,5) | Every 5 packets can recover any 1 lost packet |
| Send Window | snd_wnd=128 | Controls sending rate |
| Timeout Retransmission | resend=50ms | nodelay=1 mode |
| Packet Type | 0=control frame, 1=video frame, 2=IMU data | Differentiated processing |

**Deviations from Standard KCP** (documented in code header):
1. Simplified congestion control: Full slow start and congestion avoidance not implemented (bandwidth is deterministic in wired scenarios)
2. Relaxed fast retransmission trigger: 3 consecutive duplicate ACKs → immediate retransmission
3. Flow control window only limits sender; receiver does not perform flow control
4. MTU does not implement IP fragment reassembly (application layer ensures not exceeding MTU)

### 5.9 Madgwick AHRS (Arrow End, ESP32-P4)

The Arrow End chose Madgwick (rather than Mahony) as the IMU fusion algorithm based on the following engineering rationale:

| Comparison Dimension | Mahony | Madgwick |
|----------|--------|----------|
| Fused Sensors | Accelerometer + Gyroscope | Accelerometer + Gyroscope + **Magnetometer** |
| Yaw Drift | Present (no magnetometer correction) | **None** (magnetometer provides absolute heading) |
| Computation | ~150 floating-point multiplications | ~200 floating-point multiplications |
| Parameter Tuning | Kp, Ki dual parameters | β (single parameter) |
| Maturity | Classic in aviation field | Mainstream AHRS algorithm in recent years |

At the Arrow End ESP32-P4 @ 200Hz sampling rate, Mahony requires ~2.5μs/update and Madgwick requires ~3.3μs/update, both well within the microcontroller's capacity, while Madgwick's drift-free yaw angle is critical for arrow attitude estimation.

### 5.10 Visualization Rendering Pipeline

[visualizer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/visualizer.c) implements a complete 2D rendering pipeline on the CPU:

```
visualizer_render_detection_view(frame, track_objects, ...):
    1. memcpy(frame → vis_buffer)                   // Frame copy (6MB@1080p)
    2. for each tracked object:
       ├── draw_rect(vis_buffer, bbox, ID_color)    // Bounding box
       ├── draw_label(vis_buffer, track_id + conf)  // ID + confidence label
       ├── draw_skeleton(vis_buffer, pose.keypoints) // COCO skeleton lines
       └── draw_trajectory(vis_buffer, trajectory)   // Motion trajectory lines
    3. draw_info_bar(vis_buffer, FPS, frame_count)  // Top info bar
    4. draw_top_down(vis_buffer, spatial_positions) // Top-down view area
```

**Rendering Primitive Implementation**:
- `draw_rect`: Bresenham line algorithm ×4 edges → supports variable line width (per-line fill)
- `draw_line`: Bresenham line → supports thickness (vertical expansion)
- `draw_circle`: Midpoint circle algorithm → O(R) per-point
- `draw_text`: 5×7 pixel bitmap font → per-character per-bit traversal

### 5.11 AI Acceleration Adapter Layer

[ai_accel_adapter.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/ai_accel_adapter.c) provides a unified abstraction for RISC-V AI instruction acceleration. Important note: K1 X60 **does not have an independent NPU**; AI acceleration comes from RISC-V custom AI instructions (RVV 1.0 + IME 16 instructions) integrated within the CPU cores.

**Acceleration Path Decision Tree**:

```
Detect backend type (ai_accel / cpu)
    │
    ├── "ai_accel" → ai_accel_context_create()
    │       │
    │       ├── HAS_SPACENGINE_AI → Load Spacengine SDK
    │       │       └── ai_accel_load_model() → Spacengine INT8 model
    │       │
    │       └── else → Fallback to ONNX Runtime CPU (libonnxruntime.so)
    │
    └── "cpu" → Use onnxruntime pure CPU inference
            │
            └── HAS_ONNX_RUNTIME → ort_run_session()
                    │
                    └── else → HEURISTIC FALLBACK (pure C algorithm)
```

### 5.12 SpacemiT Execution Provider Call Chain

SpacemiT EP is the core path for K1 AI acceleration:

```
ort_global_init()
    │
ort_create_session(model_path, 4)  ← 4 threads
    │
CreateSessionOptions
    ├── SetGraphOptimizationLevel(ORT_ENABLE_ALL)
    ├── SetIntraOpNumThreads(4)
    ├── spacemit_ort_session_options_init()  ← C++ bridge (spacemit_ort_bridge.cpp)
    │       └── SessionOptionsSpaceMITEnvInit(opts)
    │             └── Register libspacemit_ep.so
    └── CreateSession(model_path)
            └── Auto-load libspacemit_ep.so
                 → RVV 1.0 + IME hardware-accelerated inference
```

The [C→C++ bridge layer](file:///d:/shool/大三下/embedded/lingqi_tantong_c/src/spacemit_ort_bridge.cpp) (`spacemit_ort_bridge.cpp`) is responsible for calling SpacemiT's C++ SDK API from the C call chain, exporting C-compatible interfaces through `extern "C"`.

---

## 6. Usage Guide

### 6.1 Environment Requirements

| Dependency | Minimum Version | Description |
|--------|---------|------|
| GCC (riscv64) | 13.2+ | K1 native compilation or cross-compilation `riscv64-linux-gnu-gcc` |
| CMake | ≥3.16 | Build system |
| SpacemiT ONNX Runtime | 2.0.2 | [Official Download](https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.2.tar.gz) |
| ESP-IDF | v5.1+ | ESP32-P4 firmware compilation |
| Python 3 | ≥3.8 | Development/debugging utilities |

### 6.2 Installation Steps (K1 Native Build)

```bash
# Step 1: Install SpacemiT ONNX Runtime
sudo cp spacemit-ort.riscv64.2.0.2/lib/* /usr/lib/
sudo ldconfig
sudo cp -r spacemit-ort.riscv64.2.0.2/include/* /usr/local/include/

# Step 2: CMake configuration and compilation
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ONNX_RUNTIME=ON \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2
make -j$(nproc)

# Step 3: Configure hardware access permissions
sudo chmod 777 /dev/tcm           # IME acceleration resource

# Step 4: Run
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
./lingqi_tantong --video_path test_video.mp4
```

### 6.3 Cross-Compilation (x86 Linux → RISC-V K1)

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
  -DBIANBU_SYSROOT=/opt/bianbu-sysroot \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)

# Deploy to K1
scp build/lingqi_tantong k1@192.168.1.x:/home/k1/
```

### 6.4 Building for Different Platforms

```bash
# RISC-V cross-compilation
cmake -B build_riscv \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
  -DBIANBU_SYSROOT=/path/to/bianbu-sysroot \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_riscv -j$(nproc)

# x86 Linux development build
cmake -B build -DUSE_ONNX_RUNTIME=ON -DONNX_RUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j$(nproc)
```

### 6.5 Command-Line Parameters Reference

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `--video_path <path>` | string | — | Offline mode input video file path |
| `--realtime` | flag | false | Enable realtime Arrow pipeline mode |
| `--uart-A <path>` | string | `/dev/ttyS0` | Arrow UART primary link device node |
| `--uart-C <path>` | string | `/dev/ttyS1` | Arrow UART secondary link (IMU data) |
| `--baudrate <rate>` | int | `3000000` | UART baud rate |
| `--camera <path>` | string | `/dev/video0` | V4L2 camera device node |
| `--output_path <path>` | string | `output/reports` | Result output directory |
| `--max_frames <N>` | int | `0` | Maximum frames to process (0=no limit) |
| `--save_frame_interval <N>` | int | `10` | Save to output video every N frames |
| `--config <path>` | string | `configs/default.yaml` | YAML configuration file path |
| `--help` | flag | — | Display full help information |

### 6.6 Running Examples

```bash
# Minimal run (offline mode)
./lingqi_tantong --video_path test_video.mp4

# Full-parameter offline run
./lingqi_tantong \
    --video_path test_video.mp4 \
    --output_path output/results \
    --config configs/default.yaml \
    --max_frames 500 \
    --save_frame_interval 1

# K1 realtime pipeline mode (Arrow dual-link)
./lingqi_tantong \
    --realtime \
    --uart-A /dev/ttyS0 \
    --uart-C /dev/ttyS1 \
    --baudrate 3000000 \
    --config configs/default.yaml
```

### 6.7 Testing

Unit tests and integration tests are under development. Run the application with a test video to verify functionality:

```bash
./lingqi_tantong --video_path test_video.mp4 --max_frames 50
```

For production deployment, use the systemd service configuration below.

### 6.8 systemd Service Deployment (Production)

```ini
# /etc/systemd/system/lingqi.service
[Unit]
Description=LingQi TanTong Detection Service
After=network.target

[Service]
Type=simple
ExecStart=/opt/lingqi/lingqi_tantong --realtime --uart-A /dev/ttyS0 \
          --config /opt/lingqi/configs/default.yaml
Restart=on-failure
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lingqi
sudo systemctl start lingqi
sudo systemctl status lingqi
```

---

## 7. Configuration Options

### 7.1 Configuration File Format

The system uses a YAML-style key-value configuration file ([configs/default.yaml](file:///d:/shool/大三下/embedded/lingqi_tantong_c/configs/default.yaml)). The configuration manager uses a built-in minimal YAML parser (not libyaml). The supported YAML feature subset includes: scalar values (string/int/float/bool), nested mappings (up to 3 levels), and sequences (bracket lists). **Unsupported YAML features**: anchors/aliases (`&anchor`, `*alias`), multiline strings (`|`/`>`), complex tags.

### 7.2 Complete Configuration Parameters

#### 7.2.1 System Parameters `system`

| Parameter | Type | Default | Valid Range | Description |
|------|------|--------|----------|------|
| `use_onnx` | bool | `true` | — | Whether to enable ONNX Runtime inference (required for all AI models) |
| `log_level` | string | `INFO` | DEBUG/INFO/WARN/ERROR | Global log level |
| `startup_mode` | string | `realtime` | offline/realtime/benchmark | Default startup mode |
| `max_frames` | int | `0` | ≥0 | Maximum frames to process (0=unlimited) |
| `ring_buffer_size` | int | `16` | 4-256 | Arrow UART ring buffer slot count |
| `target_fps` | float | `15.0` | 1.0-60.0 | Target frame rate |
| `worker_threads` | int | `2` | 1-8 | Worker thread count (non-K1 Pipeline mode) |

#### 7.2.2 Video Parameters `video`

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `source` | string | `camera` | Video source type (camera/file/arrow) |
| `camera_device` | string | `/dev/video0` | V4L2 camera device |
| `camera_width` | int | `640` | Capture resolution width |
| `camera_height` | int | `480` | Capture resolution height |
| `camera_fps` | float | `15.0` | Capture frame rate |
| `camera_format` | string | `MJPEG` | Capture pixel format |
| `save_frame_interval` | int | `15` | Output video frame save interval |

#### 7.2.3 Arrow Communication `arrow`

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `uart_device_A` | string | `/dev/ttyS0` | Primary UART link |
| `uart_device_C` | string | `/dev/ttyS1` | Secondary UART link |
| `baudrate` | int | `3000000` | Baud rate (bps) |
| `dual_link` | bool | `true` | Whether to enable dual-link mode |

#### 7.2.4 Detection Parameters `detection`

| Parameter | Type | Default | Valid Range | Description |
|------|------|--------|----------|------|
| `backend` | string | `cpu` | cpu/ai_accel | Inference backend |
| `model_path` | string | — | — | YOLO11n ONNX model path |
| `confidence_threshold` | float | `0.25` | 0.0-1.0 | Raw detection confidence threshold |
| `iou_threshold` | float | `0.45` | 0.0-1.0 | NMS IoU threshold |
| `input_size` | int | `320` | 320/640 | Model input size |

#### 7.2.5 Tracking Parameters `tracking`

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `max_lost` | int | `30` | Maximum frames to retain after target loss |
| `min_iou` | float | `0.30` | Minimum IoU for tracking match |
| `max_distance` | float | `5.0` | Maximum association distance (meters) |
| `max_track_history` | int | `300` | Maximum trajectory history length |
| `ema_alpha` | float | `0.25` | EMA smoothing coefficient (smaller = smoother) |

#### 7.2.6 Spatial Localization Parameters `spatial`

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `fx` | float | `500.0` | Horizontal focal length (pixels) |
| `fy` | float | `500.0` | Vertical focal length (pixels) |
| `cx` | float | `320.0` | Principal point X offset |
| `cy` | float | `240.0` | Principal point Y offset |
| `avg_human_height` | float | `1.70` | Assumed average human height (meters) |
| `min_depth` | float | `0.3` | Minimum depth (meters) |
| `max_depth` | float | `120.0` | Maximum depth (meters) |
| `imu_pitch_correction` | bool | `true` | Whether to enable IMU pitch correction |

#### 7.2.7 K1 Hardware Parameters `k1_hardware`

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `enabled` | bool | `true` | Whether to enable K1 hardware acceleration |
| `cluster0_ai_cores` | string | `"0-3"` | AI inference core binding |
| `cluster1_io_cores` | string | `"4-7"` | I/O task core binding |
| `tcm.enabled` | bool | `true` | Enable TCM tightly-coupled memory |
| `tcm.tcm_device` | string | `/dev/tcm` | TCM device node |
| `tcm.tcm_size_kb` | int | `512` | TCM capacity (KB) |
| `vpu.enabled` | bool | `true` | Enable VPU hardware video acceleration |
| `jpu.enabled` | bool | `true` | Enable JPU hardware JPEG decoding |
| `gpu.enabled` | bool | `true` | Enable GPU (currently flag only) |
| `pipeline.ring_buffer_size` | int | `4` | K1 pipeline ring buffer slot count |

---

## 8. Troubleshooting Guide

### 8.1 Compilation Errors

| Error Message | Possible Cause | Solution |
|----------|----------|----------|
| `fatal error: onnxruntime_c_api.h: No such file` | ONNX Runtime not installed or incorrect path | Install SpacemiT ORT 2.0.2 and specify `-DONNX_RUNTIME_DIR` |
| `undefined reference to '__atomic_*'` | RISC-V linking missing `-latomic` | CMakeLists.txt auto-adds this; check toolchain |
| `error: unrecognized command line option '-march=native'` | Compiling on non-x86 platform | Use RISC-V cross-compilation toolchain file |
| `spacemit_ort_env.h not found` | SpacemiT EP header file missing | Confirm `/usr/local/include/onnxruntime/spacemit_ort_env.h` exists |
| `cannot find -lonnxruntime` | ORT library not in link path | `export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH` |

### 8.2 Runtime Errors

| Error Message | Possible Cause | Solution |
|----------|----------|----------|
| `Failed to load model: models/...` | Model file missing | Confirm `models/` directory is complete with 4 required models |
| `imgdecode failed` | Video format not supported | Confirm input is MP4/H.264 encoded |
| `Segmentation fault (core dumped)` | Memory out-of-bounds or null pointer | Use Address Sanitizer: `cmake -DCMAKE_BUILD_TYPE=Debug` |
| `Cannot open /dev/tcm: Permission denied` | IME device no permission | `sudo chmod 777 /dev/tcm` |
| `ONNX Runtime Error: Session creation failed` | Model format incompatible | Confirm model exported with ONNX opset 11-17 |
| `UART read timeout` | Arrow link disconnected | Check physical wiring, baud rate matching, ESP32-P4 firmware status |

### 8.3 Performance Issues

| Symptom | Diagnostic Method | Solution |
|------|----------|----------|
| FPS < 5 (K1 platform) | Check inference backend: `detection.backend` | Confirm SpacemiT EP is properly registered, `/dev/tcm` is accessible |
| Frame processing time >500ms | Use benchmark mode to analyze per-module latency | Consider reducing resolution or disabling face recognition |
| Memory continuously growing | Observe RSS with `top` / `htop` | Check for FrameData leaks, ensure `frame_data_destroy` per frame |
| IMU attitude severe jitter | Check `mahony.kp` parameter | Reduce kp to 0.3-0.5, increase ki to 0.02-0.05 |

### 8.4 Frequently Asked Questions (FAQ)

**Q: How to verify SpacemiT EP is correctly enabled?**

A: Check the build log for `SpacemiT EP: YES (RVV 1.0 + IME)` output. At runtime, check logs for successful `SessionOptionsSpaceMITEnvInit` call records.

**Q: K1 has no independent NPU, how to understand the 2.0 TOPS compute?**

A: The K1 X60 core achieves AI acceleration through RVV 1.0 (256-bit vector) + IME (16 custom matrix instructions) within the CPU core. "2.0 TOPS" is the equivalent compute of these instruction extensions in INT8 inference scenarios. The core advantage is zero copy overhead (data processed within the CPU core, no DMA transfer to a separate NPU required).

---

## 9. Performance Metrics

### 9.1 Expected Performance (Based on K1 X60 + SpacemiT EP 2.0.2)

| Model | CPU-only (FP32) | SpacemiT EP (FP32) | EP (INT8 Quantized) |
|------|----------------|--------------------|-------------------|
| YOLO11n (320×320) | ~500ms | ~200ms | **~60ms** |
| YOLOv8-Pose (640×640) | ~800ms | ~350ms | **~100ms** |
| YOLOv5-Face (320×320) | ~300ms | ~120ms | **~40ms** |
| ArcFace (112×112) | ~100ms | ~40ms | **~15ms** |

> ⚠ **Data Note**: The above are **estimated data** based on similar RISC-V platforms. Actual performance must be measured on K1 hardware using `onnxruntime_perf_test -e spacemit`. After FP32→INT8 quantization, IME matrix instructions have special optimizations for INT8, expected to achieve **3-5x** inference acceleration.

### 9.2 Reference Performance (x86-64, Intel i7-12700H, Development Test Platform)

| Scenario | FPS | Latency (ms) | Memory (MB) |
|------|-----|-----------|-----------|
| ONNX CPU (FP32) | 18-25 | 40-55 | 180 |
| ONNX CPU (INT8) | 35-50 | 20-28 | 120 |

### 9.3 Memory Usage Analysis

| Component | Estimated Memory Usage | Notes |
|------|-------------|------|
| Frame buffer (640×480×3) | ~0.9 MB | Single frame RGB |
| Arrow ring buffer (64KB×16) | ~1.0 MB | UART reception |
| Tracker (100 targets × ~3.5KB) | ~0.35 MB | Including Kalman state |
| Trajectory history (32 targets × 300 points) | ~0.1 MB | SpatialPosition |
| ONNX Runtime (4 models) | ~400-600 MB | Runtime peak |
| **Total (estimated peak)** | **< 800 MB** | Meets K1 4GB memory constraint |

### 9.4 Optimization Strategies

#### 9.4.1 Model INT8 Quantization (Highest Priority)

SpacemiT EP's IME instructions provide 3-5x acceleration for INT8 quantized models:

```python
# Using ONNX Runtime quantization tool
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("yolov8n.onnx", "yolov8n.q.onnx", weight_type=QuantType.QInt8)
```

#### 9.4.2 K1 Thread and Core Binding

```yaml
performance:
  openmp_threads: 2        # Inference thread count (K1 recommended 2-4)
  pipeline_pin_to_core: true
  imu_core: 0              # IMU processing bound to core 0
  inference_core: 1        # Inference bound to core 1
  visualization_core: 2    # Visualization bound to core 2
```

#### 9.4.3 System-Level Optimization

```bash
# IME hardware access permissions
sudo chmod 777 /dev/tcm

# Disable CPU frequency scaling
sudo cpufreq-set -g performance

# Confirm EP dynamic library path
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
```

### 9.5 Target Performance (K1 Muse Pi Pro End-to-End)

| Metric | Target Value | Current Estimate |
|------|--------|----------|
| End-to-end FPS (EP INT8, YOLO11n only) | ≥30 | ~17 (4-model cascade) |
| End-to-end Latency (Arrow → Display) | <200ms | ~150ms (excluding Arrow) |
| Memory Usage | <800 MB | ~600 MB (estimated) |
| Model Loading Time | <10s | Pending real-world testing |

---

## 10. Limitations and Future Work

### 10.1 Current Known Limitations

1. **Depth Estimation Accuracy Limited by Monocular Prior**: The current spatial engine uses the formula $Z = f_y \cdot H_{avg} / h_{bbox}$, assuming all people have a constant height of 1.70m. This assumption produces 30-50% depth errors for children, bending, or squatting targets in real-world scenarios. Integration of the MiDaS depth estimation model (interface already reserved) will significantly improve this issue.

2. **ATW Not Embedded in GPU Rendering Pipeline**: `ar_renderer_compensate_motion()` currently performs O(W×H) 3D rotation transforms per-pixel on CPU, taking >200ms at 1080p resolution, far exceeding the Motion-to-Photon ≤ 17.8ms human perception threshold. GPU fragment shader implementation is the only path to resolve this.

3. **Visualization Rendering Entirely on CPU**: Bounding boxes, skeleton lines, text annotations, and all other rendering are completed per-pixel on the CPU. At 1920×1080 resolution, expected latency is 40-80ms, accounting for 30-50% of total frame processing time. Migration to OpenGL ES 3.2 rendering pipeline could reduce this to <2ms.

4. **ESP32-P4 ↔ K1 Only Supports Wired UART**: When the Arrow End cannot connect to the Shooter End via cable (e.g., wireless throwing scenarios), an additional KCP-Lite over Wi-Fi/UDP wireless transmission channel is needed.

5. **VINS-Mono Visual-Inertial Odometry Not Implemented**: The current system lacks 6-DOF precise pose estimation capability. When the shooter-end camera moves, all target world coordinates experience systematic drift.

6. **Insufficient Test Coverage (12 Unit Tests)**: Lacking ONNX inference integration tests, Arrow protocol end-to-end tests, K1 hardware performance benchmarks, and long-running memory leak detection.

7. **YAML Parser Feature Subset Limitations**: Does not support anchors/aliases, multiline strings, complex tags, and other advanced YAML features; not fully compatible with libyaml.

8. **AVI Video Output Uncompressed**: Raw RGB24 format, 640×480@15fps generates approximately 800MB per minute. K1 hardware VPU (H.264/H.265) hardware encoding not integrated.

### 10.2 Development Roadmap

#### v1.2 (Near-term High Priority)

- [x] **Full INT8 Model Quantization**: All key models (YOLO11n / YOLOv8-Pose / YOLOv5-Face / ArcFace) are INT8-quantized; ST-GCN remains FP32
- [ ] **K1 Hardware Benchmarking**: `onnxruntime_perf_test -e spacemit` to obtain real performance data
- [ ] **EP vs CPU-only Real-world Comparison Report**
- [ ] **RVV Hand-written Vectorization**: letterbox, NMS, matrix multiplication critical path RVV intrinsic optimization
- [ ] **KCP-Lite + Mahony Arrow End Integration Verification**

#### v2.0 (Mid-to-Long-term Core Features)

- [ ] **VINS-Mono Visual-Inertial Odometry**: IMU pre-integration + sliding window Bundle Adjustment
- [ ] **ATW Asynchronous Time Warp**: OpenGL ES 3.2 fragment shader implementation, MTP ≤ 17.8ms
- [ ] **MiDaS Depth Estimation ONNX INT8**: Replace current height-prior depth estimation
- [ ] **ICP Point Cloud Registration**: Arrow End → Shooter End spatial coordinate system alignment

#### v3.0 (Long-term Engineering Optimization)

- [ ] **GPU Visualization Rendering Pipeline**: All rendering migrated to OpenGL ES 3.2
- [ ] **K1 VPU H.264/H.265 Hardware Encoding**: Replace Raw RGB24 AVI
- [ ] **libyaml Integration**: Replace minimal YAML parser
- [ ] **GLTF 2.0 3D Scene Export**
- [ ] **CI/CD Automation**: GitHub Actions + K1 hardware-in-the-loop testing
- [ ] **Unit Test Coverage ≥ 80%**

### 10.3 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------|------|----------|
| SpacemiT EP SDK instability | Medium | Blocks AI inference acceleration | Fallback to ONNX Runtime CPU + RVV |
| RVV 0.7.1 compiler support immature | High | Low hand-written vectorization development efficiency | Start with scalar C + OpenMP, gradually migrate |
| ESP32-P4 MIPI CSI driver instability | Medium | Unreliable image capture | Degrade to DVP parallel + external ISP |
| K1 4GB memory insufficient for 4 models | Medium | System OOM | Model LRU lazy loading |
| UART 3Mbps high-speed stability | Low | Data packet loss | KCP-Lite FEC already available, add hardware flow control |

---

## 11. References

### Academic Papers

1. Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637-646. DOI: 10.1109/JIOT.2016.2579198

2. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2001)*, 1, I-511–I-518. DOI: 10.1109/CVPR.2001.990517

3. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *Proceedings of the 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2005)*, 1, 886-893. DOI: 10.1109/CVPR.2005.177

4. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. *Proceedings of the 2014 IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2014)*, 580-587. DOI: 10.1109/CVPR.2014.81

5. Girshick, R. (2015). Fast R-CNN. *Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV 2015)*, 1440-1448. DOI: 10.1109/ICCV.2015.169

6. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *Advances in Neural Information Processing Systems (NIPS 2015)*, 28, 91-99.

7. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, 779-788. DOI: 10.1109/CVPR.2016.91

8. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016). SSD: Single Shot MultiBox Detector. *European Conference on Computer Vision (ECCV 2016)*, 21-37. DOI: 10.1007/978-3-319-46448-0_2

9. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019)*, 4690-4699. DOI: 10.1109/CVPR.2019.00482

10. Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and Computation Redistribution for Efficient Face Detection. *International Conference on Learning Representations (ICLR 2022)*. arXiv: 2105.04714

11. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple Online and Realtime Tracking. *2016 IEEE International Conference on Image Processing (ICIP 2016)*, 3464-3468. DOI: 10.1109/ICIP.2016.7533003

12. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *2017 IEEE International Conference on Image Processing (ICIP 2017)*, 3645-3649. DOI: 10.1109/ICIP.2017.8296962

13. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *European Conference on Computer Vision (ECCV 2022)*, 1-21. DOI: 10.1007/978-3-031-20047-2_1

14. Mahony, R., Hamel, T., & Pflimlin, J. M. (2008). Nonlinear Complementary Filters on the Special Orthogonal Group. *IEEE Transactions on Automatic Control*, 53(5), 1203-1218. DOI: 10.1109/TAC.2008.923738

15. Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011). Estimation of IMU and MARG orientation using a gradient descent algorithm. *2011 IEEE International Conference on Rehabilitation Robotics (ICORR 2011)*, 1-7. DOI: 10.1109/ICORR.2011.5975346

### Technical Specifications and Standards

16. RISC-V International. (2021). *RISC-V "V" Vector Extension Specification Version 1.0*. https://github.com/riscv/riscv-v-spec

17. Microsoft Corporation. (2023). *ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator*. https://onnxruntime.ai/

18. Ultralytics. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics

19. Khronos Group. (2021). *GLTF™ 2.0 Specification*. https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

20. MISRA C:2012. *Guidelines for the use of the C language in critical systems*. MISRA Consortium Ltd, 2013.

### Hardware Platform Documentation

21. SpacemiT. (2024). *SpacemiT K1/M1 Key Stone AI CPU Chip Overview*. https://www.spacemit.com/

22. Duan Jiahui. (2024). "K1 is an AI CPU, not CPU+NPU". *4th Dishui Lake China RISC-V Industry Forum (2024.08.19)*. https://view.inews.qq.com/a/20240819A07C7M00

23. SpacemiT. (2024). *Bianbu OS — SpacemiT K1 Official Linux Distribution*. https://bianbu.spacemit.com/

24. SpacemiT. (2024). *ONNX Runtime with SpacemiT Execution Provider*. https://archive.spacemit.com/spacemit-ai/onnxruntime/

25. SpacemiT. (2024). *Model Deployment on Bianbu — C++ Inference Example*. https://bianbu.spacemit.com/en/brdk/Model_deployment/4.3_CPP_Inference_Example/

26. Espressif Systems. (2024). *ESP32-P4 Datasheet*. https://www.espressif.com.cn/sites/default/files/documentation/esp32-p4_datasheet_cn.pdf

27. Espressif Systems. (2024). *ESP-IDF Programming Guide v5.4*. https://docs.espressif.com/projects/esp-idf/

28. Espressif Systems. (2024). *esp32-camera Component*. https://components.espressif.com/components/espressif/esp32-camera

29. Banana Pi. (2024). *BPI-F3 SpacemiT K1 Datasheet*. https://docs.banana-pi.org/en/BPI-F3/SpacemiT_K1

30. SpacemiT. (2024). *Muse Pi Pro User Manual*. https://github.com/spacemit-com/docs-product

### Open Source Projects and Software Dependencies

31. skywind3000. (2024). *KCP — A Fast and Reliable ARQ Protocol*. https://github.com/skywind3000/kcp

32. YAML. (2021). *libyaml — A C library for parsing and emitting YAML*. https://github.com/yaml/libyaml

33. Madgwick, S. (2011). *MadgwickAHRS — Open-source IMU and AHRS algorithms*. https://github.com/arduino-libraries/MadgwickAHRS

34. Arduino. (2024). *Mahony AHRS Filter Implementation*. https://github.com/arduino-libraries/MadgwickAHRS

35. OpenCV. (2024). *OpenCV — Open Source Computer Vision Library*. https://opencv.org/

---

## Appendix A: Project File Listing

```
lingqi_tantong_c/
├── include/                              # Header files (30)
│   ├── core_types.h                      #   Core data types + inline functions
│   ├── system_controller.h               #   System main controller interface
│   ├── inference_pipeline.h              #   AI inference pipeline interface
│   ├── tracking_manager.h                #   Multi-object tracking interface
│   ├── spatial_engine.h                  #   3D spatial localization interface
│   ├── yolov8_detector.h                 #   YOLO11 detector interface
│   ├── yolov8_pose_estimator.h           #   YOLOv8 pose estimation interface
│   ├── yolov5_face_detector.h            #   YOLOv5-Face face detection interface
│   ├── scrfd_detector.h                  #   SCRFD face detection interface (legacy)
│   ├── arcface_recognizer.h              #   ArcFace face recognition interface
│   ├── stgcn_action_recognizer.h         #   ST-GCN action recognizer interface
│   ├── keypoint_validator.h              #   Keypoint anatomical validator interface
│   ├── visualizer.h                      #   Visualization rendering interface
│   ├── ar_renderer.h                     #   AR rendering interface
│   ├── video_processor.h                 #   Video frame reading interface
│   ├── video_writer.h                    #   Video output writing interface
│   ├── imu_handler.h                     #   IMU data processing interface
│   ├── arrow_receiver.h                  #   Arrow protocol receiver interface
│   ├── kcp_lite.h                        #   KCP reliable transport protocol interface
│   ├── mahony_filter.h                   #   Mahony complementary filter interface
│   ├── ort_common.h                      #   ONNX Runtime shared module
│   ├── ort_inference_context.h           #   ONNX inference context interface
│   ├── spacemit_ort_bridge.h             #   SpacemiT EP C↔C++ bridge
│   ├── ai_accel_adapter.h                #   RISC-V AI acceleration adapter layer
│   ├── config_manager.h                  #   Configuration manager interface
│   ├── logger.h                          #   Logging system interface
│   ├── utils.h                           #   Utility functions interface
│   ├── result_manager.h                  #   Result manager interface
│   ├── model_store.h                     #   Model storage interface
│   ├── model_export.h                    #   3D model export interface
│   ├── k1_platform.h                     #   K1 platform detection interface
│   └── benchmark.h                       #   Performance benchmark interface
├── src/                                  # Source files (30)
│   ├── main.c                            #   Program entry
│   ├── system_controller.c               #   System controller implementation
│   ├── inference_pipeline.c              #   Inference pipeline implementation
│   ├── tracking_manager.c                #   Tracking manager implementation (Hungarian + cascade)
│   ├── spatial_engine.c                  #   Spatial engine implementation
│   ├── yolov8_detector.c                 #   YOLO11 detector (ONNX only, DFL decode)
│   ├── yolov8_pose_estimator.c           #   YOLOv8 pose estimation (ONNX only, OKS NMS)
│   ├── yolov5_face_detector.c             #   YOLOv5-Face face detection (ONNX only)
│   ├── scrfd_detector.c                  #   SCRFD face detection (ONNX only)
│   ├── arcface_recognizer.c              #   ArcFace face recognition (ONNX only)
│   ├── stgcn_action_recognizer.c         #   ST-GCN action recognition (ONNX only)
│   ├── keypoint_validator.c              #   Keypoint anatomical validation
│   ├── visualizer.c                      #   Visualization rendering
│   ├── ar_renderer.c                     #   AR rendering
│   ├── video_processor.c                 #   Video processing (ffmpeg + V4L2)
│   ├── video_writer.c                    #   Video writing
│   ├── imu_handler.c                     #   IMU processing
│   ├── arrow_receiver.c                  #   Arrow protocol reception
│   ├── kcp_lite.c                        #   KCP-Lite implementation
│   ├── mahony_filter.c                   #   Mahony filtering
│   ├── ort_common.c                      #   ONNX Runtime shared
│   ├── ort_inference_context.c           #   ONNX inference context
│   ├── spacemit_ort_bridge.cpp           #   SpacemiT EP C++ bridge
│   ├── ai_accel_adapter.c               #   AI acceleration adapter
│   ├── config_manager.c                  #   Configuration manager
│   ├── logger.c                          #   Logging system
│   ├── utils.c                           #   Utility functions
│   ├── result_manager.c                  #   Result manager
│   ├── model_store.c                     #   Model storage
│   ├── model_export.c                    #   3D model export
│   ├── core_types.c                      #   Core type helpers
│   ├── k1_platform.c                     #   K1 platform detection
│   └── benchmark.c                       #   Performance benchmark
├── cmake/                                # CMake toolchain files
│   ├── riscv64-toolchain.cmake           #   RISC-V cross-compilation toolchain
│   └── esp32p4-toolchain.cmake           #   ESP32-P4 toolchain
├── configs/
│   └── default.yaml                      #   Default full-parameter configuration
├── models/                               # ONNX model files
│   ├── Human Recognition/
│   │   └── yolo11n.q.onnx                #   YOLO11n person detection (quantized)
│   ├── Face Recognition/
│   │   ├── yolov5n-face_cut.q.onnx       #   YOLOv5-Face face detection (quantized)
│   │   └── arcface_mobilefacenet_cut.q.onnx  # ArcFace feature extraction (quantized)
│   └── Action Prediction/
│       └── Skeleton Recognition/
│           ├── yolov8n-pose.q.onnx       #   YOLOv8-Pose 17 keypoints (quantized)
│           └── stgcn.fp32.onnx           #   ST-GCN action recognition (FP32)
├── docs/                                 # Design documents
│   ├── ARCHITECTURE.md                   #   Architecture design details
│   ├── CODE_CHANGE_LOG.md                #   Code change log (v2.0 optimization)
│   ├── IMPLEMENTATION_GAPS.md            #   Unimplemented module list
│   ├── OPTIMIZATION_DESIGN.md            #   Optimization design document
│   ├── PROBLEM_ANALYSIS_REPORT.md        #   Problem analysis report
│   ├── REFACTORING_REPORT.md             #   Refactoring report
│   ├── TESTING_GUIDE.md                  #   Testing guide
│   └── TESTING_GUIDE_V2.md               #   Testing guide v2
├── receive/                              # ESP32-P4 / Arrow-end test scripts
│   ├── test-new-noface.py                #   Python test receiver
│   ├── test-ov-imu.py                    #   IMU test script
│   ├── test-ov-imu.ino                   #   Arduino IMU test
│   └── test-udp.py                       #   UDP test script
├── CMakeLists.txt                        #   Top-level CMake build configuration
├── configs/default.yaml                  #   Default configuration
├── test_video.mp4                        #   Test video file
└── README.md                             #   This document
```

## Appendix B: CMake Build Options Complete List

| Option | Default | Description |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Release` | Release / Debug / RelWithDebInfo |
| `CMAKE_TOOLCHAIN_FILE` | — | Cross-compilation toolchain file |
| `BIANBU_SYSROOT` | — | Bianbu OS sysroot path (cross-compilation) |
| `ONNX_RUNTIME_DIR` | — | SpacemiT ONNX Runtime 2.0.2 installation path |
| `USE_ONNX_RUNTIME` | `ON` | Enable ONNX Runtime (including SpacemiT EP) |
| `USE_SPACENGINE_AI` | `OFF` | Enable RISC-V AI instruction acceleration adapter layer |
| `USE_OPENMP` | `ON` | Enable OpenMP multi-core parallelism |
| `ENABLE_RVV_OPT` | `OFF` | Enable RVV hand-written vectorization intrinsics |
| `ENABLE_K1_PIPELINE` | `ON` | Enable K1 dual-cluster pipeline parallelism |
| `ENABLE_K1_TCM` | `ON` | Enable K1 TCM tightly-coupled memory (weight preloading) |
| `ENABLE_K1_VPU` | `ON` | Enable K1 VPU hardware video acceleration |
| `ENABLE_K1_JPU` | `ON` | Enable K1 JPU hardware JPEG decoding |
| `MUSE_PI_ARCH` | `rv64gcv0p7` | RISC-V target architecture |
| `SPACENGINE_DIR` | — | Spacengine AI acceleration SDK path |
| `K1_MPP_DIR` | — | K1 MPP media processing SDK path |
| `K1_JPU_DIR` | — | K1 JPU hardware decoding library path |

---

> **Project Repository**: lingqi_tantong_c
> **License**: Proprietary
> **Last Updated**: 2026-06
> **Total Code**: ~12,000 lines C/C++ (Shooter End, K1 Muse Pi Pro)
> **Models**: 5 ONNX models (4 INT8-quantized + 1 FP32)
