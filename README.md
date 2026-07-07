# LingQi TanTong (灵柒探瞳) — Edge AI Inference Pipeline

> **Version**: v2.5 — Muse Pi Pro (SpacemiT K1 X60) Fully Adapted
> **Target Platform**: SpacemiT K1 X60 (Muse Pi Pro) — RISC-V 64 (Bianbu Linux 2.1+)
> **Development Platform**: Linux (Cross-compilation) / K1 Native Build
> **Language Standard**: C11 + C++17 (SpacemiT EP Bridge)
> **Build System**: CMake ≥3.16
> **Total Code**: ~15,000 lines C/C++ (Shooter End, K1 Muse Pi Pro)
> **Models**: 5 ONNX models (4 INT8-quantized + 1 FP32)

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

In the field of modern tactical reconnaissance and intelligent perception, rapid detection of unknown spaces and real-time information backhaul constitute core technical challenges. Traditional detection methods (static surveillance cameras, handheld detectors) have inherent limitations in dynamic, unstructured environments: large perception blind spots, high response latency, and insufficient spatial positioning accuracy. The **LingQi TanTong** (灵柒探瞳) project builds an intelligent detection system based on dual-device heterogeneous computing, achieving real-time 3D perception and intelligent target analysis through deep integration of visual perception, inertial navigation, deep learning inference, and augmented reality visualization.

The core innovation lies in adopting the **SpacemiT K1 X60 RISC-V chip** as the primary computing platform, combined with an **ESP32 microcontroller** as the front-end perception node, to build a low-cost, low-power, high-real-time embedded AI vision system. This design aligns with the macro trend of embedded AI migrating toward edge computing [Shi et al., 2016], and validates the feasibility of domestic autonomous RISC-V chips in intelligent perception applications.

### 1.2 System Feature Overview

The LingQi TanTong system consists of two physical computing nodes:

| Computing Node | Hardware Platform | Core Responsibilities |
|---|---|---|
| **Arrow End** | ESP32 + OV3660 + GY85 | Image capture (OV3660 VGA), IMU attitude estimation (ADXL345 + ITG3205), CoAP/UDP video streaming + IMU telemetry (WiFi AP mode) |
| **Shooter End** | SpacemiT K1 Muse Pi Pro | AI inference acceleration (RVV 1.0 + IME 2.0 TOPS), multi-object tracking (ByteTrack + Hungarian + cascade matching), 3D world-coordinate localization (K1 odometry + pinhole model), Web UI visualization (Three.js 3D) |

The system provides the following **10 core capabilities**:

1. **Person Detection & Pose Estimation (Unified Model)**: YOLOv8-Pose (PRIMARY) performs BOTH person detection AND 17-keypoint COCO pose estimation in a single forward pass. YOLO11n-Pose available as alternative variant. Hardware-accelerated via ONNX Runtime + SpacemiT Execution Provider (RVV 1.0 + IME).

2. **Face Detection & Recognition**: YOLOv5-Face lightweight face detection (320×320 INT8) cascaded with ArcFace MobileFaceNet-cuted deep feature extraction (128-dimensional embedding vector). Cosine similarity matching for identity re-identification.

3. **Action Recognition**: ST-GCN (Spatial-Temporal Graph Convolutional Network) for skeleton-based action recognition. Supports 7-class NTU-RGB+D action subset. INT8-quantized for SpacemiT EP acceleration.

4. **Multi-Object Tracking**: ByteTrack-inspired cascade matching + Kuhn-Munkres Hungarian algorithm + 7-state Kalman filtering + EMA smoothing (α=0.35). Includes re-identification pool, occlusion handling, and multi-person detection with new-person grace frames.

5. **3D World-Coordinate Localization**: K1 INS odometry (strapdown inertial navigation + ZUPT EKF) provides dynamic world origin. Pinhole camera model monocular depth estimation fused with IMU pitch correction. Multi-point anatomical depth sampling with MAD outlier rejection. World coordinate system with adaptive anchor strategy (SEVIS-3D inspired).

6. **Web UI (Three.js 3D Visualization)**: Embedded HTTP + WebSocket server (Mongoose). Real-time frame streaming with bounding boxes, skeleton overlays, 3D world-coordinate projection. Pipeline lifecycle control (start/stop) from browser. Ceramic-white design system with Chinese localization.

7. **Terminal UI (3-mode adaptive)**: Auto-detects TTY/CI/pipe. Three output modes: HUMAN (Unicode color), PLAIN (ASCII), MACHINE (JSON Lines). Checklist spinner for model loading, progress bar for offline video, status line for realtime mode.

8. **Real-time Data Link**: CoAP/UDP wireless protocol (RFC 7252) implementing ESP32 → K1 Block2 video streaming + IMU telemetry over WiFi. Adaptive burst pacing to optimize throughput.

9. **K1 Hardware Acceleration**: RISC-V Vector 1.0 (256-bit VLEN) auto-vectorization + IME matrix extension instructions (2.0 TOPS). Dual-cluster pipeline parallelism (Cluster0 AI cores 0-3, Cluster1 I/O cores 4-7). TCM tightly-coupled memory (512KB) for model weight preloading. VPU hardware H.264 encode, JPU hardware JPEG decode.

10. **Adaptive Frame Skip**: Grid-based subsampled MAD (Mean Absolute Difference) frame differencing. Three activity levels (STATIC/LOW_MOTION/ACTIVE) with adaptive skip limits. Up to 10× effective throughput improvement on static scenes (~0.2ms comparison cost vs ~150ms inference savings).

### 1.3 Three Run Modes

| Mode | Trigger | Description |
|---|---|---|
| **GUI Mode** (default) | No arguments, or `--web [PORT]` | Embedded Web UI at `http://localhost:8080`. Pipeline starts IDLE; user controls lifecycle from browser. Three.js 3D visualization with real-time WebSocket frame streaming. |
| **Realtime CLI** | `--realtime` | K1 dual-cluster pipeline (Capture→Inference→PostProcess threads). V4L2 camera or CoAP WiFi input. Terminal UI status display. |
| **Offline CLI** | `--video PATH` | Video file frame-by-frame processing. Progress bar with FPS/ETA. Full inference→tracking→localization→visualization pipeline. AVI output. |

### 1.4 Technical Specifications

| Metric | Target Value | Current Status |
|---|---|---|
| Object Detection FPS (K1 EP INT8) | ≥25 | ~10-15 (480×480, 4-model cascade) |
| Object Detection mAP50 (YOLOv8-Pose) | ≥48% | Model integrated (INT8 quantized) |
| Face Detection AP (YOLOv5-Face) | ≥90% | Model integrated (INT8 quantized) |
| Action Recognition Accuracy (ST-GCN) | ≥70% | Model integrated (INT8, 7-class) |
| Tracking ID Switch Rate | <5% | Implemented (cascade + Hungarian + re-id) |
| Spatial Localization Error (<10m) | <20% | Implemented (anatomical depth + MAD outlier) |
| End-to-end Latency (Arrow→Display) | <200ms | ~150ms (excluding Arrow WiFi) |
| System Memory Usage | <800 MB | ~600 MB (4 ONNX models loaded) |

### 1.5 Project Significance and Contributions

1. **Heterogeneous Embedded AI System Paradigm**: Validated the "low-power front-end perception + RISC-V back-end inference" dual-device architecture, providing a reusable reference for embedded AI system design.

2. **RISC-V AI Acceleration Practice**: Systematically applied SpacemiT K1's RVV 1.0 + IME instruction set for computer vision inference acceleration, accumulating practical experience for the RISC-V AI ecosystem.

3. **Multi-sensor Fusion Localization**: Implemented a spatial localization scheme fusing INS odometry (strapdown + ZUPT EKF), visual monocular depth estimation, and adaptive world anchors — effectively improving robustness in GPS-denied environments.

4. **Full C Language Embedded Implementation**: Entire system written in C11 with a minimal C++17 bridge for SpacemiT EP. Zero dependency on heavyweight frameworks (Python, OpenCV, ROS). Portable to various resource-constrained platforms.

5. **Browser-Based 3D Visualization**: Novel Web UI approach using Mongoose embedded server + Three.js rendering, eliminating the need for a physical display while providing rich 3D world-coordinate visualization.

---

## 2. Literature Review

### 2.1 Evolution of Object Detection Algorithms

Object detection has undergone a profound transformation from traditional handcrafted feature methods to deep learning end-to-end paradigms. Early Viola-Jones detectors [Viola & Jones, 2001] and HOG + SVM [Dalal & Triggs, 2005] relied on sliding windows and manually designed features. The R-CNN series [Girshick et al., 2014; Girshick, 2015; Ren et al., 2015] pioneered region-proposal-based two-stage detection, known for high accuracy but insufficient real-time performance.

The emergence of single-stage detectors — YOLO [Redmon et al., 2016], SSD [Liu et al., 2016] — transformed detection into a regression problem. **YOLOv8** [Jocher et al., 2023] introduced an anchor-free detection head, C2f cross-stage partial connection module, and Decoupled Head architecture. The nano variant achieves 48.0% AP on COCO-Pose with only 3.3M parameters.

**Key Design Decision — Unified Detection + Pose**: This project uses YOLOv8-Pose as the PRIMARY model, performing both person detection and 17-keypoint pose estimation in a single forward pass. This eliminates the need for a separate person detector, saving ~150ms/frame (one full EP inference) and one TCM slot. The alternative YOLO11n-Pose variant offers improved architecture (C3k2 blocks, anchor-free design) with higher mAP.

### 2.2 Face Detection and Recognition

For face detection, **YOLOv5-Face** (~0.8M parameters, INT8 quantized) balances accuracy and inference efficiency at 320×320 input resolution. For face recognition, **ArcFace** [Deng et al., 2019] uses Additive Angular Margin Loss in angular space to significantly improve inter-class separability and intra-class compactness. This project uses MobileFaceNet-cuted architecture outputting 128-dimensional feature vectors, matched via cosine similarity.

### 2.3 Multi-Object Tracking Methods

The mainstream paradigms of MOT are divided into Tracking-by-Detection and Joint Detection and Tracking. **SORT** [Bewley et al., 2016] constructed an efficient tracking framework using Kalman filter prediction and Hungarian algorithm association. **DeepSORT** [Wojke et al., 2017] introduced appearance features for occlusion robustness. **ByteTrack** [Zhang et al., 2022] innovatively proposed using low-confidence detection boxes for secondary matching, achieving SOTA performance on MOT17/MOT20.

This project implements a ByteTrack-inspired tracker with:
- **Cascade matching**: Group tracks by time-since-update, matching freshest first
- **Hungarian algorithm** (Kuhn-Munkres): Optimal global assignment with combined IoU (80%) + appearance (20%) cost matrix
- **7-state Kalman filter**: State vector [cx, cy, area, aspect_ratio, vx, vy, vs]
- **EMA smoothing** (α=0.35): Spatial coordinate smoothing to suppress inter-frame jitter
- **Re-identification pool**: Deleted tracks retained for 60 frames for re-association
- **Occlusion handling**: Upper-body (≥3 keypoints) and side-body (≥2 keypoints) fallback validation

### 2.4 Inertial Navigation and Attitude Estimation

**Madgwick filter** [Madgwick et al., 2011] uses gradient descent optimization to fuse accelerometer, gyroscope, and magnetometer (9-DOF) data, providing drift-free yaw angle estimation with a single tunable parameter β. On the K1 Shooter End, raw IMU readings from the Arrow End (or K1's onboard ICM-20948) are processed through Madgwick filtering for attitude quaternion output.

For K1 self-motion estimation, the system implements **Strapdown Inertial Navigation** (INS) with **ZUPT (Zero-Velocity Update)** via an Error-State Kalman Filter (9-state EKF: position error ×3, velocity error ×3, attitude error ×3). **GLRT (Generalized Likelihood Ratio Test)** zero-velocity detection uses a 5-sample sliding window on accelerometer and gyroscope readings. The TRIAD algorithm provides gravity-aligned initialization during a 2-second stationary period at startup.

**Yaw Unobservability Note**: ZUPT EKF only observes velocity error (zero-velocity pseudo-measurement). Yaw angle is completely unobservable from velocity measurements alone [Ilyas et al., Sensors 2016], resulting in ~1-5°/min gyro Z-axis drift. The WorldCoord adaptive anchor strategy mitigates this for short runs (<60s) by locking stationary person positions.

### 2.5 RISC-V Vector Extension and AI Acceleration

The **RISC-V Vector Extension (RVV 1.0)** [RISC-V International, 2021] is a variable-length SIMD ISA supporting VLEN from 32 to 65536 bits. The SpacemiT X60 core implements RVV 1.0 with VLEN=256 bits — a single vector instruction processes 8 float32 or 16 int16 operands simultaneously. The proprietary **IME (Intelligent Matrix Extension)** adds 16 custom AI acceleration instructions (matrix multiply, sliding window), achieving 2.0 TOPS of in-core AI compute when working with RVV 1.0 [SpacemiT, 2024].

**Important**: K1 X60 has NO independent NPU. AI acceleration comes entirely from RVV 1.0 + IME instructions within the CPU cores. The core advantage is zero-copy: data is processed inside the CPU core without DMA transfers to an external accelerator.

### 2.6 Embedded Deep Inference Framework

**ONNX Runtime** [Microsoft, 2023] is a cross-platform deep learning inference engine supporting multiple hardware Execution Providers (EP). SpacemiT provides a customized EP (`libspacemit_ep.so`) registered through `SessionOptionsSpaceMITEnvInit()`, automatically mapping convolutions and matrix multiplications to RVV 1.0 + IME instructions. The EP only supports INT8-quantized models (.q.onnx produced by SpacemiT's xquant tool); FP32 models silently fall back to CPU EP.

---

## 3. Methodology

### 3.1 Research Paradigm and Design Philosophy

The project follows a four-stage spiral iterative methodology:

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

Core design principles:

1. **Minimal Dependency Principle**: Core algorithm libraries have zero external dependencies beyond the C standard library and ONNX Runtime. ONNX Runtime is required for all AI inference — there is no heuristic/pure-C fallback path.

2. **ONNX Runtime Required**: All AI inference modules require ONNX Runtime with either SpacemiT EP (RVV 1.0 + IME) or CPU EP (real ONNX inference, no hardware acceleration).

3. **Configuration-Driven Architecture**: All tunable parameters managed uniformly through a YAML-style configuration file (`configs/default.yaml`), supporting runtime dynamic loading. Detection thresholds, tracking parameters, and visualization options adjustable without recompilation.

4. **Modular Composition Pattern**: Strict `create → process → destroy` lifecycle management. Each module encapsulates internal state through structs, eliminates global variables, and ensures thread safety.

### 3.2 Development Environment and Toolchain

| Tool/Component | Version/Specification | Purpose |
|---|---|---|
| C Compiler | GCC 13.2+ / Clang 18+ (riscv64) | K1 native/cross-compilation |
| C++ Compiler | G++ 13.2+ / Clang++ 18+ (riscv64) | SpacemiT EP C++ bridge |
| CMake | ≥3.16 | Cross-platform build management |
| Python 3 | ≥3.8 | Development/debugging utilities |
| ESP-IDF | v5.1+ | ESP32 firmware development |
| ONNX Runtime | SpacemiT ORT 2.0.2 | RISC-V AI inference acceleration |
| SpacemiT EP | libspacemit_ep.so | RVV 1.0 + IME hardware acceleration |
| Mongoose | cesanta/mongoose (embedded) | HTTP + WebSocket server |
| libjpeg-turbo | ≥2.0 | JPEG decode (Arrow UART/CoAP frames) |
| Three.js | r160+ (ES module) | Web UI 3D visualization |

### 3.3 Development Process and Quality Control

The project adopts a quality assurance system driven by systematic code reviews. Across the v1.0 → v2.5 iteration history, reviews covering all 32 source files and 32 header files identified and fixed critical defects including:

- **YOLOv8 ONNX Output Format Correction**: (x1,y1,x2,y2) → (cx,cy,w,h) format correction in pose estimator
- **K1 No Independent NPU Cognitive Correction**: Comprehensively cleaned up all improper "NPU" references, replacing with "RISC-V AI instruction acceleration"
- **xquant INT8 Classification Head**: Discovered that SpacemiT's INT8 quantization destroys the YOLO classification head (all sigmoid outputs ≈0.5). Innovated DFL peakiness as a confidence signal proxy.
- **GLRT ZUPT Threshold**: Corrected from 300,000 (never fired) to 500 (proper 4× headroom above baseline)
- **Velocity Clamp**: Tightened from 50 m/s (180 km/h) to 5 m/s (18 km/h) for physically meaningful pedestrian/mobile-robot speeds
- **GCC X60 Auto-Vectorization Safety**: Disabled `-ftree-vectorize` on GCC <15 to prevent misaligned vector load/store SIGBUS (X60 lacks Zicclsm hardware support)

For detailed change records, see [docs/CODE_CHANGE_LOG.md](docs/CODE_CHANGE_LOG.md).

---

## 4. System Architecture

### 4.1 Overall Architecture Overview

The LingQi TanTong system adopts a **dual-device heterogeneous computing architecture**, consisting of the Arrow End (ESP32 front-end perception node) and the Shooter End (K1 Muse Pi Pro primary computing node), interconnected via WiFi (CoAP/UDP).

```
┌────────────────────────── Arrow End ──────────────────────────┐
│                   ESP32 (Xtensa LX7 240MHz ×2)                 │
│                                                                 │
│  ┌──────────┐   DVP (8-bit)   ┌──────────────────┐             │
│  │  OV3660  │───────────────▶│  ISP + JPEG Enc  │             │
│  │  Camera  │                │  (HW Pipeline)   │             │
│  └──────────┘                └────────┬─────────┘             │
│                                       │ JPEG frames            │
│  ┌──────────┐   I²C (400kHz) ┌───────▼─────────┐             │
│  │  GY85    │───────────────▶│  Accel + Gyro   │             │
│  │  IMU     │  ADXL345+      │  Raw Readings   │             │
│  │          │  ITG3205       │  → JSON         │             │
│  └──────────┘                └────────┬─────────┘             │
│                                       │                        │
│                              ┌────────▼─────────┐             │
│                              │   CoAP Server    │             │
│                              │ /stream + /imu   │             │
│                              │ + /servo (PWM)   │             │
│                              └────────┬─────────┘             │
│                                       │                        │
│                     WiFi AP (CoAP/UDP :5683)                  │
└───────────────────────────────────────┼────────────────────────┘
                                        │
           ═══════════════════════════════╪══════════════════════
                                        │
┌────────────────────────── Shooter End ─────────────────────────┐
│          Muse Pi Pro — SpacemiT K1 X60 (8-core, RVV 1.0 + IME) │
│                                        │                        │
│  ┌─────────────────────────────────────▼──────────────────────┐│
│  │              SystemController (Main Scheduler)              ││
│  │                                                              ││
│  │  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  ││
│  │  │  Data Layer  │  │ Business Logic │  │  Presentation  │  ││
│  │  │              │  │                │  │                │  ││
│  │  │VideoProcessor│  │InferencePipeline│  │ Terminal UI    │  ││
│  │  │CoapReceiver  │  │ TrackingManager│  │ Web UI Server  │  ││
│  │  │ IMUHandler   │  │ SpatialEngine  │  │ VideoWriter    │  ││
│  │  │ ModelStore   │  │  WorldCoord    │  │ (AVI Export)   │  ││
│  │  │ FrameDiff    │  │ K1Odometry     │  │                │  ││
│  │  │ K1IMU        │  │ ORT+SpacemiTEP │  │                │  ││
│  │  └──────────────┘  └────────────────┘  └────────────────┘  ││
│  │                                                              ││
│  │  ┌────────────────────────────────────────────────────────┐ ││
│  │  │ Support: Logger │ ConfigManager │ PipelineState │ ResMgr│ ││
│  │  └────────────────────────────────────────────────────────┘ ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Web UI (HTTP :8080) │ HDMI 1080p60 │ Framebuffer / SDL   │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 Shooter End Module Architecture

The Shooter End (K1) software system is organized into functional module groups:

```
┌──────────────────────────────────────────────────────────────┐
│                     Application Layer                          │
│  ┌─────────────────────┐                                     │
│  │      main.c         │  CLI → Mode dispatch → Main loop    │
│  │  + terminal_ui.c    │  TUI: banner, checklist, progress   │
│  └─────────┬───────────┘                                     │
├────────────┼─────────────────────────────────────────────────┤
│                     Controller Layer                           │
│  ┌─────────▼───────────┐                                     │
│  │ system_controller.c │  3-mode orchestration:              │
│  │                     │  GUI(web idle) / Realtime(K1 3-thread│
│  │ pipeline_state.c    │  pipeline) / Offline(video loop)    │
│  └─────────────────────┘                                     │
├──────────────────────────────────────────────────────────────┤
│                   Business Logic Layer                         │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐  │
│  │InferencePipeline │ │ TrackingManager  │ │SpatialEngine │  │
│  │                  │ │                  │ │              │  │
│  │ YOLOv8-Pose(1°)  │ │ Cascade+Hungarian│ │ Pinhole model│  │
│  │ YOLOv5-Face→ArcFace│ │ + 7-state Kalman│ │ + IMU correct│  │
│  │ ST-GCN(action)   │ │ + EMA + re-id    │ │ + MAD fusion │  │
│  └──────────────────┘ └──────────────────┘ └──────────────┘  │
│  ┌──────────────────┐ ┌──────────────────┐                   │
│  │   WorldCoord     │ │   K1Odometry     │                   │
│  │                  │ │                  │                   │
│  │ Adaptive anchors │ │ INS+ZUPT EKF     │                   │
│  │ 3D skeleton proj │ │ GLRT zero-vel    │                   │
│  │ ENU world coords │ │ TRIAD init       │                   │
│  └──────────────────┘ └──────────────────┘                   │
├──────────────────────────────────────────────────────────────┤
│                    Data Processing Layer                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │VideoProcessor │ │  IMUHandler   │ │  ModelStore   │       │
│  │               │ │               │ │               │       │
│  │ MP4/V4L2/CoAP │ │ Sliding window│ │ Model file    │       │
│  │ input sources │ │ smoothing     │ │ management    │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │ CoapReceiver  │ │   FrameDiff   │ │    K1IMU      │       │
│  │               │ │               │ │               │       │
│  │ Block2 JPEG   │ │ Grid MAD      │ │ ICM-20948     │       │
│  │ reassembly    │ │ adaptive skip │ │ I²C driver    │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
├──────────────────────────────────────────────────────────────┤
│                    Presentation Layer                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │  WebServer    │ │   Web UI      │ │  VideoWriter  │       │
│  │  (Mongoose)   │ │  (Three.js)   │ │               │       │
│  │ HTTP+WS server│ │ 3D world view │ │ AVI encoding  │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
├──────────────────────────────────────────────────────────────┤
│                      Support Layer                             │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │ConfigManager  │ │    Logger     │ │ResultManager  │       │
│  │ YAML parser   │ │ Leveled+JSON  │ │ Session mgmt  │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐       │
│  │PipelineState  │ │  JsonWriter   │ │  K1Platform   │       │
│  │ 5-state FSM   │ │ JSON builder  │ │ Cap detection │       │
│  └───────────────┘ └───────────────┘ └───────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Core Data Flow

#### 4.3.1 Single-Frame Processing Pipeline

```
FrameData (uint8*, width, height, channels, timestamp)
    │
    ├──▶ FrameDiff ──▶ Skip? (adaptive frame differencing)
    │
    ├──▶ YOLOv8-Pose (PRIMARY, unified detection + pose)
    │       │
    │       ├── Detection[] (bbox, conf, class_id, keypoints)
    │       ├── PoseEstimation[] (17 COCO keypoints)
    │       └── Filtering (confidence/area/aspect ratio/NMS/OKS-NMS)
    │
    ├──▶ YOLOv5-Face (periodic, every 5-60 frames)
    │       └── ArcFace ──▶ FaceIdentity[] (128-d feature vectors)
    │
    ├──▶ ST-GCN (periodic, every ~15 frames)
    │       └── ActionPrediction[] (7-class NTU-RGB+D subset)
    │
    └──▶ SpatialEngine ──▶ depth → camera coords → WorldCoord
            │
            └── K1Odometry (INS position) ──▶ world coordinates
```

#### 4.3.2 Target Association Graph

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
        │
        ▼
  WorldCoord (K1-relative world coordinates)
```

### 4.4 Inter-Module Data Transfer Contract

| Producer | Data Product | Consumer | Transfer Method |
|---|---|---|---|
| VideoProcessor | FrameData* | SystemController | Function return (heap allocated) |
| CoapReceiver | ArrowSourceFrame | SystemController | Thread-safe get_latest |
| InferencePipeline | InferenceResult (value) | SystemController | Stack return value |
| SpatialEngine | SpatialResult | SystemController | Value return |
| TrackingManager | TrackingResult | SystemController | Stack return value |
| K1Odometry | pose, velocity | WorldCoord, SpatialEngine | Thread-safe getter |
| WorldCoord | world positions | WebServer, SystemController | Thread-safe getter |
| IMUHandler | IMUExternalPose | SpatialEngine | Function call parameter |
| SystemController | vis_buffer (uint8_t*) | VideoWriter | Function call parameter |
| SystemController | frame_json | WebServer | web_server_push_frame() ring buffer |

### 4.5 K1 Dual-Cluster Pipeline Architecture

The K1 X60 processor has 8 cores divided into two clusters. The system enables a **multi-thread pipeline** through the `HAS_K1_PIPELINE` compile definition:

```
┌─────────────────── Cluster0 (AI, cores 0-3, 512KB shared TCM) ─┐
│                                                                    │
│  CPU 1: Inference Thread                                          │
│    RingBuffer[slot].rgb → YOLOv8-Pose → face detect → ArcFace    │
│    → ST-GCN dispatch → RingBuffer[slot].inference                 │
│                                                                    │
│  CPU 0: PostProcess Thread                                        │
│    RingBuffer[slot].inference → tracking → spatial → WorldCoord  │
│    → serialize JSON → WebServer ring buffer → release slot        │
│                                                                    │
│  CPU 2: OpenMP worker / spare                                     │
│  CPU 3: OpenMP worker                                             │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────── Cluster1 (I/O, cores 4-7) ────────────────────┐
│                                                                    │
│  CPU 4: Capture Thread                                            │
│    CoAP receive → JPEG decode (JPU) → RingBuffer[slot].rgb_data  │
│    K1IMU polling → K1Odometry update                              │
│                                                                    │
│  CPU 5: VPU/JPU offload / spare                                   │
│  CPU 6: Viz Thread (visualizer rendering + WebServer broadcast)   │
│  CPU 7: OpenMP worker / OS background                             │
└────────────────────────────────────────────────────────────────────┘
```

**Ring Buffer Design** (4 slots):
```c
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

Slot acquisition and release adopt a **producer-consumer model** with `pthread_mutex_lock` + `pthread_cond_wait` for busy-wait-free synchronization.

### 4.6 Memory Management Strategy

| Strategy | Description | Implementation |
|---|---|---|
| **Stack First** | Small structs (BBox, Detection, SpatialPosition) passed by value | C value semantics |
| **Fixed Upper Bounds** | All array capacities hard-constrained via macros | `MAX_DETS=40`, `MAX_TRACKS=40` |
| **Explicit Lifecycle** | Each module follows `create → process → destroy` | No implicit memory leak paths |
| **Zero Global Variables** | All runtime state encapsulated in module structs | Thread safety guarantee |
| **Frame Assembly Buffer** | CoAP Block2 data assembled in 128KB dynamic buffer | Realloc on large frames, zero-copy output |
| **Frame Data Heap Allocation** | FrameData malloc'd in video_processor, freed in controller | Clear ownership transfer |
| **WebSocket Ring Buffer** | 32-slot lock-protected ring buffer for frame JSON | PostProcess → WebServer thread |

---

## 5. Technical Implementation

### 5.1 System Initialization and Dependency Management

The entry file [main.c](src/main.c) handles CLI parsing, TUI initialization, and mode dispatch. System initialization follows a strict "configuration first, dependency injection, bottom-up" order via `system_controller_create()`:

```c
SystemController* system_controller_create(const char* config_path, const char* pose_model) {
    // Step 1: Configuration manager (initialized first)
    // Step 2: Pipeline state machine (5-state FSM: IDLE→STARTING→RUNNING→STOPPING→ERROR)
    // Step 3: Logger (JSON format, thread-safe)
    // Step 4: Model file manager (validate model directory integrity)
    // Step 5: IMU handler (window_size=10, min_interval=0.01s, max_gap=0.1s)
    // Step 6: Frame differencing (adaptive frame skip)
    // Step 7: Inference pipeline (YOLOv8-Pose + Face + ArcFace + ST-GCN)
    // Step 8: Tracking manager (cascade matching + Hungarian + 7-state Kalman)
    // Step 9: Spatial engine (pinhole model + anatomical depth sampling)
    // Step 10: K1 odometry (INS+ZUPT EKF, if K1 IMU available)
    // Step 11: World coordinate system (adaptive anchors)
    // Step 12: Result manager (session tracking, JSON/CSV export)
    // Step 13: Web server (Mongoose HTTP+WebSocket, if GUI/--web mode)
}
```

### 5.2 Pipeline State Machine

[pipeline_state.c](src/pipeline_state.c) implements a 5-state FSM enabling GUI-driven pipeline lifecycle management:

```
IDLE ──→ STARTING ──→ RUNNING ──→ STOPPING ──→ IDLE
  ↓         ↓           ↓           ↓
  └─────── ERROR ◄──────┴───────────┘
              │
              └──→ IDLE (recovery)
```

**Valid transitions**: IDLE→STARTING, STARTING→RUNNING, STARTING→ERROR, RUNNING→STOPPING, RUNNING→ERROR, STOPPING→IDLE, STOPPING→ERROR, ERROR→IDLE (recovery).

Thread-safe with `pthread_mutex_t` + `pthread_cond_t` for blocking wait (`psm_wait_for`). Each transition records a timestamp and reason string for diagnostics.

### 5.3 System Controller: Three-Mode Orchestration

[system_controller.c](src/system_controller.c) orchestrates three run modes:

#### GUI Mode (Default)
```
main() → web_server_start() → idle wait loop (pause())
  │
  └── User clicks "Start" in Web UI
        → WebSocket command → psm_transition(STARTING)
        → system_controller_start_async() spawns pipeline thread
        → psm_transition(RUNNING)
        → User clicks "Stop" → psm_transition(STOPPING)
        → system_controller_stop_async() joins pipeline thread
        → psm_transition(IDLE)
```

#### Realtime CLI Mode (`--realtime`)
```
system_controller_process_realtime_k1():
  → Spawn Capture thread (CPU4)
  → Spawn Inference thread (CPU1)
  → Spawn PostProcess thread (CPU0)
  → Spawn Viz thread (CPU6)
  → Terminal UI status line updates
```

#### Offline CLI Mode (`--video PATH`)
```
for each frame in video:
    1. video_processor_read_frame_raw()           → FrameData*
    2. frame_diff_should_process()                → Skip check
    3. inference_pipeline_process_frame()          → InferenceResult
    4. spatial_engine_calculate_position()         → positions[]
    5. object_tracker_update(detections, positions)→ TrackingResult
    6. associate_poses_with_objects()              → IoU match
    7. associate_faces_with_objects()              → IoU match
    8. world_coord_register_person()               → World coordinates
    9. visualizer_render_detection_view()          → vis_buffer
    10. video_writer_write_frame()                 → Output AVI
    11. FPS statistics + TUI progress update
    12. frame_data_destroy()
```

### 5.4 Inference Pipeline: Cascaded AI Model Execution

[inference_pipeline.c](src/inference_pipeline.c) implements cascaded multi-model AI inference:

```
Pipeline Stage                    Model                    Frequency         Backend
────────────────────────────────────────────────────────────────────────────────────
1. Frame Differencing            MAD grid comparison      Every frame       CPU (0.2ms)
2. Person Detection + Pose       YOLOv8-Pose (PRIMARY)    Every frame       SpacemiT EP
3. Face Detection                YOLOv5-Face              Every 5 frames*   SpacemiT EP
4. Face Recognition              ArcFace                  Per detected face SpacemiT EP
5. Action Recognition            ST-GCN                   Every 15 frames   SpacemiT EP
```

*Face detection runs every 5 frames when people are detected, every 60 frames otherwise.

#### 5.4.1 YOLOv8-Pose Unified Detection + Pose Estimation

The PRIMARY model performs BOTH person detection AND 17-keypoint COCO pose estimation in a single forward pass. This eliminates the separate YOLO11n secondary detector, saving ~150ms/frame and one TCM slot.

**xquant-split Format Decoding**: SpacemiT's INT8 quantization splits the model output into per-stride groups, each containing:
- DFL regression head [1, 64, H, W] — 4 coordinates × 16 distribution bins
- Objectness score [1, 1, H, W] — BROKEN by INT8 quantization (all sigmoid ≈0.5)
- Keypoint head [1, 51, H, W] — 17 keypoints × 3 (x, y, confidence)

**DFL Peakiness as Confidence**: Since the classification head is destroyed by INT8 quantization, the system innovatively uses DFL distribution peakiness (ratio of max bin to mean) as the confidence signal. This is the ONLY reliable signal from quantized models.

**Keypoint Anatomical Validation** (3 tiers):
- Tier 1: Full-body (paired shoulders/hips/knees required)
- Tier 2: Upper-body fallback (keypoints 0-10, ≥4 valid)
- Tier 3: Side-body fallback (one-side chain, ≥3 valid)

Partial-body detections are marked but NOT rejected — they're passed to the tracker with `is_partial_body=true` for downstream handling.

**OKS-NMS**: Object Keypoint Similarity NMS replaces standard IoU NMS for pose estimation, using keypoint distances to suppress duplicate detections of the same person.

#### 5.4.2 Face Detection and Recognition Cascade

```c
// Step 1: Full-frame face detection (YOLOv5-Face, 320×320 INT8)
yolov5_face_detector_detect(face_detector, frame, width, height, detected_faces, 20);

// Step 2: Per-face cropping (112×112) + ArcFace feature extraction
for each detected_face:
    face_crop = crop_face(frame, bbox, 112, 112);
    identity = arcface_recognizer_recognize(face_recognizer, face_crop, 112, 112);
    // 128-dimensional feature vector → cosine similarity matching
```

#### 5.4.3 ST-GCN Action Recognition

ST-GCN processes skeleton sequences using spatial-temporal graph convolutions. The INT8 model accepts NTU-RGB+D standard format: 300 frames, 25 keypoints, 2 persons. Keypoint count is auto-converted from COCO-17 to NTU-25 format. Inference runs every ~15 pose frames (≈1.5s at 10 FPS).

#### 5.4.4 Adaptive Cascade State Machine

```
CASCADE_SEARCHING: No confirmed tracks.
  → Run YOLOv8-Pose at full resolution (480×480).
  → Apply full anatomical + OKS-NMS validation.
  → Maximum recall, higher latency.

CASCADE_TRACKING: ≥1 confirmed track.
  → Run YOLOv8-Pose every frame.
  → Reduced resolution option (320×320).
  → New people detected within ~1 second.

CASCADE_VALIDATING: Periodic full check (1 frame every N).
  → Same as SEARCHING: full resolution.
  → Triggered by validation_interval timer or multi-person trigger.
```

### 5.5 Adaptive Frame Differencing

[frame_diff.c](src/frame_diff.c) implements grid-based subsampled MAD (Mean Absolute Difference) for adaptive frame skip — the single most impactful optimization for real-world video:

```
Algorithm:
  1. Divide frame into grid_w × grid_h cells (default 8×8 = 64 cells)
  2. In each cell, sample every subsample-th pixel (default stride=4)
  3. Compute per-channel MAD for each cell
  4. Cell is "changed" if MAD > cell_threshold (default 8.0)
  5. Frame needs processing if changed_ratio > change_threshold (default 0.15)

Activity Classification:
  STATIC      (< 5% cells changed):  skip up to 20 frames
  LOW_MOTION  (5-15% cells changed): skip up to 5 frames
  ACTIVE      (> 15% cells changed): process every frame

Cost:    ~0.2ms per comparison (640×480, stride=4, 8×8 grid)
Savings: ~150ms per skipped frame → up to 10× effective throughput
```

A `force_process_every` (default 30) guard prevents unlimited drift in slowly-changing scenes.

### 5.6 ByteTrack Multi-Object Tracking

[tracking_manager.c](src/tracking_manager.c) implements a ByteTrack-inspired tracker:

#### 5.6.1 7-State Kalman Filter

**State Vector**: x = [cx, cy, area, aspect_ratio, vx, vy, vs]

**Process Model** (constant velocity):
```
F = [[1, 0, 0, 0, dt, 0,  0 ],
     [0, 1, 0, 0, 0,  dt, 0 ],
     [0, 0, 1, 0, 0,  0,  dt],
     [0, 0, 0, 1, 0,  0,  0 ],
     [0, 0, 0, 0, 1,  0,  0 ],
     [0, 0, 0, 0, 0,  1,  0 ],
     [0, 0, 0, 0, 0,  0,  0 ]]
```

**Measurement Model**: z = [cx, cy, area, aspect_ratio] (4 observed, 7 state)

#### 5.6.2 Cascade Matching + Hungarian Algorithm

The tracker uses cascade matching (grouping tracks by time-since-update, matching freshest first) with the Kuhn-Munkres Hungarian algorithm for optimal global assignment. The cost matrix combines:
- **IoU (80% weight)**: Spatial overlap between predicted and detected boxes
- **Appearance (20% weight)**: Cosine distance between feature vectors

#### 5.6.3 Key Features

- **Re-identification pool**: Deleted tracks retained for 60 frames for re-association
- **Occlusion handling**: Upper-body (≥3 keypoints) and side-body (≥2 keypoints) fallback
- **New-person grace period**: 1 frame before confirming a new track
- **EMA smoothing** (α=0.35): Spatial coordinate smoothing for reduced jitter
- **Spatial jump gate** (5.0m): Reject physically impossible position jumps

### 5.7 3D Spatial Localization

[spatial_engine.c](src/spatial_engine.c) implements monocular depth estimation and 3D coordinate computation.

#### 5.7.1 Multi-Point Anatomical Depth Sampling

Instead of using only the bounding box height (which assumes standing posture), the system samples depth from multiple anatomical keypoints:

```
Depth estimation:
  Z = fy × Havg / h_bbox                    (bounding box height method)
  Z_kpt = fy × Hsegment / kpt_pair_distance  (per-keypoint-pair method)

Fusion:
  Collect depth estimates from shoulder-hip, hip-knee, shoulder-shoulder pairs
  MAD (Median Absolute Deviation) outlier rejection (2.5× multiplier)
  Weighted average of inlier estimates
  EMA smoothing (α=0.30) across frames per tracked person
```

#### 5.7.2 Pinhole Camera Model

**3D Back-projection** (pixels → camera coordinate system):
```
X_cam = (u - cx) × Z / fx
Y_cam = (v - cy) × Z / fy
Z_cam = Z
```

**IMU Pitch Angle Depth Correction**:
```
Z_corrected = Z × cos(θ_pitch)
```

### 5.8 K1 Odometry: INS + ZUPT EKF

[k1_odometry.c](src/k1_odometry.c) implements strapdown inertial navigation with zero-velocity update:

```
Algorithm:
  1. TRIAD gravity alignment initialization (2s stationary)
  2. Quaternion attitude integration (first-order)
  3. Accelerometer world-frame transform + gravity compensation
  4. Velocity/position integration
  5. GLRT zero-velocity detection (5-sample sliding window)
  6. ZUPT EKF error correction (9-state: pos×3, vel×3, attitude×3)
```

**GLRT Statistic** (Skog et al., IEEE TBME 2010):
```
T(z_accel, z_gyro) = (1/N) × Σ[ ||a_k - g·ā/||ā|| ||² / σ_a² + ||ω_k||² / σ_w² ]
```
- Per-sample baseline: ~25.5 (accel) + 0.5 (gyro) ≈ 26
- 5-sample window baseline: ~130
- Threshold 500: ~4× headroom above baseline, catches slow walking (<1 m/s)

**Velocity safety clamp**: 5 m/s (18 km/h) — physically meaningful for pedestrian/mobile-robot speeds.

### 5.9 World Coordinate System

[world_coord.c](src/world_coord.c) maintains persistent world-coordinate positions of detected persons relative to the moving K1 platform:

**Transform Chain**:
```
P_world = T_W_B(t) × T_B_C × P_cam
```
where:
- `T_W_B(t)` = K1Odometry real-time pose (R_W_B, p_W_B)
- `T_B_C` = K1→camera extrinsic (Wahba alignment quaternion + mounting offset)
- `P_cam` = depth estimation + inverse projection camera coordinates

**Adaptive Anchor Strategy** (SEVIS-3D inspired):
- First detection: fix anchor at projected world position
- Stationary K1 (motion < 0.30m): lock anchor — prevent odometry drift contamination
- Moving K1: slow momentum update (α=0.15) for smooth anchor tracking
- Person timeout: 60s after last sighting

### 5.10 Madgwick AHRS Filter

[madgwick_filter.c](src/madgwick_filter.c) (in `src/` via IMU handler) implements the Madgwick orientation filter, fusing accelerometer, gyroscope, and magnetometer data to estimate device attitude with a single tunable parameter β=0.08.

### 5.11 CoAP/UDP Communication Protocol

#### 5.11.1 Endpoints

| Endpoint | Method | Direction | Content | Frequency |
|---|---|---|---|---|
| `/stream` | GET | K1→ESP (sub), ESP→K1 (push) | Block2 NON blocks, pure JPEG | ~15 FPS |
| `/imu` | GET | K1→ESP (poll) | JSON `{"ax":...,"ay":...,"az":...,"gx":...,"gy":...,"gz":...}` | 1 Hz |
| `/servo` | PUT | K1→ESP (control) | JSON `{"angle": N}` (0=CCW, 90=STOP, 180=CW) | On demand |

#### 5.11.2 Block2 Transfer

JPEG frames transmitted using CoAP Block2 (RFC 7959) block-wise transfer with adaptive pacing (4-12 blocks/tick, 0-2ms inter-block gap).

### 5.12 Web UI System

[web_server.c](src/web_server.c) + [web/index.html](web/index.html) provide a complete browser-based control interface:

**Architecture**:
```
┌──────────────┐   HTTP GET /app/*   ┌──────────────┐
│   Browser    │◄───────────────────│   Mongoose   │
│  (Three.js)  │                    │   HTTP+WS    │
│              │   WebSocket /ws    │   Server     │
│  3D Render   │◄──────────────────►│  (pthread)   │
└──────────────┘                    └──────┬───────┘
                                           │
                              Frame JSON ring buffer
                              (32 slots, mutex-protected)
                                           │
                              ┌────────────▼───────┐
                              │  PostProcess Thread │
                              └────────────────────┘
```

**Features**:
- Pipeline start/stop control from browser
- Real-time WebSocket frame streaming (JSON + binary JPEG)
- 3D world-coordinate visualization (Three.js)
- Bounding box + skeleton overlay on video
- Status indicators (WebSocket connection, pipeline state, FPS)
- Ceramic-white design system, Chinese localization

### 5.13 Terminal UI System

[terminal_ui.c](src/terminal_ui.c) provides a rich CLI interface with automatic TTY/CI/pipe detection:

| Mode | Trigger | Features |
|---|---|---|
| **HUMAN** | Interactive TTY | Unicode box-drawing, ANSI colors, spinners, progress bars |
| **PLAIN** | CI, pipe, `--quiet` | ASCII only, no colors, minimal output |
| **MACHINE** | `--json` | JSON Lines (`{"type":"...","data":{...}}`) for scripting |

**UI Components**: Banner, section headers, checklist spinners (model loading), progress bars (offline video), status lines (realtime mode), key-value pairs, intro/outro summaries.

### 5.14 Visualization Rendering Pipeline

CPU-based 2D rendering (in `system_controller.c` and related modules):
- Bounding boxes: Bresenham line algorithm ×4 edges, variable line width
- Skeleton lines: COCO-17 bone connections with per-track color
- Trajectory lines: Historical position trace with fading
- Text: 5×7 pixel bitmap font, per-character per-bit rendering
- Top-down view: Spatial position projection on overhead map
- Colors: 8-track rotating color palette (configurable)

### 5.15 SpacemiT Execution Provider Call Chain

```
ort_global_init()
    │
ort_create_session(model_path, intra_threads)
    │
CreateSessionOptions
    ├── SetGraphOptimizationLevel(ORT_ENABLE_ALL)
    ├── SetIntraOpNumThreads(intra_threads)
    ├── spacemit_ort_session_options_init()  ← C++ bridge
    │       └── SessionOptionsSpaceMITEnvInit(opts)
    │             └── Register libspacemit_ep.so
    └── CreateSession(model_path)
            └── Auto-load libspacemit_ep.so
                 → RVV 1.0 + IME hardware-accelerated inference
```

The [C→C++ bridge](src/spacemit_ort_bridge.cpp) exports C-compatible interfaces through `extern "C"`, auto-detecting INT8-quantized models (scans for QuantizeLinear/QLinearConv ops) and only registering the EP for quantized models. FP32 models silently use CPU EP.

---

## 6. Usage Guide

### 6.1 Environment Requirements

| Dependency | Minimum Version | Description |
|---|---|---|
| GCC (riscv64) or Clang | GCC 13.2+ / Clang 18+ | K1 native/cross-compilation. Clang recommended for X60 (no SIGBUS risk). |
| CMake | ≥3.16 | Build system |
| SpacemiT ONNX Runtime | 2.0.2 | [Official Download](https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.2.tar.gz) |
| libjpeg-turbo | ≥2.0 | REQUIRED for JPEG decode |
| ESP-IDF | v5.1+ | ESP32 firmware compilation (Arrow End only) |
| Python 3 | ≥3.8 | Development/debugging utilities |

### 6.2 Quick Start (K1 Native Build)

```bash
# Step 1: Install system dependencies
sudo apt install build-essential cmake libturbojpeg0-dev linux-libc-dev

# Step 2: Install SpacemiT ONNX Runtime
wget https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.2.tar.gz
tar xzf spacemit-ort.riscv64.2.0.2.tar.gz -C /opt/
sudo cp /opt/spacemit-ort.riscv64.2.0.2/lib/*.so /usr/lib/
sudo ldconfig
sudo cp -r /opt/spacemit-ort.riscv64.2.0.2/include/* /usr/local/include/

# Step 3: Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Step 4: Configure hardware access
sudo chmod 777 /dev/tcm          # IME acceleration
sudo chmod 777 /dev/mpp_service  # VPU hardware encode

# Step 5: Run (GUI mode — open http://localhost:8080 in browser)
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
./lingqi_tantong
```

### 6.3 Build for Different Platforms

```bash
# K1 native build (auto-detects RISC-V, enables RVV + K1 Pipeline)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# RISC-V cross-compilation (x86 → K1)
cmake -B build_riscv \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
  -DBIANBU_SYSROOT=/opt/bianbu-sysroot \
  -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_riscv -j$(nproc)

# x86 Linux development build (ONNX Runtime CPU EP only, no SpacemiT EP)
cmake -B build -DONNX_RUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j$(nproc)

# K1 with Clang (recommended — better X60 support, avoids SIGBUS)
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 6.4 Command-Line Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--video PATH` | string | — | Offline video file processing (CLI mode) |
| `--realtime` | flag | false | K1 dual-cluster realtime pipeline (CLI mode) |
| `--web [PORT]` | int | 8080 | Web UI port (implied in GUI mode) |
| `--config PATH` | string | `configs/default.yaml` | YAML configuration file |
| `--output PATH` | string | `output` | Output directory |
| `--max-frames N` | int | `0` | Frame limit (0=unlimited) |
| `--save-interval N` | int | `0` | Save frame every N frames |
| `--pose-model NAME` | string | `yolov8n-pose` | Pose model variant: `yolov8n-pose` or `yolo11n-pose` |
| `--camera DEV` | string | `/dev/video1` | Camera device (K1: /dev/video1=CCIC1 MIPI CSI) |
| `--log-level LVL` | string | `info` | trace/debug/info/warn/error |
| `--json` | flag | false | Machine-readable JSON Lines output |
| `--quiet` | flag | false | Minimal ASCII output |
| `--coap` | flag | false | Enable CoAP/UDP receiver (ESP32 WiFi) |
| `--coap-ip IP` | string | `192.168.4.1` | ESP32 CoAP server IP |
| `--coap-port PORT` | int | `5683` | CoAP/UDP port |
| `--wifi-ssid SSID` | string | `ESP32-Camera-AP` | WiFi SSID |
| `--wifi-password PW` | string | `12345678` | WiFi password |
| `--save-video` | flag | false | Save output to MP4 video file |
| `--help` | flag | — | Display help information |

### 6.5 Running Examples

```bash
# GUI Mode (default) — browser-based control
./lingqi_tantong
./lingqi_tantong --web 9000                          # Custom port

# Offline video processing (CLI mode)
./lingqi_tantong --video test_video.mp4
./lingqi_tantong --video test.mp4 --max-frames 500 --save-interval 1

# K1 realtime pipeline with CoAP WiFi
./lingqi_tantong --realtime --coap --config configs/default.yaml

# JSON machine output (for scripting)
./lingqi_tantong --video test.mp4 --json --quiet --max-frames 100

# With custom pose model variant
./lingqi_tantong --video test.mp4 --pose-model yolo11n-pose
```

### 6.6 systemd Service Deployment (Production)

```ini
# /etc/systemd/system/lingqi.service
[Unit]
Description=LingQi TanTong Edge AI Pipeline
After=network.target

[Service]
Type=simple
ExecStart=/opt/lingqi/lingqi_tantong --web 8080 --config /opt/lingqi/configs/default.yaml
Restart=on-failure
RestartSec=5
User=root
WorkingDirectory=/opt/lingqi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable lingqi
sudo systemctl start lingqi
sudo systemctl status lingqi
```

### 6.7 Model Files

Models must be placed in the `models/` directory with this structure:

```
models/
├── Human Recognition/
│   └── yolo11n.q.onnx                          # YOLO11n secondary detector (optional)
├── Face Recognition/
│   ├── yolov5n-face_320_cut.q.onnx             # YOLOv5-Face face detection (INT8)
│   └── arcface_mobilefacenet_cut.q.onnx        # ArcFace feature extraction (INT8)
└── Action Prediction/
    ├── Skeleton Recognition/
    │   ├── yolov8n-pose.q.onnx                 # YOLOv8-Pose PRIMARY (INT8)
    │   └── yolo11n-pose.q.onnx                 # YOLO11n-Pose alternative (INT8)
    └── Skeleton-based Action Prediction/
        └── stgcn_int8.onnx                     # ST-GCN action recognition (INT8)
```

---

## 7. Configuration Options

### 7.1 Configuration File Format

The system uses a YAML-style key-value configuration file ([configs/default.yaml](configs/default.yaml)) with a built-in minimal YAML parser. Supported features: scalar values (string/int/float/bool), nested mappings (up to 3 levels), sequences (bracket lists). **Unsupported**: anchors/aliases, multiline strings (`|`/`>`), complex tags.

### 7.2 Complete Configuration Reference

#### 7.2.1 System `system`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_onnx` | bool | `true` | Enable ONNX Runtime inference |
| `use_spacemit_ep` | bool | `true` | Enable SpacemiT EP (RVV 1.0 + IME). Only works for INT8-quantized models. |
| `spacemit_ep_intra_threads` | int | `1` | Per-session intra-op threads. 1 maximizes EP model coverage on 512KB TCM. |
| `log_level` | string | `INFO` | DEBUG/INFO/WARN/ERROR |
| `startup_mode` | string | `realtime` | offline/realtime/benchmark |
| `max_frames` | int | `0` | Max frames (0=unlimited) |
| `ring_buffer_size` | int | `16` | Pipeline ring buffer slots |
| `target_fps` | float | `15.0` | Target frame rate |
| `worker_threads` | int | `4` | Worker thread count |

#### 7.2.2 Video `video`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | `camera` | camera/file/arrow |
| `camera_device` | string | `/dev/video1` | K1: /dev/video0=VPU, /dev/video1=CCIC1 MIPI CSI |
| `camera_width` | int | `640` | Capture width |
| `camera_height` | int | `480` | Capture height |
| `camera_fps` | float | `15.0` | Capture frame rate |
| `camera_format` | string | `MJPEG` | Capture pixel format |
| `camera_buffer_count` | int | `4` | V4L2 buffer count |
| `save_frame_interval` | int | `1` | Output video save interval |

#### 7.2.3 CoAP `coap`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable CoAP/UDP receiver |
| `esp_ip` | string | `192.168.4.1` | ESP32 WiFi AP IP |
| `esp_port` | int | `5683` | CoAP/UDP port |
| `wifi_ssid` | string | `ESP32-Camera-AP` | WiFi SSID |
| `wifi_password` | string | `12345678` | WiFi password |
| `frame_timeout_s` | int | `10` | Auto-exit after N seconds idle |

#### 7.2.4 Detection `detection`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | string | `ai_accel` | Inference backend |
| `confidence_threshold` | float | `0.12` | Detection confidence threshold |
| `iou_threshold` | float | `0.40` | NMS IoU threshold |
| `input_size` | int[2] | `[640, 640]` | Model input size |
| `cascade_enabled` | bool | `false` | Enable cascade state machine |
| `cascade_validation_interval` | int | `15` | Full-check interval |
| `keypoint_min_count` | int | `2` | Minimum visible keypoints |
| `keypoint_min_confidence` | float | `0.05` | Keypoint confidence floor |

#### 7.2.5 Pose `pose`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | string | `ai_accel` | Inference backend |
| `model_variant` | string | `yolov8n-pose` | `yolov8n-pose` or `yolo11n-pose` |
| `confidence_threshold` | float | `0.04` | Ultra-low for INT8 model |
| `iou_threshold` | float | `0.55` | OKS-NMS IoU threshold |
| `input_size` | int[2] | `[480, 480]` | Model input size |
| `num_keypoints` | int | `17` | COCO keypoint count |
| `skeleton_type` | string | `coco` | Keypoint layout |

#### 7.2.6 Face `face`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detection_backend` | string | `ai_accel` | Face detection backend |
| `recognition_backend` | string | `ai_accel` | Face recognition backend |
| `confidence_threshold` | float | `0.50` | Face detection threshold |
| `similarity_threshold` | float | `0.55` | Identity matching threshold |
| `input_size` | int[2] | `[320, 320]` | Face detection input |
| `embedding_dim` | int | `128` | ArcFace feature vector dimension |
| `liveness_detection` | bool | `false` | Anti-spoofing |

#### 7.2.7 Action `action`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable action recognition |
| `backend` | string | `ai_accel` | Inference backend |
| `confidence_threshold` | float | `0.50` | Action confidence threshold |
| `inference_interval` | int | `15` | Run every N pose frames |

#### 7.2.8 Tracking `tracking`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_lost` | int | `30` | Max frames before track deletion |
| `max_occluded_frames` | int | `45` | Max occlusion tolerance |
| `min_iou` | float | `0.18` | Minimum IoU for match |
| `max_distance` | float | `40` | Max association distance (pixels) |
| `max_track_history` | int | `300` | Trajectory history length |
| `confirmation_frames` | int | `1` | Hits to confirm new track |
| `ema_alpha` | float | `0.35` | EMA smoothing coefficient |
| `appearance_weight` | float | `0.20` | Appearance cost weight |
| `reid_pool_max_age` | int | `60` | Re-ID pool retention |
| `spatial_jump_max_m` | float | `5.0` | Max position jump (meters) |
| `new_person_grace_frames` | int | `1` | New person confirmation delay |

#### 7.2.9 Spatial Localization `spatial`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fx` | float | `650.0` | Horizontal focal length |
| `fy` | float | `650.0` | Vertical focal length |
| `avg_human_height` | float | `1.70` | Assumed average height (m) |
| `min_depth` | float | `0.3` | Minimum depth (m) |
| `max_depth` | float | `120.0` | Maximum depth (m) |
| `imu_pitch_correction` | bool | `true` | IMU pitch correction |
| `depth_ema_alpha` | float | `0.30` | Depth EMA smoothing |
| `depth_outlier_mad_mult` | float | `2.5` | MAD outlier multiplier |
| `depth_min_keypoint_conf` | float | `0.25` | Min keypoint confidence for depth |

#### 7.2.10 K1 IMU `k1_imu`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable K1 onboard IMU (ICM-20948) |
| `i2c_bus` | int | `4` | I²C bus number |
| `sample_rate_hz` | int | `100` | IMU sampling rate |
| `calibration_samples` | int | `200` | Bias calibration samples |

#### 7.2.11 Odometry `odometry`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable K1 INS odometry |
| `init_duration_s` | float | `2.0` | Gravity alignment duration |
| `zupt_accel_thresh` | float | `0.15` | ZUPT accelerometer threshold (m/s²) |
| `zupt_gyro_thresh` | float | `0.15` | ZUPT gyroscope threshold (rad/s) |
| `zupt_window_size` | int | `5` | GLRT window size |
| `sigma_accel` | float | `0.01` | Accelerometer noise σ (m/s²) |
| `sigma_gyro` | float | `0.015` | Gyroscope noise σ (rad/s) |
| `glrt_threshold` | float | `500.0` | GLRT statistic threshold |
| `export_csv` | bool | `true` | Export trajectory CSV |

#### 7.2.12 World Coordinates `world_coord`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable world coordinate system |
| `max_persons` | int | `64` | Max tracked persons |
| `person_timeout_s` | float | `60.0` | Person removal timeout |
| `skeleton_fixed_ratio` | bool | `true` | Fixed-ratio skeleton 3D |
| `export_csv` | bool | `true` | Export world coordinate CSV |
| `anchor_motion_threshold` | float | `0.30` | Anchor motion detection (m) |
| `anchor_moving_alpha` | float | `0.15` | Moving anchor momentum |

#### 7.2.13 AR Render `ar_render`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skeleton_3d_mode` | bool | `true` | World-coordinate skeleton projection |
| `distance_labels` | bool | `true` | Show K1-relative distance labels |
| `show_k1_compass` | bool | `true` | Show K1 compass (yaw indicator) |
| `show_k1_speed` | bool | `true` | Show K1 speed bar (km/h) |

#### 7.2.14 Frame Differencing `frame_diff`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch |
| `grid_w` | int | `8` | Horizontal grid cells |
| `grid_h` | int | `8` | Vertical grid cells |
| `subsample` | int | `4` | Pixel stride within cell |
| `cell_threshold` | float | `8.0` | MAD per channel threshold |
| `change_threshold` | float | `0.15` | Changed cell ratio trigger |
| `adaptive_enabled` | bool | `true` | Adaptive skip limits |
| `max_static_skip` | int | `20` | Max skips in static scenes |
| `max_low_motion_skip` | int | `5` | Max skips in low-motion |
| `force_process_every` | int | `30` | Force inference every N frames |

#### 7.2.15 Visualization `visualization`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `show_info_bar` | bool | `true` | Info bar overlay |
| `corner_markers` | bool | `true` | Corner markers |
| `crosshair` | bool | `true` | Center crosshair |
| `render_size` | int[2] | `[1920, 1080]` | Render resolution |
| `line_width` | int | `2` | Box line width |
| `show_ids` | bool | `true` | Show track IDs |
| `show_trajectory` | bool | `true` | Show motion trails |
| `trajectory_length` | int | `45` | Trail length (frames) |
| `draw_skeleton` | bool | `true` | Draw COCO skeleton |
| `show_fps` | bool | `true` | FPS counter |
| `show_depth` | bool | `true` | Depth values |
| `show_imu_overlay` | bool | `true` | IMU data overlay |
| `record_to_video` | bool | `true` | Save output video |

#### 7.2.16 K1 Hardware `k1_hardware`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable K1 acceleration |
| `cluster0_ai_cores` | string | `"0-3"` | AI inference cores |
| `cluster1_io_cores` | string | `"4-7"` | I/O task cores |
| `tcm.enabled` | bool | `true` | TCM tightly-coupled memory |
| `tcm.tcm_device` | string | `/dev/tcm` | TCM device node |
| `tcm.tcm_size_kb` | int | `512` | TCM capacity |
| `vpu.enabled` | bool | `true` | VPU hardware encode |
| `gpu.enabled` | bool | `true` | GPU (flag only) |
| `pipeline.ring_buffer_size` | int | `4` | K1 pipeline slots |
| `pipeline.capture_thread_cpu` | int | `4` | Capture thread core |
| `pipeline.infer_thread_cpu` | int | `1` | Inference thread core |
| `pipeline.postprocess_thread_cpu` | int | `0` | PostProcess thread core |
| `pipeline.viz_thread_cpu` | int | `6` | Viz thread core |
| `power.governor` | string | `performance` | CPU frequency governor |

#### 7.2.17 Performance `performance`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rvv_enabled` | bool | `true` | RVV vectorization |
| `rvv_vlen` | int | `256` | Vector length (bits) |
| `openmp_threads` | int | `6` | OpenMP thread count |
| `pipeline_pin_to_core` | bool | `true` | Thread core pinning |
| `realtime_scheduling` | bool | `true` | SCHED_FIFO scheduling |
| `rt_priority` | int | `50` | Real-time priority |

---

## 8. Troubleshooting Guide

### 8.1 Compilation Errors

| Error | Cause | Solution |
|---|---|---|
| `fatal error: onnxruntime_c_api.h` | ORT not installed | Install SpacemiT ORT 2.0.2, set `-DONNX_RUNTIME_DIR` |
| `undefined reference to '__atomic_*'` | Missing `-latomic` | CMake auto-adds on RISC-V; check toolchain |
| `libjpeg-turbo is REQUIRED` | turbojpeg not found | `sudo apt install libturbojpeg0-dev` |
| `V4L2 header not found` | linux-libc-dev missing | `sudo apt install linux-libc-dev` |
| `libmpp.so not found` | K1 MPP SDK not installed | `sudo apt install k1x-mpp` or set `-DK1_MPP_DIR` |
| `spacemit_ort_env.h not found` | SpacemiT EP headers missing | Confirm in `/usr/local/include/onnxruntime/` |
| `libspacemit_ep.so not found` | EP library not in search path | Install `spacemit-onnxruntime` package |

### 8.2 Runtime Errors

| Error | Cause | Solution |
|---|---|---|
| `Failed to load model` | Model file missing or wrong path | Verify `models/` directory structure |
| `tcm buffer alloc failed` | TCM exhausted (INT8 model fed as FP32) | Use xquant INT8-quantized models (.q.onnx) |
| `Segmentation fault` | Memory or null pointer | Build with Debug, use Address Sanitizer |
| `Cannot open /dev/tcm: Permission denied` | IME device permissions | `sudo chmod 777 /dev/tcm` |
| `CoAP receive timeout` | ESP32 WiFi disconnected | Check ESP32 AP, K1 WiFi connection |
| `SIGBUS on K1` | Misaligned vector instruction (GCC <15 + X60) | Use Clang, or disable `-ftree-vectorize` |
| `Web UI: failed to start on port` | Port already in use | Use `--web 9000` for alternative port |

### 8.3 Performance Issues

| Symptom | Diagnostic | Solution |
|---|---|---|
| FPS < 5 (K1) | Check `spacemit_ep_intra_threads` | Set to 1 for max EP model coverage |
| Frame time >500ms | Benchmark mode analysis | Reduce resolution, disable face recognition |
| Memory growing | `htop` RSS monitor | Check FrameData destroy per frame |
| IMU attitude jitter | Adjust Madgwick `beta` | Lower β to 0.04-0.06 |
| ZUPT never fires | Check GLRT threshold | Lower to 500 (was 300,000 in earlier versions) |

### 8.4 Frequently Asked Questions

**Q: How to verify SpacemiT EP is correctly enabled?**

A: Check CMake output for `SpacemiT EP: YES (RVV 1.0 + IME hardware acceleration)`. At runtime, check logs for successful `SessionOptionsSpaceMITEnvInit` calls. The EP auto-detects INT8-quantized models; FP32 models silently fall back to CPU EP.

**Q: Why does K1 use Clang instead of GCC?**

A: GCC <15 on X60 may emit misaligned vector load/store instructions that cause SIGBUS because X60 lacks Zicclsm hardware support. Clang handles this correctly. GCC 15+ should also be safe.

**Q: What is the TCM allocation strategy?**

A: K1 Cluster0 has 512KB shared TCM. With `spacemit_ep_intra_threads=1` (~170KB per EP session), the system fits 3 EP models: YOLOv8-Pose, YOLOv5-Face, and ArcFace (total ~510KB). ST-GCN uses CPU EP (FP32, not quantized).

**Q: How does the DFL peakiness confidence work?**

A: SpacemiT's xquant INT8 quantization destroys the YOLO classification head (all sigmoid outputs ≈0.5). Instead, the system computes the ratio of the max DFL bin value to the mean across all 16 bins. A peaked distribution (one bin much higher than others) indicates a confident bounding box regression, which serves as a proxy for detection confidence.

---

## 9. Performance Metrics

### 9.1 Expected Performance (K1 X60 + SpacemiT EP 2.0.2)

| Model | CPU-only (FP32) | SpacemiT EP (INT8) |
|---|---|---|
| YOLOv8-Pose (480×480) | ~800ms | **~100ms** |
| YOLOv5-Face (320×320) | ~300ms | **~40ms** |
| ArcFace (112×112) | ~100ms | **~15ms** |
| ST-GCN (300 frames) | ~200ms | **~60ms** |

> **Note**: These are estimates based on similar RISC-V platforms. The INT8→FP32 speedup is expected to be 3-5× due to IME matrix instruction optimizations for INT8.

### 9.2 Memory Usage Analysis

| Component | Estimated Usage | Notes |
|---|---|---|
| Frame buffer (640×480×3) | ~0.9 MB | Single frame RGB |
| CoAP assembly buffer | ~0.13 MB | 128KB Block2 reassembly |
| Tracker (40 targets × ~3.5KB) | ~0.14 MB | Including Kalman state |
| Trajectory history (40 × 60 points) | ~0.03 MB | SpatialPosition |
| ONNX Runtime (4 models loaded) | ~400-600 MB | Runtime peak |
| **Total (estimated peak)** | **< 800 MB** | Within K1 4GB constraint |

### 9.3 Optimization Strategies

1. **Model INT8 Quantization** (Highest Priority): SpacemiT EP's IME instructions provide 3-5× acceleration for INT8 models. Use SpacemiT's xquant tool for quantization.

2. **Adaptive Frame Skip**: Frame differencing saves ~150ms per skipped frame at ~0.2ms cost. STATIC scenes achieve up to 10× effective throughput.

3. **Unified Detection+Pose**: Using YOLOv8-Pose as PRIMARY eliminates the separate YOLO11n secondary detector, saving one full EP inference (~100ms) and one TCM slot per frame.

4. **K1 Thread Core Binding**: Pinning threads to specific cores (described in config) maximizes cache locality and minimizes cross-cluster traffic.

5. **System-Level Tuning**:
   ```bash
   sudo chmod 777 /dev/tcm
   sudo cpufreq-set -g performance
   export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
   ```

---

## 10. Limitations and Future Work

### 10.1 Current Known Limitations

1. **Depth Estimation Limited by Monocular Prior**: The spatial engine uses height-prior depth estimation (Z = fy × Havg / h_bbox), producing 30-50% errors for non-standing poses. Multi-point anatomical sampling + MAD fusion improves this but does not fully resolve it. MiDaS depth estimation integration would significantly improve accuracy.

2. **Yaw Unobservability in ZUPT EKF**: The K1 odometry's yaw angle is completely unobservable from velocity-only ZUPT measurements (formally proven by Ilyas et al., 2016). Gyro Z-axis drift causes ~1-5°/min yaw divergence. The WorldCoord adaptive anchor strategy mitigates this for short runs (<60s).

3. **CPU-Only Visualization**: All rendering (bounding boxes, skeleton lines, text) is performed on CPU. At 1920×1080 this takes 40-80ms. Migration to OpenGL ES 3.2 could reduce this to <2ms.

4. **AVI Video Output Uncompressed**: Raw RGB24 format generates ~800MB/min at 640×480@15fps. K1 VPU H.264/H.265 hardware encoding integration would dramatically reduce file sizes.

5. **ATW Not GPU-Accelerated**: Asynchronous Time Warp currently performs O(W×H) per-pixel transforms on CPU (>200ms at 1080p), far exceeding the MTP ≤17.8ms threshold.

6. **No Visual-Inertial Odometry**: When the K1 camera moves, target world coordinates experience drift without VINS-Mono style visual-inertial bundle adjustment.

7. **Test Coverage Gaps**: Limited to unit tests. Missing ONNX inference integration tests, Arrow protocol end-to-end tests, K1 hardware benchmarks, and long-running memory leak detection.

8. **WiFi Range Limitation**: ESP32 ↔ K1 uses CoAP/UDP over WiFi. Longer-range radio links (LoRa, UART direct) not implemented.

### 10.2 Development Roadmap

#### v2.6 (Near-term)
- [ ] K1 hardware benchmarking (`onnxruntime_perf_test -e spacemit`)
- [ ] EP vs CPU-only real-world comparison report
- [ ] RVV hand-written vectorization: letterbox, NMS, matrix multiply critical paths
- [ ] JPU hardware JPEG decode integration for Arrow UART frames
- [ ] Web UI 3D skeleton rendering in world coordinate view

#### v3.0 (Mid-term)
- [ ] VINS-Mono visual-inertial odometry (IMU pre-integration + sliding window BA)
- [ ] MiDaS depth estimation ONNX INT8 replacement
- [ ] OpenGL ES 3.2 GPU visualization pipeline
- [ ] K1 VPU H.264/H.265 hardware video encoding
- [ ] ATW asynchronous time warp GPU fragment shader

#### v4.0 (Long-term)
- [ ] Multi-camera sensor fusion (Arrow + Shooter cameras)
- [ ] LoRa long-range telemetry link
- [ ] GLTF 2.0 3D scene export
- [ ] CI/CD with K1 hardware-in-the-loop
- [ ] Unit test coverage ≥80%

### 10.3 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| SpacemiT EP SDK instability | Medium | Blocks AI acceleration | Fallback to ONNX Runtime CPU EP |
| GCC X60 misaligned vector SIGBUS | Medium | Random crashes | Use Clang, or GCC 15+, or disable tree-vectorize |
| ESP32 WiFi congestion at high FPS | Medium | Block2 packet loss | Lower JPEG quality, increase block pacing |
| K1 4GB memory insufficient for all models | Low | System OOM | Model lazy loading, LRU eviction |
| TCM allocation failure with 4 EP models | Medium | EP session creation fails | Limit to 3 EP models, FP32 fallback for 4th |

---

## 11. References

### Academic Papers

1. Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637-646.
2. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR 2001*.
3. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR 2005*.
4. Girshick, R., et al. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. *CVPR 2014*.
5. Girshick, R. (2015). Fast R-CNN. *ICCV 2015*.
6. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN. *NIPS 2015*.
7. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR 2016*.
8. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV 2016*.
9. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR 2019*.
10. Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and Computation Redistribution for Efficient Face Detection. *ICLR 2022*.
11. Bewley, A., et al. (2016). Simple Online and Realtime Tracking. *ICIP 2016*.
12. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP 2017*.
13. Zhang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV 2022*.
14. Mahony, R., Hamel, T., & Pflimlin, J. M. (2008). Nonlinear Complementary Filters on the Special Orthogonal Group. *IEEE TAC*, 53(5).
15. Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011). Estimation of IMU and MARG orientation using a gradient descent algorithm. *ICORR 2011*.
16. Skog, I., et al. (2010). Zero-Velocity Detection — An Algorithm Evaluation. *IEEE TBME*, 57(11).
17. Black, H. (1964). A Passive System for Determining the Attitude of a Satellite. *AIAA Journal*, 2(7). [TRIAD algorithm]
18. Ilyas, M., et al. (2016). Drift Reduction in Pedestrian Navigation. *Sensors*, 16(5). [Yaw unobservability in ZUPT EKF]
19. Savage, P. G. (2000). *Strapdown Analytics*. Strapdown Associates.

### Technical Specifications and Standards

1. RISC-V International. (2021). *RISC-V "V" Vector Extension Specification Version 1.0*.
2. Microsoft Corporation. (2023). *ONNX Runtime*.
3. Ultralytics. (2023). *Ultralytics YOLOv8*.
4. Khronos Group. (2021). *GLTF™ 2.0 Specification*.

### Hardware Platform Documentation

1. SpacemiT. (2024). *SpacemiT K1/M1 Key Stone AI CPU Chip Overview*.
2. SpacemiT. (2024). *Bianbu OS — SpacemiT K1 Official Linux Distribution*.
3. SpacemiT. (2024). *ONNX Runtime with SpacemiT Execution Provider*.
4. SpacemiT. (2024). *Model Deployment on Bianbu — C++ Inference Example*.
5. Espressif Systems. (2024). *ESP32 Technical Reference Manual*.
6. Espressif Systems. (2024). *ESP-IDF Programming Guide*.
7. Banana Pi. (2024). *BPI-F3 SpacemiT K1 Datasheet*.

---

## Appendix A: Project File Listing

```
lingqi_tantong/
├── include/                                   # Header files (32)
│   ├── core_types.h                           #   Core data types: BBox, Detection, Pose, Face,
│   │                                          #     TrackedObject, InferenceResult, etc.
│   ├── system_controller.h                    #   System main controller interface
│   ├── pipeline_state.h                       #   5-state pipeline FSM (IDLE→STARTING→RUNNING→STOPPING→ERROR)
│   ├── inference_pipeline.h                   #   AI inference pipeline (4-model cascade)
│   ├── tracking_manager.h                     #   ByteTrack-style multi-object tracking
│   ├── spatial_engine.h                       #   3D spatial localization (pinhole model)
│   ├── world_coord.h                          #   World coordinate system (K1 dynamic origin)
│   ├── k1_odometry.h                          #   K1 INS+ZUPT EKF odometry
│   ├── k1_imu.h                               #   K1 onboard IMU driver (ICM-20948, I²C)
│   ├── k1_platform.h                          #   K1 platform detection + capability query
│   ├── yolov8_pose_estimator.h                #   YOLOv8-Pose unified detection + 17-keypoint pose
│   ├── yolov5_face_detector.h                 #   YOLOv5-Face face detection (320×320 INT8)
│   ├── arcface_recognizer.h                   #   ArcFace face recognition (128-d embedding)
│   ├── stgcn_action_recognizer.h              #   ST-GCN skeleton-based action recognition
│   ├── yolo_postprocess.h                     #   YOLO output decoding (DFL, NMS, grid assembly)
│   ├── keypoint_validator.h                   #   Keypoint anatomical validation (3-tier)
│   ├── frame_diff.h                           #   Adaptive frame differencing (grid MAD)
│   ├── video_processor.h                      #   Video frame reading (MP4/V4L2/CoAP)
│   ├── video_writer.h                         #   AVI video output writing
│   ├── imu_handler.h                          #   IMU data processing (sliding window smoothing)
│   ├── coap_receiver.h                        #   CoAP/UDP receiver (Block2 JPEG + IMU poll)
│   ├── web_server.h                           #   Embedded HTTP+WebSocket server (Mongoose)
│   ├── terminal_ui.h                          #   Terminal UI (3-mode: HUMAN/PLAIN/MACHINE)
│   ├── ort_common.h                           #   ONNX Runtime shared utilities
│   ├── ort_inference_context.h                #   ONNX inference context wrapper
│   ├── spacemit_ort_bridge.h                  #   SpacemiT EP C→C++ bridge
│   ├── config_manager.h                       #   YAML configuration manager
│   ├── logger.h                               #   Leveled logging (JSON format, thread-safe)
│   ├── json_writer.h                          #   JSON document builder
│   ├── utils.h                                #   Utility functions (timing, file I/O)
│   ├── result_manager.h                       #   Session-level result management
│   └── model_store.h                          #   ONNX model file discovery + validation
├── src/                                       # Source files (32 C + 1 C++)
│   ├── main.c                                 #   Program entry (CLI parsing, mode dispatch)
│   ├── system_controller.c                    #   System controller (3-mode orchestration)
│   ├── pipeline_state.c                       #   5-state pipeline FSM implementation
│   ├── inference_pipeline.c                   #   Cascaded AI inference (4 models)
│   ├── tracking_manager.c                     #   ByteTrack: cascade+Hungarian+Kalman
│   ├── spatial_engine.c                       #   Monocular depth + 3D back-projection
│   ├── world_coord.c                          #   World coordinate system + adaptive anchors
│   ├── k1_odometry.c                          #   INS strapdown + ZUPT EKF + GLRT
│   ├── k1_imu.c                               #   ICM-20948 I²C driver + bias calibration
│   ├── k1_platform.c                          #   K1 capability detection
│   ├── yolov8_pose_estimator.c                #   YOLOv8-Pose: DFL decode + OKS-NMS
│   ├── yolov5_face_detector.c                 #   YOLOv5-Face: 320×320 INT8 face detection
│   ├── arcface_recognizer.c                   #   ArcFace: MobileFaceNet-cuted embedding
│   ├── stgcn_action_recognizer.c              #   ST-GCN: skeleton action classification
│   ├── yolo_postprocess.c                     #   Shared YOLO decode: DFL, NMS, grid assembly
│   ├── keypoint_validator.c                   #   3-tier anatomical validation
│   ├── frame_diff.c                           #   Grid-based MAD frame differencing
│   ├── video_processor.c                      #   Video capture: MP4/V4L2/CoAP sources
│   ├── video_writer.c                         #   AVI file output (Raw RGB24)
│   ├── imu_handler.c                          #   IMU sliding window + Madgwick filter
│   ├── coap_receiver.c                        #   CoAP Block2 JPEG + IMU poll + WiFi mgmt
│   ├── web_server.c                           #   Mongoose HTTP+WS: REST API + frame streaming
│   ├── terminal_ui.c                          #   TUI: spinners, progress bars, status lines
│   ├── ort_common.c                           #   ORT environment + session management
│   ├── ort_inference_context.c                #   ORT inference context (memory planning)
│   ├── spacemit_ort_bridge.cpp                #   C++ bridge: SessionOptionsSpaceMITEnvInit
│   ├── config_manager.c                       #   Minimal YAML parser + key-value store
│   ├── logger.c                               #   JSON-formatted leveled logger
│   ├── json_writer.c                          #   JSON document builder (no allocation)
│   ├── utils.c                                #   Time, file, string utilities
│   ├── result_manager.c                       #   Session tracking + JSON/CSV export
│   ├── model_store.c                          #   Model file discovery + validation
│   └── core_types.c                           #   Data structure initializers
├── web/                                       # Web UI frontend (SPA)
│   ├── index.html                             #   Ceramic-white design, pipeline control
│   └── three.module.js                        #   Three.js ES module (3D rendering)
├── lib/mongoose/                              # Embedded HTTP+WebSocket library
│   └── mongoose.c                             #   Single-file networking library
├── cmake/                                     # CMake toolchain files
│   └── riscv64-toolchain.cmake                #   RISC-V cross-compilation toolchain
├── configs/
│   └── default.yaml                           #   Full-parameter default configuration
├── models/                                    # ONNX model files
│   ├── Human Recognition/
│   │   └── yolo11n.q.onnx                     #   YOLO11n secondary detector (optional, INT8)
│   ├── Face Recognition/
│   │   ├── yolov5n-face_320_cut.q.onnx        #   YOLOv5-Face face detection (INT8)
│   │   └── arcface_mobilefacenet_cut.q.onnx   #   ArcFace feature extraction (INT8)
│   └── Action Prediction/
│       ├── Skeleton Recognition/
│       │   ├── yolov8n-pose.q.onnx            #   YOLOv8-Pose PRIMARY detection+pose (INT8)
│       │   └── yolo11n-pose.q.onnx            #   YOLO11n-Pose alternative variant (INT8)
│       └── Skeleton-based Action Prediction/
│           └── stgcn_int8.onnx                #   ST-GCN action recognition (INT8)
├── docs/                                      # Design documents
│   ├── 核心算法详解.md                         #   Core algorithm detailed explanation
│   ├── ARCHITECTURE.md                        #   Architecture design details
│   ├── CODE_CHANGE_LOG.md                     #   Code change log
│   ├── COMPREHENSIVE_PROJECT_DOCUMENTATION.md #   Comprehensive project documentation
│   ├── MODEL_SPECIFICATION.md                 #   Model specification
│   ├── K1_HARDWARE_OPTIMIZATION_PLAN.md       #   K1 hardware optimization plan
│   ├── INFERENCE_OPTIMIZATION_ANALYSIS.md     #   Inference optimization analysis
│   ├── OPTIMIZATION_DESIGN.md                 #   Optimization design document
│   ├── 3D_WORLD_COORDINATE_SYSTEM_PLAN.md     #   3D world coordinate system design
│   ├── CODE_AUDIT_REPORT.md                   #   Code audit report
│   ├── REFACTORING_REPORT.md                  #   Refactoring report
│   ├── TESTING_GUIDE.md                       #   Testing guide
│   └── TESTING_GUIDE_V2.md                    #   Testing guide v2
├── receive/                                   # ESP32 Arrow-end firmware + test scripts
│   ├── coap_server.h                          #   ESP32 CoAP server header
│   ├── ov-imu-pwm.py                          #   Python receiver script
│   ├── receive.ino                            #   Arduino receiver sketch
│   └── test-udp.py                            #   UDP test script
├── CMakeLists.txt                             #   Top-level CMake build configuration
└── README.md                                  #   This document
```

## Appendix B: CMake Build Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Release / Debug / RelWithDebInfo |
| `CMAKE_TOOLCHAIN_FILE` | — | Cross-compilation toolchain file |
| `BIANBU_SYSROOT` | — | Bianbu OS sysroot path (cross-compilation) |
| `ONNX_RUNTIME_DIR` | — | SpacemiT ONNX Runtime 2.0.2 installation path (auto-probes /opt) |
| `USE_ONNX_RUNTIME` | `ON` | Enable ONNX Runtime (REQUIRED for all AI inference) |
| `USE_OPENMP` | `ON` | Enable OpenMP multi-core parallelism |
| `ENABLE_RVV_OPT` | `ON` (RISC-V) | Enable RVV vectorization (auto-enabled on RISC-V targets) |
| `ENABLE_K1_PIPELINE` | `ON` | Enable K1 dual-cluster pipeline parallelism |
| `ENABLE_K1_TCM` | `ON` | Enable K1 TCM tightly-coupled memory |
| `ENABLE_K1_VPU` | `ON` | Enable K1 VPU hardware video/JPEG acceleration |
| `MUSE_PI_ARCH` | `rv64gcv0p7` | RISC-V target architecture (auto-detected at build time) |
| `K1_MPP_DIR` | — | K1 MPP media processing SDK path (auto-detects system libmpp.so) |

---

> **Project Repository**: lingqi_tantong
> **License**: Proprietary
> **Last Updated**: 2026-07
> **Total Code**: ~15,000 lines C/C++ (Shooter End, K1 Muse Pi Pro)
> **Models**: 5 ONNX models (4 INT8-quantized + 1 FP32)
