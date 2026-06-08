# LingQi TanTong Architecture Design Document

> This document describes the software architecture, inter-module data flow, core algorithm principles, and design decisions of the `lingqi_tantong_c` project.

---

## 1. System Layered Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Application Layer                           │
│  ┌─────────────────────┐                                             │
│  │      main.c         │  CLI parsing → Module creation → Main loop → Cleanup      │
│  └─────────┬───────────┘                                             │
├────────────┼─────────────────────────────────────────────────────────┤
│            │                    Controller Layer                      │
│  ┌─────────▼───────────┐                                             │
│  │ system_controller.c │  system_controller_process_video(frames)    │
│  │                     │  ┌── _process_single_frame() ──────────┐    │
│  │                     │  │ 1. video_processor_read()           │    │
│  │                     │  │ 2. inference_pipeline_process()     │    │
│  │                     │  │ 3. spatial_engine_init_coord()      │    │
│  │                     │  │ 4. spatial_engine_calculate_pos()   │    │
│  │                     │  │ 5. tracking_manager_update()        │    │
│  │                     │  │ 6. [pose/face association]          │    │
│  │                     │  │ 7. spatial_engine_update_trajectory │    │
    │  │                     │  │ 8. imu_handler → spatial_engine     │    │
    │  │                     │  │ 9. visualizer_render()              │    │
    │  │                     │  │ 10. result_manager_record()         │    │
│  │                     │  └─────────────────────────────────────┘    │
│  └─────────────────────┘                                             │
├──────────────────────────────────────────────────────────────────────┤
│                      Business Logic Layer                            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐      │
│  │inference_pipeline│ │ tracking_manager │ │  spatial_engine  │      │
│  │                  │ │                  │ │                  │      │
│  │ Cascade:         │ │ ByteTrack:       │ │ Pinhole Model:   │      │
│  │ YOLOv8-Pose (PRIMARY) │ │ Cascade+Hungarian  │ │ 2D box → 3D pos  │      │
│  │ YOLO11n (SECONDARY)   │ │ + 7-state Kalman │ │ + depth est.     │      │
│  │ YOLOv5-Face ─► ArcFace │ │ + EMA smoothing  │ │ + trajectory     │      │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘      │
├──────────────────────────────────────────────────────────────────────┤
│                        Data Processing Layer                         │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │ video_processor│ │  imu_handler  │ │  model_store  │               │
│  │               │ │               │ │               │               │
│  │ Frame read/scale │ │ Validate/parse │ │ Model file mgmt │               │
│  │ Color convert   │ │ Sliding window  │ │ Path resolve    │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                      Presentation Layer                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │  visualizer   │ │  ar_renderer  │ │  video_writer │               │
│  │               │ │               │ │               │               │
│  │ BBox rendering  │ │ Motion comp.    │ │ MP4 encode out  │               │
│  │ Skeleton/traj   │ │ Marker overlay  │ │               │               │
│  │ 5×7 bitmap   │ │               │ │               │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                        Support Layer                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │config_manager │ │    logger     │ │result_manager │               │
│  │               │ │               │ │               │               │
│  │ YAML key-value  │ │ Leveled logging │ │ Session track   │               │
│  │ Prog. defaults  │ │ File/terminal   │ │ JSON/CSV report │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Data Flow

### 2.1 Frame Data Structure

```
FrameData (uint8*,width,height,channels,timestamp)
    │
    ├──▶ YOLO11     ──▶ Detection[] (bbox,conf,class,label)
    │
    ├──▶ YOLOv8-Pose ──▶ PoseEstimation[] (keypoints[17],bbox,conf)
    │
    ├──▶ YOLOv5-Face  ──▶ FaceIdentity[] (bbox,id,sim,feature[512])
    │
    └──▶ ArcFace     backfill feature_vector into FaceIdentity
```

### 2.2 Object Association Graph (Data Convergence)

```
                    Frame
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        Detection   Pose    FaceIdentity
            │         │         │
            └────┬────┘    IoU associate to
                 │         PoseEstimation
            IoU matching          │
                 │    IoU associate to FaceIdentity
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

### 2.3 Tracking Manager Internal Flow

```
New Detection[] ──▶ ByteTrack Three-stage Matching ──▶ TrackedObject[] ──▶ EMA Smoothing
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  HIGH conf(≥0.6)   LOW conf(0.1-0.6)  UNMATCHED
  → confirmed track → IoU matching    → new track
  → unconfirmed      → lost track      → (if conf≥0.7)
```

---

## 3. Core Algorithm Details

### 3.1 Spatial Localization (Pinhole Camera Model)

$$
Z = \frac{f_y \cdot H_{\text{avg}}}{h_{\text{bbox}}}
$$

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad
Y = \frac{(v - c_y) \cdot Z}{f_y}
$$

- $f_x, f_y$: Focal length (pixels), default 960
- $c_x, c_y$: Principal point offset (pixels), default 960, 540 (for 1920×1080)
- $H_{\text{avg}}$: Average human height (m), default 1.70
- $h_{\text{bbox}}$: Bounding box height (pixels)
- $(u, v)$: Bounding box bottom-center pixel coordinates

**World coordinate system initialization:** The (X,Z) of the first detected object in the first frame is set as the world origin (0,0,0).

### 3.2 Kalman Filter (7-State)

**State vector:**
$$
\mathbf{x} = [cx, cy, area, aspect\_ratio, \dot{cx}, \dot{cy}, \dot{area}]
$$

- $cx, cy$: Bounding box center pixel coordinates
- $area$: Bounding box area
- $aspect\_ratio$: Width / Height
- $\dot{cx}, \dot{cy}, \dot{area}$: Velocities of center and area

**State transition (constant velocity model):**
$$
cx_{k+1} = cx_k + \dot{cx}_k \cdot \Delta t
$$
$$
cy_{k+1} = cy_k + \dot{cy}_k \cdot \Delta t
$$
$$
area_{k+1} = area_k + \dot{area}_k \cdot \Delta t
$$

**Measurement observation:** $[cx, cy, area, aspect\_ratio]$ (4 observed, 3 latent)

### 3.3 IMU Data Sliding Smoothing

Reference: `imu_handler.c` — Moving average filter, default window size 10.

### 3.4 Letterbox Preprocessing

Aspect-ratio-preserving scaling + padding:

```
ratio = min(target_w / src_w, target_h / src_h)
new_w = src_w * ratio
new_h = src_h * ratio
pad_w = (target_w - new_w) / 2
pad_h = (target_h - new_h) / 2
```

---

## 4. Module Interface Design

### 4.1 Primary Interface Conventions

All modules follow the **create → process → destroy** lifecycle:

```c
// Create (resource allocation + config injection)
Module* module_create(Config* config);

// Process (main logic, frame-driven)
Result* module_process(Module* m, FrameData* frame);

// Destroy (resource release)
void module_destroy(Module* m);
```

### 4.2 Inter-Module Data Transfer Contract

| Producer | Data | Consumer | Transfer Method |
|----------|------|----------|-----------------|
| VideoProcessor | FrameData | SystemController | Direct call return |
| InferencePipeline | Detection[], PoseEstimation[], FaceIdentity[] | TrackingManager | Array + count |
| SpatialEngine | SpatialPosition | TrackingManager | struct assignment |
| TrackingManager | TrackedObject[] | Visualizer, ResultManager | Array + count |
| IMUHandler | IMUExternalPose | SpatialEngine | Function call |

### 4.3 ONNX Runtime Conditional Compilation

```c
#ifdef HAS_ONNX_RUNTIME
    // Real ONNX model inference path (ONNX Runtime required)
    ort_create_session(model_path, num_threads, use_ep);
    ort_ctx_run(ctx, output_values);
    // DFL decoding for xquant-split models
    yolo_dfl_decode_position(reg_data, pix, hw, dists);
#else
    // Build-time error: project has no heuristic fallback
    #error "Module requires HAS_ONNX_RUNTIME"
#endif
```

---

## 5. Memory Management Strategy

| Strategy | Description |
|----------|-------------|
| **Stack-first** | Small structs (BoundingBox, Detection) passed by value |
| **Frame buffer pool** | VideoProcessor pre-allocates fixed-size ring buffer |
| **Explicit lifecycle** | create/destroy pattern, no implicit memory leak paths |
| **Fixed upper bounds** | Detections ≤100, TrackedObjects ≤100, Trajectory ≤300, Tracks ≤256 |
| **No global variables** | All state encapsulated in module structs |

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| C11 over C99 | Static assertions `_Static_assert`, anonymous unions, alignment macros |
| Dual coordinate systems (2D+3D) | Decoupled 2D fast display + 3D spatial analysis |
| EMA smoothing (α=0.30) | Biased toward new value at 0.30, reduces jitter, improves localization stability |
| Independent top-down view area | Front view rendering unaffected by 3D trajectories |
| ONNX Runtime required | Real ONNX inference only; no heuristic fallback exists |
| YAML-style configuration | Human-readable, compatible with Python version config |
| Fixed 128-dim face features | Aligned with ArcFace MobileFaceNet-cuted model output |

---

## 7. Extension Points (Reserved Interfaces)

| Interface Location | Purpose | Status |
|--------------------|---------|--------|
| `depth_map` parameter | SpatialEngine accepts external depth map | Interface ready, MiDaS not implemented |
| `IMUData` struct | Sensor data container | Type definition ready, IMU→SpatialEngine link connected |
| `ArMotionState` | AR rendering motion state | Interface ready, motion compensation implemented |
| `ArrowReceiver` | ESP32P4 UART protocol receiver | Implemented, pending real-time UART input source integration |

---

## 8. File Dependency Graph

```
main.c
 └── system_controller.c
      ├── video_processor.c
      ├── video_writer.c
      ├── imu_handler.c
      ├── model_store.c
      ├── inference_pipeline.c
      │    ├── yolov8_detector.c
      │    ├── yolov8_pose_estimator.c
      │    ├── yolov5_face_detector.c
      │    └── arcface_recognizer.c
      ├── tracking_manager.c
      ├── spatial_engine.c
      ├── visualizer.c
      ├── ar_renderer.c
      ├── arrow_receiver.c
      ├── mahony_filter.c
      ├── kcp_lite.c
      ├── ort_common.c
      ├── result_manager.c
      ├── model_export.c
      ├── config_manager.c
      ├── logger.c
      ├── utils.c
      └── core_types.c
```

---

## Related Documents

| Document | Path |
|----------|------|
| Project README | `README.md` |
| Build Info | See `CMakeLists.txt` and `cmake/` directory |
| Unimplemented Modules | `docs/IMPLEMENTATION_GAPS.md` |
| Configuration File | `configs/default.yaml` |
