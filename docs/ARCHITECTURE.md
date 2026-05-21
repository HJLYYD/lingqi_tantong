# 灵柒·探瞳 架构设计文档

> 本文档描述 `lingqi_tantong_c` 项目的软件架构、模块间数据流、核心算法原理与设计决策。

---

## 1. 系统分层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Application Layer                           │
│  ┌─────────────────────┐                                             │
│  │      main.c         │  命令行解析 → 模块创建 → 主循环 → 清理      │
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
│  │ YOLOv8n ─►       │ │ 三阶段匹配       │ │ 2D box → 3D pos  │      │
│  │ YOLOv8-Pose ─►   │ │ + 7-state Kalman │ │ + depth est.     │      │
│  │ SCRFD ─► ArcFace │ │ + EMA smoothing  │ │ + trajectory     │      │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘      │
├──────────────────────────────────────────────────────────────────────┤
│                        Data Processing Layer                         │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │ video_processor│ │  imu_handler  │ │  model_store  │               │
│  │               │ │               │ │               │               │
│  │ 帧读取/缩放   │ │ 验证/解析     │ │ 模型文件管理 │               │
│  │ 颜色转换     │ │ 滑动窗口平滑  │ │ 路径解析     │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                      Presentation Layer                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │  visualizer   │ │  ar_renderer  │ │  video_writer │               │
│  │               │ │               │ │               │               │
│  │ 边界框渲染   │ │ 运动补偿     │ │ MP4 编码输出 │               │
│  │ 骨架/轨迹   │ │ 标记叠加     │ │               │               │
│  │ 5×7 bitmap   │ │               │ │               │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
├──────────────────────────────────────────────────────────────────────┤
│                        Support Layer                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│  │config_manager │ │    logger     │ │result_manager │               │
│  │               │ │               │ │               │               │
│  │ YAML键值解析 │ │ 分级日志     │ │ Session跟踪  │               │
│  │ 可编程默认值 │ │ 文件/终端    │ │ JSON/CSV报告 │               │
│  └───────────────┘ └───────────────┘ └───────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心数据流

### 2.1 帧数据结构

```
FrameData (uint8*,width,height,channels,timestamp)
    │
    ├──▶ YOLOv8n     ──▶ Detection[] (bbox,conf,class,label)
    │
    ├──▶ YOLOv8-Pose ──▶ PoseEstimation[] (keypoints[17],bbox,conf)
    │
    ├──▶ SCRFD       ──▶ FaceIdentity[] (bbox,id,sim,feature[512])
    │
    └──▶ ArcFace     回填 feature_vector 到 FaceIdentity
```

### 2.2 目标关联图（数据汇聚）

```
                    Frame
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        Detection   Pose    FaceIdentity
            │         │         │
            └────┬────┘    IoU 关联到
                 │         PoseEstimation
            IoU 匹配          │
                 │    IoU 关联到 FaceIdentity
                 ▼              │
           TrackedObject ◄──────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
  spatial_pos  pose   face_identity
        │
        ▼
  trajectory[] (历史 SpatialPosition[])
```

### 2.3 跟踪管理器内部流程

```
New Detection[] ──▶ ByteTrack 三阶段匹配 ──▶ TrackedObject[] ──▶ EMA 平滑
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  HIGH conf(≥0.6)   LOW conf(0.1-0.6)  UNMATCHED
  → confirmed track → IoU matching    → new track
  → unconfirmed      → lost track      → (if conf≥0.7)
```

---

## 3. 核心算法详解

### 3.1 空间定位（针孔相机模型）

$$
Z = \frac{f_y \cdot H_{\text{avg}}}{h_{\text{bbox}}}
$$

$$
X = \frac{(u - c_x) \cdot Z}{f_x}, \quad 
Y = \frac{(v - c_y) \cdot Z}{f_y}
$$

- $f_x, f_y$: 焦距（像素），默认 500
- $c_x, c_y$: 主点偏移（像素），默认 320, 240
- $H_{\text{avg}}$: 平均人体身高 (m)，默认 1.70
- $h_{\text{bbox}}$: 检测框高度 (像素)
- $(u, v)$: 检测框底部中心像素坐标

**世界坐标系初始化：** 首帧第一个检测目标的 (X,Z) 设为世界原点 (0,0,0)。

### 3.2 Kalman 滤波器（7 状态）

**状态向量：**
$$
\mathbf{x} = [u, v, w, h, \dot{u}, \dot{v}, s]
$$

- $u, v$: 边界框中心像素坐标
- $w, h$: 边界框宽度、高度
- $\dot{u}, \dot{v}$: 速度分量
- $s$: 尺度因子

**状态转移（匀速运动模型）：**
$$
u_{k+1} = u_k + \dot{u}_k \cdot \Delta t
$$

### 3.3 IMU 数据滑动平滑

参考：`imu_handler.c` — 移动平均滤波器，窗口默认 10。

### 3.4 Letterbox 预处理

保持宽高比的缩放+填充：

```
ratio = min(target_w / src_w, target_h / src_h)
new_w = src_w * ratio
new_h = src_h * ratio
pad_w = (target_w - new_w) / 2
pad_h = (target_h - new_h) / 2
```

---

## 4. 模块接口设计

### 4.1 主要接口约定

所有模块遵循 **create → process → destroy** 生命周期：

```c
// 创建（资源分配 + 配置注入）
Module* module_create(Config* config);

// 处理（主要逻辑，帧驱动）
Result* module_process(Module* m, FrameData* frame);

// 销毁（资源释放）
void module_destroy(Module* m);
```

### 4.2 模块间数据传递契约

| 生产方 | 数据 | 消费方 | 传递方式 |
|--------|------|--------|----------|
| VideoProcessor | FrameData | SystemController | 直接调用返回 |
| InferencePipeline | Detection[], PoseEstimation[], FaceIdentity[] | TrackingManager | 数组+计数 |
| SpatialEngine | SpatialPosition | TrackingManager | struct 赋值 |
|│ TrackingManager | TrackedObject[] | Visualizer, ResultManager | 数组+计数 |
| IMUHandler | IMUExternalPose | SpatialEngine | 函数调用 |

### 4.3 ONNX Runtime 条件编译

```c
#ifdef HAS_ONNX_RUNTIME
    // 真实 ONNX 模型推理路径
    ort_run_session(session, input_tensor);
#else
    // 启发式回退（纯 C 算法，无外部依赖）
    heuristic_detect(frame, &results);
#endif
```

---

## 5. 内存管理策略

| 策略 | 说明 |
|------|------|
| **栈优先** | 小型结构体（BoundingBox, Detection）按值传递 |
| **帧缓冲区池** | VideoProcessor 预分配固定大小环形缓冲 |
| **显式生命周期** | create/destroy 模式，无隐式内存泄漏路径 |
| **固定上限** | Detections ≤100, TrackedObjects ≤32, Trajectory ≤90 |
| **无全局变量** | 所有状态封装在模块 struct 中 |

---

## 6. 关键设计决策

| 决策 | 理由 |
|------|------|
| C11 而非 C99 | 静态断言 `_Static_assert`，匿名 union，对齐宏 |
| 双坐标系 (2D+3D) | 2D 快速显示 + 3D 空间分析解耦 |
| EMA 平滑 (α=0.25) | 偏向新值 0.25，减少抖动，提升定位稳定性 |
| 俯视图独立区域 | 正面渲染不受 3D 轨迹干扰 |
| ONNX/Heuristic 双路径 | 确保无外部依赖时仍可构建和运行 |
| YAML 风格配置 | 人类可读，且兼容 Python 版配置 |
| 固定 512 维人脸特征 | 与 ArcFace glintr100 输出对齐 |

---

## 7. 扩展点（预留接口）

| 接口位置 | 用途 | 状态 |
|----------|------|------|
| `depth_map` 参数 | SpatialEngine 接受外部深度图 | 接口就绪，MiDaS 未实现 |
| `IMUData` 结构体 | 传感器数据容器 | 类型定义就绪，已接入 IMU→SpatialEngine 链路 |
| `ArMotionState` | AR 渲染运动状态 | 接口就绪，运动补偿已实现 |
| `ArrowReceiver` | ESP32P4 UART 协议接收 | 已实现，待接入实时 UART 输入源 |

---

## 8. 文件依赖图

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
      │    ├── scrfd_detector.c
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

## 相关文档

| 文档 | 路径 |
|------|------|
| 项目 README | `README.md` |
| 构建指南 | `docs/BUILD_GUIDE.md` |
| 未实现模块 | `docs/IMPLEMENTATION_GAPS.md` |
| 配置文件 | `configs/default.yaml` |