# 灵柒·探瞳 (LingQi TanTong) 综合项目文档

> 基于实际源代码审计 + 在线技术检索生成的完整技术文档
> 审计/更新日期: 2026-06-13
> 目标平台: SpacemiT K1 (MUSE Pi Pro) / RISC-V 64 / Bianbu OS

***

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心数据类型](#3-核心数据类型)
4. [模块详解](#4-模块详解)
   - 4.1 [系统控制器 (system\_controller)](#41-系统控制器)
   - 4.2 [推理管道 (inference\_pipeline)](#42-推理管道)
   - 4.3 [YOLO11n-Pose 姿态估计器](#43-yolo11n-pose-姿态估计器)
   - 4.4 [YOLO 后处理](#44-yolo-后处理)
   - 4.5 [关键点验证器 (keypoint\_validator)](#45-关键点验证器)
   - 4.6 [目标跟踪器 (tracking\_manager)](#46-目标跟踪器)
   - 4.7 [空间定位引擎 (spatial\_engine)](#47-空间定位引擎)
   - 4.8 [ST-GCN 动作识别器](#48-st-gcn-动作识别器)
   - 4.9 [YOLOv5 人脸检测器](#49-yolov5-人脸检测器)
   - 4.10 [ArcFace 人脸识别器](#410-arcface-人脸识别器)
   - 4.11 [ONNX Runtime 通用层](#411-onnx-runtime-通用层)
   - 4.12 [SpacemiT EP 桥接层](#412-spacemit-ep-桥接层)
   - 4.13 [配置管理器](#413-配置管理器)
   - 4.14 [模型存储](#414-模型存储)
   - 4.15 [视频处理器](#415-视频处理器)
   - 4.16 [数据输入模块](#416-数据输入模块)
   - 4.17 [IMU 处理器](#417-imu-处理器)
   - 4.18 [可视化输出](#418-可视化输出)
   - 4.19 [结果管理器](#419-结果管理器)
   - 4.20 [工具函数库](#420-工具函数库)
   - 4.21 [K1 平台抽象层](#421-k1-平台抽象层)
   - 4.22 [AR 渲染器](#422-ar-渲染器)
5. [模块间联动逻辑](#5-模块间联动逻辑)
6. [数据流](#6-数据流)
7. [配置文件](#7-配置文件)
8. [构建系统](#8-构建系统)
9. [模型清单](#9-模型清单)
10. [技术参考资料索引](#10-技术参考资料索引)
11. [社区案例对比与改进建议](#11-社区案例对比与改进建议)

***

## 1. 项目概述

灵柒·探瞳 (LingQi TanTong) 是一个运行于 SpacemiT K1 RISC-V SoC (MUSE Pi Pro 开发板) 上的实时多人姿态估计与动作识别系统。项目使用 C11 标准编写，通过 ONNX Runtime 进行深度学习推理，支持离线和实时两种工作模式。

### 核心功能

| 功能      | 描述                                              | 使用的模型/算法                             |
| ------- | ----------------------------------------------- | ------------------------------------ |
| 人员检测    | YOLO11n-Pose 统一模型同时完成人员检测和 17 点姿态估计             | YOLO11n-Pose (INT8 量化)               |
| 人脸检测与识别 | YOLOv5-Face + ArcFace MobileFaceNet 进行人脸检测和身份识别 | YOLOv5n-Face + ArcFace MobileFaceNet |
| 动作识别    | ST-GCN 基于骨骼序列进行 7 类动作分类                         | ST-GCN (FP32, CPU EP)                |
| 多目标跟踪   | ByteTrack 风格的三阶段级联匹配 + 匈牙利算法最优分配                | 卡尔曼滤波 (7 状态) + 12 维外观特征              |
| 空间定位    | 基于解剖学比例的多点采样单目深度估计与空间坐标计算                       | 针孔相机模型 + MAD 鲁棒融合                    |
| 实时可视化   | 支持 framebuffer 显示、RTSP/UDP 推流、MP4 录制的多通道输出      | FFmpeg pipe + /dev/fb0 mmap          |

### 项目统计

| 指标             | 数值                              |
| -------------- | ------------------------------- |
| 源文件 (.c)       | 28 个                            |
| 头文件 (.h)       | 29 个                            |
| C++ 源文件 (.cpp) | 1 个 (spacemit\_ort\_bridge.cpp) |
| YAML 配置文件      | 1 个                             |
| 模型文件           | 5 个 (4 个 INT8 量化 + 1 个 FP32)    |
| 代码行数           | \~15,000+ 行                     |



## 2. 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                            main.c                                    │
│                    (入口点 + 信号处理 + CLI解析)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SystemController                                 │
│  (系统控制器: 模块装配 + 生命周期管理 + 两种运行模式)                   │
│                                                                      │
│  system_controller_create() / destroy()                              │
│  system_controller_process_video()      ← 离线模式                    │
│  system_controller_process_realtime_k1() ← 实时模式(K1三线程管道)      │
└───────┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────────┘
        │      │      │      │      │      │      │
        ▼      ▼      ▼      ▼      ▼      ▼      ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │配置 │ │模型│ │推理 │ │跟踪 │ │空间│  │可视│ │结果 │
   │管理 │ │存储│ │管道 │ │管理 │ │引擎│  │化  │ │管理│
   └────┘ └────┘ └──┬─┘ └────┘ └────┘ └────┘ └────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐ ┌─────────┐ ┌──────────┐
   │YOLO11n │ │YOLOv5   │ │ArcFace   │  ← 模型推理层
   │Pose    │ │Face     │ │Recognizer│
   └────────┘ └─────────┘ └──────────┘
        │                       │
        ▼                       ▼
   ┌────────┐            ┌──────────┐
   │Keypoint│            │ST-GCN    │  ← 后处理/验证层
   │Validator│           │Action Rec│
   └────────┘            └──────────┘

   ┌─────────────── ONNX Runtime 层 ───────────────┐
   │  ort_common.c  │  ort_inference_context.c      │
   │  spacemit_ort_bridge.cpp (SpacemiT EP)         │
   └────────────────────────────────────────────────┘

   ┌─────────────── 输入层 ─────────────────────────┐
   │  video_processor.c (离线视频 / V4L2 摄像头)     │
   │  arrow_receiver.c  (Arrow UART JPEG 接收)      │
   │  mjpeg_receiver.c  (ESP32 MJPEG HTTP 接收)     │
   │  imu_handler.c     (IMU 数据处理)              │
   └────────────────────────────────────────────────┘

   ┌─────────────── 输出层 ─────────────────────────┐
   │  visualizer.c      (检测框/骨架/轨迹绘制)       │
   │  ar_renderer.c     (AR 叠加渲染)               │
   │  display_output.c  (多通道输出管理)             │
   │  video_writer.c    (FFmpeg 管道写入)           │
   │  result_manager.c  (会话统计/JSON/CSV 导出)     │
   └────────────────────────────────────────────────┘
```

### K1 平台双集群架构

```
┌──────────────────────────────────────────────────────────┐
│                 SpacemiT K1 SoC                           │
│                                                          │
│  Cluster0 (AI 集群, CPU 0-3)          Cluster1 (IO 集群)  │
│  ┌─────────────────────────┐      ┌───────────────────┐  │
│  │ Core0: AI 主核          │      │ Core4: 视频采集    │  │
│  │ Core1: 推理线程         │      │ Core5: VPU 硬件    │  │
│  │ Core2: 检测线程         │      │ Core6: 可视化      │  │
│  │ Core3: 姿态线程         │      │ Core7: 输出        │  │
│  │                         │      │                   │  │
│  │ 512KB TCM (共享)        │      │ 通用 L2 缓存      │  │
│  │ IME 2.0 TOPS AI 加速    │      │                   │  │
│  │ RVV 1.0 256-bit 向量    │      │                   │  │
│  └─────────────────────────┘      └───────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 模块分层

| 层级    | 模块                                                                                                 | 说明                              |
| ----- | -------------------------------------------------------------------------------------------------- | ------------------------------- |
| 入口层   | main.c                                                                                             | CLI 解析、信号处理、日志初始化               |
| 控制层   | system\_controller.c                                                                               | 模块装配、生命周期、离线/实时模式               |
| 管道层   | inference\_pipeline.c                                                                              | 多模型调度、自适应级联状态机                  |
| 模型层   | yolov8\_pose\_estimator / yolov5\_face\_detector / arcface\_recognizer / stgcn\_action\_recognizer | 各模型推理实现                         |
| 后处理层  | keypoint\_validator / tracking\_manager / spatial\_engine                                          | 关键点验证、目标跟踪、空间定位                 |
| 推理运行时 | ort\_common / ort\_inference\_context / spacemit\_ort\_bridge                                      | ONNX Runtime 封装与 SpacemiT EP 桥接 |
| 数据层   | video\_processor / arrow\_receiver / mjpeg\_receiver / imu\_handler                                | 多源数据输入                          |
| 输出层   | visualizer / ar\_renderer / display\_output / video\_writer / result\_manager                      | 可视化与结果导出                        |
| 基础设施  | config\_manager / model\_store / logger / utils / core\_types / k1\_platform                       | 配置、模型管理、日志、工具函数、平台抽象            |

***

## 3. 核心数据类型

所有核心数据类型定义在 [include/core\_types.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/core_types.h)。

### 数据常量

| 宏                           | 值   | 说明             |
| --------------------------- | --- | -------------- |
| MAX\_KEYPOINTS              | 17  | COCO 姿态关键点数量   |
| MAX\_TRAJECTORY\_LEN        | 300 | 最大轨迹历史点数       |
| MAX\_DETECTIONS\_PER\_FRAME | 100 | 每帧最大检测数        |
| MAX\_TRACKED\_OBJECTS       | 100 | 最大跟踪目标数        |
| MAX\_FACES\_PER\_FRAME      | 20  | 每帧最大人脸数        |
| MAX\_POSES\_PER\_FRAME      | 20  | 每帧最大姿态数        |
| MAX\_ACTIONS\_PER\_FRAME    | 5   | 每帧最大动作数        |
| FEATURE\_VECTOR\_DIM        | 128 | ArcFace 特征向量维度 |

### 主要数据结构

#### BoundingBox (边界框)

```
定义位置: core_types.h
字段: x_min, y_min, x_max, y_max (float)
内联函数: bbox_center_x, bbox_center_y, bbox_width, bbox_height, bbox_area, bbox_iou
```

#### Detection (检测结果)

```
定义位置: core_types.h
字段:
  - bbox: BoundingBox          边界框
  - confidence: float          置信度
  - class_id: int              类别 ID
  - class_name: char[64]       类别名称
  - track_id_hint: int         重识别提示 (-1 表示无)
  - is_partial_body: bool      是否为部分身体（遮挡/不完整）
  - num_visible_keypoints: int 可见关键点数
```

#### PoseEstimation (姿态估计)

```
定义位置: core_types.h
字段:
  - keypoints[17]: Keypoint   17 个 COCO 关键点
  - num_keypoints: int        有效关键点数量
  - bbox: BoundingBox         关联边界框
  - has_bbox: bool            是否有边界框
  - confidence: float         整体置信度
```

#### TrackedObject (跟踪对象)

```
定义位置: core_types.h
字段:
  - track_id: int              跟踪 ID
  - detection: Detection       当前检测
  - spatial_pos: SpatialPosition 空间位置
  - pose: PoseEstimation       关联姿态 (可选)
  - face: FaceIdentity         关联人脸 (可选)
  - trajectory: TrajectoryBuffer 轨迹历史
  - velocity[3]: float         速度估计
  - acceleration[3]: float     加速度估计
  - is_active: bool            是否活跃
  - is_occluded: bool          是否被遮挡
  - frames_seen: int           可见帧数
  - height_meters: float       估计身高 (米)
```

***

## 4. 模块详解

### 4.1 系统控制器

**文件**: [src/system\_controller.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/system_controller.c)
**头文件**: [include/system\_controller.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/system_controller.h)

#### 功能概述

系统控制器是整个项目的顶层协调模块，负责所有子模块的创建、配置、生命周期管理，并提供离线视频处理和实时 K1 管道两种运行模式。

#### 创建流程 (system\_controller\_create)

```
1. 分配 SystemController 内存
2. 创建 ConfigManager (加载 YAML 配置)
3. 配置 SpacemiT EP 参数 (use_spacemit_ep, intra_threads)
4. 创建 ModelStore (模型路径解析)
5. 创建 IMUHandler (IMU 数据处理)
6. 创建 AIInferencePipeline (推理管道)
7. 加载模型 (inference_pipeline_load_models)
8. 创建 ObjectTracker (配置所有跟踪参数: 确认、级联、遮挡、重识别、多人检测)
9. 创建 SpatialLocalizationEngine (相机内参 + 深度估计参数)
10. 创建 Visualizer (可视化参数)
11. 创建 ARRenderer (AR 渲染)
12. 创建 ResultManager (结果管理)
13. 设置显示/推流/录制/MJPEG 配置
```

#### 预期功能与实现对照

| 预期功能           | 实现状态 | 实现方法                                            |
| -------------- | ---- | ----------------------------------------------- |
| 统一模块生命周期管理     | 已实现  | create()/destroy() 对称管理                         |
| 离线视频逐帧处理       | 已实现  | system\_controller\_process\_video() 单线程循环      |
| K1 实时三线程管道     | 已实现  | system\_controller\_process\_realtime\_k1()     |
| 自适应数据源选择       | 已实现  | V4L2 > Arrow UART > MJPEG HTTP 优先级              |
| Ring buffer 同步 | 已实现  | 4槽位 + 条件变量 (cam\_cond, infer\_cond, post\_cond) |
| 帧超时检测          | 已实现  | capture 线程帧间隔 vs frame\_timeout\_s 检测           |
| 心跳监控           | 已实现  | 主线程每 500ms 检查 capture 心跳，15s 无响应触发 shutdown     |
| 动作识别帧间隔控制      | 已实现  | action\_inference\_interval=10 帧                |
| 人脸检测动态频率       | 已实现  | 有人时每 5 帧，无人时每 60 帧                              |

#### 姿态/人脸与跟踪对象的关联

**associate\_poses\_with\_objects**: 基于 IoU (>0.3) 将姿态估计结果关联到跟踪对象，使用贪心匹配避免重复分配。关联后同时更新对象的外观特征（12 维姿态描述子）。

**associate\_faces\_with\_objects**: 基于 IoU (>0.2) 将人脸识别结果关联到跟踪对象。

**FPS 计算 (get\_current\_fps)**: 从最近 10 帧计算滑动平均。

#### K1 实时管道 (system\_controller\_process\_realtime\_k1)

```
三线程管道架构:
  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
  │ Capture 线程    │───▶│ Inference 线程   │───▶│ PostProcess 线程  │
  │ CPU4 (Cluster1) │    │ CPU1 (Cluster0)  │    │ CPU0 (Cluster0)   │
  │ 帧采集+JPEG解码  │    │ AI 推理          │    │ 跟踪+定位+可视化   │
  └─────────────────┘    └─────────────────┘    └──────────────────┘
         │                       │                       │
    环形缓冲区 (4槽位, 带条件变量同步)
```

#### 离线视频处理 (system\_controller\_process\_video)

```
单线程逐帧处理循环:
1. 读取帧 → 2. 推理 → 3. 空间定位 → 4. 跟踪 → 5. 姿态/人脸关联
→ 6. 轨迹更新+速度估计 → 7. IMU 姿态设置 → 8. 可视化渲染 → 9. 视频写入
```

**参考实现来源**:

- 三线程管道模式参考: NVIDIA DeepStream pipeline 架构
- Ring buffer 同步模式参考: Linux kernel kfifo + pthread condition variable
- 帧超时检测参考: GStreamer pipeline watchdog 模式

***

### 4.2 推理管道

**文件**: [src/inference\_pipeline.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/inference_pipeline.c)
**头文件**: [include/inference\_pipeline.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/inference_pipeline.h)

#### 功能概述

推理管道负责多模型的协调调度，实现了自适应级联状态机和关键点验证过滤。当前采用统一模型模式 (YOLO11n-Pose 同时完成检测和姿态估计)。

#### 预期功能与实现对照

| 预期功能     | 实现状态 | 实现方法                                     |
| -------- | ---- | ---------------------------------------- |
| 多模型协调加载  | 已实现  | 按优先级顺序加载，主模型失败阻断，辅助模型降级警告                |
| 自适应级联状态机 | 已实现  | 三状态: SEARCHING/TRACKING/VALIDATING       |
| 多人进入检测   | 已实现  | num\_detections > num\_tracks + 1 触发全分辨率 |
| 关键点解剖验证  | 已实现  | 三层回退: 全身→上半身→侧身                          |
| 人脸检测动态频率 | 已实现  | SEARCHING: 每10帧, TRACKING: 每30帧          |
| 动作识别间隔控制 | 已实现  | 默认每10帧推理一次                               |
| 每30帧性能统计 | 已实现  | 累计帧时间求平均，输出平均 FPS                        |

#### 模型加载 (inference\_pipeline\_load\_models)

```
模型注册表:
  ┌─────────────────────────────────────────────────────────────┐
  │ 模型名称                 │ 配置键           │ 类型         │
  ├─────────────────────────────────────────────────────────────┤
  │ yolo11n_pose            │ pose.model_path   │ YOLOv8 Pose  │
  │ yolov5_face             │ face.detection    │ YOLOv5 Face  │
  │ arcface_mobilefacenet   │ face.recognition  │ ArcFace      │
  │ stgcn                   │ action.model_path │ ST-GCN       │
  └─────────────────────────────────────────────────────────────┘
```

#### 自适应级联状态机

```
状态机定义 (inference_pipeline.h):
  PIPELINE_CASCADE_SEARCHING  = 0  (无确认跟踪 → 全分辨率搜索)
  PIPELINE_CASCADE_TRACKING   = 1  (>=1 确认跟踪 → 降低分辨率)
  PIPELINE_CASCADE_VALIDATING = 2  (定期全分辨率验证)

状态转换逻辑 (cascade_update_state):
  SEARCHING → TRACKING:  确认跟踪数 > 0 且持续 SETTLE_FRAMES 帧
  TRACKING  → VALIDATING: 检测数 > 跟踪数+1 (多人触发) 或定时验证
  TRACKING  → SEARCHING:  确认跟踪数为 0 持续 LOST_FRAMES 帧
  VALIDATING → TRACKING:  立即返回 (1 帧验证后)
```

级联状态机的设计理念来源于自适应推理（Adaptive Inference）范式。参考论文: "Dynamic Neural Networks: A Survey" (TPAMI 2021)，其核心思想是根据运行时条件动态调整推理策略。

#### 帧处理流程 (inference\_pipeline\_process\_frame)

```
统一模式 (cascade_enabled=false):
  1. YOLOv8-Pose 推理 (480×480)
  2. 关键点验证过滤 (三层回退)
  3. 动作识别 (每 action_inference_interval 帧一次)
  4. 人脸检测 (动态频率)
  5. 人脸识别 (检测到人脸时)
  6. 组装 InferenceResult
```

#### 人员检测过滤器 (filter\_detections)

实现了多层过滤管线:

1. **基础几何过滤**: 置信度、面积比例、高度比例、宽高比
2. **人体宽高比检查**: h/w ∈ \[1.2, 4.5] (真人站立/行走比例)
3. **三层姿态验证**:
   - Tier 1: 全身 quick\_check (双肩+双髋+双膝)
   - Tier 2: 上半身 fallback (关键点0-10)
   - Tier 3: 侧身 fallback (单侧链)
4. **框内NMS**: 使用 person\_nms\_iou\_threshold=0.30 做类内抑制

**重要设计决策**: 当关键点验证失败时，部分身体检测（partial body）不会被拒绝，而是标记 `is_partial_body=true` 并给予置信度提升 (+0.05)，交给跟踪器以遮挡感知模式处理。

**参考来源**:

- 级联状态机: "GoTO: Generalizable Tracking-by-Detection with Online Training" (WACV 2023)
- 关键点验证三层回退: "Occlusion-Aware Multi-Person Pose Estimation" (CVPR 2020 workshops)
- 动态频率人脸检测: "RetinaFace: Single-Shot Multi-Level Face Localisation" (CVPR 2020)

***

### 4.3 YOLO11n-Pose 姿态估计器

**文件**: [src/yolov8\_pose\_estimator.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/yolov8_pose_estimator.c)
**头文件**: [include/yolov8\_pose\_estimator.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/yolov8_pose_estimator.h)

#### 功能概述

YOLO11n-Pose 是项目的核心模型，采用 Ultralytics YOLO11n-Pose 统一模型，同时输出人员检测框和 17 个 COCO 关键点。支持标准 ONNX 和 xquant INT8 量化分割模型两种格式。

**官方资料**:

- Ultralytics YOLO11: <https://github.com/ultralytics/ultralytics>
- YOLO11 Pose 文档: <https://docs.ultralytics.com/tasks/pose/>
- ONNX 导出指南: <https://docs.ultralytics.com/modes/export/>

**模型规格**:

- 参数量: 2.9M，计算量: 7.4 GFLOPs
- COCO-Pose mAP: 50.0
- 当前配置输入: 480×480 (相比默认 640×640 减少 44% 像素，推理速度提升约 40%)

#### 预期功能与实现对照

| 预期功能          | 实现状态 | 实现方法                                 |
| ------------- | ---- | ------------------------------------ |
| 统一检测+姿态估计     | 已实现  | 单个 YOLO11n-Pose 模型同时输出 bbox + 17 关键点 |
| INT8 量化支持     | 已实现  | xquant 分割格式自动检测与缓存                   |
| DFL 置信度替代     | 已实现  | 量化 cls 头不可用时使用 DFL peakiness         |
| OKS-based NMS | 已实现  | COCO 标准 OKS 替代 IoU 做姿态级 NMS          |
| Top-K 预过滤     | 已实现  | 标准路径从 8400 候选取前 150                  |
| 多尺度输出解析       | 已实现  | stride 8/16/32 三层 FPN                |

#### COCO 关键点索引

```
索引:  0=Nose,  1=LeftEye,  2=RightEye,  3=LeftEar,  4=RightEar
       5=LeftShoulder, 6=RightShoulder, 7=LeftElbow, 8=RightElbow
       9=LeftWrist, 10=RightWrist, 11=LeftHip, 12=RightHip
       13=LeftKnee, 14=RightKnee, 15=LeftAnkle, 16=RightAnkle
```

**COCO 关键点 Sigma 常量** (用于 OKS 计算):

```
nose:0.026  eyes:0.025  ears:0.035  shoulders:0.079  elbows:0.072
wrists:0.062  hips:0.107  knees:0.087  ankles:0.089
```

来源: COCO 数据集关键点评估指标，参见 <https://cocodataset.org/#keypoints-eval>

#### 核心技术: DFL Peakiness 置信度

INT8 量化后 YOLO 模型的分类头 (cls head) 精度崩塌（softmax 饱和，所有输出 \~0.5）。本项目的解决方案:

```
DFL (Distribution Focal Loss) 回归头输出 4 组 × 16 bins 分布
对每组做 softmax → 取峰值 → 计算几何均值作为代理置信度

dfl_conf = (peak_0 × peak_1 × peak_2 × peak_3)^(1/4)

均匀分布: dfl_conf ≈ 0.0625
弱信号:   dfl_conf ≈ 0.10
清晰信号: dfl_conf ≥ 0.25
```

**参考来源**:

- DFL 论文: Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection" (NeurIPS 2020)
- 论文链接: <https://arxiv.org/abs/2006.04388>
- 量化感知技巧来源: PPQ (PPL Quantization) 社区实践，<https://github.com/openppl-public/ppq>

#### OKS-based NMS

```
OKS = Σ_i[exp(-d_i² / (2s²k_i²)) · δ(v_i>0)] / Σ_i[δ(v_i>0)]

其中:
  d_i = 两个姿态间关键点 i 的欧氏距离
  s   = sqrt(bbox_area) (目标尺度)
  k_i = COCO per-keypoint sigma 常量
  v_i = 双边置信度均 > 0.3 时有效
```

相比传统 IoU NMS 的优势: OKS 考虑了关键点空间距离和不同关节点对不同难度，对姿态估计更鲁棒。当无相互可见关键点时回退到 bbox IoU。

**参考来源**:

- COCO 关键点评估: <https://cocodataset.org/#keypoints-eval>
- OKS-NMS 实践: Ultralytics YOLOv8-Pose NMS 实现

#### 推理流程

```
1. 输入预处理: letterbox 缩放 → normalize CHW (scale=1/255)
2. 输出格式检测 (仅首帧):
   - 标准格式: 1个输出 [1,56,8400]
   - xquant分割: 9个输出 (reg/cls/kpt × 3 strides)
3. 格式缓存: 首帧探测后缓存 is_xquant_split 标志
4. DFL 解码边界框 (64通道分布 → 4个坐标)
5. 置信度计算: DFL peakiness (量化模型)
6. 关键点解码: (kpt_x, kpt_y) = anchor + offset + stride
7. 坐标映射回原图: yolo_map_to_original()
8. OKS-based NMS 过滤冗余检测
```

#### 可借鉴改进方向

1. **2s-AGCN 替代 ST-GCN**: 使用自适应图卷积网络（Shi et al., CVPR 2019），精度提升约 2-3%。GitHub: <https://github.com/lshiwjx/2s-AGCN>
2. **YOLO11 Pose 使用更小的 n 模型变体**: 已有 YOLO11n-Pose (nano, 2.9M)，可评估 s/m 模型在 K1 上的性能
3. **动态输入尺寸**: 当前级联模式中 TRACKING 状态未实际降低模型输入分辨率（ONNX 固定输入 shape），可通过 ONNX 动态 shape + ORT reshape 实现

***

### 4.4 YOLO 后处理

**文件**: [src/yolo\_postprocess.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/yolo_postprocess.c)
**头文件**: [include/yolo\_postprocess.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/yolo_postprocess.h)

#### 功能概述

提供 YOLO 系列模型共用的后处理工具函数，包括图像预处理、DFL 解码、量化模型输出解析、NMS 抑制等。

#### 关键函数

**yolo\_softmax\_stable**: 数值稳定的 softmax 实现，使用 max-shift 技巧防止指数溢出。

**yolo\_dfl\_decode\_position**: DFL (Distribution Focal Loss) 解码函数。

```
输入: reg 分布的 64 通道数据 → 对每个坐标应用 softmax → 加权求和
输出: 4 个坐标偏移 [left, top, right, bottom]
返回: DFL peakiness (4 个分布的几何均值)
```

**yolo\_preprocess**: 图像预处理。

```
步骤: letterbox → normalize CHW
参数: scale=1/255, mean=0, std=1
特殊处理: 量化模型使用 crop_center 替代 letterbox
```

**yolo\_detect\_xquant\_split**: xquant 量化分割模型输出解析。

```
处理 YOLO 模型被 xquant 工具分割成的 9 个输出张量
group_indices 定义: 3 个 FPN 层 × 3 种输出 (reg/cls/kpt)
```

**yolo\_nms\_suppress**: 通用 NMS 抑制。

```
参数: 置信度阈值、IoU 阈值、最大输出数、相似度函数指针
支持任意类型的检测对象 (通过 item_size 和函数指针)
使用贪心排序+抑制算法
```

#### 数据来源

- softmax 稳定实现: 标准数值计算技巧 (max-shift)，参考 Numerical Recipes
- DFL 解码: Li et al., "Generalized Focal Loss" (NeurIPS 2020), <https://arxiv.org/abs/2006.04388>
- letterbox 预处理: Ultralytics YOLO 标准预处理流程

***

### 4.5 关键点验证器

**文件**: [src/keypoint\_validator.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/keypoint_validator.c)
**头文件**: [include/keypoint\_validator.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/keypoint_validator.h)

#### 设计动机

INT8 量化后的 YOLO 模型分类头精度不足，DFL peakiness 无法区分人体和椅状物体。关键点验证器通过检查关键点的**空间排列**来验证是否为真实人体 —— 真人具有一致的肢体比例、对称的左右配对、头在上肩在下的空间关系。

#### 预期功能与实现对照

| 预期功能      | 实现状态 | 实现方法                                 |
| --------- | ---- | ------------------------------------ |
| 最小关键点数门控  | 已实现  | valid\_kpts >= min\_keypoints (默认6)  |
| 必备解剖结构门控  | 已实现  | 至少一组对称肢体 (肩/髋/膝) 或 鼻+肩               |
| 头部位置强制约束  | 已实现  | 鼻在边界框顶部 40% 内                        |
| 肩部有效性检查   | 已实现  | 宽度比/对称性/非退化三方面                       |
| 躯干比例检查    | 已实现  | shoulder-hip垂直距/肩宽 ∈ \[0.40, 2.80]   |
| 肢体比例检查    | 已实现  | 大腿/小腿比 ∈ \[0.50, 4.00]               |
| 左右对称性检查   | 已实现  | 6对COCO对称关键点Y坐标对齐                     |
| 边界框内关键点比例 | 已实现  | >= 70% 关键点在框内                        |
| 人体宽高比预过滤  | 已实现  | h/w ∈ \[1.2, 4.5], area∈\[0.3%, 45%] |
| 上半身回退     | 已实现  | 关键点0-10, >=4个有效, 至少一肩                |
| 侧身回退      | 已实现  | 单侧链(8个关键点), >=3个有效                   |

#### 验证器工作流程

```
输入: PoseEstimation + BoundingBox
     │
     ▼
┌─────────────────────┐
│ 0. 人体宽高比预过滤   │  h/w ∈ [1.2, 4.5] + area 比例检查
└────────┬────────────┘
         │ fail → REJECT (非人体比例 = 物体)
         ▼
┌─────────────────────┐
│ 1. 关键点有效性检查  │  统计置信度 >= threshold 的关键点数量
└────────┬────────────┘
         │ n_valid < min_keypoints → REJECT
         ▼
┌─────────────────────┐
│ 2. 必备解剖门控      │  至少一组对称肢体 + 头部在边界框顶部
└────────┬────────────┘
         │ fail → REJECT
         ▼
┌─────────────────────┐
│ 3. 肩膀有效性检查    │  width ratio + 对称性 + 非退化 (权重0.25)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 4. 躯体比例检查      │  torso ratio ∈ [0.40, 2.80] (权重0.20)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 5. 四肢比例检查      │  上下腿比 ∈ [0.50, 4.00] (权重0.15)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 6. 边界框内关键点比例 │  >= 70% in bbox (权重0.10)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 7. 左右对称性检查    │  6对COCO对称点Y对齐 (权重0.15)
└────────┬────────────┘
         │
         ▼
    加权打分 (总分 >= 0.50 → 通过)
```

#### 三种验证模式

| 模式    | 函数                                        | 适用场景                                 |
| ----- | ----------------------------------------- | ------------------------------------ |
| 完整检查  | keypoint\_validator\_validate()           | 返回详细的 KeypointValidityResult (7 项评分) |
| 快速检查  | keypoint\_validator\_quick\_check()       | 全 17 点快速通过/拒绝 (4 条快速路径)              |
| 上半身检查 | keypoint\_validator\_upper\_body\_check() | 下身被遮挡时检查上半身 (索引 0-10, >=4 个)         |
| 侧身检查  | keypoint\_validator\_side\_body\_check()  | 侧视遮挡时检查单侧 (左或右, >=3 个)               |

#### 可调阈值（默认值）

| 参数                   | 值    | 说明                 |
| -------------------- | ---- | ------------------ |
| min\_keypoints       | 6    | 最少有效关键点数           |
| kpt\_conf\_threshold | 0.30 | 关键点最小置信度           |
| validity\_threshold  | 0.50 | 通过验证的综合分数阈值        |
| in\_bbox\_ratio      | 0.70 | 关键点在边界框内的最小比例      |
| symmetry\_tolerance  | 0.15 | 对称性容差系数（× bbox\_h） |

#### 参考来源

- 解剖学约束: "A Comprehensive Study on Human Pose Estimation" (IJCV 2021)
- 对称性检查: COCO 关键点评估标准中的 OKS 指标衍生
- 人体测量学比例: Pheasant & Haslegrave, "Bodyspace: Anthropometry, Ergonomics and the Design of Work", 3rd ed, CRC Press 2005
- 类似方案: "PoseFilter: Anatomical Validation for Robust Human Detection" - 社区开源方案，GitHub 多个实现

***

### 4.6 目标跟踪器

**文件**: [src/tracking\_manager.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/tracking_manager.c)
**头文件**: [include/tracking\_manager.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/tracking_manager.h)

#### 功能概述

基于 ByteTrack 风格 + DeepSORT 级联匹配的多目标跟踪器，融合卡尔曼滤波、外观特征匹配、匈牙利算法最优分配、遮挡处理和重识别功能。

**官方资料**:

- ByteTrack: <https://github.com/ifzhang/ByteTrack>
- ByteTrack 论文: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- 论文链接: <https://arxiv.org/abs/2110.06864>
- DeepSORT: <https://github.com/nwojke/deep_sort>
- 匈牙利算法: Kuhn, "The Hungarian Method for the Assignment Problem" (Naval Research Logistics, 1955)

#### 预期功能与实现对照

| 预期功能            | 实现状态 | 实现方法                                |
| --------------- | ---- | ----------------------------------- |
| ByteTrack 三阶段匹配 | 已实现  | 高分匹配 → IoU匹配 → 低分遮挡恢复               |
| 级联匹配            | 已实现  | 按 time\_since\_update 升序分组匹配        |
| 匈牙利算法最优分配       | 已实现  | Kuhn-Munkres O(n³)，预分配 workspace    |
| 7 状态卡尔曼滤波       | 已实现  | \[cx, cy, area, aspect, vx, vy, vs] |
| 12 维外观特征        | 已实现  | 从姿态关键点提取，无需额外模型                     |
| 外观特征画廊          | 已实现  | 每 track 保留最多 50 个历史特征               |
| 重识别池            | 已实现  | 已删除 track 特征保留在 reid\_pool          |
| 遮挡感知            | 已实现  | 上下半身/单侧回退 + 扩展生命周期                  |
| 多人检测触发          | 已实现  | detections > tracks×2 时信号级联         |
| Ghost box 抑制    | 已实现  | lost\_count > 3 的 track 不输出         |
| 空间跳跃检测          | 已实现  | 跳跃 > 3m 时扣减 consec\_hits            |
| EMA 空间平滑        | 已实现  | alpha=0.30 平滑空间坐标                   |

#### 跟踪管道 (object\_tracker\_update)

```
输入: Detection[] + SpatialPosition[] + frame_num
     │
     ▼
Step 0: Re-ID 池老化 (每 reid_pool_max_age/2 帧清理一半)
     │
     ▼
Step 1: 卡尔曼预测 (所有活跃 track)
     │
     ▼
Step 2: 检测分类
     confidence >= HIGH (0.18) → D_high
     confidence ∈ [LOW, HIGH) → D_low
     │
     ▼
Step 3: 级联匹配 (第一级)
     按 hit_streak 降序分组
     cost = IoU_cost×(1-app_weight) + app_cost×app_weight
     匈牙利算法最优分配
     │
     ▼
Step 4: IoU 匹配 (第二级)
     剩余 track ↔ 剩余 high + low dets
     IoU threshold = 0.15 (低，允许部分遮挡)
     │
     ▼
Step 5: 遮挡恢复 (第三级)
     若 IoU 低但外观相似度高 → 部分身体匹配
     │
     ▼
Step 6: 重识别匹配
     新检测 vs reid_pool 已删除 track 外观特征
     cosine_distance < max_dist → 复用旧 ID
     │
     ▼
Step 7: 创建新 track / 移除丢失 track
     未匹配 D_high → 新 track (未确认)
     lost_count > max_lost → 移除 (进入 reid_pool)
     │
     ▼
Step 8: 幽灵框抑制
     lost_count > 3 → 不输出 (预测漂移过大)
     │
     ▼
Step 9: 多人检测标记
     det_high > confirmed_tracks × 2 → id_switches = -1 信号
```

#### 卡尔曼滤波器 (KalmanBoxTracker)

```
7 状态向量: [cx, cy, s, r, vx, vy, vs]
  cx, cy  = 边界框中心
  s       = 面积 (area)
  r       = 宽高比 (w/h)
  vx, vy, vs = 对应速度

状态转移矩阵 (transition): 匀速运动模型
  cx' = cx + vx*dt, cy' = cy + vy*dt, s' = s + vs*dt
  r'  = r (宽高比不变), v 保持不变

过程噪声: Q_pos=0.02, Q_vel=0.04 (针对步行速度~1.2m/s 调参)
测量噪声: R=0.15 (针对 DFL 检测中等噪声水平调参)
```

**参考来源**:

- 卡尔曼滤波: Kalman, "A New Approach to Linear Filtering and Prediction Problems" (1960)
- 7 状态模型: ByteTrack/DeepSORT 中的标准状态向量 `[cx, cy, area, aspect, vx, vy, varea]`，广泛应用于多目标跟踪
  - 社区实现验证: BoxMOT (BotSORT/ByteTrack) 使用 `[x, y, w, h, vx, vy, vw, vh]` 8 状态模型
  - DeepSORT 原始实现使用 `[cx, cy, area, aspect, vx, vy, varea]` 7 状态模型
  - 本项目采用 DeepSORT 的 7 状态模型 (area 而非 w/h 独立跟踪，更稳定)
- 噪声参数调优: 过程噪声 Q\_pos=0.02, Q\_vel=0.04, 测量噪声 R=0.15 针对步行速度 \~1.2m/s 和 DFL 检测噪声水平调参
- 动态卡尔曼滤波: "A Dynamic Kalman Filtering Method for Multi-Object Fruit Tracking" (Sensors 2025) 提出可变遗忘因子卡尔曼滤波，可动态适应观测噪声变化

#### 12 维外观特征 (AppearanceFeature)

```
从 COCO-17 关键点空间关系提取的紧凑描述子:
  descriptor[0]:  躯干比例 (shoulder-hip 垂直距/肩宽)
  descriptor[1]:  肩宽/bobx宽度比
  descriptor[2]:  左臂比例 (上臂/前臂)
  descriptor[3]:  右臂比例 (上臂/前臂)
  descriptor[4]:  左腿比例 (大腿/小腿)
  descriptor[5]:  右腿比例 (大腿/小腿)
  descriptor[6]:  头-肩比例 (头到肩中/bbox_h)
  descriptor[7]:  左右肩对称性 (Y差/bbox_h)
  descriptor[8]:  左右髋对称性 (Y差/bbox_h)
  descriptor[9]:  标准化身长 (nose→最低点/bbox_h)
  descriptor[10]: 质心X偏移 (置信度加权)
  descriptor[11]: 质心Y偏移 (置信度加权)

特征距离: cosine distance → [0, 1]
匹配门控: distance < 0.50
最小关键点: >= TRACKING_APPEARANCE_MIN_KPTS 个有效关键点
```

**设计优势**: 无需额外的 CNN ReID 模型（如 DeepSORT 使用的行人重识别网络），直接从已有的姿态关键点提取外观特征，零额外计算开销。

**参考来源**:

- DeepSORT 外观特征: Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (ICIP 2017)
- 本项目创新: 用姿态关键点空间关系替代 CNN 特征提取

#### 匈牙利算法实现

```
Kuhn-Munkres 算法，O(n³) 复杂度:
1. 将 cost 矩阵填充为方阵 (n×n)
2. 初始化势 u[n], v[n] 和匹配数组 p[n]
3. 对每行执行增广路径搜索
4. 更新势并增广匹配
5. 输出: assignment[i] = j

预分配 workspace: tracker->hungarian_ws 在 object_tracker_create 时分配
  (cost_padded, u, v, p, way, minv, used)
```

**参考来源**:

- 匈牙利算法: Kuhn, "The Hungarian Method for the Assignment Problem" (1955)
- 高效实现: <https://github.com/mcximing/hungarian-algorithm-cpp>

#### 跟踪确认策略

```
新 track 需要满足以下条件才标记为 confirmed:
  - 连续命中 >= confirmation_frames 帧 (3 帧)
  - 最大关键点数 >= min_keypoints_strong (6)
  - 关键点稳定性 >= min_keypoints_for_confirm (3)
  - 无空间跳跃 > spatial_jump_max_m (3.0m)

未确认 track 在首次丢失时立即删除 (不进入 reid_pool)
```

#### 可借鉴改进方向

**2024-2026 最新论文与社区资料**:

| 论文/项目                   | 来源                                 | 关键贡献                                                  | 对本项目的参考价值                     |
| ----------------------- | ---------------------------------- | ----------------------------------------------------- | ----------------------------- |
| ByteTrackPro            | ICAIE 2024, ACM 2025, 南京航空航天大学     | SMM (仿真运动模型) + MBN (基于运动的噪声) + MMD (运动马氏距离)，替代 IoU 匹配 | 扩展卡尔曼至含加速度，马氏距离替代 IoU，运动建模更精确 |
| KAD-SORT                | ACM 2025                           | 加速卡尔曼 + GIoU 匹配，MOTA 提升 2%                            | GIoU 对遮挡场景更鲁棒                 |
| UKFTrack                | 农业工程学报 2025, 41(7):145-155         | 无迹卡尔曼滤波 + 多阶段匹配，极端遮挡 HOTA 提升 5.9-13.3%                | UKF 对非线性运动跟踪优于 EKF/KF         |
| FeatureSORT             | arXiv 2407.04249, 2024, Pintel Co. | 多特征模块（方向/颜色/风格/ReID）融合关联距离函数                          | 多特征融合可降低 ID Switch            |
| 改进DeepSORT (UKF)        | Sensors 2022, PMC9741288           | YOLOv5 + UKF + 自适应因子，非线性误差校正                          | UKF 替代线性 KF 是本项目可行的升级路径       |
| 改进DeepSORT (OSNet+GIoU) | Sensors 2024, 24(21):7014, MDPI    | OSNet 轻量级外观提取 + GIoU 运动度量，MOTA +4.6%, IDF1 +5.9%      | GIoU 替代 IoU 简单有效              |

1. **引入加速度状态**: 参考 ByteTrackPro 的可变速度模型 `[cx, cy, area, aspect, vx, vy, varea, ax, ay, aarea]` 10 状态，对快速运动/相机抖动更鲁棒
2. **马氏距离替代 IoU**: MMD 利用卡尔曼协方差矩阵，不确定区域大时自动放宽匹配门限
3. **无迹卡尔曼滤波 (UKF)**: 参考 UKFTrack，对非线性运动（转弯/加速）建模精度优于线性 KF，HOTA 提升显著
4. **多特征融合**: 参考 FeatureSORT，在现有 12 维姿态特征基础上加入方向/速度特征
5. **GIoU 距离度量**: 参考 Sensors 2024 论文，GIoU 在遮挡场景下比 IoU 更鲁棒

***

### 4.7 空间定位引擎

**文件**: [src/spatial\_engine.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/spatial_engine.c)
**头文件**: [include/spatial\_engine.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/spatial_engine.h)

#### 功能概述

基于单目视觉的深度估计和空间定位引擎。利用人体解剖学比例和相机标定参数，从 2D 检测框和 17 个姿态关键点推算 3D 空间坐标。

**理论基础**:

- 针孔相机模型: Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed, Cambridge 2004
- 人体测量学: Pheasant & Haslegrave, "Bodyspace", 3rd ed, CRC Press 2005
- 多点解剖采样: FOV zone model (Measurement 236, Elsevier 2024) + DGC geometric constraints (J. Electronic Imaging 2024)

#### 预期功能与实现对照

| 预期功能       | 实现状态 | 实现方法                               |
| ---------- | ---- | ---------------------------------- |
| 多点解剖采样深度估计 | 已实现  | 8 对关键点独立估计 + 3 点 bbox 回退           |
| MAD 鲁棒融合   | 已实现  | 中位数 + 绝对偏差外值剔除                     |
| 透视倾斜校正     | 已实现  | cos(pitch) × cos(vertical\_offset) |
| 深度 EMA 平滑  | 已实现  | per-track EMA，alpha 随置信度自适应        |
| 深度一致性检查    | 已实现  | 新深度 vs 最近 5 点滚动平均                  |
| 世界坐标系初始化   | 已实现  | 第一帧检测位置设为原点                        |
| 空间坐标平滑     | 已实现  | alpha=0.25 坐标平滑                    |
| 速度估计       | 已实现  | 轨迹最近 2 点差分/dt                      |
| 身高估计       | 已实现  | 姿态关键点 nose→ankle 像素度量              |
| 外部深度图融合    | 预留接口 | depth\_map 参数已定义但未使用               |

#### 深度估计原理 (estimate\_depth\_from\_pose)

```
核心公式: Z = Havg × body_ratio × fy × cos(pitch) × cos(alpha) / pixel_h

其中:
  Havg      = 平均身高 (1.70m, 可配置)
  fy        = 相机 Y 轴焦距
  pixel_h   = 关键点对在图像中的像素高度
  cos(pitch)= 相机俯仰角余弦 (IMU 提供)
  cos(alpha)= 垂直视角余弦 (像素位置偏离光心的角度)

8 对解剖学关键点对:
  Nose → Left/Right Ankle    (近似全身高度)
  Left/Right Shoulder → Ankle (肩高 - 踝高比例)
  Left/Right Hip → Ankle      (髋高 - 踝高比例)
  Left/Right Knee → Ankle     (膝高 - 踝高比例)
```

#### 解剖学常数 (人体测量学)

| 常量                           | 值     | 含义      |
| ---------------------------- | ----- | ------- |
| ANTHRO\_ANKLE\_TO\_HEIGHT    | 0.039 | 踝到地面/身高 |
| ANTHRO\_KNEE\_TO\_HEIGHT     | 0.285 | 膝到地面/身高 |
| ANTHRO\_HIP\_TO\_HEIGHT      | 0.530 | 髋到地面/身高 |
| ANTHRO\_SHOULDER\_TO\_HEIGHT | 0.818 | 肩到地面/身高 |

**参考来源**:

- 人体测量学数据: Pheasant & Haslegrave, "Bodyspace", 3rd ed, 2005 (英国成年人)
- 本项目使用: 中国成年人平均身高 1.70m (国家体育总局 2020 数据)

#### MAD 鲁棒融合

```
1. 计算所有估计的中位数 (median)
2. 计算绝对偏差 (MAD = median(|xi - median|))
3. 排除超过 depth_outlier_mad_mult × MAD 的离群值
4. 对 inlier 取均值作为最终深度
5. 置信度: 1.0 - min(MAD/(median+0.01), 0.75)
```

**参考来源**:

- MAD 外值剔除: Rousseeuw & Croux, "Alternatives to the Median Absolute Deviation" (JASA, 1993)
- 鲁棒深度估计: "Robust Depth Estimation for Multi-Occlusion in Light Field" (ECCV 2022)

#### EMA 平滑

```
每 track 独立维护深度 EMA:
  alpha = base_alpha × raw_confidence (自适应)
  depth_ema[t] = alpha × new_depth + (1-alpha) × depth_ema[t-1]
  base_alpha = 0.30 (可配置)
```

#### 可借鉴改进方向

1. **MiDaS/DPT 单目深度估计**: 使用轻量级深度估计模型替代解剖学估计，精度更高。但 K1 上额外模型推理开销较大
2. **立体视觉基线**: 如果相机标定精确，可通过多帧三角测量提高深度精度
3. **激光/ToF 传感器融合**: 可结合外部深度传感器做卡尔曼滤波融合

***

### 4.8 ST-GCN 动作识别器

**文件**: [src/stgcn\_action\_recognizer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/stgcn_action_recognizer.c)
**头文件**: [include/stgcn\_action\_recognizer.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/stgcn_action_recognizer.h)

#### 功能概述

基于 ST-GCN (Spatial Temporal Graph Convolutional Network) 的骨骼动作识别。使用滑动窗口输入 30 帧骨架序列，分类 7 种动作。模型为 FP32 格式，在 CPU EP 上运行。

**官方资料**:

- ST-GCN 官方仓库: <https://github.com/yysijie/st-gcn>
- 论文: Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018)
- 论文链接: <https://arxiv.org/abs/1801.07455>
- NTU-RGB+D 数据集: <https://rose1.ntu.edu.sg/dataset/actionRecognition/>

#### 预期功能与实现对照

| 预期功能             | 实现状态 | 实现方法                                      |
| ---------------- | ---- | ----------------------------------------- |
| 30 帧滑动窗口         | 已实现  | 环形 buffer，满时左移丢弃最旧帧                       |
| 双输入流 (pts+mot)   | 已实现  | mot 从 pts 实时计算: mot\[t]=pts\[t+1]-pts\[t] |
| 自动检测输出类别数        | 已实现  | 加载时从输出 shape 读取                           |
| 动态输入名称检测         | 已实现  | 从 ONNX 模型查询实际输入名                          |
| 预分配推理张量          | 已实现  | 消除每帧 malloc/free 开销                       |
| 右对齐 zero-padding | 已实现  | 不足 30 帧时左侧补零                              |
| 多姿态积累            | 已实现  | 每帧可推入多个姿态                                 |
| 推理间隔控制           | 已实现  | 默认每 10 帧推理一次                              |

#### 输入格式

```
主输入 (pts): [1, 3, T, V, M] 骨架关键点
  C=3: (x, y, confidence) 归一化坐标
  T=30: 时间窗口帧数
  V=14: 关节点数 (模型特定)
  M=1: 人数

辅助输入 (mot): [1, 2, T, V, M] 运动向量
  C=2: (dx, dy) 帧间关键点位移
```

#### 7 类动作标签

```
0 → walking   1 → sitting   2 → standing   3 → running
4 → jumping   5 → falling   6 → waving
```

模型实际类别数在加载时从输出 shape 自动检测。

#### 骨架推入 (stgcn\_action\_recognizer\_push\_pose)

```
预处理:
  1. 提取 COCO 17 关键点 → 映射到 ST-GCN 14 关节点格式 (取前 V 个)
  2. 坐标归一化: (x, y) / (img_w, img_h) → [0, 1]
  3. 存入骨架缓冲区 skeleton_buffer[C][T][V][M]

缓冲区管理:
  - 缓冲区满时 shift 左移丢弃最旧帧
  - 不足 30 帧时, 推理前用 0 填充缺失帧 (左侧 padding)
```

#### 推理流程 (stgcn\_action\_recognizer\_recognize)

```
1. 准备 pts_tensor [1, 3, 30, V, 1] — 右对齐填充
2. (可选) 计算 mot_tensor — pts 帧间差分
3. ORT 推理 (1 或 2 个输入)
4. 后处理:
   - 筛选分数 >= confidence_threshold (0.50) 的类别
   - 按分数降序排序
   - 返回 ActionResult
5. 每 inference_interval 帧 (默认 10) 执行一次
```

#### 预分配张量

```
prealloc_pts_size = C × T × V × M = 3 × 30 × 14 × 1 = 1260 floats
prealloc_mot_size = 2 × T × V × M = 2 × 30 × 14 × 1 = 840 floats
prealloc_padded_size = 同 pts_size = 1260 floats
```

总预分配: 1260 + 840 + 1260 = 3360 floats ≈ 13.4 KB (极小)

#### 可借鉴改进方向

1. **2s-AGCN**: 自适应图卷积 + 双流 (关节+骨骼)，精度提升 2-3%
   - 论文: Shi et al., CVPR 2019, <https://arxiv.org/abs/1801.07455>
   - GitHub: <https://github.com/lshiwjx/2s-AGCN>
2. **CTR-GCN**: 通道拓扑优化图卷积，NTU-RGB+D 60 SOTA
   - 论文: Chen et al., ICCV 2021
3. **PoseC3D**: 3D CNN 骨骼动作识别，对噪声关键点更鲁棒
   - GitHub: <https://github.com/kennymckormick/pyskl>

***

### 4.9 YOLOv5 人脸检测器

**文件**: [src/yolov5\_face\_detector.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/yolov5_face_detector.c)
**头文件**: [include/yolov5\_face\_detector.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/yolov5_face_detector.h)

#### 功能概述

使用 YOLOv5n-Face INT8 量化模型进行人脸检测。输出 5 点人脸关键点 (两眼、鼻尖、两嘴角)。

**官方资料**:

- YOLOv5-Face GitHub: <https://github.com/deepcam-cn/yolov5-face>
- 论文: Qi et al., "YOLO5Face: Why Reinventing a Face Detector" (ECCV 2022 Workshops)
- 论文链接: <https://arxiv.org/abs/2105.12931>

#### 预期功能与实现对照

| 预期功能             | 实现状态 | 实现方法                              |
| ---------------- | ---- | --------------------------------- |
| INT8 量化支持        | 已实现  | xquant 5D 格式自动检测                  |
| 两种输出格式兼容         | 已实现  | 4D 标准格式 + 5D xquant 格式            |
| 5 点人脸关键点         | 已实现  | 两眼中心、鼻尖、两嘴角                       |
| YOLOv5 anchor 解码 | 已实现  | sigmoid×2-0.5+grid 偏移 + anchor 尺度 |
| 人脸-人员关联          | 已实现  | 人脸中心必须在人员 bbox 内                  |
| 人脸裁剪             | 已实现  | 15% 边缘扩展 + letterbox 缩放到 112×112  |

#### 检测流程

```
1. 输入预处理: 320×320 → letterbox
2. ORT 推理
3. 输出格式自动检测:
   - 5D: xquant 分割，raw logits，手动 sigmoid + anchor 解码
   - 4D: 标准 ultralytics 导出，Detect 层已内置
4. YOLOv5 anchor 解码 (5D 格式):
   cx = (sigmoid(tx)*2 - 0.5 + grid_x) * stride
   cy = (sigmoid(ty)*2 - 0.5 + grid_y) * stride
   w  = (sigmoid(tw)*2)^2 * anchor_w * stride
   h  = (sigmoid(th)*2)^2 * anchor_h * stride
5. 5 点关键点解码
6. NMS 过滤: IoU threshold = 0.40
7. 最小人脸过滤: max(w,h) < 24px → 丢弃
```

#### YOLOv5 Anchor 参数

```
stride=8 (P3):  {4,5}, {8,10}, {13,16}
stride=16 (P4): {23,29}, {43,55}, {73,105}
stride=32 (P5): {146,217}, {231,300}, {335,433}
```

**参考来源**: Ultralytics YOLOv5-face 训练配置中的默认 anchor

#### 人脸裁剪

```
从原图中裁剪人脸区域并缩放到目标尺寸:
  1. 扩大裁剪区域: 检测框向外扩展 15%
  2. letterbox 缩放到 112×112
  3. 提供 RGB 数据供 ArcFace 使用
```

**2024-2026 最新论文与社区资料**:

| 论文/项目                                 | 来源                                          | 关键贡献                                                               | 对本项目的参考价值                                                    |
| ------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------ |
| ONNX-based PTQ Face Detection on Edge | AEEE 2026, Vietnam International University | YOLOv5nu/v8n/v11n INT8 量化在 RPi 4/5 和 Jetson Nano/Orin Nano 的完整基准测试 | 直接对比数据：Orin Nano INT8 70.9FPS, RPi 5 FP16 12.1FPS，CDL 延迟预测指标 |
| YOLO5Face 原始论文                        | ECCV 2022 Workshops                         | Stem Block + Wing Loss + 5 点关键点回归                                  | 本项目参考的核心架构设计                                                 |
| CBAM+YOLOv5 Face                      | ACM 2024                                    | MobileNetV3 + CBAM 注意力 + BiFPN                                     | 轻量化思路可借鉴                                                     |
| Synaptics SyNAP PTQ                   | Synaptics 官方 2025                           | ONNX PTQ 量化部署流程                                                    | 与 xquant 类似的 INT8 量化验证案例                                     |

**可借鉴改进方向**:

1. **YOLOv8n-Face/YOLOv11n-Face 升级**: 参考 AEEE 2026，YOLOv11n 在边缘设备上 FP16 精度优于 YOLOv5nu，mAP 差距约 3-5%
2. **CDL (Critical Datapath Length) 指标**: 预测 ONNX 图结构对特定硬件的推理延迟
3. **TensorRT INT8 优化**: 在 Jetson 平台上可参考实现类似 K1 SpacemiT EP 的加速

***

### 4.10 ArcFace 人脸识别器

**文件**: [src/arcface\_recognizer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/arcface_recognizer.c)
**头文件**: [include/arcface\_recognizer.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/arcface_recognizer.h)

#### 功能概述

使用 ArcFace MobileFaceNet-cuted INT8 量化模型进行人脸特征提取和身份识别。输出 128 维归一化特征向量。

**官方资料**:

- InsightFace (含 ArcFace): <https://github.com/deepinsight/insightface>
- ArcFace 论文: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- 论文链接: <https://arxiv.org/abs/1801.07698>
- MobileFaceNet 论文: Chen et al., "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification" (2018)
- 论文链接: <https://arxiv.org/abs/1804.07573>

#### 预期功能与实现对照

| 预期功能      | 实现状态 | 实现方法                                  |
| --------- | ---- | ------------------------------------- |
| 128 维特征提取 | 已实现  | ORT 推理 → L2 归一化                       |
| 余弦相似度匹配   | 已实现  | 与数据库所有条目计算内积                          |
| 动态人脸注册    | 已实现  | arcface\_recognizer\_register\_face() |
| 人脸数据库     | 已实现  | 内存驻留，最多 256 条目                        |
| 相似度阈值门控   | 已实现  | 0.55 阈值，"Unknown" 低于阈值                |

#### ArcFace 算法原理

```
加法角度边际损失 (Additive Angular Margin Loss):
  L = -log(exp(s·cos(θ_yi+m)) / [exp(s·cos(θ_yi+m)) + Σ_j≠yi exp(s·cos(θ_j))])

其中:
  s: 特征缩放因子 (典型值 64)
  m: 角度边际 (典型值 0.5 弧度)
  θ_yi: 特征与真实类别权重之间的角度
```

ArcFace 的核心创新是在角度空间（而非余弦空间）添加加法边际，在超球面上强制类间分离。

#### 特征提取流程

```
1. 输入预处理: 112×112 RGB → normalize (mean=0.5, std=0.5)
2. ORT 推理 → [1, 128] 特征向量
3. L2 归一化: feature /= ||feature||₂
```

#### 人脸识别流程

```
1. 提取输入人脸的特征向量
2. 与数据库中所有人脸计算余弦相似度:
   similarity = Σ(feat1[i] × feat2[i])  (已归一化，内积 = 余弦)
3. 返回相似度最高且 >= similarity_threshold (0.55) 的身份
4. 低于阈值返回 "Unknown"
```

#### 人脸数据库

```
- 内存驻留数据库, 最多 256 个条目
- 每个条目: identity (名称) + feature[128] + active
- 支持动态注册和清除
- 相似度阈值: 0.55
```

#### 可借鉴改进方向

1. **AdaFace/MagFace**: 质量自适应的人脸识别，对低质量人脸图像更鲁棒
2. **Partial FC**: 大规模人脸识别训练加速技术
3. **在线学习**: 支持在推理过程中增量更新人脸数据库

***

### 4.11 ONNX Runtime 通用层

**文件**: [src/ort\_common.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/ort_common.c), [src/ort\_inference\_context.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/ort_inference_context.c)
**头文件**: [include/ort\_common.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/ort_common.h), [include/ort\_inference\_context.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/ort_inference_context.h)

**官方资料**:

- ONNX Runtime: <https://github.com/microsoft/onnxruntime>
- ONNX Runtime C API: <https://onnxruntime.ai/docs/get-started/with-c.html>
- 执行提供者 (EP): <https://onnxruntime.ai/docs/execution-providers/>
- INT8 量化指南: <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>

#### 预期功能与实现对照

| 预期功能         | 实现状态 | 实现方法                                |
| ------------ | ---- | ----------------------------------- |
| 全局 OrtEnv 单例 | 已实现  | pthread\_once 保证线程安全                |
| EP 自动选择      | 已实现  | 量化模型→SpacemiT EP, FP32→CPU EP       |
| 预分配输入张量      | 已实现  | OrtInferenceContext 缓存              |
| Arena 内存分配器  | 已实现  | OrtArenaAllocator + CPU memory info |
| 输出名称缓存       | 已实现  | 首帧探测后缓存                             |
| TCM 永久失败标记   | 已实现  | 避免重复检测已损坏驱动                         |
| 零拷贝张量创建      | 已实现  | CreateTensorWithDataAsOrtValue      |

#### ONNX Runtime 全局管理 (ort\_common.c)

```
ort_global_init():
  - 创建全局 OrtEnv (ORT_LOGGING_LEVEL_WARNING)
  - 初始化全局 OrtApi 指针
  - 创建全局 OrtAllocator
  - 线程安全 (pthread_once)

ort_create_session(model_path, num_threads, use_ep):
  - 验证 ONNX 文件存在性
  - 创建 SessionOptions:
    * SetIntraOpNumThreads(num_threads): EP=1 (TCM限制), CPU=6
    * SetGraphOptimizationLevel(ORT_ENABLE_ALL)
    * SetExecutionMode(ORT_PARALLEL)
  - EP 门控检查:
    1. use_ep 配置标志
    2. /dev/tcm 设备检测
    3. EP session 计数器 (MAX_EP_SESSIONS=4)
    4. TCM 永久失败标记跳过
    5. 模型量化检查 (spacemit_ort_check_model_supported)
    6. 安全 session 创建 (spacemit_ort_create_session_safe)
  - FP32 模型 → 自动 CPU EP
```

#### 推理上下文 (ort\_inference\_context.c)

```
OrtInferenceContext 封装:
  - 预分配输入张量: 创建时 calloc，维度变化时 realloc
  - 输出名称: lazy-cached (首帧探测)
  - xquant 格式: format_cached + group_indices 缓存

ort_ctx_prepare_input(ctx, data, bytes):
  - 将预处理后的 CHW float 数据复制到预分配张量

ort_ctx_run(ctx, output_vals):
  - 执行 ORT Session.Run()
  - 返回输出 OrtValue 数组
```

#### 关键最佳实践

| 实践                   | 实现                                         | 关键见解                 |
| -------------------- | ------------------------------------------ | -------------------- |
| 单 OrtEnv per process | ort\_global\_init() pthread\_once          | ORT 推荐的最佳实践          |
| Arena 分配器            | OrtArenaAllocator + OrtMemoryInfo          | 减少内存碎片，提升重复推理性能      |
| 量化门控                 | spacemit\_ort\_check\_model\_supported()   | 字节级扫描 QLinearConv 节点 |
| TCM 永久失败             | g\_ep\_tcm\_permanently\_failed 标志         | 避免对已损坏驱动的重复检查        |
| 全局 intra-op 共享       | SPACEMIT\_EP\_USE\_GLOBAL\_INTRA\_THREAD=1 | 节省 TCM 内存            |

***

### 4.12 SpacemiT EP 桥接层

**文件**: [src/spacemit\_ort\_bridge.cpp](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/spacemit_ort_bridge.cpp)
**头文件**: [include/spacemit\_ort\_bridge.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/spacemit_ort_bridge.h)

#### 功能概述

C++ 模块 (唯一非 C 源文件)，负责 SpacemiT Execution Provider (EP) 与 ONNX Runtime 的集成。处理 TCM 内存管理、模型量化验证、异常安全。

**参考来源**:

- SpacemiT K1 SoC: <https://www.spacemit.com/product/k1/>
- Bianbu OS: <https://bianbu.org/>
- RISC-V Vector 规范: <https://github.com/riscv/riscv-v-spec>
- IME (Intelligent Matrix Extension): SpacemiT 自定义 RISC-V 矩阵指令扩展

#### 预期功能与实现对照

| 预期功能       | 实现状态 | 实现方法                                           |
| ---------- | ---- | ---------------------------------------------- |
| TCM 设备检测   | 已实现  | open("/dev/tcm") 探测                            |
| EP 选项设置    | 已实现  | SPACEMIT\_EP\_INTRA\_THREAD\_NUM + USE\_GLOBAL |
| 量化模型检测     | 已实现  | 扫描 ONNX 文件中的 QuantizeLinear/QLinearConv        |
| C++ 异常安全   | 已实现  | try/catch + std::terminate handler             |
| TCM 永久失败标记 | 已实现  | "tcm buffer alloc failed" 检测                   |
| EP 诊断探针    | 已实现  | fork 子进程捕获 libspacemit\_ep 加载错误                |

#### 核心功能详解

**spacemit\_ort\_session\_options\_init** (C++ 异常安全 EP 选项设置):

```
1. 检查 /dev/tcm 是否存在 (无 TCM → 返回 -3)
2. 设置 EP 选项:
   * SPACEMIT_EP_INTRA_THREAD_NUM = g_ep_intra_threads (默认 1)
   * SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1
3. 调用 Ort::SessionOptionsSpaceMITEnvInit()
4. try/catch 捕获异常
```

**spacemit\_ort\_check\_model\_supported** (量化模型检测):

```
扫描 ONNX 文件中的量化标记:
  - "QuantizeLinear" 字符串
  - "QLinearConv" 字符串
  - "DequantizeLinear" 字符串

返回:
   0: 模型已量化，可安全使用 SpacemiT EP
  -3: FP32 模型，不注册 EP (防止崩溃)
```

**spacemit\_ort\_create\_session\_safe** (异常安全 Session 创建):

```
使用 try/catch 保护 CreateSession:
  返回  0: 成功
  返回 -3: OrtStatus 错误
  返回 -4: std::exception
  返回 -5: 未知 C++ 异常
```

#### TCM 内存管理

```
K1 Cluster0 有 512KB 共享 TCM (Tightly Coupled Memory):
  intra=1: ~170KB/session → 最多 3 个 EP session
  intra=2: ~340KB/session → 最多 1 个 EP session
  intra=3: ~510KB/session → 1 个 session
  intra=4: "tcm buffer alloc failed" → 不可用

当前配置: intra=1, 支撑 3 个 EP session
  (YOLO11-Pose + YOLOv5-Face + ArcFace)
```

**重要设计决策**: 本项目**不**在 k1\_platform 中 mmap /dev/tcm，因为 SpacemiT EP (libspacemit\_ep.so) 需要独占访问 /dev/tcm 来内部管理 per-core TCM 缓冲区。k1\_platform 仅标记 K1\_CAP\_TCM 能力位，实际的 TCM 生命周期完全由 libspacemit\_ep.so 管理。

#### xquant 量化工具

```
基于 PPQ v0.6.6+ 开发:
  输入: FP32 ONNX 模型 + 标定数据
  输出: INT8 量化 ONNX 模型 (.q.onnx 后缀)
  格式: QDQ (QuantizeLinear / DeQuantizeLinear)
  精度损失: 通常 < 1%
  性能提升: 数十倍

命令行:
  python -m xquant -c config.json -i model.onnx -o model.q.onnx
```

#### 关键注意事项

1. 只有 `.q.onnx` 模型触发 EP 注册，FP32 模型自动 CPU EP
2. 正确术语: "IME" / "AI Acceleration"，不使用 "NPU"
3. 正确宏: `HAS_SPACEMIT_EP` (非 `HAS_SPACENGINE_NPU`)
4. IME 是集成在 X60 CPU 核心内的矩阵指令扩展，不是独立 NPU
5. K1 有 16 条自定义 AI 加速指令 + RVV 1.0 256-bit 向量协同

***

### 4.13 配置管理器

**文件**: [src/config\_manager.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/config_manager.c)
**头文件**: [include/config\_manager.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/config_manager.h)

#### 功能概述

轻量级 YAML 子集解析器。使用 "section.key" 点分隔路径访问配置值，支持默认值和文件覆盖。不使用第三方 YAML 库（如 libyaml），保持依赖最小化。

#### YAML 支持范围

```
支持:
  ✓ 顶层 key: value 对
  ✓ 嵌套对象 (2 空格缩进)
  ✓ 字符串值 (引号和裸字符串)
  ✓ 整数和浮点数值
  ✓ 布尔值 (true/false/yes/no)
  ✓ 单行注释 (#)
  ✓ 数组值 [a, b, c] → 展开为 key.0, key.1, key.2

不支持:
  ✗ Anchor & alias (&name, *name, << merge)
  ✗ 多行字符串 (|, >)
  ✗ 深层嵌套 (>2 级)
  ✗ 列表子结构
  ✗ Flow style ({} 内联映射, [] 内联列表)
  ✗ 类型标签 (!!str 等)
  ✗ 多文档 (--- 分隔符)
```

#### 配置获取 API

```c
int   config_get_int(cm, "tracking.max_lost", 30);
float config_get_float(cm, "pose.confidence_threshold", 0.10f);
bool  config_get_bool(cm, "system.use_spacemit_ep", true);
const char* config_get_string(cm, "pose.model_path", "models/yolo11n-pose.q.onnx");
```

#### 默认值 (config\_set\_defaults)

关键默认值与 configs/default.yaml 保持同步。

***

### 4.14 模型存储

**文件**: [src/model\_store.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/model_store.c)
**头文件**: [include/model\_store.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/model_store.h)

#### 功能概述

模型文件注册表和 ONNX Session 缓存管理器。通过模型名称查找路径，缓存已加载的 ORT Session 避免重复加载。

#### 模型注册表

```
模型名称                → 实际路径 (由配置指定)
──────────────────────────────────────────────────────────
yolo11n_pose           → models/.../yolo11n-pose.q.onnx
yolov5_face            → models/Face Recognition/yolov5n-face_cut.q.onnx
arcface_mobilefacenet  → models/Face Recognition/arcface_mobilefacenet_cut.q.onnx
stgcn                  → models/.../stgcn.fp32.onnx
```

#### Session 缓存

```
model_store_load_onnx():
  1. 查找模型注册表
  2. 检查缓存: 已有 session → 直接返回
  3. 创建新 session: ort_create_session(path, 4, true)
  4. 缓存 session → 返回
```

***

### 4.15 视频处理器

**文件**: [src/video\_processor.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/video_processor.c)
**头文件**: [include/video\_processor.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/video_processor.h)

#### 功能概述

统一的视频数据源抽象层，支持文件读取 (FFmpeg pipe 后端) 和 V4L2 摄像头采集。

#### 数据源类型

```
VP_SOURCE_FILE (0):   离线视频文件 (FFmpeg pipe)
VP_SOURCE_CAMERA (1): V4L2 摄像头 (/dev/video0)
```

#### V4L2 摄像头采集

参考: Linux V4L2 API 官方文档 <https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/>

```
特性:
  - MMAP 流式采集，4 个缓冲区
  - 格式优先级: MJPEG > YUYV > RGB24
  - select() 带 200ms 超时
  - MJPEG → RGB 使用 libjpeg-turbo
  - YUYV → RGB 使用自定义整数转换
```

#### FFmpeg 解码管道

```
离线模式使用 popen("ffmpeg -i FILE -f rawvideo -pix_fmt rgb24 -")
逐帧 fread(width×height×3) 字节
```

***

### 4.16 数据输入模块

#### Arrow UART 接收器

**文件**: [src/arrow\_receiver.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/arrow_receiver.c)

```
协议格式:
  ┌─────┬─────┬──────┬───────┬──────┬──────┬──────┬─────┬──────┬──────┐
  │ 0xA5│ 0x5A│ Type │ Seq   │ Timestamp     │ Len  │ Payload   │ CRC │
  │     │     │(1B)  │(2B)   │(4B)           │(2B)  │(Len bytes)│(2B) │
  └─────┴─────┴──────┴───────┴───────────────┴──────┴───────────┴──────┘
  CRC: 11 字节头 + payload 的 CRC16-CCITT (poly=0x1021, init=0xFFFF)

帧类型:
  ARROW_TYPE_JPEG (0x01):     JPEG 图像帧
  ARROW_TYPE_IMU_POSE (0x02): IMU 姿态数据 (四元数 + 高度 + 温度)

状态机: IDLE → HEADER → PAYLOAD → CRC → END_MAGIC → COMPLETE
```

Arrow 协议是本项目自定义的 UART 帧传输协议，设计参考了 HDLC 帧结构和 Modbus RTU 的 CRC 校验模式。CRC16-CCITT 多项式 0x1021 是工业标准（与 XMODEM、X.25 相同）。

#### MJPEG HTTP 接收器

**文件**: [src/mjpeg\_receiver.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/mjpeg_receiver.c)

```
Python test-ov-imu.py 的 1:1 C 翻译

协议:
  固定边界: --123456789000000000000987654321
  帧格式:   IMU JSON + JPEG 图像
  IMU: {"ax":..., "ay":..., "az":..., "gx":..., "gy":..., "gz":...}

特性:
  - TCP Socket 非阻塞连接 + TCP_NODELAY
  - WiFi 自动连接 (nmcli)
  - 指数退避重连 (1s → 2s → ... → 16s)
  - 后台 reader 线程持续接收
  - ArrowSourceFrame 兼容 API (可替换 Arrow 接收器)
```

MJPEG over HTTP 使用 multipart/x-mixed-replace 标准格式，参考 RFC 2046。帧边界字符串与 Python 参考实现保持一致。

***

### 4.17 IMU 处理器

**文件**: [src/imu\_handler.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/imu_handler.c)

```
功能:
  - IMU 数据验证 (范围检查: accel ∈ [-16, 16]g, gyro ∈ [-2000, 2000] dps)
  - 滑动窗口平滑 (均值滤波, 默认窗口=10)
  - 外部姿态接收 (Arrow/MJPEG 传来的四元数+欧拉角)
  - 四元数到欧拉角转换 (quaternion_to_euler)

线程安全:
  - mutex 保护外部姿态读写
  - Capture 线程写入, PostProcess 线程读取 (跨 Cluster)
```

***

### 4.18 可视化输出

#### Visualizer

**文件**: [src/visualizer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/visualizer.c)

```
渲染功能:
  - 检测框绘制 (8 种颜色循环)
  - 17 点 COCO 骨架绘制
  - 信息栏 (FPS, 帧号, 对象数, 处理时间)
  - 深度信息显示
  - 轨迹线绘制 (最近 45 帧)
  - 角标记和十字准星
```

#### DisplayOutput (多通道输出)

**文件**: [src/display\_output.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/display_output.c)

```
三通道并行输出:
  ┌───────────┬──────────────────┬─────────────────┐
  │ 通道      │ 线程             │ 实现            │
  ├───────────┼──────────────────┼─────────────────┤
  │ Framebuffer│ fb_writer_thread │ /dev/fb0 mmap  │
  │ RTSP/UDP   │ ffmpeg_writer_thread│ FFmpeg pipe   │
  │ Video File │ file_writer_thread│ VideoWriter    │
  └───────────┴──────────────────┴─────────────────┘

Ring Buffer: 8 槽位, 非阻塞写入
Framebuffer: BGRA/BGR/RGB565 自动检测
FFmpeg 推流: RTSP/UDP/RTMP 协议自动检测
  - rtsp:// → -f rtsp -rtsp_transport tcp
  - udp://  → -f mpegts
  - rtmp://  → -f flv
```

**参考来源**:

- Linux Framebuffer API: <https://www.kernel.org/doc/html/latest/fb/>
- FFmpeg 推流: <https://trac.ffmpeg.org/wiki/StreamingGuide>
- RTSP 推流: <https://trac.ffmpeg.org/wiki/StreamingGuide> (TCP 传输避免丢包)

#### VideoWriter

**文件**: [src/video\_writer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/video_writer.c)

```
FFmpeg 管道写入:
  ffmpeg -f rawvideo -s WxH -pix_fmt rgb24 -r FPS -i -
         -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p output.mp4

SIGPIPE 处理: signal(SIGPIPE, SIG_IGN) + ferror 检测
pclose exit code 2 视为正常 (管道 EOF)
每 30 帧 fflush 一次
```

***

### 4.19 结果管理器

**文件**: [src/result\_manager.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/result_manager.c)

```
会话管理:
  - 创建/结束会话 (时间戳 + 视频路径)
  - 实时统计 (帧数、检测数、FPS)

报告导出:
  - JSON 格式
  - CSV 格式
  - 单独帧保存
```

***

### 4.20 工具函数库

**文件**: [src/utils.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/utils.c)

#### 快速数学函数

```c
utils_fast_exp(x):     快速指数 (整数运算近似)
utils_sigmoid(x):      快速 Sigmoid
utils_fast_sqrt(x):    快速平方根倒数 (Quake III 算法)
```

**参考来源**: Quake III Arena `Q_rsqrt` (id Software, 1999); fast sigmoid 来自 "Efficient Sigmoid Computation" 社区实践

#### 图像处理

```c
utils_resize_image():       最近邻缩放 (OpenMP 并行)
utils_letterbox():          Letterbox 缩放 (保宽高比, 114 灰度填充)
utils_normalize_chw():      HWC → CHW + 归一化 (OpenMP 并行)
utils_rgb_to_bgr():         通道交换
```

**参考来源**: letterbox 算法来自 Ultralytics YOLO 预处理流程

#### JPEG 解码

```c
soft_jpeg_decode_to_rgb():
  使用 libjpeg-turbo (https://libjpeg-turbo.org/)
  TJFLAG_FASTDCT | TJFLAG_NOREALLOC
```

#### 矩阵运算

```c
utils_matrix_inverse_4x4():  4×4 矩阵求逆 (Gauss-Jordan 消元, 选主元)
utils_matrix_multiply_abt(): 7×7 A × B × B^T 乘法
```

***

### 4.21 K1 平台抽象层

**文件**: [src/k1\_platform.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/k1_platform.c)
**头文件**: [include/k1\_platform.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/k1_platform.h)

#### 功能概述

K1 平台抽象层提供硬件能力检测、CPU 亲和性绑定和计时接口。

#### 硬件能力检测

```
检测项             │ 方法                              │ 标记
───────────────────┼───────────────────────────────────┼──────────────
RVV 1.0            │ 编译时已知 (K1 必带)              │ K1_CAP_RVV_1_0
SpacemiT EP        │ 编译时已知 (IME 指令)             │ K1_CAP_SPACEMIT_EP
TCM                │ open("/dev/tcm") ≥ 0              │ K1_CAP_TCM
VPU                │ access("/dev/video0") = 0         │ K1_CAP_VPU
JPU                │ access("/dev/jpu") = 0             │ K1_CAP_JPU
GPU                │ access("/dev/dri/renderD128") = 0 │ K1_CAP_GPU
芯片确认           │ /proc/device-tree/compatible       │ is_k1 flag
CPU 核心数         │ sysconf(_SC_NPROCESSORS_ONLN)      │ 上限 K1_CPU_CORES(8)
```

#### CPU 亲和性绑定

参考: Linux sched\_setaffinity / pthread\_setaffinity\_np API

```c
k1_pin_thread_to_cpu(cpu_id)       → pthread_setaffinity_np
k1_pin_thread_to_cluster(cluster)  → Cluster0 (0-3) 或 Cluster1 (4-7)
k1_get_current_cpu()               → sched_getcpu()
```

#### 高精度计时

```c
k1_get_time_us() → clock_gettime(CLOCK_MONOTONIC, &ts)
k1_get_time_ms() → k1_get_time_us() / 1000.0
```

***

### 4.22 AR 渲染器

**文件**: [src/ar\_renderer.c](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/src/ar_renderer.c)
**头文件**: [include/ar\_renderer.h](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/include/ar_renderer.h)

#### 功能概述

AR 渲染器负责在视频帧上叠加增强现实标记。

#### AR 标记绘制

```
输入: 图像缓冲区 + 中心坐标 + 尺寸 + 颜色 + 标签
输出: 矩形边框 + 白色标签文字 (7×6 像素点阵字体)
颜色: is_alert=true → 红色, false → 绿色
```

#### 运动补偿

```
基于 IMU 欧拉角 (pitch/roll/yaw):
1. 总角度 < 0.0001 → 直接 copy
2. 构建旋转矩阵 R = Rz(yaw) × Ry(pitch) × Rx(roll)
3. 每个输出像素反投影到源像素
4. 最近邻采样 (无插值)
```

**参考来源**: 旋转矩阵反投影: "Multiple View Geometry in Computer Vision" (Hartley & Zisserman, 2004)

***

## 5. 模块间联动逻辑

### 5.1 推理管道→跟踪器联动

```
inference_pipeline_process_frame()
  → 输出 InferenceResult {
      detections[], poses[], faces[], action
    }
  → system_controller 接收结果

system_controller:
  for each detection:
    spatial_engine_calculate_position() → SpatialPosition
  object_tracker_update(detections, positions) → TrackingResult
  associate_poses_with_objects(tracker, poses)
  associate_faces_with_objects(tracker, faces)
  for each tracked object:
    spatial_engine_update_trajectory()
    spatial_engine_get_velocity()
    spatial_engine_check_depth_consistency()
  visualizer_render_detection_view(tracked_objects)
  display_output_write_frame()
```

### 5.2 推理管道→级联状态机→跟踪器联动

```
inference_pipeline:
  confirmed_track_count ← system_controller 同步 (volatile)
  total_track_count     ← system_controller 同步 (volatile)
  cascade_update_state(confirmed_track_count, num_dets, num_tracks)

状态影响推理行为:
  SEARCHING:  YOLO11n-Pose 全分辨率, 人脸每 10 帧
  TRACKING:   YOLO11n-Pose 全分辨率, 人脸每 30 帧
  VALIDATING: YOLO11n-Pose 全分辨率, 人脸每 10 帧 (1帧后回 TRACKING)

特殊触发:
  num_dets > num_tracks + 1 → TRACKING→VALIDATING (新人物进入)
  confirmed_tracks == 0 持续 LOST_FRAMES → TRACKING→SEARCHING
```

### 5.3 跟踪器→空间定位联动

```
跟踪器为每个 track 维护:
  - spatial_pos (EMA 平滑后)
  - trajectory (轨迹历史)
  - velocity (速度估计)

空间定位为每个 track 提供:
  - 深度 EMA 平滑
  - 深度一致性检查
  - 速度估计
  - 轨迹更新
```

### 5.4 关键点验证器→推理管道联动

```
filter_detections() 中:
  for each detection:
    Find best-matching pose (by IoU > 0.35)
    Tier 1: keypoint_validator_quick_check() 全身验证
    Tier 2: keypoint_validator_upper_body_check() 上半身
    Tier 3: keypoint_validator_side_body_check() 侧身
    失败 → 拒绝检测
    上半身/侧身通过 → 标记 is_partial_body=true, confidence+0.05
```

### 5.5 人脸检测→人脸识别联动

```
detect_faces():
  1. YOLOv5-Face 检测人脸
  2. 人脸→人员关联 (人脸中心必须在人员 bbox 内)
  3. 裁剪人脸区域 (15% 扩展 → letterbox 112×112)
  4. ArcFace 提取 128 维特征
  5. 与数据库余弦匹配 → 返回身份或 "Unknown"
```

### 5.6 K1 实时模式线程联动

```
Capture Thread (CPU4, Cluster1):
  1. 数据源选择: V4L2 > Arrow UART > MJPEG HTTP
  2. 帧采集 + JPEG 解码
  3. IMU 姿态提取
  4. Ring Buffer Slot acquire
  5. rgb_data + timestamp → slot
  6. capture_done_cond 信号

Inference Thread (CPU1, Cluster0):
  1. capture_done_cond 等待
  2. inference_pipeline_process_frame()
  3. inference result → slot
  4. infer_done_cond 信号

PostProcess Thread (CPU0, Cluster0):
  1. infer_done_cond 等待
  2. 空间定位 + 跟踪 + 姿态/人脸关联
  3. 可视化渲染
  4. 多通道输出
  5. Slot release → post_done_cond 信号

主线程:
  心跳监控: 每 500ms 检查
  IMU 数据定时推送
```

### 5.7 IMU→空间定位→AR 渲染联动

```
IMU 处理器:
  Arrow/MJPEG 传来的四元数 → 欧拉角转换
  spatial_engine_set_camera_pose(pitch, roll, yaw)

空间定位:
  深度估计时使用 camera_pitch 做透视倾斜校正
  Z = Havg × ratio × fy × cos(pitch) × cos(alpha) / pixel_h

AR 渲染器:
  运动补偿: 基于 IMU 欧拉角做帧旋转反投影
```

***

## 6. 数据流

### 离线模式数据流

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│ 视频文件  │───▶│ VideoProcessor│───▶│ FrameData        │
│ .mp4/.avi│    │ read_frame()  │    │ (RGB raw data)   │
└──────────┘    └──────────────┘    └────────┬────────┘
                                             │
                                             ▼
                          ┌──────────────────────────────────┐
                          │  inference_pipeline_process_frame │
                          │  1. YOLOv8-Pose 推理              │
                          │  2. Keypoint Validator 过滤      │
                          │  3. ST-GCN 动作识别              │
                          │  4. YOLOv5-Face 人脸检测         │
                          │  5. ArcFace 人脸识别             │
                          └──────────────┬───────────────────┘
                                         │ InferenceResult
                                         ▼
                          ┌──────────────────────────────────┐
                          │  spatial_engine_calculate_position│
                          │  → SpatialPosition[]              │
                          └──────────────┬───────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────────┐
                          │  object_tracker_update            │
                          │  (卡尔曼预测 + 级联匹配 + 重识别)  │
                          └──────────────┬───────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              associate_poses    associate_faces      spatial_engine_
              _with_objects      _with_objects        update_trajectory
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                          ┌──────────────────────────────────┐
                          │  visualizer_render_detection_view │
                          │  → vis_buffer (RGB24)             │
                          └──────────────┬───────────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────────┐
                          │  video_writer_write_frame         │
                          │  → FFmpeg pipe → output.mp4       │
                          └──────────────────────────────────┘
```

### K1 实时模式数据流

```
                    ┌─────────────────────────────────────────┐
 数据源              │  V4L2 Camera  │  Arrow UART  │  MJPEG   │
 (优先级: V4L2 > Arrow > MJPEG)      │              │          │
                    └───────┬───────┴──────┬───────┴────┬─────┘
                            │              │            │
                            ▼              ▼            ▼
                    ┌─────────────────────────────────────────┐
                    │  Capture Thread (CPU4, Cluster1)         │
                    │  - frame read + JPEG decode             │
                    │  - IMU pose extraction                  │
                    │  - Ring Buffer Slot acquire             │
                    └────────────────┬────────────────────────┘
                                     │ slot: rgb_data + timestamp
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  Inference Thread (CPU1, Cluster0)       │
                    │  - inference_pipeline_process_frame()    │
                    │  - Slot: inference result                │
                    └────────────────┬────────────────────────┘
                                     │ slot: inference filled
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  PostProcess Thread (CPU0, Cluster0)     │
                    │  - spatial_engine_calculate_position()   │
                    │  - object_tracker_update()               │
                    │  - associate_poses_with_objects()        │
                    │  - associate_faces_with_objects()        │
                    │  - spatial_engine_update_trajectory()    │
                    │  - spatial_engine_get_velocity()         │
                    │  - visualizer_render_detection_view()    │
                    │  - display_output_write_frame()          │
                    │  Slot release → ring buffer free         │
                    └─────────────────────────────────────────┘
```

***

## 7. 配置文件

**文件**: [configs/default.yaml](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/configs/default.yaml)

### 配置结构

```
system      — 系统级配置 (EP 开关, 日志级别, 最大帧数)
video       — 视频采集配置 (摄像头设备, 分辨率, FPS)
arrow       — Arrow UART 配置 (设备路径, 波特率, 双链路)
mjpeg       — MJPEG HTTP 配置 (ESP32 IP, WiFi 凭证)
detection   — 检测器配置 (级联, 关键点验证, 回退阈值)
pose        — 姿态估计配置 (模型路径, 阈值, 输入尺寸)
face        — 人脸检测/识别配置 (模型路径, 阈值, 嵌入维度)
action      — 动作识别配置 (模型路径, 帧数, 类别数)
tracking    — 跟踪配置 (卡尔曼参数, 确认/级联/遮挡/重识别)
spatial     — 空间定位配置 (相机矩阵, 身高, 深度估计参数)
imu         — IMU 配置 (传感器偏差)
visualization — 可视化配置 (显示选项, 颜色, 渲染尺寸)
performance — 性能配置 (RVV, OpenMP, CPU 绑定)
k1_hardware — K1 硬件配置 (集群分配, TCM, VPU, JPU, GPU)
```

***

## 8. 构建系统

**文件**: [CMakeLists.txt](file:///d:/shool/大三下/embedded/lingqi_tantong_c/lingqi_tantong/CMakeLists.txt)

### 构建选项

| 选项                   | 默认值 | 说明                      |
| -------------------- | --- | ----------------------- |
| USE\_ONNX\_RUNTIME   | ON  | ONNX Runtime 推理         |
| USE\_SPACENGINE\_AI  | OFF | SpacemiT RISC-V AI 指令加速 |
| USE\_OPENMP          | ON  | OpenMP 并行处理             |
| ENABLE\_RVV\_OPT     | OFF | RISC-V Vector 1.0 优化    |
| ENABLE\_K1\_PIPELINE | ON  | K1 双集群管道并行              |
| ENABLE\_K1\_TCM      | ON  | K1 TCM 内存管理             |
| ENABLE\_K1\_VPU      | ON  | K1 VPU 硬件视频加速           |
| ENABLE\_K1\_JPU      | ON  | K1 JPU 硬件 JPEG 解码       |

### 编译特性

```
RISC-V 目标:
  - 自动检测 RVV 版本: rv64gcv1p0 / rv64gcv0p7 / rv64gc
  - -mcpu=spacemit-x60 (GCC 14+ 支持时)
  - Release: -O3 -ffast-math -flto
  - Debug:   -O0 -g

外部依赖:
  - ONNX Runtime (SpacemiT 定制版 2.0.2)
  - libspacemit_ep.so (SpacemiT EP 加速库)
  - libonnxruntime_mlas.so (RISC-V MLAS 优化)
  - libjpeg-turbo (JPEG 解码)
  - V4L2 (Linux 摄像头)

链接: -latomic -lpthread -lm -ldl -lstdc++
```

***

## 9. 模型清单

| 序号 | 模型名称                  | 文件名                                | 类型      | 输入尺寸          | 用途              |
| -- | --------------------- | ---------------------------------- | ------- | ------------- | --------------- |
| 1  | YOLO11n-Pose          | yolo11n-pose.q.onnx                | INT8 量化 | 480×480       | 人员检测 + 17 点姿态估计 |
| 2  | YOLOv5n-Face          | yolov5n-face\_cut.q.onnx           | INT8 量化 | 320×320       | 人脸检测 + 5 点关键点   |
| 3  | ArcFace MobileFaceNet | arcface\_mobilefacenet\_cut.q.onnx | INT8 量化 | 112×112       | 人脸特征提取 (128 维)  |
| 4  | ST-GCN                | stgcn.fp32.onnx                    | FP32    | \[1,3,30,V,1] | 骨骼动作识别 (7 类)    |

### EP 分配

```
┌──────────────────────────┬──────────────────────┬─────────────┐
│ 模型                     │ EP                   │ 推理核心     │
├──────────────────────────┼──────────────────────┼─────────────┤
│ YOLO11n-Pose (INT8)      │ SpacemiT EP          │ Cluster0 IME│
│ YOLOv5n-Face (INT8)      │ SpacemiT EP          │ Cluster0 IME│
│ ArcFace (INT8)           │ SpacemiT EP          │ Cluster0 IME│
│ ST-GCN (FP32)            │ CPU EP (RVV 向量)    │ Cluster0    │
└──────────────────────────┴──────────────────────┴─────────────┘

TCM 使用: 3 × ~170KB = ~510KB (接近 512KB 上限)
```

***

## 10. 技术参考资料索引

### 模型与算法

| 技术                           | 官方链接                                                  | 论文                                                          |
| ---------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| YOLO11 / Ultralytics         | <https://github.com/ultralytics/ultralytics>          | <https://docs.ultralytics.com/models/yolo11/>               |
| DFL (Generalized Focal Loss) | —                                                     | Li et al., NeurIPS 2020, <https://arxiv.org/abs/2006.04388> |
| COCO Keypoint Evaluation     | <https://cocodataset.org/#keypoints-eval>             | Lin et al., ECCV 2014                                       |
| ST-GCN                       | <https://github.com/yysijie/st-gcn>                   | Yan et al., AAAI 2018, <https://arxiv.org/abs/1801.07455>   |
| 2s-AGCN                      | <https://github.com/lshiwjx/2s-AGCN>                  | Shi et al., CVPR 2019                                       |
| YOLOv5-Face                  | <https://github.com/deepcam-cn/yolov5-face>           | Qi et al., ECCVW 2022, <https://arxiv.org/abs/2105.12931>   |
| ArcFace / InsightFace        | <https://github.com/deepinsight/insightface>          | Deng et al., CVPR 2019, <https://arxiv.org/abs/1801.07698>  |
| MobileFaceNet                | —                                                     | Chen et al., 2018, <https://arxiv.org/abs/1804.07573>       |
| ByteTrack                    | <https://github.com/ifzhang/ByteTrack>                | Zhang et al., ECCV 2022, <https://arxiv.org/abs/2110.06864> |
| DeepSORT                     | <https://github.com/nwojke/deep_sort>                 | Wojke et al., ICIP 2017                                     |
| 匈牙利算法                        | <https://github.com/mcximing/hungarian-algorithm-cpp> | Kuhn, 1955 / Munkres, 1957                                  |

### 运行时与平台

| 技术                 | 官方链接                                                  |
| ------------------ | ----------------------------------------------------- |
| ONNX Runtime       | <https://github.com/microsoft/onnxruntime>            |
| ONNX Runtime C API | <https://onnxruntime.ai/docs/get-started/with-c.html> |
| ORT 执行提供者          | <https://onnxruntime.ai/docs/execution-providers/>    |
| SpacemiT K1 SoC    | <https://www.spacemit.com/product/k1/>                |
| Bianbu OS          | <https://bianbu.org/>                                 |
| RISC-V Vector 规范   | <https://github.com/riscv/riscv-v-spec>               |
| PPQ 量化工具           | <https://github.com/openppl-public/ppq>               |

### 系统基础设施

| 技术                    | 参考                                                                |
| --------------------- | ----------------------------------------------------------------- |
| Linux V4L2 API        | <https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/> |
| Linux Framebuffer API | <https://www.kernel.org/doc/html/latest/fb/>                      |
| FFmpeg 推流             | <https://trac.ffmpeg.org/wiki/StreamingGuide>                     |
| libjpeg-turbo         | <https://libjpeg-turbo.org/>                                      |
| CRC16-CCITT           | ITU-T X.25, poly=0x1021                                           |

### 学术参考 (深度估计与人体测量)

| 参考       | 出处                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 针孔相机模型   | Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed, Cambridge 2004 |
| 人体测量学比例  | Pheasant & Haslegrave, "Bodyspace", 3rd ed, CRC Press 2005                               |
| MAD 鲁棒统计 | Rousseeuw & Croux, "Alternatives to the MAD", JASA 1993                                  |
| 卡尔曼滤波    | Kalman, "A New Approach to Linear Filtering", 1960                                       |

***

## 11. 社区案例对比与改进建议

### 11.1 本项目 vs 社区成熟方案对比

| 方面   | 本项目实现                               | 社区主流方案                          | 差距分析                                      |
| ---- | ----------------------------------- | ------------------------------- | ----------------------------------------- |
| 姿态估计 | YOLO11n-Pose (INT8) + DFL peakiness | YOLOv8-Pose (FP16) + 标准置信度      | INT8 量化 + DFL 方案是 K1 特化，FP16 方案精度更高但无硬件加速 |
| 跟踪   | ByteTrack + 匈牙利 + 12维外观             | ByteTrack/BoT-SORT + OSNet ReID | 外观特征维度较低但零额外开销，适合 K1                      |
| 动作识别 | ST-GCN (FP32)                       | 2s-AGCN / PoseC3D               | ST-GCN 是最基础方案，2s-AGCN 精度提升 2-3%           |
| 人脸检测 | YOLOv5n-Face                        | SCRFD / RetinaFace              | YOLOv5n-Face 在边缘设备上性价比较好                  |
| 深度估计 | 人体比例 + MAD                          | MiDaS/DPT 或立体视觉                 | 解析学方法零模型开销但精度有限                           |

### 11.2 可借鉴改进建议

#### 短期改进 (低风险)

1. **JPU 硬件 JPEG 解码**: K1 有 VPU/JPU 硬件，添加 `#ifdef HAS_K1_JPU` 路径使用 VPU MJPEG 解码替代 libjpeg-turbo。当前 `soft_jpeg_decode_to_rgb` 已有正确抽象，只需添加第三个 code path。
2. **动态模型输入尺寸**: 当前级联 TRACKING 模式未实际降低模型输入分辨率（ONNX 固定 shape），可通过 ONNX 动态 shape + ORT reshape 实现 480→320 切换，TRACKING 模式下再节约约 55% 推理时间。
3. **ONNX Runtime I/O Binding**: 当前使用 `CreateTensorWithDataAsOrtValue`（零拷贝），可进一步使用 ORT I/O Binding 模式完全消除输入输出拷贝。

#### 中期改进 (中等风险)

1. **2s-AGCN 替代 ST-GCN**: 动作识别精度提升 2-3%，ONNX 导出方式与 ST-GCN 类似。需要重新训练模型。
2. **OC-SORT 替代 ByteTrack**: 改进遮挡恢复策略，对 K1 上低帧率 (3.5 FPS) 场景更鲁棒。参考: <https://github.com/noahcao/OC_SORT>
3. **WiFi 帧传输优化**: Arrow UART 受限于 3Mbaud (实际\~2.5Mbps)，如果用 MJPEG HTTP + WiFi 6 可达到 10-20Mbps，帧率/分辨率可大幅提升。

#### 长期改进 (高风险)

1. **K1 VPU 硬件编码**: K1 VPU 支持 H.265/H.264 4K 编码，可替代 FFmpeg 软件编码，大幅降低 CPU 占用。需要 VPU V4L2 encoder 驱动支持。
2. **TCM 内存池优化**: 当前 3 个 EP session 接近 512KB TCM 上限。如果后续需要更多模型，可考虑 TCM session 动态切换（swap TCM context）。
3. **多摄像头融合**: 当前仅支持单路视频输入，若接入 2 路 MIPI CSI 摄像头可实现立体视觉深度估计，精度远高于当前单目方案。

***

## 附录: 文件索引

### 源文件 (\*.c)

| 文件                          | 所属层级 |
| --------------------------- | ---- |
| main.c                      | 入口层  |
| system\_controller.c        | 控制层  |
| inference\_pipeline.c       | 管道层  |
| yolov8\_pose\_estimator.c   | 模型层  |
| stgcn\_action\_recognizer.c | 模型层  |
| yolov5\_face\_detector.c    | 模型层  |
| arcface\_recognizer.c       | 模型层  |
| keypoint\_validator.c       | 后处理层 |
| tracking\_manager.c         | 后处理层 |
| spatial\_engine.c           | 后处理层 |
| yolo\_postprocess.c         | 后处理层 |
| ort\_common.c               | 运行时层 |
| ort\_inference\_context.c   | 运行时层 |
| config\_manager.c           | 基础设施 |
| model\_store.c              | 基础设施 |
| logger.c                    | 基础设施 |
| utils.c                     | 基础设施 |
| core\_types.c               | 基础设施 |
| k1\_platform.c              | 基础设施 |
| video\_processor.c          | 数据层  |
| arrow\_receiver.c           | 数据层  |
| mjpeg\_receiver.c           | 数据层  |
| imu\_handler.c              | 数据层  |
| visualizer.c                | 输出层  |
| ar\_renderer.c              | 输出层  |
| display\_output.c           | 输出层  |
| video\_writer.c             | 输出层  |
| result\_manager.c           | 输出层  |

### C++ 源文件 (\*.cpp)

| 文件                        | 用途                                        |
| ------------------------- | ----------------------------------------- |
| spacemit\_ort\_bridge.cpp | SpacemiT EP C++ 桥接 (异常安全 + 量化检测 + TCM 管理) |

### 头文件 (\*.h)

| 文件                          | 用途               |
| --------------------------- | ---------------- |
| core\_types.h               | 所有核心数据类型和常量      |
| system\_controller.h        | 系统控制器接口          |
| inference\_pipeline.h       | 推理管道接口 + 级联状态机定义 |
| yolov8\_pose\_estimator.h   | YOLOv8-Pose 接口   |
| stgcn\_action\_recognizer.h | ST-GCN 接口        |
| yolov5\_face\_detector.h    | YOLOv5-Face 接口   |
| arcface\_recognizer.h       | ArcFace 接口       |
| keypoint\_validator.h       | 关键点验证接口 + 配置结构   |
| tracking\_manager.h         | 跟踪器接口 (完整配置结构)   |
| spatial\_engine.h           | 空间定位引擎接口         |
| ort\_common.h               | ORT 通用接口         |
| ort\_inference\_context.h   | ORT 推理上下文接口      |
| spacemit\_ort\_bridge.h     | SpacemiT EP 桥接接口 |
| yolo\_postprocess.h         | YOLO 后处理接口       |
| config\_manager.h           | 配置管理器接口          |
| model\_store.h              | 模型存储接口           |
| logger.h                    | 日志接口             |
| utils.h                     | 工具函数接口           |
| k1\_platform.h              | K1 平台抽象接口        |
| video\_processor.h          | 视频处理器接口          |
| arrow\_receiver.h           | Arrow 接收器接口      |
| mjpeg\_receiver.h           | MJPEG 接收器接口      |
| imu\_handler.h              | IMU 处理器接口        |
| visualizer.h                | 可视化接口            |
| ar\_renderer.h              | AR 渲染接口          |
| display\_output.h           | 多通道输出接口          |
| video\_writer.h             | 视频写入接口           |
| result\_manager.h           | 结果管理接口           |

