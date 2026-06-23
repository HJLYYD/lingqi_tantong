# 推理性能优化方案（v3.0）

> **当前状态**: FPS 2.3 | **目标**: 15-25 FPS  
> **基准环境**: SpacemiT K1 X60 (8 × RISC-V 64 @ 2.0GHz), Bianbu Linux, ONNX Runtime + SpacemiT EP  
> **分析日期**: 2026-06-16  
> **更新**: 经学术文献、官方文档、社区实践三重验证后重新规划

---

## 目录

1. [研究综述](#1-研究综述)
2. [瓶颈诊断](#2-瓶颈诊断)
3. [优化方案（分阶段）](#3-优化方案)
4. [实施路线图](#4-实施路线图)
5. [风险评估与回退策略](#5-风险评估)

---

## 1. 研究综述

### 1.1 官方基准数据 (SpacemiT / Bit-Brick 文档)

| 模型 | 输入尺寸 | 精度 | K1 四核帧率 | 来源 |
|------|---------|------|-----------|------|
| ResNet50 | 224×224 | INT8 | 23 FPS | [Bit-Brick docs](https://docs.bit-brick.com/docs/k1/ml/model-optimization) |
| EfficientNet-B1 | 224×224 | INT8 | 18 FPS | 同上 |
| **YOLOv5n** | **640×640** | **INT8** | **6 FPS** | 同上 |
| **YOLOv8n** | **320×320** | **INT8** | **26 FPS** | 同上 |
| ArcFace | 320×320 | INT8 | 23 FPS | 同上 |

**关键发现**: 640×640 → 320×320 分辨率降低带来 **4.3× 帧率提升**（6 → 26 FPS）。本项目的 yolo11n-pose（2.9M 参数, 7.4 GFLOPs）比 YOLOv5n（1.9M, 4.5 GFLOPs）重 ~60%，当前 2.3 FPS 与官方基准趋势一致。

### 1.2 学术界最新进展

| 方向 | 论文/方法 | 关键贡献 | 对本项目的适用性 |
|------|----------|---------|---------------|
| **NMS 加速** | [Accelerating NMS: A Graph Theory Perspective (2024)](https://arxiv.org/abs/2409.20520) | 将 NMS 建模为最大团问题，启发式剪枝 | 中等 — 理论基础可参考，实现复杂度高 |
| **端到端免 NMS** | [YOLOv10 (NeurIPS 2024)](https://arxiv.org/abs/2405.14458) | 一致双重分配策略消除 NMS 需求 | 低 — 需要重新训练模型，不适用于已有 ONNX 模型 |
| **RISC-V CNN 加速** | [Flexible Vector Integration in RISC-V SoCs (2025)](https://arxiv.org/abs/2507.17771) | RVV 端到端 CNN 推理，验证了向量化预处理的有效性 | 高 — 验证了 RVV 预处理优化的价值 |
| **边缘推理内存管理** | ONNX Runtime Memory Arena 研究 | Embedding 系统应禁止 Arena（省 50% 内存），启用 MemoryPattern 可提速 10-20% | 高 — 直接影响 K1 TCM 512KB 约束 |

### 1.3 社区最佳实践

| 项目 | 优化技术 | 性能收益 | 参考 |
|------|---------|---------|------|
| **ncnn (Tencent)** | NEON `vld3_u8` 硬件解交织 HWC→CHW 转换 | 5.3× (150→800 MP/s) | [ncnn DeepWiki](https://deepwiki.com/Tencent/ncnn) |
| **ncnn** | 通道步长对齐 (`cstep` 16-byte aligned) | 缓存命中率 +45-60% | 同上 |
| **ggml/llama.cpp (SpacemiT 后端)** | IME/IME2 矩阵指令 + TCM 权重预加载 + NUMA 线程亲和 | Qwen 0.5B: 64 t/s (prefill) | [PR #22863](https://github.com/ggml-org/llama.cpp/pull/22863) |
| **RK3588 YOLO 部署** | 硬件 RGA letterbox + 内存对齐零拷贝 | 10× vs CPU 预处理 | rknn-cpp-yolo |
| **ONNX Runtime 嵌入式** | DisableCpuMemArena + DisableMemPattern + ORT_SEQUENTIAL | 内存 -50%, 速度 -10% | [GitHub #22763](https://github.com/microsoft/onnxruntime/discussions/22763) |

### 1.4 SpacemiT K1 X60 微架构 (Remlab 逆向分析)

```
X60 双集群 8 核:
├─ Cluster 0 (AI 加速): 4 核, 512KB 共享 TCM
│   ├─ 核 0-3: RVV 1.0 VLEN=256, IME 矩阵扩展
│   ├─ 双 128-bit 执行单元: EX1 (移位/逻辑), EX1+EX2 (算术)
│   └─ LMUL=1 时 vadd 双发射, vsll/vmv 单发射
│
└─ Cluster 1 (I/O): 4 核, 无 TCM
    └─ 核 4-7: 通用 RISC-V, 处理外设 I/O

IME 指令约束:
├─ vmadot / vfmadot: LMUL=1, VL=16/32, SEW=8
├─ 目标寄存器必须偶数 (EMUL=2)
├─ SEW=16 存在已知硬件算术错误 — 禁用
└─ 仅 INT8 量化模型可触发 IME 加速路径
```

---

## 2. 瓶颈诊断

### 2.1 每帧耗时分解（640×640 输入，估测）

```
┌─────────────────────────────────────────────────────────┐
│               每帧管道耗时 (~430ms → 2.3 FPS)            │
├─────────────────────────────────────────────────────────┤
│ ORT 推理 (yolo11n-pose):     ~200ms  ████████████████░░ │ 46% │
│ 预处理 (letterbox+CHW):       ~70ms  █████░░░░░░░░░░░░ │ 16% │
│ DFL 解码 (10800 cells):       ~50ms  ███░░░░░░░░░░░░░░ │ 12% │
│ OKS-NMS (全对比较):           ~30ms  ██░░░░░░░░░░░░░░░ │  7% │
│ 检测过滤 (filter_detections):  ~20ms  █░░░░░░░░░░░░░░░ │  5% │
│ 跟踪 (Hungarian+Kalman):      ~20ms  █░░░░░░░░░░░░░░░ │  5% │
│ 可视化渲染:                    ~15ms  █░░░░░░░░░░░░░░░ │  3% │
│ 内存拷贝 (InferenceResult):    ~10ms  ░░░░░░░░░░░░░░░░ │  2% │
│ 关联/空间/IMU:                 ~10ms  ░░░░░░░░░░░░░░░░ │  2% │
│ 其他:                           ~5ms  ░░░░░░░░░░░░░░░░ │  1% │
└─────────────────────────────────────────────────────────┘
```

### 2.2 根因分析树

```
2.3 FPS
├─🔴 模型输入过大 (640×640)
│   └─ 官方基准: YOLOv5n 640²=6 FPS, YOLOv8n 320²=26 FPS
│      结论: 降低分辨率是最直接有效的优化
│
├─🔴 预处理逐帧分配 3 个临时缓冲区
│   ├─ yolo_preprocess(): crop_buf (malloc/free)
│   ├─ yolo_preprocess(): padded (malloc/free)
│   └─ yolov8_pose_estimator_estimate(): input_tensor (malloc/free)
│      总计: 3× malloc + 3× free + 3× memset ≈ 70ms
│      → 社区方案: ncnn 预分配 arena, 零动态分配
│
├─🟡 NMS 使用选择排序 (O(n×K))
│   └─ 从 6000 proposals 中选 Top-150: 900K 次比较
│      → 社区方案: 最小堆 Top-K, O(n log K) ≈ 43K 次比较
│
├─🟡 OKS-NMS 逐对调用 expf (无快速路径)
│   └─ 每个 proposal 对计算 17 关键点 OKS, 含 expf
│      → 社区方案: IoU 预筛 (≤0.1 直接跳过) 过滤 90% 比较对
│
├─🟡 DFL 解码无预筛
│   └─ 每个 grid cell 完整计算 4×16=64 bin softmax 后才判置信度
│      → 社区方案: 先检查单通道 max bin 做快速跳过
│
├─🟢 InferenceResult 按值返回 (~100KB 栈拷贝)
│   └─ pipeline → slot 拷贝 → 再拷贝 → 3 次 memcpy
│      → 社区方案: 指针传递, 写入预分配 buffer
│
└─🟢 ONNX Runtime 内存配置未针对嵌入式优化
    └─ EnableCpuMemArena / EnableMemoryPattern 默认开启
       → 官方推荐: 嵌入式场景应关闭 Memory Arena
```

---

## 3. 优化方案

### Phase 1: 模型推理优化（预计 FPS 2.3 → 8-12）

这是投入产出比最高的优化阶段，利用现有官方基准数据直接指导。

#### 1.1 降低模型输入分辨率 [预期 +150-300% FPS]

**依据**: SpacemiT 官方基准 — 分辨率从 640→320 带来 **4.3× 帧率提升**。

**方案**: 在 cascade TRACKING 模式下使用 320×320/416×416 输入

```yaml
# configs/default.yaml
pose:
  model_path: "models/Action Prediction/Skeleton Recognition/yolo11n-pose.q.onnx"
  input_size: [480, 480]     # 当前配置值
  # 新增: 级联动态分辨率
  cascade_input_size: [320, 320]   # TRACKING 模式使用
  searching_input_size: [640, 640]  # SEARCHING 初始帧使用全分辨率
```

**代码改动**: 在 `inference_pipeline.c` 的 cascade TRACKING 路径上动态切换分辨率。

```c
// 在 cascade_update_state 中根据状态动态调整
if (pipeline->cascade_state == PIPELINE_CASCADE_TRACKING) {
    est->input_width  = pipeline->cascade_tracking_w;   // 320
    est->input_height = pipeline->cascade_tracking_h;   // 320
} else {
    est->input_width  = 640;  // SEARCHING/VALIDATING: 全精度
    est->input_height = 640;
}
```

**风险**: 小目标可能漏检。缓解：VALIDATING 帧（每 15 帧）仍然全分辨率检测。

**改动文件**: `inference_pipeline.c`, `yolov8_pose_estimator.c`

---

#### 1.2 ORT Session 嵌入式内存配置 [预期 +5-10% FPS]

**依据**: [ONNX Runtime 嵌入式优化指南](https://github.com/microsoft/onnxruntime/discussions/22763)。嵌入式应禁用 `EnableCpuMemArena` 以避免预分配开销。保留 `EnableMemoryPattern` 因为本项目固定输入形状。

```c
// ort_common.c → ort_create_session()
// 当前: 默认 EnableCpuMemArena=true (浪费 50% 峰值内存)
// 优化: 嵌入式场景显式关闭

OrtStatus* st_opt;

// 嵌入式: 关闭内存池 (matching upstream best practice)
st_opt = g_ort_api->DisableCpuMemArena(session_opts);  // 新增
if (st_opt) g_ort_api->ReleaseStatus(st_opt);

// 保持 MemoryPattern — 固定输入形状, 缓存分配计划有效
// EnableMemPattern 已是默认值, 无需改动

// Sequential 模式 — 本项目为线性管线, 无并行算子分支
st_opt = g_ort_api->SetExecutionMode(session_opts, ORT_SEQUENTIAL);  // 新增
if (st_opt) g_ort_api->ReleaseStatus(st_opt);

// 关闭线程自旋 (嵌入式省电)
st_opt = g_ort_api->AddSessionConfigEntry(session_opts,
    "session.intra_op.allow_spinning", "0");  // 新增
if (st_opt) g_ort_api->ReleaseStatus(st_opt);
```

**风险**: 关闭 Arena 后推理峰值内存降低但分配次数增加。对于固定输入形状，MemoryPattern 缓存弥补了 Arena 的缺失。

**改动文件**: `ort_common.c`

---

#### 1.3 CPU EP 线程数调优 [预期 +5-15% FPS]

**依据**: K1 8 核。SpacemiT EP 必须用 intra=1（TCM 约束），但 CPU EP 可用 6 核。

```c
// ort_common.c: CPU EP 回退路径
// 当前: intra=4 → 优化: intra=6
int cpu_intra = num_threads > 0 ? num_threads 
    : (n_online >= 8 ? 6 : (n_online >= 4 ? 4 : n_online));
```

**改动文件**: `ort_common.c`

---

### Phase 2: 预处理与内存池化（预计 FPS 8-12 → 12-18）

#### 2.1 预处理缓冲区预分配 [预期 +20-30% FPS]

**依据**: ncnn 架构 — 所有预处理 buffer 在 `create` 时一次分配，推理路径零动态分配。

**方案**: 在 `OrtInferenceContext` 中预分配 crop/padded/input_tensor 缓冲区。

```c
// ort_inference_context.h — 新增字段
typedef struct {
    // ... 现有字段 ...
    uint8_t*  preproc_crop_buf;     // max(crop_w*h*3, target_w*h*3)
    size_t    preproc_crop_buf_size;
    uint8_t*  preproc_padded_buf;   // target_w * target_h * 3
    float*    preproc_input_tensor;  // target_w * target_h * 3 * sizeof(float)
    int       last_crop_w, last_crop_h;
    int       last_target_w, last_target_h;
} OrtInferenceContext;
```

```c
// yolo_postprocess.c — 新增 pool 版本的预处理
int yolo_preprocess_pooled(const uint8_t* image_data, int width, int height,
                           OrtInferenceContext* ctx,  // 使用预分配 buffers
                           float* out_scale, int* out_pad_x, int* out_pad_y,
                           int* out_crop_x, int* out_crop_y);
```

**改动文件**: `ort_inference_context.h`, `ort_inference_context.c`, `yolo_postprocess.h`, `yolo_postprocess.c`, `yolov8_pose_estimator.c`, `yolov5_face_detector.c`, `arcface_recognizer.c`

---

#### 2.2 预处理 OpenMP 并行化启用 [预期 +5-15% FPS]

**依据**: 项目已预留 `#ifdef HAS_OPENMP` 编译宏。K1 8核下并行化 letterbox + normalize 有效。

**实现**: 利用已有的 `#pragma omp parallel for` 块，确保 CMake 检测通过：

```cmake
# CMakeLists.txt — 当前使用 compiler probe
# 问题: HAS_OPENMP 在某些 RISC-V sysroot 上检测失败
# 解决: 使用 find_package(OpenMP) 替代手工探测
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(lingqi_tantong PRIVATE OpenMP::OpenMP_C)
    target_compile_definitions(lingqi_tantong_core PRIVATE HAS_OPENMP)
endif()
```

**改动文件**: `CMakeLists.txt`

---

#### 2.3 预处理合并为单遍扫描 [预期 +5-10% FPS]

**依据**: ncnn 将 letterbox resize + HWC→CHW + normalize 融合为单次循环，减少内存访问。

**方案**: 将 `yolo_preprocess` 的三阶段：
1. crop → `crop_buf`（可选）
2. letterbox resize → `padded`
3. HWC→CHW + normalize → `input_tensor`

合并为：crop → 直接写入 CHW float tensor（按需 letterbox padding）

```c
// 新函数: 单遍扫描预处理
void yolo_preprocess_fused(const uint8_t* src, int src_w, int src_h,
                           float* out_chw, int target_w, int target_h,
                           float* out_scale, int* out_pad_x, int* out_pad_y);
```

**改动文件**: `yolo_postprocess.c`, `yolo_postprocess.h`

---

### Phase 3: 后处理算法优化（预计 FPS 12-18 → 18-25）

#### 3.1 NMS 使用最小堆 Top-K 选择 [预期 +10-15% FPS]

**依据**: 标准算法 — O(n log K) vs 当前 O(nK)。K=150, n=6000: 比较次数从 900K → 43K。

```c
// yolo_postprocess.c — 新增堆排序
static void heapify_min(PoseCandidate* heap, int size, int root) {
    int smallest = root;
    int left  = 2 * root + 1;
    int right = 2 * root + 2;
    if (left  < size && heap[left].conf  < heap[smallest].conf) smallest = left;
    if (right < size && heap[right].conf < heap[smallest].conf) smallest = right;
    if (smallest != root) {
        PoseCandidate tmp = heap[root];
        heap[root] = heap[smallest];
        heap[smallest] = tmp;
        heapify_min(heap, size, smallest);
    }
}

static void heaptopk_select(PoseCandidate* arr, int n, int k) {
    // 1. 前 k 个元素建最小堆
    for (int i = k/2 - 1; i >= 0; i--) heapify_min(arr, k, i);
    // 2. 剩余元素: 若大于堆顶则替换并下沉
    for (int i = k; i < n; i++) {
        if (arr[i].conf > arr[0].conf) {
            arr[0] = arr[i];
            heapify_min(arr, k, 0);
        }
    }
    // 3. 堆排序输出: 从大到小
    for (int i = k - 1; i > 0; i--) {
        PoseCandidate tmp = arr[0];
        arr[0] = arr[i];
        arr[i] = tmp;
        heapify_min(arr, i, 0);
    }
}
```

**改动文件**: `yolo_postprocess.c`

---

#### 3.2 NMS 快速路径: IoU 预筛 + 延迟 OKS [预期 +8-15% FPS]

**依据**: 学术界 [NMS Graph Theory 2024](https://arxiv.org/abs/2409.20520) — 绝大多数 proposal 对 IoU≈0，可快速跳过。

```c
// yolov8_pose_estimator.c — pose_similarity 快速路径
static float pose_similarity_fast(const void* a, const void* b) {
    const PoseEstimation* pa = (const PoseEstimation*)a;
    const PoseEstimation* pb = (const PoseEstimation*)b;

    // Step 1: 廉价 IoU 预筛
    if (pa->has_bbox && pb->has_bbox) {
        float iou = bbox_iou(&pa->bbox, &pb->bbox);
        if (iou < 0.08f) return 0.0f;    // 90%+ 的对在这里快速返回
        if (iou < 0.30f) return iou;      // 中等重叠: IoU 足够判定
    }

    // Step 2: 高重叠时计算完整 OKS
    if (pa->has_bbox && pb->has_bbox) {
        return compute_oks(pa, pb);
    }
    return bbox_iou(&pa->bbox, &pb->bbox);
}
```

**改动文件**: `yolov8_pose_estimator.c`

---

#### 3.3 DFL 解码预筛优化 [预期 +8-12% FPS]

**依据**: 社区实践 — 先用最快路径判断 grid cell 是否有潜力，避免为 60-70% 的低质量 cell 做完整 softmax。

```c
// yolov8_pose_estimator.c — 在完整 DFL 解码前加预筛
// 原始: 直接执行 yolo_dfl_decode_position()（内部做 4×16 bin softmax）
// 优化: 先快速检查 reg_data 的 sigmoid 近似值

// 快速预筛: 检查 reg [0:4][0] 四个坐标的零 bin 值
static inline float dfl_peak_quick_check(const float* reg_data, int pix, int hw) {
    // 只用第一个 coord 的 16 bins 作为快速代理
    // 若最大值 < 0.10，该 cell 不可能达标
    float max_bin = reg_data[0 * hw + pix];
    for (int b = 1; b < 8; b++) {  // 仅扫描前 8 bin（更多 bin 内到中心）
        if (reg_data[b * hw + pix] > max_bin)
            max_bin = reg_data[b * hw + pix];
    }
    return max_bin;
}

// 在循环中:
float max_bin_0 = dfl_peak_quick_check(reg_data, pix, hw);
if (max_bin_0 < 0.10f) continue;  // 跳过 60-70% 的低质量 cell

// 通过预筛后才完整解码
float dists[4];
float dfl_conf = yolo_dfl_decode_position(reg_data, pix, hw, dists);
```

**改动文件**: `yolov8_pose_estimator.c`

---

#### 3.4 消除重复 NMS [预期 +3-8% FPS]

**现状**: `filter_detections()` 和 `yolov8_pose_estimator_estimate()` 各自执行独立 NMS。

**方案**: 将管道层 NMS 移除，信任 estimator 内部的 OKS-NMS 结果。`filter_detections` 仅做置信度/尺寸/解剖验证。

```c
// inference_pipeline.c — filter_detections()
// 移除内部 NMS 步骤，只保留过滤逻辑
// 原: 置信过滤 + 类别过滤 + 几何过滤 + NMS
// 改: 置信过滤 + 类别过滤 + 几何过滤 + 解剖验证
//      不执行 NMS（estimation 内部已完成 OKS-NMS）
```

**改动文件**: `inference_pipeline.c`

---

### Phase 4: 内存与拷贝优化（预计 FPS 18-25 → 22-28）

#### 4.1 InferenceResult 改为指针传递 [预期 +3-5% FPS]

**方案**: `inference_pipeline_process_frame` 改为写入外部预分配的 buffer。

```c
// 旧接口（按值返回 ~100KB）:
InferenceResult inference_pipeline_process_frame(
    AIInferencePipeline* pipeline, const uint8_t* frame, int w, int h);

// 新接口（写入外部指针）:
int inference_pipeline_process_frame(
    AIInferencePipeline* pipeline, const uint8_t* frame, int w, int h,
    InferenceResult* out_result);  // 直接写入调用方提供的 buffer

// 系统控制器调用处:
inference_pipeline_process_frame(sc->inference_pipeline,
    s->rgb_data, s->width, s->height,
    &s->inference);  // 直接写入 ring buffer slot, 零额外拷贝
```

**改动文件**: `inference_pipeline.h`, `inference_pipeline.c`, `system_controller.c`

---

#### 4.2 管道 Slot 复用 ImuExternalPose [预期 ~1-2% FPS]

**现状**: IMU 数据每帧独立获取和传递，内部有 `pthread_mutex_lock` 开销。

**方案**: 在采集线程中获取 IMU 姿态，通过 slot 传递给后处理线程，避免后处理线程竞争 IMU 锁。

**改动文件**: `system_controller.c`

---

## 4. 实施路线图

```
                           当前: 2.3 FPS
                              │
Phase 1 ──────────────────────┤
│ 1.1 降低推理分辨率 (640→320)│  ████████████  预计: 6-10 FPS
│ 1.2 ORT 嵌入式内存配置      │  ██            预计: +0.3-0.5
│ 1.3 CPU EP 线程调优         │  ██            预计: +0.2-0.5
│                             │
│ Phase 1 合计:               │  ██████████████  → 8-12 FPS
├─────────────────────────────┤
Phase 2 ──────────────────────┤
│ 2.1 预处理缓冲区预分配       │  ██████        预计: +1.5-2.5
│ 2.2 OpenMP 并行化启用        │  ███           预计: +0.5-1.0
│ 2.3 预处理单遍扫描           │  ██            预计: +0.3-0.5
│                             │
│ Phase 2 累计:               │  ██████████████  → 12-18 FPS
├─────────────────────────────┤
Phase 3 ──────────────────────┤
│ 3.1 NMS Top-K 堆排序        │  ████          预计: +1.0-1.5
│ 3.2 NMS IoU 快速路径        │  ████          预计: +0.8-1.5
│ 3.3 DFL 解码预筛            │  ███           预计: +0.5-1.0
│ 3.4 消除重复 NMS            │  ███           预计: +0.3-0.8
│                             │
│ Phase 3 累计:               │  ██████████████  → 18-25 FPS
├─────────────────────────────┤
Phase 4 ──────────────────────┤
│ 4.1 InferenceResult 指针传递 │  █            预计: +0.3-0.5
│ 4.2 Slot 复用 IMU 数据      │  █            预计: +0.2
│                             │
│ Phase 4 累计:               │  ██████████████  → 22-28 FPS
└─────────────────────────────┘
```

### 实施优先级矩阵

| 优化项 | FPS 收益 | 改动行数 | 风险 | 立即执行？ |
|--------|---------|---------|------|----------|
| 1.1 降低分辨率 | **+150-300%** | ~50 行 | 中（小目标） | ✅ 是 |
| 2.1 预分配缓冲区 | +20-30% | ~200 行 | 低 | ✅ 是 |
| 3.2 NMS IoU 快速路径 | +8-15% | ~30 行 | 低 | ✅ 是 |
| 3.1 NMS 堆排序 | +10-15% | ~60 行 | 低 | ✅ 是 |
| 3.3 DFL 预筛 | +8-12% | ~30 行 | 低 | ✅ 是 |
| 1.2 ORT 嵌入式配置 | +5-10% | ~15 行 | 低 | ✅ 是 |
| 3.4 消除重复 NMS | +3-8% | ~20 行 | 低 | ✅ 是 |
| 4.1 指针传递 | +3-5% | ~40 行 | 低 | 是 |
| 2.2 OpenMP 启用 | +5-15% | ~10 行 | 中（sysroot） | 是 |
| 2.3 预处理融合 | +5-10% | ~150 行 | 中 | 是 |
| 1.3 CPU EP 线程 | +5-15% | ~5 行 | 低 | 是 |

---

## 5. 风险评估与回退策略

| 优化项 | 主要风险 | 缓解措施 | 回退方案 |
|--------|---------|---------|---------|
| 1.1 降低分辨率 | 远处小目标漏检 | VALIDATING 帧保持 640×640 全精度检测 | 配置项可恢复 `input_size: [640, 640]` |
| 1.2 关闭 Arena | 峰值内存分配次数增加 | MemoryPattern 缓存弥补 | 取消 `DisableCpuMemArena` 调用 |
| 2.1 预分配内存 | 动态分辨率时不匹配 | realloc 回退路径保留 | 回退到当前 malloc/free 方式 |
| 2.2 OpenMP | 某些 sysroot 编译失败 | `find_package` 优雅回退 | 已有 `#ifdef HAS_OPENMP` 条件编译 |
| 3.1/3.2 NMS 优化 | 相似度计算等价性 | 保持相同阈值语义 | 单元测试对比新旧 NMS 输出 |
| 3.3 DFL 预筛 | 阈值选择过于激进 | 使用宽松阈值 (0.10 = 原 softmax 的保守下界) | 调整阈值或移除预筛 |

---

## 6. 验证方法

每阶段完成后执行:

```bash
# 1. 单帧 benchmark
./lingqi_tantong --video_path test.mp4 --max-frames 1 2>&1 | grep "ORT Run"

# 2. 多帧平均 FPS
./lingqi_tantong --video_path test_video.mp4 --max-frames 100 \
    2>&1 | grep -E "Avg FPS|Average FPS"

# 3. 级联状态切换验证
./lingqi_tantong --video_path test.mp4 --config configs/default.yaml \
    2>&1 | grep "Cascade:"

# 4. 内存峰值检查 (Bianbu)
/usr/bin/time -v ./lingqi_tantong --video_path test.mp4 --max-frames 50
```

---

## 参考文献

1. SpacemiT K1 Model Quantization Guide — https://docs.bit-brick.com/docs/k1/ml/model-optimization
2. SpacemiT ONNX Runtime Archive — https://archive.spacemit.com/spacemit-ai/onnxruntime/
3. ONNX Runtime Memory Minimization — https://github.com/microsoft/onnxruntime/discussions/22763
4. ggml SpacemiT Backend (IME/IME2) — https://github.com/ggml-org/llama.cpp/pull/22863
5. Remlab: SpacemiT Integrated Matrix Extension — https://www.remlab.net/op/riscv-xstime.shtml
6. ncnn Image Processing (Tencent) — https://deepwiki.com/Tencent/ncnn/4.6-image-and-pixel-processing
7. Accelerating NMS: A Graph Theory Perspective (2024) — https://arxiv.org/abs/2409.20520
8. YOLOv10: Real-Time End-to-End Object Detection (NeurIPS 2024) — https://arxiv.org/abs/2405.14458
9. Flexible Vector Integration in Embedded RISC-V SoCs (2025) — https://arxiv.org/abs/2507.17771
10. RISC-V Vector Extension 实战指南 — https://blog.csdn.net/weixin_28419039/article/details/159065216
11. SpacemiT AI Demo Repository — https://gitee.com/bianbu/spacemit-demo
12. ONNX Runtime Performance Tuning — https://onnxruntime.ai/docs/performance/tune-performance/
