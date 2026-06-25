# 推理优化实施计划

> 基于 [INFERENCE_OPTIMIZATION_ANALYSIS.md](INFERENCE_OPTIMIZATION_ANALYSIS.md) 的发现

---

## 总体策略

- **每个 Phase 独立可交付**，完成后立即测试验证
- **全局性修改**（影响多个模块的 API 变更）集中在 Phase 2、Phase 4
- **每个 Phase 附带回滚方案**
- 优先级: P0 (正确性) → P1 (高收益) → P2 (中期) → P3 (锦上添花)

---

## Phase 1: YOLOv5-Face Landmark Decode 修正 [P0]

### 影响范围
- `src/yolov5_face_detector.c` — `decode_5d_proposal()` 函数
- 全局性：否（仅影响人脸 landmark 坐标计算）

### 修改内容

**文件**: `src/yolov5_face_detector.c:209-216`

**当前代码**:
```c
/* ── Landmark decode (5 keypoints × 2 coords, at feat[5..14]) ── */
for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
    float lx = feat[5 + k * 2 + 0];
    float ly = feat[5 + k * 2 + 1];
    float slx = utils_sigmoid(lx);
    float sly = utils_sigmoid(ly);

    out_kpts[k][0] = (slx * 2.0f - 0.5f + (float)grid_x) * (float)stride;
    out_kpts[k][1] = (sly * 2.0f - 0.5f + (float)grid_y) * (float)stride;
}
```

**修改为** (对齐 deepcam-cn 官方 + 社区共识):
```c
/* ── Landmark decode (5 keypoints × 2 coords, at feat[5..14]) ──
 * Official formula (deepcam-cn/yolov5-face):
 *   lm_x = raw_pred_x * anchor_w + grid_x * stride
 *   lm_y = raw_pred_y * anchor_h + grid_y * stride
 * Landmarks use LINEAR regression (no sigmoid) with anchor_w/h as the
 * scale reference — this makes predictions scale-invariant to face size. */
const float* anchor = YOLOV5_FACE_ANCHORS[stride_idx][anchor_idx];
float anchor_w = anchor[0];
float anchor_h = anchor[1];

for (int k = 0; k < YOLOV5_FACE_NUM_KEYPOINTS; k++) {
    float lx = feat[5 + k * 2 + 0];  /* raw logit — NO sigmoid */
    float ly = feat[5 + k * 2 + 1];

    out_kpts[k][0] = (lx * anchor_w + (float)grid_x * (float)stride);
    out_kpts[k][1] = (ly * anchor_h + (float)grid_y * (float)stride);
}
```

### 注意事项
- `anchor_w/anchor_h` 已经在上面的 bbox decode 中计算过了，可以直接复用
- 确认 `YOLOV5_FACE_ANCHORS` 三个 stride 的 anchor 值正确（当前值已验证 OK）
- `stride` 变量在函数中已计算：`int stride = (stride_idx == 0) ? 8 : ...`
- 4D 标准格式路径也需要同样修改（`yolov5_face_detector.c:507-512`）

### 测试策略
1. 在 PC 上用 ONNX Runtime 跑同一张人脸图，对比修改前后的 landmark 坐标
2. 用 OpenCV 可视化 landmark 点，人工判断是否落在正确位置
3. 对比 ArcFace 识别率（如果 landmark 更准，face alignment 更好，识别率应提升）

### 回滚
- `git revert` 单次提交，零副作用

---

## Phase 2: ONNX Runtime Arena 共享配置 [P1]

### 影响范围
- `src/ort_common.c` — `ort_global_init()` 和 `ort_create_session()`
- `include/ort_common.h` — 可能需要新增 API
- **全局性：是** — 所有模型 session 创建路径都受影响

### 修改内容

**Step 1**: 在 `ort_global_init()` 中创建共享 Arena

```c
// ort_common.c — ort_global_init() 中添加
#ifdef HAS_ONNX_RUNTIME
static OrtArenaCfg* g_shared_arena_cfg = NULL;

bool ort_global_init(void) {
    // ... 现有初始化 ...

    // 创建共享 Arena 配置
    // 50MB 总上限，8MB 初始块（K1 4模型约需 15-20MB）
    const char* arena_keys[] = {
        "max_mem",                    // 硬上限
        "arena_extend_strategy",      // kSameAsRequested = 精确分配
        "initial_chunk_size_bytes"    // 初始分配大小
    };
    const char* arena_vals[] = {
        "52428800",    // 50MB
        "kSameAsRequested",
        "8388608"      // 8MB
    };
    OrtCreateArenaCfgV2(arena_keys, arena_vals, 3, &g_shared_arena_cfg);

    // ... 继续现有初始化 ...
}
```

**Step 2**: 在 `ort_create_session()` 中应用 Arena 配置

```c
// ort_create_session() 中 — 创建 SessionOptions 后添加
if (g_shared_arena_cfg) {
    // 将 Arena 配置应用到 session
    // 注意: ORT C API 中 Arena 通过 SessionOptions 的 AddConfigEntry 设置
    OrtStatus* st = g_ort_api->AddSessionConfigEntry(
        session_opts,
        "session.use_env_allocators", "1"  // 使用环境级分配器
    );
    if (st) {
        log_warning("Failed to set shared arena for session");
        g_ort_api->ReleaseStatus(st);
    }
}
```

### 注意事项
- `OrtCreateArenaCfgV2` 在某些 ONNX Runtime 版本中可能不可用（需要 ≥1.16）
- 先检查 `ort_get_api()` 是否有此函数，若无则跳过（graceful degradation）
- SpacemiT EP 可能有自己的内存管理，与 Arena 交互需测试

### 测试策略
1. 启动程序，用 `htop` 监控内存占用
2. 修改前后跑 100 帧，对比 RSS 内存峰值
3. 确认 TCM 分配不受影响

### 回滚
- `#ifdef` 宏保护，编译时或运行时可通过环境变量禁用
- Fallback: 如果 `OrtCreateArenaCfgV2` 不可用，自动退回到默认行为

---

## Phase 3: Face ROI 推理加速 [P1]

### 影响范围
- `src/inference_pipeline.c` — `detect_faces()` 函数
- `src/yolov5_face_detector.c` — 可能需要新增 ROI 模式
- 全局性：否（仅人脸检测路径）

### 修改内容

**核心思路**: 当有人体检测结果时，只在人体 bbox 扩展区域内搜索人脸。

```c
// inference_pipeline.c — detect_faces() 修改

static int detect_faces(YOLOv5FaceDetector* face_detector, 
                        ArcFaceRecognizer* face_recognizer,
                        const uint8_t* frame, int width, int height,
                        const Detection* person_dets, int num_person_dets,
                        FaceIdentity* out_faces, int max_faces) {
    
    if (num_person_dets > 0) {
        // ── ROI 模式: 只在人体上半身区域搜索人脸 ──
        for (int p = 0; p < num_person_dets && num_faces < max_faces; p++) {
            const BoundingBox* pb = &person_dets[p].bbox;
            
            // 人脸在整个人的上半部分 (头部区域)
            float person_height = bbox_height(pb);
            BoundingBox head_roi;
            head_roi.x_min = pb->x_min - person_height * 0.1f;  // 扩展 10%
            head_roi.y_min = pb->y_min;                           // 从顶部开始
            head_roi.x_max = pb->x_max + person_height * 0.1f;
            head_roi.y_max = pb->y_min + person_height * 0.35f;   // 到 35% 高度（头+肩）
            
            // Clamp to frame bounds
            head_roi.x_min = UTILS_MAX(0.0f, head_roi.x_min);
            head_roi.y_min = UTILS_MAX(0.0f, head_roi.y_min);
            head_roi.x_max = UTILS_MIN((float)width, head_roi.x_max);
            head_roi.y_max = UTILS_MIN((float)height, head_roi.y_max);
            
            // 裁剪 ROI → 缩放到模型输入大小 → 检测
            // ... ROI crop + resize + detect ...
        }
    } else {
        // ── 回退: 全帧检测 ──
        // ... 当前全帧逻辑 ...
    }
}
```

### 实现选择
- **方案 A**: 新增 `yolov5_face_detector_detect_roi()` 函数（接受 ROI bbox）
- **方案 B**: 在 `detect_faces()` 中做 ROI 裁剪，调用现有 `detect_faces()` 全帧接口

**推荐方案 A**: 避免重复的 letterbox/resize，ROI 裁剪后直接送入模型。

### 注意事项
- ROI 区域太小（<24px）时跳过
- 多个人体 bbox 重叠时合并 ROI 避免重复检测
- ROI 模式下置信度阈值可以适当降低（因为是预筛选区域）

### 测试策略
1. 单人和多人场景分别测试
2. 对比 ROI 模式和全帧模式的人脸检测数量
3. 监控推理时间变化

### 回滚
- 通过 config flag `face.roi_enabled = false` 控制

---

## Phase 4: TCM 动态分配优先级 [P1]

### 影响范围
- `src/spacemit_ort_bridge.cpp` — TCM 管理逻辑
- `src/ort_common.c` — EP session 创建路径
- `src/inference_pipeline.c` — 模型调度顺序
- **全局性：是** — 影响所有 EP 模型的内存管理

### 修改内容

**当前状态**:
```
TCM 槽位静态分配:
  Slot 0: YOLOv8n-Pose  (~170KB) — 常驻
  Slot 1: YOLOv5n-Face  (~170KB) — 常驻
  Slot 2: ArcFace       (~170KB) — 常驻
  Slot 3: 预留
总占用: ~510KB / 512KB
```

**问题**: Face Detect 和 ArcFace 只在每 N 帧运行一次（N=10~120），却始终占用 TCM。

**优化方案**:
```
TCM 优先级调度:
  Slot 0: YOLOv8n-Pose  (~170KB, intra=1) — 永久常驻
  Slot 1: 动态槽位 — face_detect 和 arcface 按需加载/释放
  Slot 2: 动态槽位 — 级联 SEARCHING 时给 YOLOv11n 使用

Face 推理前:
  1. 加载 face_detect model weight 到 TCM Slot 1
  2. 推理 face_detect
  3. 如果需要识别: 加载 arcface weight 到 TCM Slot 2 (或复用 Slot 1)
  4. 推理 arcface
  5. 释放 TCM Slot 1 和 Slot 2

收益: YOLOv8n-Pose 可以使用 intra=2 (~340KB TCM)
      → 推理速度提升 40-60%
```

### 实现要点
- 需要 `spacemit_ort_bridge` 暴露 `spacemit_ort_tcm_load_model(slot, model_path)` 和 `spacemit_ort_tcm_unload_slot(slot)` 接口
- 加载权重到 TCM 耗时约 5-10ms（DDR→TCM DMA），需要评估是否值得
- 如果 face 每 10 帧运行一次（~2.5s），额外的 10ms 加载是可接受的

### 注意事项
- TCM 加载可能在 SpacemiT EP 内部有锁，需确认线程安全
- 如果 TCM API 不支持按需加载，此优化可能无法实施
- 备选方案：通过配置控制 TCM 静态分配 (`face.tcm_persistent: false`)

### 测试策略
1. 验证 YOLOv8n-Pose 在 intra=2 下的推理延迟
2. 验证 TCM 加载/卸载的耗时
3. 测试 face 检测/识别是否正常

### 回滚
- Config flag `tcm.dynamic_allocation: false` 恢复静态分配

---

## Phase 5: ONNX Runtime 输出张量预分配 (IOBinding) [P2]

### 影响范围
- `src/ort_common.c` — session 创建和管理
- `src/ort_inference_context.c` — 推理上下文
- `include/ort_inference_context.h` — 上下文结构
- 所有调用 `ort_ctx_run()` 的模块
- **全局性：是** — 改变推理 API

### 修改内容

**核心**: 将 `ort_ctx_run()` 从动态分配输出改为预分配 + IOBinding。

```c
// ort_inference_context.h — 扩展结构体
typedef struct OrtInferenceContext {
    // ... 现有字段 ...
    
    // 新增: 预分配输出
    void**   output_buffers;        // 每个输出的预分配内存
    size_t*  output_buffer_sizes;   // 每个输出的大小
    size_t   num_outputs;
    bool     use_iobinding;         // 是否启用 IOBinding
} OrtInferenceContext;

// ort_common.c — 新增函数
int ort_ctx_init_outputs(OrtInferenceContext* ctx) {
    // 1. 获取模型输出数量和 shape
    // 2. 为每个输出预分配最大所需内存
    // 3. 创建 OrtValue 包装这些内存
    // 4. 存储到 ctx->output_buffers
}

int ort_ctx_run_with_binding(OrtInferenceContext* ctx, ...) {
    // 1. 创建 IOBinding
    // 2. 绑定输入 (ctx->input_tensor)
    // 3. 绑定预分配的输出
    // 4. RunWithBinding → 结果直接写入预分配内存
    // 5. 用户代码直接读取 ctx->output_buffers[i] 即可
}
```

### Notes
- xquant-split 模型输出 shape 固定，适合此优化
- 需要逐步迁移每个模型模块
- ST-GCN 的输入 shape 固定 (30×14)，同样适合

### 测试策略
- 对比优化前后单次推理的 malloc/free 次数（`strace -e trace=mmap,brk`）
- 确认输出值与优化前完全一致（bit-exact）

---

## Phase 6: 推理-后处理流水线并行 [P2]

### 影响范围
- `src/inference_pipeline.c` — 主循环
- `src/system_controller.c` — 线程管理
- **全局性：是** — 改变处理流水线架构

### 修改内容

**当前串行**:
```
Frame N: [Pre→Infer YOLO→Post→Track→Face→Action]
Frame N+1:                                  [Pre→Infer→Post→...]
```

**优化为双缓冲流水线**:
```
Cluster0 (AI cores):    [Post N-1] [Infer N]   [Post N]   [Infer N+1]
Cluster1 (IO cores):    [Pre N]     [Capture]   [Pre N+1]  [Capture]
```

K1 有 8 个核心分为 2 个 Cluster，可以真正并行。

### 实现要点
- 双缓冲 Frame 结构 (`FrameBuffer[2]`)
- 生产者-消费者模式，用 `atomic` flag 同步
- 后处理使用 GPU/VPU 时注意资源竞争

### 回滚
- Config flag `pipeline.double_buffer: false`

---

## Phase 7: DFL 解码 RVV 向量化 [P3]

### 影响范围
- `src/yolo_postprocess.c` — `yolo_dfl_decode_position()`
- 全局性：否

### 修改内容

```c
#ifdef __riscv_vector
#include <riscv_vector.h>

float yolo_dfl_decode_position_rvv(const float* restrict reg_data, 
                                    int pix, int hw, float dists_out[4]) {
    float peaks[4];
    for (int coord = 0; coord < 4; coord++) {
        float bins[16];
        int base = coord * 16;
        // RVV vectorized load + max + exp
        size_t vl;
        vfloat32m8_t bins_v;
        for (int b = 0; b < 16; b += vl) {
            vl = vsetvl_e32m8(16 - b);
            bins_v = vle32_v_f32m8(&reg_data[(base + b) * hw + pix], vl);
            // ... vectorized softmax ...
        }
        // ... weighted sum ...
    }
}
#endif
```

### 注意事项
- 保持 C 标量版本作为回退（`#ifdef __riscv_vector`）
- 编译器 flag: `-march=rv64gcv_zvl256b`
- K1 RVV vlen=256 → 单指令处理 8 个 float

---

## 实施顺序建议

```
Week 1: Phase 1 (正确性修正) → 立即测试
Week 1: Phase 2 (Arena 配置) → 内存验证
Week 2: Phase 3 (Face ROI) → 速度验证
Week 2-3: Phase 4 (TCM 动态) → 需要仔细测试 TCM 行为
Week 3-4: Phase 5 (IOBinding) → API 变更，逐步迁移
Week 4+: Phase 6 (并行流水线) → 架构变更，需要充分设计
Future: Phase 7 (RVV) → 低优先级，收益有限
```

---

## 全局变更依赖图

```
Phase 2 (Arena)
    ↓ (共享内存)
Phase 5 (IOBinding)
    ↓ (预分配 API)
Phase 6 (并行流水线)

Phase 4 (TCM 动态)
    ↓ (EP 管理)
Phase 6 (并行流水线)

Phase 1 (Landmark) ─── 独立，零依赖
Phase 3 (Face ROI) ─── 独立，零依赖
Phase 7 (RVV)      ─── 独立，零依赖
```
