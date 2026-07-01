# Inference Pipeline Rewrite Plan

> 基于学术论文和官方推荐的边缘AI推理管线最佳实践
> 日期: 2026-06-26

## 目录

1. [研究方法论](#1-研究方法论)
2. [当前架构审计](#2-当前架构审计)
3. [差距分析与最佳实践对照](#3-差距分析与最佳实践对照)
4. [分阶段重写计划](#4-分阶段重写计划)

---

## 1. 研究方法论

本文档基于对以下资料的全面审查:

### 学术论文来源 (40+)
- **边缘AI管线设计**: 基于BSP的系统框架 (arXiv:2605.26119), ChiPy DSL, Quadric GPNPU架构
- **量化管线**: PTQ vs QAT对比 (IEEE Access 2024), 自适应逐层量化 (TinyML Symposium)
- **编排模式**: Ayo框架 (ACM SIGMOD), InfraMind (ICML), MLRun服务图, SNN门控多模型管线
- **跟踪/可观测性**: OpenTelemetry语义约定 (CNCF), Langfuse追踪, MLflow评估
- **推理优化**: rknnPool线程池 (CSDN), DEEPX NPU加速, ARM NEON优化指南

### 官方文档来源
- **ONNX Runtime**: 嵌入式部署指南, 自定义EP注册, 内存管理最佳实践
- **NVIDIA TensorRT**: Jetson优化 (FP16/INT8/FP8), 管线化, 多流并发
- **SpacemiT Bianbu**: 模型量化 (xquant), TCM管理, EP会话限制 (4个max, 512KB TCM)
- **Rockchip RKNN**: NPU多核调度, 零拷贝DMA, SRAM权重存储
- **NCNN/MNN/TNN**: 图优化 (算子融合 15-20%), Winograd卷积, 内存池化 90%降低

---

## 2. 当前架构审计

### 2.1 整体管线架构

```
                    ┌──────────────── K1 Dual-Cluster Pipeline ────────────────┐
                    │                                                         │
  Camera/Video ──► [CPU4:Capture] ──► [CPU1:Inference] ──► [CPU0:PostProc] ──► Output
                    │    JPEG解码        YOLO-Pose+Face      Tracking+World     │
                    │    V4L2/CoAP       +ArcFace+ST-GCN     +Trajectory        │
                    │                                                         │
                    └── Ring Buffer (4 slots) + Mutex/Cond synchronization ────┘
```

### 2.2 当前实现的优势
- ✅ K1双集群线程亲和性手动调度 (Cluster0 AI + Cluster1 I/O)
- ✅ 池化预处理 (零malloc热路径)
- ✅ 级联状态机 (SEARCHING→TRACKING→VALIDATING) 省电
- ✅ OKS-NMS替代IoU-NMS (姿势感知)
- ✅ 解剖学关键点验证 (7项检查, 部分身体回退)
- ✅ xquant-split格式自动检测与缓存
- ✅ DFL峰值置信度作为INT8损坏分类器的回退
- ✅ Top-K堆选择 (O(n log k)替代O(n²))

### 2.3 关键差距 (按严重程度排序)

| 严重度 | 差距 | 当前状态 | 目标状态 | 影响 |
|--------|------|----------|----------|------|
| **P0** | 无结构化日志/追踪 | 纯文本printf+fprintf, 全局互斥锁 | OpenTelemetry兼容跨度, 环缓冲日志 | 无法在生产环境中观测性能回归 |
| **P0** | 手写JSON生成(有注入漏洞) | fprintf手动拼接字符串 | cJSON库或FlatBuffers序列化 | 输出损坏, 字符串未转义 |
| **P0** | 无逐阶段延迟分解 | 仅总帧时间+每30帧平均值 | OpenTelemetry跨度: 预处理/推理/NMS/滤波 | 无法定位瓶颈 |
| **P1** | 错误处理不一致 | 混合返回码+log_error, 无分类 | 错误分类 (FATAL/TRANSIENT/MODEL/IO)，优雅降级 | 静默失败阻碍调试 |
| **P1** | 无模型预热 | 首次推理时进行格式探测 | 显式预热阶段 (3次假数据推理) | 首帧延迟不可预测 |
| **P1** | `__sync_synchronize()` 重量级屏障 | 每个槽都有完全内存屏障 | C11 `atomic_store_explicit(memory_order_release)` | RISC-V弱序下不必要的开销 |
| **P1** | 日志级别不可热重载 | 编译时或启动时固定 | SIGHUP或配置监听 | 无法在生产中切换至DEBUG级别 |
| **P2** | 无可观察性导出器 | 仅本地日志文件 | OTLP/Prometheus remote_write (可选) | 无集中式监控 |
| **P2** | 无日志轮转/保留 | 无限追加 (无大小限制) | 10MB文件轮转, 保留最近7天 | 磁盘会填满 |
| **P2** | 人脸批处理未使用 | 每张脸单独提取嵌入 | 批处理ArcFace推理 (每帧最多20张脸) | 人脸识别吞吐量降低 |
| **P2** | 无量化校准元数据 | 仅模型文件 | 每个.onnx文件附带calibration.json | QA/QP节点验证不可能 |

---

## 3. 差距分析与最佳实践对照

### 3.1 日志系统重写

**当前**: `logger.c` — 全局互斥锁, 纯文本格式, `printf`+`fprintf`, 无限追加

**参考实现**:
- [spdlog](https://github.com/gabime/spdlog) (C++但模式适用于C) — 异步日志, 环缓冲, 自定义格式器
- [zlog](https://github.com/HardySimpson/zlog) — 纯C, 分类日志, 运行时配置
- [nanolog](https://github.com/PlatformLab/NanoLog) — 亚微秒延迟, 无锁MPSC队列

**目标架构**:
```
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│ 调用线程      │───►│ 无锁SPSC环缓冲    │───►│ 后台写入线程  │
│ log_info(...) │    │ (每线程, 64KB)    │    │ (CPU 7)      │
└──────────────┘    └──────────────────┘    └──────┬──────┘
                                                   │
                              ┌─────────────────────┼──────────────┐
                              ▼                     ▼              ▼
                         stdout/stderr         日志文件         OTLP导出器
                         (彩色, 格式:          (轮转:          (可选gRPC,
                          HH:MM:SS.MMM│LVL│msg)  10MB×10)      OpenTelemetry)
```

**重写要点**:
1. **无锁SPSC环缓冲**: 每线程一个缓冲避免互斥锁争用。使用C11 `atomic_fetch_add` 进行槽位获取。
2. **结构化日志格式**: `{"ts":"...","level":"INFO","thread":"infer","msg":"frame processed","frame":42,"lat_ms":12.3}`
3. **日志轮转**: 按大小 (10MB) 自动轮转，保留最近10个文件。
4. **运行时级别切换**: `SIGUSR1` 信号处理程序切换 `TRACE/DEBUG/INFO/WARNING/ERROR`。
5. **逐帧上下文ID**: 每个推理帧获取一个 `frame_seq_id` 以关联跨线程日志。

### 3.2 结果输出重写

**当前**: `result_manager.c` — 手动 `fprintf` JSON, 无转义, 无模式版本, 无压缩

**参考实现**:
- [cJSON](https://github.com/DaveGamble/cJSON) — 广泛使用的纯C JSON库 (麻省理工学院许可)
- [FlatBuffers](https://google.github.io/flatbuffers/) — Google零拷贝序列化, 3-5倍速度提升, 模式演变
- [MessagePack](https://msgpack.org/) — 二进制JSON, 嵌入式紧凑 (比JSON小50-70%)

**推荐**: 两阶段采用:
- **第一阶段**: cJSON用于JSON/CSV输出 (最低风险, 向后兼容)
- **第二阶段**: FlatBuffers模式用于高性能二进制轨迹输出

**重写要点**:
1. **cJSON替换**: 所有 `fprintf` JSON替换为 `cJSON_CreateObject()/Print()`。
2. **模式版本控制**: 在报告和轨迹CSV中嵌入 `"schema_version": "1.0.0"`。
3. **正确转义**: cJSON自动处理特殊字符, 防止损坏。
4. **增量结果**: 每N帧 `result_manager_flush()` 而不是仅在会话结束时。
5. **MsgPack轨迹**: 可选二进制格式, 使用 `mpack` 库用于大型轨迹 (>10K点)。

### 3.3 推理管线重写

**当前**: `inference_pipeline.c` — 顺序: 姿势→人脸→动作, 级联状态机, 无预热

**参考实现**:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 官方Python管线, 模型预热, 动态批处理
- [rknn_yolov5_demo](https://github.com/airockchip/rknn_model_zoo) — Rockchip官方C++参考
- [TNN examples](https://github.com/Tencent/TNN/tree/master/examples) — 多模型管线, 异步推理

**重写要点**:

1. **模型预热阶段** (在 `load_models` 之后):
```c
// 使用零填充输入运行3次推理以预热TCM/NPU缓存
int pipeline_warmup(AIInferencePipeline* p) {
    uint8_t dummy[640*640*3] = {0};
    for (int i = 0; i < 3; i++) {
        InferenceResult r;
        pipeline_process_frame(p, dummy, 640, 640, &r);
    }
    p->warmed_up = true;
}
```

2. **逐阶段延迟追踪**:
```c
typedef struct {
    int64_t t_preprocess_us;   // 预处理 (缩放+归一化)
    int64_t t_inference_us;    // ONNX运行
    int64_t t_decode_us;       // DFL解码+Top-K
    int64_t t_nms_us;          // OKS-NMS
    int64_t t_filter_us;       // 检测过滤
    int64_t t_face_detect_us;  // 人脸检测
    int64_t t_face_recog_us;   // 人脸识别 (ArcFace)
    int64_t t_action_push_us;  // ST-GCN缓冲区推送
    int64_t t_total_us;
} PipelineTiming;
```
添加到 `InferenceResult` 作为 `pipeline_timing`。

3. **动态帧跳过** (当管线饱和时):
```c
// 如果队列深度 > 2且当前FPS > 目标, 跳过非关键帧
bool should_skip_face_detection(pipeline, queue_depth, current_fps);
bool should_skip_action_push(pipeline, queue_depth);
```

4. **C11原子内存排序** 替代 `__sync_synchronize()`:
```c
// 之前: __sync_synchronize();
// 之后:
atomic_store_explicit(&slot->has_frame, true, memory_order_release);
// 消费者:
if (atomic_load_explicit(&slot->has_frame, memory_order_acquire)) { ... }
```

### 3.4 预处理管线重写

**当前**: CPU字母盒调整大小→CHW转换→`/255.0` 归一化, 在 `yolo_postprocess.c` 中

**参考**:
- ONNX Composer: 将预处理嵌入模型图 (无需外部代码)
- RK3588 RGA: 硬件加速调整大小 (比CPU快10倍)
- NVIDIA DALI: GPU加速预处理管线

**重写要点**:

1. **单遍 CHW 转换+归一化**: 消除字母盒中间缓冲区:
```c
// 当前: 裁剪→字母盒 (malloc)→CHW转换 (循环)
// 目标: 单遍: 对每个输出像素, 计算源坐标, 采样/填充, 写入CHW
// 节省: 640*640*3 = 1.2MB malloc + 1遍
```

2. **预计算查找表 (LUT)**: 对于固定输入尺寸 (例如320×320人脸检测器), 预计算 `x_src→y_src` 映射:
```c
typedef struct {
    int16_t src_x[320];  // 每列: 源x坐标或-1 (填充)
    int16_t src_y[320];  // 每行: 源y坐标或-1 (填充)
    float   scale_x, scale_y;
    int     pad_x, pad_y;
} PrecomputeLUT;
```

3. **类型双关归一化**: `pixel * (1.0f/255.0f)` 替代 `/255.0f` — GCC对此进行向量化处理。

### 3.5 后处理管线重写

**当前**: `yolo_postprocess.c` — DFL解码 + 堆 Top-K + qsort NMS, 运行良好

**重写要点** (增量, 当前大多数代码正确):

1. **批量人脸ArcFace推理**: 收集所有人脸裁剪→调整为112×112→堆叠为 `[N,3,112,112]` 张量→一次ONNX运行。
   当前: 每张脸一次推理 (N次运行)。目标: 一次批处理运行。节省: ~(N-1)*ORT调用开销。

2. **NMS优化微调**: OKS NMS已经优化 (IoU快速路径, 堆Top-K), 但可以调整为:
   - 仅使用躯干关键点 (5,6,11,12) 进行OKS, 而不是所有17个。节省: 9个expf调用/对。

3. **xquant解码快速路径**: 在零置信度情况下完全跳过关键点网格采样。

4. **反量化元数据**: 存储每个张量的 `scale/zp` 值以允许运行时验证。

### 3.6 可观测性基础设施

**新添加**: 当前缺失。为生产就绪而关键。

**架构**:
```c
// 追踪跨度系统 (OpenTelemetry-compatible)
typedef struct {
    uint64_t trace_id;       // 跨请求的全局唯一ID
    uint64_t span_id;        // 跨度本地ID
    uint64_t parent_span_id;
    char     name[64];
    int64_t  start_time_ns;  // CLOCK_MONOTONIC
    int64_t  end_time_ns;
    int      status;         // 0=OK, 1=ERROR
    char     attributes[512]; // 键值对 (JSON编码)
} TraceSpan;

// 指标计数器
typedef struct {
    // 推理
    atomic_uint_fast64_t frames_processed;
    atomic_uint_fast64_t inference_errors;
    atomic_uint_fast64_t timeout_count;

    // 延迟直方图 (桶: <10, <20, <50, <100, <200, <500, <1000ms)
    atomic_uint_fast64_t latency_buckets_us[8];

    // 检测
    atomic_uint_fast64_t detections_total;
    atomic_uint_fast64_t false_positives_filtered;
} PipelineMetrics;
```

---

## 4. 分阶段重写计划

### 阶段 1: 日志+追踪基础设施 (最高优先级)
**文件**: `src/logger.c` → 完全重写
**新增文件**: `include/trace_span.h`, `src/trace_span.c`, `include/pipeline_metrics.h`
**原因**: 所有其他改进都依赖可观测性来验证有效性
**预计影响**: 日志延迟从 ~50μs 降至 ~5μs (无锁环缓冲), 每帧追踪跨度可见

### 阶段 2: 结果输出 (高优先级)
**文件**: `src/result_manager.c` → 使用cJSON重写
**新增文件**: `third_party/cJSON.c`, `third_party/cJSON.h`
**原因**: 当前手工JSON生成有注入漏洞, 且不支持增量刷新
**预计影响**: 3倍更快的JSON生成, UTF-8正确, 模式版本控制

### 阶段 3: 推理管线优化 (高优先级)
**文件**: `src/inference_pipeline.c` → 增强
**原因**: 模型预热, 逐阶段延迟, 动态跳帧改善系统行为
**预计影响**: 首帧延迟可预测, 管线饱和时更平滑的性能

### 阶段 4: 预处理+后处理优化 (中优先级)
**文件**: `src/yolo_postprocess.c`, `src/yolov8_pose_estimator.c`
**原因**: 单遍预处理, 批量ArcFace, xquant快速路径
**预计影响**: 人脸识别批处理节省30-50%, 预处理节省10-15%

### 阶段 5: 可观测性导出器 (低优先级)
**文件**: `src/otel_exporter.c`, `src/prometheus_writer.c`
**原因**: 仅在生产部署时需要; 可选HTTP/gRPC组件
**预计影响**: 集中式监控, Grafana仪表板, 告警

---

## 5. 参考文献

| 论文/来源 | 年份 | 关键贡献 | 相关阶段 |
|-----------|------|----------|----------|
| 基于BSP的AI推理系统框架 (arXiv:2605.26119) | 2025 | 5层分解, 硬件→OS→运行时→应用→验证 | 阶段1,3 |
| Ayo: 细粒度调度框架 (ACM SIGMOD) | 2025 | 基本数据流图, 4遍优化, 两级调度 | 阶段3 |
| InfraMind (ICML) | 2025 | 基础设施感知规划, 实时负载优化, 99.9% SLO | 阶段3 |
| OpenTelemetry语义约定 (CNCF) | 2025 | LLM跨度类型, 属性键, 追踪传播标准 | 阶段1,5 |
| rknnPool线程池架构 (CSDN) | 2024 | 权重共享, NPU核绑定, 3.4倍吞吐量提升 | 阶段3 |
| 自适应逐层量化 (TinyML) | 2024 | INT8/INT4混合, 65%大小减少, 45%延迟改善 | 阶段4 |
| 边缘设备的生产模型服务 (NimbleEdge) | 2024 | 多执行器基础, 硬件感知调度, OTA | 阶段3 |
| ONNX Runtime嵌入式部署 (官方) | 2024 | 自定义构建, 静态量化, NHWC转换 | 阶段4 |
| NVIDIA TensorRT Jetson优化 (NVIDIA博客) | 2024 | FP16 2倍, INT8 5倍, 管线化模式 | 阶段3,4 |
| zlog (HardySimpson): 分类C日志框架 | 2023 | 运行时配置, 分类, 环缓冲 | 阶段1 |
| cJSON (DaveGamble): 超轻量JSON解析器 | 2023 | 纯C, MIT许可, 无依赖, 广泛部署 | 阶段2 |
