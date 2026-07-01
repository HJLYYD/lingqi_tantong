# LingQi TanTong — 审计修复计划

> 基于 2026-06-26 全面代码审计，逐项研究学术文献后制定

---

## ═══════════ P0 — 跟踪管线数学正确性 ═══════════

### P0-1 [C1] Kalman 协方差预测: F·P·F^T 计算错误

**问题**: `utils_matrix_multiply_abt` 计算的是 `F·P·P^T`，而非 Kalman 协方差预测所需的 `F·P·F^T`。

**文献依据**:
- Kalman (1960), *"A New Approach to Linear Filtering and Prediction Problems"* — 协方差传播公式: P̄ = F·P·F^T + Q
- 线性变换的方差传播规则: 若 x ~ N(μ, P)，则 Fx ~ N(Fμ, FPF^T)。左乘 F 右乘 F^T 是矩阵标量规则 Var(ax)=a²σ² 的推广
- Bar-Shalom, Li, Kirubarajan (2001), *"Estimation with Applications to Tracking and Navigation"* — 第 5 章对协方差外推的严格推导

**当前代码** (`src/utils.c:326-343`):
```c
void utils_matrix_multiply_abt(const float a[7][7], const float b[7][7], float out[7][7]) {
    float temp[7][7] = {{0}};
    // temp = a * b        →   temp = F * P
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            for (int k = 0; k < 7; k++)
                temp[i][j] += a[i][k] * b[k][j];

    // out = temp * b^T    →   out = (F * P) * P^T   ← 错误！ b^T = P^T
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            for (int k = 0; k < 7; k++)
                out[i][j] += temp[i][k] * b[j][k];  // b[j][k] 是 b^T[k][j]
}
```

函数名说 `abt` = `a * b^T`，调用时传入 `a=F, b=P`，结果为 `F·P^T`（b 的转置是 P^T）。但对角协方差矩阵 P = P^T，所以实际计算了 `F·P·P^T = F·P·P = F·P²`，多了一个因子 P。

**修复方案**:

1. 在 `src/utils.c` 中添加新函数 `utils_matrix_multiply_afaT`（A·F·A^T 的正确实现）:

```c
/**
 * Compute out = A * F * A^T
 * Correct Kalman covariance propagation: P' = F * P * F^T
 * Reference: Kalman (1960), Bar-Shalom et al. (2001)
 */
void utils_matrix_multiply_afaT(const float a[7][7], const float f[7][7], float out[7][7]) {
    float temp[7][7] = {{0}};
    // temp = a * f
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            for (int k = 0; k < 7; k++)
                temp[i][j] += a[i][k] * f[k][j];

    // out = temp * a^T
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) {
            out[i][j] = 0.0f;
            for (int k = 0; k < 7; k++)
                out[i][j] += temp[i][k] * a[j][k];  // a[j][k] = a^T[k][j]
        }
}
```

2. 在 `src/utils.h` 中声明新函数。

3. 修改 `src/tracking_manager.c:115`:
```c
// 旧: utils_matrix_multiply_abt(kf->transition, kf->covariance, new_cov);
// 新:
utils_matrix_multiply_afaT(kf->transition, kf->covariance, new_cov);
```

4. 检查 `utils_matrix_multiply_abt` 是否在其他地方被调用:
```bash
grep -rn "utils_matrix_multiply_abt" src/
```
若仅此一处调用，可删除旧函数。

**验证**: 在 cov 为单位矩阵 I 时，`F·I·F^T` 应等于 `F·F^T`，可手算一个 7×7 的参考值对比。

---

### P0-2 [C2] 检测/Track 外观特征维度不兼容

**问题**: 检测特征使用 bbox 几何量 (aspect, norm_x, norm_y, norm_area)，但 track 特征使用姿态关键点比例（躯干比、手臂比、腿比等）。二者在 `appearance_feature_distance` 中计算 cosine 距离时维度完全不同。

**文献依据**:
- Wojke, Bewley, Paulus (2017), *"Simple Online and Realtime Tracking with a Deep Association Metric"* (DeepSORT) — 外观特征必须来自**同一特征空间**，通常是一个 CNN 提取的 L2 归一化 embedding
- DeepSORT 使用 wide residual network 提取 128-D 特征向量，对所有检测和 track 使用同一网络
- ByteTrack (Zhang et al., 2022) 故意**不使用**外观特征，仅依赖 IoU 运动匹配——在没有 CNN 特征提取器的嵌入式平台上这是更合理的选择
- 若确实需要外观匹配，BoT-SORT (Aharon et al., 2022) 采用 `C = λ·d_motion + (1-λ)·d_cos` 的加权方案，但前提是检测和 track 特征来自同一提取器

**修复方案（二选一）**:

**方案 A（推荐）: 关闭外观匹配，纯 IoU + 运动模型**

对于嵌入式 K1 平台（无 GPU CNN），ByteTrack 的纯运动方案更合适:
- 在配置中将 `appearance_weight` 默认值设为 `0.0f`
- 在 `cascade_matching` 中，当 `appearance_weight == 0.0f` 时跳过外观距离计算
- 在 `object_tracker_update` 的 Step 4 中跳过 `det_features` 的构建

```c
// cascade_matching (line ~1127): 当 appearance_weight == 0 时直接使用 IoU
if (tracker->appearance_weight < 0.001f) {
    cost_val = iou_cost;  // 纯 IoU 匹配 (ByteTrack 方案)
} else {
    cost_val = iou_cost * (1.0f - tracker->appearance_weight - OCM_DIRECTION_WEIGHT) +
               app_cost * tracker->appearance_weight +
               ocm_cost;
}
```

**方案 B: 统一特征空间（需额外工作）**

如果必须保留外观匹配功能:
- 对检测和 track 都使用几何特征（仅 descriptor[0..3]），维度一致
- 删除 `appearance_feature_from_pose` 中姿态特征填充 descriptor[4..11] 的代码
- 将 `appearance_feature_distance` 限制为仅比较 descriptor[0..3]
- 设置 `TRACKING_APPEARANCE_DIM = 4`

**推荐方案 A**，理由:
1. ByteTrack 在 MOT17/MOT20 上证明了纯运动匹配的有效性
2. K1 无 GPU，CNN 特征提取不可行
3. 几何特征（4 维）信息量不足以支撑可靠的外观重识别
4. 减少 CPU 开销（跳过 per-frame 外观特征构建和 cosine 距离计算）

**验证**: 运行测试视频，确认 tracking ID 一致性不劣于当前状态；检查 `appearance_weight=0` 时无外观相关代码路径崩溃。

---

### P0-3 [C3] 姿态外观特征读取未初始化栈数据

**问题**: `appearance_feature_from_pose` 中 `kx/ky/kc` 数组仅填充 `pose->num_keypoints` 个元素，但后续循环遍历全部 17 个索引。

**文献依据**:
- COCO 关键点格式定义 (Lin et al., 2014, *"Microsoft COCO: Common Objects in Context"*): 每个人体实例有 17 个关键点，但 `num_keypoints`（v>0 的数量）可以小于 17
- YOLOv8-Pose 输出的关键点数量取决于模型配置和检测质量，部分遮挡时可能输出 5-10 个关键点
- 标准防御做法：对所有未填充索引设置 sentinel 值 (x=-1, y=-1, c=0)

**当前代码** (`src/tracking_manager.c:395-408`):
```c
float kx[17], ky[17], kc[17];
for (int i = 0; i < 17 && i < pose->num_keypoints; i++) {
    kc[i] = pose->keypoints[i].confidence;
    if (kc[i] >= min_conf && ...) {
        kx[i] = pose->keypoints[i].x;
        ky[i] = pose->keypoints[i].y;
        valid_kpts++;
    } else {
        kx[i] = -1.0f; ky[i] = -1.0f; kc[i] = 0.0f;
    }
}
// BUG: i >= num_keypoints 的索引未初始化！
```

**修复方案**:
在循环前对数组进行零初始化 sentinel:

```c
float kx[17], ky[17], kc[17];
// 零初始化 — 确保 num_keypoints < 17 时未填充索引为安全 sentinel
memset(kx, 0, sizeof(kx));
memset(ky, 0, sizeof(ky));
memset(kc, 0, sizeof(kc));
// 对超出的索引显式设置 sentinel
for (int i = pose->num_keypoints; i < 17; i++) {
    kx[i] = -1.0f;
    ky[i] = -1.0f;
    kc[i] = 0.0f;
}
```

同样的修复应用于 `src/spatial_engine.c:169-174` 中的 `kp_y/kp_c` 数组:

```c
float kp_y[17];
float kp_c[17];
memset(kp_y, 0, sizeof(kp_y));  // 零初始化防止 UB
memset(kp_c, 0, sizeof(kp_c));
// 显式 sentinel 填充
for (int i = pose->num_keypoints; i < 17; i++) {
    kp_y[i] = -1.0f;
    kp_c[i] = 0.0f;
}
```

后续使用关键点的循环中已有的 `kp_c[u] < DEPTH_MIN_KEYPOINT_CONF` 检查会因为 `kp_c = 0.0` 自动跳过 sentinel 索引。

**验证**: 使用仅有 10 个关键点的测试姿态（模拟部分遮挡），通过 valgrind 验证无未初始化读取。

---

## ═══════════ P1 — 深度估计数学正确性 ═══════════

### P1-1 [M1] 鼻子→脚踝深度对使用错误的 anthropometric 比率

**问题**: `estimate_depth_from_pose` 中鼻子→脚踝对使用 `ANTHRO_ANKLE_TO_HEIGHT = 0.039f`（脚踝离地高度占比），而实际应为 `~0.961f`（鼻子的高度占比约 93.9%，减去脚踝的 3.9% = 约 90% 身高的垂直距离）。

**文献依据**:
- Drillis & Contini (1966), *"Body Segment Parameters"* — 人体测量学经典参考文献
- Winter (2009), *"Biomechanics and Motor Control of Human Movement"* — 身体段比例表
- 现代人体测量数据 (PMC 2019): 脚踝离地高度均值 8.0cm / 170.4cm = 4.7% ≈ 0.047
- 但代码中的常量 `ANTHRO_ANKLE_TO_HEIGHT = 0.039` 更接近外侧踝的测量值（约 3.9%），这是正确的脚踝高度
- **关键洞察**: 鼻子→脚踝的垂直距离 = (鼻子高度占比) - (脚踝高度占比) ≈ 0.939 - 0.039 = 0.900

**修复方案**:

在 `include/spatial_engine.h` 中添加新常量:
```c
/* Nose-to-ankle effective body ratio for depth estimation.
 * Nose ≈ 93.9% of height above ground, ankle ≈ 3.9%.
 * Vertical distance between them covers ~90.0% of total height. */
#define ANTHRO_NOSE_TO_ANKLE_RATIO  0.900f
```

修改 `src/spatial_engine.c:180-182`:
```c
// 旧: {KPT_NOSE, KPT_LEFT_ANKLE,  ANTHRO_ANKLE_TO_HEIGHT},
// 新:
{KPT_NOSE, KPT_LEFT_ANKLE,  ANTHRO_NOSE_TO_ANKLE_RATIO},
{KPT_NOSE, KPT_RIGHT_ANKLE, ANTHRO_NOSE_TO_ANKLE_RATIO},
```

同时修正 `estimate_depth_from_bbox` 中的相同错误 (`src/spatial_engine.c:290`):
```c
// 旧: float ratios[] = {ANTHRO_ANKLE_TO_HEIGHT, ...};  // ≈ 0.039 — 错误
// 新:
float ratios[] = {
    1.0f,                              // 完整 bbox = 全身高度
    ANTHRO_SHOULDER_TO_HEIGHT,         // ≈ 0.818
    ANTHRO_SHOULDER_TO_HEIGHT * 0.7f   // 部分可见假设
};
```

**验证**: 对已知距离（如 5m）的标准身高人员，验证深度估计在 ±15% 误差范围内。

---

### P1-2 [H12] fx/fy 除零保护

**问题**: `spatial_engine_calculate_position`、`spatial_engine_initialize_coordinate_system`、`world_coord_register_person` 中均未验证 fx/fy > 0。

**修复方案**:

在 `spatial_engine_create` 中添加焦距参数验证:
```c
if (focal_length <= 0.0f && (!camera_matrix || camera_matrix[0][0] <= 0.0f)) {
    log_error("Spatial engine requires valid focal length or camera matrix (fx > 0)");
    free(engine);
    return NULL;
}
```

在 `world_coord_create` 中同样验证。

在各计算函数中添加运行时断言（Debug 模式）:
```c
#include <assert.h>
assert(fx > 0.0f && fy > 0.0f);
```

---

## ═══════════ P1 — 系统稳定性 ═══════════

### P1-3 [H1] NULL inference_pipeline 保护

**问题**: `system_controller_create` 在 `inference_pipeline_create` 失败时返回有效 `SystemController*`（`inference_pipeline == NULL`），但后续调用直接使用。

**修复方案**:

在 `src/system_controller.c` 的 `k1_inference_thread` 和 `system_controller_process_video` 中:

```c
// k1_inference_thread (line ~495):
if (!sc->inference_pipeline) {
    log_error("[K1 Pipeline] Inference pipeline is NULL — aborting inference thread");
    pl->thread_alive[2] = false;
    k1_pipeline_slot_infer_done(pl, slot);
    continue;  // 或 return NULL;
}

// system_controller_process_video (line ~1367):
if (!sc->inference_pipeline) {
    log_error("Inference pipeline is NULL — cannot process frames");
    video_processor_destroy(processor);
    strncpy(status.message, "Inference pipeline not loaded", sizeof(status.message) - 1);
    return status;
}
```

更根本的修复: 在 `system_controller_create` 中，若 pose estimator（PRIMARY）加载失败，应返回 NULL 而非部分初始化的 controller。

---

### P1-4 [H2] pthread_create 返回值检查

**问题**: 5 个 `pthread_create` 返回值未检查。创建失败后 `pthread_join(0)` 导致未定义行为。

**文献依据**:
- POSIX.1-2017, Section 2.9.9 — `pthread_create` 返回错误码而非设置 errno
- Butenhof (1997), *"Programming with POSIX Threads"* — 第 3 章建议始终检查线程创建返回值

**修复方案**:

```c
// 对每个 pthread_create:
int ret = pthread_create(&pl->threads[t], NULL, thread_funcs[t], &pl->thread_args[t]);
if (ret != 0) {
    log_error("[K1 Pipeline] Failed to create thread %d: %s", t, strerror(ret));
    pl->num_threads = t;  // 仅 join 已成功创建的线程
    return false;  // 或设置错误状态
}
pl->thread_args[t].started = true;
```

在 `k1_pipeline_destroy` 中:
```c
for (int i = 0; i < pl->num_threads; i++) {
    if (pl->thread_args[i].started) {
        pthread_join(pl->threads[i], NULL);
    }
}
```

---

### P1-5 [H3] k1_imu 在 ERROR 状态时泄漏

**问题**: `k1_imu_create` 返回非 NULL 但状态为 ERROR，对象未被销毁也未被传递。

**修复方案** (`src/system_controller.c:850-856`):
```c
K1Imu* k1_imu = k1_imu_create(k1_imu_bus, k1_imu_rate);
if (k1_imu && k1_imu_get_state(k1_imu) != K1_IMU_STATE_ERROR) {
    imu_handler_set_k1_imu(sc->imu_handler, k1_imu);
    log_info(...);
} else {
    if (k1_imu) {
        k1_imu_destroy(k1_imu);  // ← 防止泄漏
        k1_imu = NULL;
    }
    log_warning(...);
}
```

---

## ═══════════ P2 — 线程安全 ═══════════

### P2-1 [C4] JPU 共享 YUV 缓冲区无锁

**问题**: `g_jpu_yuv_buf` 在两个 decode 函数间无锁共享。

**文献依据**:
- POSIX 线程安全模式: 使用 `pthread_mutex_t` 保护共享可变状态
- 对于 DMA 缓冲区，mutex 持有时间应仅覆盖元数据操作（分配/释放），而非 DMA 传输本身

**修复方案**:

在 `k1_jpu.c` 顶部添加:
```c
static pthread_mutex_t g_jpu_mutex = PTHREAD_MUTEX_INITIALIZER;
```

在 `k1_jpu_decode_to_rgb` 和 `k1_jpu_decode_to_rgb_ex` 中，将缓冲区 resize 和 decode 调用包裹在 mutex 中:
```c
pthread_mutex_lock(&g_jpu_mutex);
// yuv_need 检查和 g_jpu_yuv_buf 重分配
// jpudec_Decode 调用（DMA 传输）
pthread_mutex_unlock(&g_jpu_mutex);
// YUV→RGB 转换可以在锁外进行（仅读取 decode 输出）
```

**注意**: `jpudec_Decode` 本身是否可重入？查看 JPU 驱动文档。若不可重入，mutex 必须覆盖 decode 调用。

---

### P2-2 [C5] IMU 标定数据竞争

**问题**: `gyro_bias[3]` 在 `k1_imu_read_sample` 和 `k1_imu_start_calibration` 间无锁。

**修复方案** (`src/k1_imu.c`):

将标定状态修改置于 `ring_mutex` 保护之下。添加 `calib_mutex` 或者在 `ring_mutex` 临界区内统一处理标定:

```c
// k1_imu_read_sample 中:
pthread_mutex_lock(&imu->ring_mutex);
// ... 现有环形缓冲区操作 ...
// 将标定累积也移入锁内:
if (imu->state == K1_IMU_STATE_CALIBRATING) {
    imu->calib.gyro_bias[0] += imu->gyro_x / (float)IMU_CALIB_SAMPLES;
    // ...
    imu->calib.samples_collected++;
    if (imu->calib.samples_collected >= IMU_CALIB_SAMPLES) {
        imu->calib.done = true;
        imu->state = K1_IMU_STATE_READY;
    }
}
pthread_mutex_unlock(&imu->ring_mutex);

// k1_imu_start_calibration 中同样加锁:
pthread_mutex_lock(&imu->ring_mutex);
memset(imu->calib.gyro_bias, 0, sizeof(imu->calib.gyro_bias));
imu->calib.samples_collected = 0;
imu->calib.done = false;
imu->state = K1_IMU_STATE_CALIBRATING;
pthread_mutex_unlock(&imu->ring_mutex);
```

---

### P2-3 [C6] k1_platform_init 无 CAS 保护

**问题**: 无锁检查 `g_k1_plat == NULL` 后 calloc，存在双重分配风险。

**修复方案**:

使用 "一次性初始化" 模式:
```c
static pthread_once_t g_k1_plat_once = PTHREAD_ONCE_INIT;

static void k1_platform_init_once(void) {
    g_k1_plat = (K1Platform*)calloc(1, sizeof(K1Platform));
    if (g_k1_plat) {
        detect_k1_capabilities(g_k1_plat);
    }
}

K1Platform* k1_platform_init(void) {
    pthread_once(&g_k1_plat_once, k1_platform_init_once);
    return g_k1_plat;
}
```

`pthread_once` 保证初始化函数仅在首次调用时执行一次，后续调用立即返回。

---

### P2-4 [H4] inference_pipeline frame_counter 数据竞争

**问题**: `frame_counter++` 不是 atomic/volatile，但在多线程环境中使用。

**修复方案** (`include/ai_inference_pipeline.h`):
```c
// 旧: int frame_counter;
// 新:
volatile int frame_counter;  // 与 confirmed_track_count 一致
```

或者使用 C11 atomic（更正确但可移植性稍差）:
```c
#include <stdatomic.h>
atomic_int frame_counter;
// 在 .c 中: atomic_fetch_add(&pipeline->frame_counter, 1);
```

---

## ═══════════ P2 — 资源管理 ═══════════

### P2-5 [H5] [H6] V4L2 mmap 缓冲区在错误路径泄漏

**问题**: `v4l2_open_stream` 中 mmap/QBUF 失败或 STREAMON 失败时，已成功 mmap 的缓冲区未 munmap。

**文献依据**:
- Linux man page `mmap(2)`: close(fd) **不会** 自动 munmap 内存映射——映射在进程地址空间中持续存在直到显式 munmap 或进程退出
- 正确生命周期: mmap → use → munmap → close

**修复方案** (`src/video_processor.c`):

在 `v4l2_open_stream` 中每个错误出口添加 `munmap` 循环:

```c
static void v4l2_cleanup_partial(V4L2Buffer* bufs, int num_bufs) {
    for (int i = 0; i < num_bufs; i++) {
        if (bufs[i].start != MAP_FAILED && bufs[i].start != NULL) {
            munmap(bufs[i].start, bufs[i].length);
        }
    }
}
```

在每个 `close(fd); return -1;` 之前调用:
```c
v4l2_cleanup_partial(vp->v4l2_buffers, i);
close(fd);
return -1;
```

同时在 `v4l2_close_stream` 中添加 `v4l2_buffer_count` 的正确设置:

```c
// 在 mmap 循环成功后:
vp->v4l2_buffer_count = num_bufs;  // ← 确保清理循环能看到所有缓冲区
```

---

### P2-6 [H7] is_k1 无条件为 true

**问题**: `plat->is_k1 = true` 在设备树检测之前设置，使检测逻辑无效。

**修复方案** (`src/k1_platform.c`):

将 `is_k1` 的设置移到设备树检测之后:
```c
// detect_k1_capabilities 中:
plat->is_k1 = false;  // 先假设非 K1

// 检测 /proc/device-tree/compatible 中的 "spacemit,k1" 完整匹配:
FILE* f = fopen("/proc/device-tree/compatible", "r");
if (f) {
    char buf[256] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    // 精确匹配 "spacemit,k1" 或 "k1-x" 完整兼容字符串
    if (strstr(buf, "spacemit,k1")) {
        plat->is_k1 = true;
    }
}
```

---

### P2-7 [H9] k1_tcm_alloc 总是返回 NULL

**问题**: 设计意图是不使用 TCM（留给 SpacemiT EP 独占），但 `k1_platform_has_cap(TCM)` 仍返回 true。

**修复方案**:

如果 TCM 不应被应用层使用，应从 capabilities 中移除:
```c
// detect_k1_capabilities 中:
if (access("/dev/tcm", F_OK) == 0) {
    // TCM exists but is reserved for SpacemiT EP — do NOT advertise to app layer
    log_info("TCM device present (reserved for SpacemiT EP)");
    // 不设置 K1_CAP_TCM
}
```

如果未来需要支持应用层 TCM 分配，则实现真正的 mmap 逻辑。

---

## ═══════════ P3 — 代码质量 / 性能 ═══════════

### P3-1 跟踪器栈分配优化

**问题**: `object_tracker_update` 在栈上分配约 13KB，在 K1 上存在栈溢出风险。

**方案**:
- 将大型数组移至堆分配（一次性分配，复用）:
  `BoundingBox predicted_bboxes[TRACKING_MAX_TRACKS]` → 使用 tracker 结构体中的预分配字段
- 使用 `alloca` 的替代方案: 在 tracker 结构体中添加 workspace 字段

### P3-2 级联匹配 cost 矩阵堆碎片

**问题**: 每个年龄级别 calloc/free cost 矩阵。

**方案**: 在 tracker 结构体中预分配一个 `float cascade_cost[HUNGARIAN_MAX_DIM][HUNGARIAN_MAX_DIM]` workspace。

### P3-3 M20: 里程计 EKF 小角度修正

**问题**: 姿态修正使用小角度近似，大航向误差时失效。

**文献依据**:
- Markley (2003), *"Attitude Error Representations for Kalman Filtering"* — 乘性 EKF (MEKF) 使用四元数指数映射，无小角度限制
- Trawny & Roumeliotis (2005), *"Indirect Kalman Filter for 3D Attitude Estimation"* — 误差状态 Kalman 滤波器 (ESKF) 设计指南

**方案**: 将误差角转换为四元数增量时使用 `sin(θ/2)/(θ/2)` 归一化，避免小角度假设:

```c
float angle = sqrtf(dphi[0]*dphi[0] + dphi[1]*dphi[1] + dphi[2]*dphi[2]);
float half = angle * 0.5f;
float s = (angle > 1e-10f) ? sinf(half) / angle : 0.5f;
float dq[4] = {cosf(half), dphi[0]*s, dphi[1]*s, dphi[2]*s};
```

---

## ═══════════ 执行顺序与优先级 ═══════════

| 批次 | 修复项 | 预计行变更 | 风险 |
|------|--------|-----------|------|
| **批次 1** | P0-1 (Kalman F·P·F^T) | ~25 行 | 中 — 核心算法变更 |
| **批次 1** | P0-3 (零初始化关键点数组) | ~15 行 | 低 |
| **批次 2** | P0-2 (关闭外观匹配) | ~30 行 | 中 — 影响匹配行为 |
| **批次 2** | P1-1 (ANTHRO 比率修正) | ~10 行 | 中 — 改变深度估计 |
| **批次 3** | P1-2 (除零保护) | ~15 行 | 低 |
| **批次 3** | P1-3+P1-4+P1-5 (稳定性修复) | ~40 行 | 低 |
| **批次 4** | P2-1 到 P2-4 (线程安全) | ~80 行 | 中 — 引入锁，需测试死锁 |
| **批次 4** | P2-5 到 P2-7 (资源管理) | ~60 行 | 低 |
| **批次 5** | P3-1 到 P3-3 (性能/质量) | ~80 行 | 低 |

**批次 1 建议作为第一个 PR**，修复最关键的数学错误。

---

## ═══════════ 测试策略 ═══════════

### 回归测试检查清单

1. **Kalman 协方差修正**: 运行 100 帧视频，比较修正前后的 track 中心位置轨迹
2. **深度估计**: 对 5m、10m 已知距离人员验证深度误差 < 20%
3. **线程安全**: 在 K1 上运行 10 分钟实时管线，检查无崩溃/死锁
4. **内存泄漏**: valgrind 或 /proc/pid/smaps 检查 V4L2 错误路径
5. **外观匹配关闭**: 比较 ID switch 数量（预期变化 < 10%）

### 单元测试

```c
// 测试 Kalman 协方差传播
void test_kalman_covariance_predict() {
    // 设置 F = I, P = diag(1,1,...), Q = 0
    // 期望: P' = I*I*I^T = I
    // 旧代码: P' = I*I*I = I (对角时碰巧一致)
}

void test_anthro_ratio_sanity() {
    // 验证 nose→ankle 比率在 0.85-0.95 合理范围内
    assert(ANTHRO_NOSE_TO_ANKLE_RATIO > 0.85f);
    assert(ANTHRO_NOSE_TO_ANKLE_RATIO < 0.95f);
}
```

---

*计划编写日期: 2026-06-26*
*审计依据: 16 源文件 + 对应头文件的全面审查*
*文献引用: Kalman (1960), DeepSORT (Wojke et al., 2017), ByteTrack (Zhang et al., 2022), COCO (Lin et al., 2014), Drillis & Contini (1966), POSIX.1-2017, Bar-Shalom et al. (2001)*
