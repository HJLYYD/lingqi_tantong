# 审计问题修复计划

## 参考资料

- **ByteTrack 官方实现** (Yifu Zhang et al.): 标准 Kalman `P = FPF^T + Q`, 纯 IoU 匹配 (无外观特征), 双阶段 BYTE 关联
- **Fusion AHRS Library** (mattwilliamson): 权威 Madgwick C 实现, 双滤波器实例独立可并行
- **DeepSORT / BoT-SORT**: CNN ReID (128-dim) + cosine distance + Mahalanobis 门控
- **Linux perfbook (Paul McKenney)**: SPSC 环形缓冲区的标准同步模式

---

## P0-1: Kalman 协方差传播 ✅ 已修复

**文件**: `src/tracking_manager.c:115`, `src/utils.c:326-374`

**问题**: `utils_matrix_multiply_abt` 计算 `F*P*P^T` (= `F*P²`) 而非正确的 `F*P*F^T`。

**参考验证**: ByteTrack 官方 `kalman_filter.py` — `predict()` 方法:
```python
self.covariance = np.linalg.multi_dot((self._motion_mat, self.covariance, self._motion_mat.T)) + self.motion_cov
```

**已执行修复**: 改为调用已有的正确实现 `utils_matrix_multiply_fpfT()`。

---

## P1-6: 视频写入器环形缓冲区竞争

**文件**: `src/video_writer.c:78-99`

**问题**: 编码器线程释放 mutex 后调用 `fwrite(frame, ...)` 读取 `ring_frames[X]`；
生产者线程可同时通过 `memcpy(ring_frames[X], ...)` 覆盖同一 slot。

**竞争场景** (已验证):
```
消费者: pop slot 0 → frame = ring_frames[0] → unlock → fwrite(ring_frames[0], ...)
生产者: lock → slot = write_idx (=0, 回绕) → memcpy(ring_frames[0], new_data)
       → 覆盖了消费者正在 fwrite 的缓冲区!
```

**修复方案** (线程本地暂存缓冲区, FFmpeg/媒体框架标准模式):

```c
static void* video_writer_thread(void* arg) {
    VideoWriter* vw = (VideoWriter*)arg;
    size_t frame_bytes = (size_t)vw->width * vw->height * 3;

    /* ── 线程本地暂存缓冲区: 消除与生产者的 ring slot 竞争 ── */
    uint8_t* local_buf = (uint8_t*)malloc(frame_bytes);
    if (!local_buf) return NULL;

    while (true) {
        pthread_mutex_lock(&vw->enc_mutex);
        while (vw->ring_pending == 0 && vw->enc_running)
            pthread_cond_wait(&vw->enc_cond, &vw->enc_mutex);

        if (!vw->enc_running && vw->ring_pending == 0) {
            pthread_mutex_unlock(&vw->enc_mutex);
            break;
        }

        /* 在 mutex 内拷贝到本地缓冲区 — ring slot 立即释放 */
        memcpy(local_buf, vw->ring_frames[vw->ring_read_idx], frame_bytes);
        vw->ring_read_idx = (vw->ring_read_idx + 1) % VW_RING_SIZE;
        vw->ring_pending--;
        pthread_mutex_unlock(&vw->enc_mutex);

        /* fwrite 在 mutex 外 — 不阻塞生产者 */
        if (vw->pipe_broken || !vw->pipe) break;
        clearerr(vw->pipe);
        size_t written = fwrite(local_buf, 1, frame_bytes, vw->pipe);
        if (written != frame_bytes) {
            if (ferror(vw->pipe)) {
                log_warning("[VideoWriter] pipe write error — exiting");
                vw->pipe_broken = true;
                break;
            }
        } else {
            vw->frames_encoded++;
        }
    }
    free(local_buf);
    return NULL;
}
```

**内存开销**: `720 * 1280 * 3 = 2.7MB` 线程本地分配 (堆, 一次分配/释放)。

---

## P1-2: 对齐上下文数据竞争

**文件**: `src/imu_handler.c:383-461`

**问题**: `feed_k1_imu` (K1 线程) 和 `feed_external_raw` (CoAP 线程) 同时无锁写入:
- `align_ctx.k1_accels[...][n]` / `align_ctx.samples_collected` (line 386-389)
- `align_ctx.cam_accels[...][n]` / `align_ctx.cam_idx` (line 450-454)
- `try_align_frames` 从两个线程并发调用读取全部字段 (line 395)

**参考验证**: Fusion 库每个 Madgwick 实例独立运行，滤波器状态无共享。对齐上下文是唯一跨线程共享的状态。

**修复方案** (细粒度 align_mutex):

```c
// imu_handler.h 新增字段:
pthread_mutex_t align_mutex;  // 保护 align_ctx 的并发访问

// imu_handler_create 中初始化:
pthread_mutex_init(&h->align_mutex, NULL);

// feed_k1_imu 中:
pthread_mutex_lock(&h->align_mutex);
if (!h->alignment_done && h->align_ctx.samples_collected < h->align_ctx.window_size) {
    int n = h->align_ctx.samples_collected;
    h->align_ctx.k1_accels[0][n] = data->accel_x;
    // ...
    h->align_ctx.samples_collected++;
}
if (!h->alignment_done && ...) try_align_frames(h);
pthread_mutex_unlock(&h->align_mutex);

// feed_external_raw 中同样加锁
// imu_handler_destroy 中: pthread_mutex_destroy(&h->align_mutex);
```

**开销**: 每次 IMU 采样 (~100Hz) 锁定 ~1μs，可忽略。对齐仅在启动后 2 秒窗口内访问，之后 `alignment_done=true` 时完全不锁。

---

## P2-1: 外观特征空间不兼容

**文件**: `src/tracking_manager.c:1191-1218` (检测特征) vs `:424-533` (跟踪特征)

**问题**: 检测使用 4-dim 几何特征 `[aspect, norm_x, norm_y, norm_area]`，跟踪使用 12-dim 姿态比例特征。`cosine_distance` 在完全不同的语义空间上计算，结果无意义。

**参考验证**:
- **ByteTrack** 官方: 完全不使用外观特征，纯 IoU 匹配。这是 K1 嵌入式环境下的正确选择。
- **DeepSORT**: 使用 CNN ReID (128-dim)，对检测和跟踪使用相同网络提取 → 特征空间一致。
- 我们的 12-dim 姿态特征在无 ReID CNN 的条件下是一种工程折衷，但不能与 4-dim 几何特征混合。

**修复方案** (禁用外观匹配, 回归 ByteTrack 标准做法):

在 `tracking_manager.h` 中将外观权重设为 0:
```c
#define TRACKING_APPEARANCE_WEIGHT  0.0f   // 禁用: 无一致的 ReID 特征提取器
```

在 `src/tracking_manager.c` 的级联匹配中，外观匹配门控增加检查:
```c
// 在级联匹配的成本计算中:
if (tracker->appearance_weight > 0.0f) {
    // 仅当启用外观匹配时才计算特征距离
    float app_cost = appearance_feature_distance(...);
    combined_cost = (1.0f - w) * iou_cost + w * app_cost;
} else {
    combined_cost = iou_cost;  // 纯 IoU 匹配 (ByteTrack 标准)
}
```

---

## 计划实施顺序

| 步骤 | 问题 | 影响范围 | 风险 |
|------|------|----------|------|
| 1 | P1-6 环形缓冲区竞争 | video_writer.c (1 文件) | 低 — 局部重构 |
| 2 | P1-2 对齐数据竞争 | imu_handler.c/.h (2 文件) | 低 — 新增互斥锁 |
| 3 | P2-1 外观特征禁用 | tracking_manager.c/.h (2 文件) | 中 — 影响匹配逻辑 |

**不修复 (架构级，当前不触发)**:
- P1-3/P1-4 日志信号处理器 — 需要队列化架构
- P3-1 到 P3-8 — 防御性编程，功能不受影响
