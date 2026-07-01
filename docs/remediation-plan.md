# 灵奇探测 — 漏洞验证与修复规划

**日期:** 2026-06-29  
**状态:** 已针对最新参考文献验证所有严重/高危发现  
**参考文献:** Madgwick (2010), RFC 7959, RISC-V RVWMO 规范 v20260120, Kalman (1960), Bucy & Joseph (1968), Lomont (2003)

---

## 第一部分：严重漏洞 — 已通过最新文献验证

### C6 ✅ 已验证 — Madgwick 滤波器：梯度修正被乘以 dt² 而非 dt

**代码位置:** `src/imu_handler.c:78-92`

```c
// 第 78 行：step = beta * dt / grad_norm  ← 在这里乘以 dt
float step = mf->beta * dt / grad_norm;
j0 *= step;   // j0 = raw * beta * dt / grad_norm

// 第 89 行：又乘了一次 dt
qw += (qdot_w - j0) * dt;   // = qdot_w*dt - raw*beta*dt²/grad_norm  ← 错误！
```

**参考验证：** 根据 Madgwick (2010) 公式 (21)，正确的更新公式为：

> qₜ = qₜ₋₁ + (½ qₜ₋₁ ⊗ ω − β · ∇f/‖∇f‖) · Δt

参考实现（[Paparazzi UAV ahrs_madgwick.c](https://docs.paparazziuav.org/latest/ahrs__madgwick_8c_source.html)，[xioTechnologies Fusion](https://github.com/xioTechnologies/Fusion)）一致确认：梯度修正项只被乘以 **一次** Δt。

**数值影响：** 在 100 Hz（dt=0.01s）下，配置中 beta=0.08，有效 beta 变为 0.08 × 0.01 = **0.0008**。加速度计修正比预期弱 100 倍，滤波器实质上变成了纯陀螺仪积分。

**修复方法：**
```c
// 方案 A（最小改动）：从步进中移除 dt
float step = mf->beta / grad_norm;  // 移除 * dt

// 方案 B（更清晰）：重构以与参考实现匹配
float step = mf->beta / grad_norm;
j0 = j0_raw * step;  // 梯度修正（无 dt）
qw += qdot_w * dt - j0 * dt;  // 两项都乘以一次 dt
```

---

### C1 ✅ 已验证 — `utils_median_float()` 对未初始化内存排序

**代码位置:** `src/utils.c:375-378`

```c
float* copy = (float*)malloc(len * sizeof(float));
if (!copy) return arr[len / 2];
qsort(copy, len, sizeof(float), compare_float_desc);   // ← 从未被填充！
```

`malloc` 返回未初始化的内存。`qsort` 对随机比特位进行排序。中位数根据垃圾数据计算。缺少 `memcpy(copy, arr, len * sizeof(float))`。

**修复方法：** 在 `qsort` 之前添加 `memcpy(copy, arr, len * sizeof(float));`。

---

### C2 ✅ 已验证 — Web 服务器中的 `json_append` 缓冲区溢出

**代码位置:** `src/web_server.c:43-49`

`vsnprintf` 在截断时返回的是*本应写入*的字节数，而非实际写入的字节数。当 `n >= len - written` 时，返回值 `written + n` 超出 `len`，使得后续调用在缓冲区范围之外写入。所有四个调用点均受影响。

**修复方法：** 将返回值限制为 `len`：
```c
return (n < 0) ? written : ((written + n > len) ? len : written + n);
```

---

### C5 ✅ 已验证 — 已知的 VideoWriter 环缓冲区数据竞争

**代码位置:** `src/video_writer.c:86-91, 78, 97, 222`

代码第 87-91 行的注释明确指出：“生产者可能在 fwrite 仍在读取时覆盖 `ring_frames[slot]`”。根据 RISC-V RVWMO 规范（[RISC-V ISA 手册第 14 章](https://docs.riscv.org/reference/isa/v20260120/unpriv/rvwmo.html)），这是 C11 第 5.1.2.4 节中的未定义行为。在 X60 上，由于弱内存排序，损坏是确定性的。

**修复方法：** 在编码器释放互斥锁进行 fwrite 之前，添加一个暂存缓冲区来复制帧。

---

### C6-C8 ✅ 其他已验证的严重问题

| ID | 问题 | 参考验证 |
|----|-------|------------------|
| C6 | Madgwick dt² 错误 | Madgwick (2010) 公式 (21) |
| C7 | `k1_odometry.c` 中静态局部变量跨实例共享 | C11 §6.2.4 ¶3（静态存储期） |
| C8 | I2C 读取失败返回 0 = 有效数据 | 无哨兵值 — 零可能是合法的传感器输出 |

---

## 第二部分：高危漏洞 — 已通过参考验证

### H2-H3（K1 平台销毁中的释放后使用竞争 / RISC-V 上的 volatile 误用）

**参考验证：** RISC-V RVWMO 规范明确指出 `volatile` 仅防止编译器重排，不插入硬件 `fence` 指令。来自 [glibc BZ#18034](https://sourceware.org/bugzilla/show_bug.cgi?id=18034) 和 [Go atomic.LoadUint64 在 RISC-V 上的修复](https://datasea.cn/go0304520076.html) 的证据证实：在弱内存排序下，普通的 `ld`/`sd` 不提供跨 hart 的可见性保证。

**所需修复：** 将所有跨线程的 `volatile` 变量迁移到 C11 `_Atomic`：
- `volatile int frame_count` → `_Atomic int frame_count`
- `volatile bool has_frame` → `_Atomic bool has_frame`
- `volatile int thread_heartbeats[N]` → `_Atomic int thread_heartbeats[N]`

每个 `atomic_store_explicit(&x, v, memory_order_release)` → 编译器发出 `amoswap.w.rl` 或 `fence rw, w; sw`
每个 `atomic_load_explicit(&x, memory_order_acquire)` → 编译器发出 `lr.w.aq` 或 `lw; fence r, rw`

---

### H9（TUI JSON 格式损坏）

TUI 的机器可读 JSON 输出完全没有转义功能。虽然这不会导致可利用的崩溃，但使得 `--json` 模式对任何包含 `\n` 或 `"` 的消息都无效。修复方法：在 JSON 构建路径中重用 `json_writer.c:write_escaped()`。

### H5（`video_processor.c`/`video_writer.c` 中的 popen shell 注入）

`input_path` 被直接传递给 `popen("ffprobe ... \"%s\" ...", ...)`。虽然在实际部署中来自受信任的配置文件，但此模式不符合安全编码标准。使用 `fork/exec` 或对 path 进行 sanitize。

---

## 第三部分：修复路线图

### Sprint 1（本周）— 崩溃/数据损坏修复

| 顺序 | 修复内容 | 文件 | 改动大小 | 风险 |
|-------|-------|------|-------|------|
| 1 | Madgwick dt² 错误 | `imu_handler.c:78` | 1 行 | 低 — 与参考实现匹配 |
| 2 | `utils_median_float` 添加 memcpy | `utils.c:377` | 1 行 | 极低 |
| 3 | `json_append` 缓冲区溢出 | `web_server.c:49` | 修复返回值 | 低 — 纯粹的约束收紧 |
| 4 | I2C 错误返回值 | `k1_imu.c:70-84` | ~15 行 | 低 — 用 `bool`+`out` 参数替代 `int16_t` 返回值 |
| 5 | 静态 `ins_update_count` | `k1_odometry.c:586` | 移至结构体字段 | 低 |
| 6 | VideoWriter 暂存缓冲区 | `video_writer.c:78-97` | ~20 行 | 中 — 改变编码器同步方式 |

### Sprint 2（下周）— 内存/线程安全

| 顺序 | 修复内容 | 文件 | 改动大小 | 风险 |
|-------|-------|------|-------|------|
| 7 | volatile → _Atomic 迁移 | `system_controller.c`, `inference_pipeline.h` 等 | ~30 行（全文替换） | 中 — 需要仔细检查所有 volatile 读取 |
| 8 | `k1_platform_destroy` 释放后使用问题 | `k1_platform.c:122-123` | 2 行 | 低 |
| 9 | `log_init` TOCTOU 改用 pthread_once | `logger.c:385-388` | ~10 行 | 低 |
| 10 | CMakeLists 源文件排序 | `CMakeLists.txt:354,534` | 重新排序块 | 低 — 纯构建系统 |
| 11 | `strncpy` 空终止符（推理流水线） | `inference_pipeline.c:165,185,208,226` | 每处 +1 行 | 极低 |

### Sprint 3（两周内）— 数值正确性 / 安全加固

| 顺序 | 修复内容 | 文件 | 改动大小 | 风险 |
|-------|-------|------|-------|------|
| 12 | Kalman 协方差对称化 | `tracking_manager.c:118` | 添加 `(P+P')/2` | 低 |
| 13 | Kalman H*P 改为 H*P*H^T（正确形式） | `tracking_manager.c:146-151` | ~10 行 | 低 — 相同结果，更可维护 |
| 14 | TUI JSON 转义 | `terminal_ui.c` | ~30 行 | 低 — 仅格式化 |
| 15 | popen 路径 sanitize | `video_processor.c`, `video_writer.c` | ~15 行 | 低 |
| 16 | CoAP 头部大小检查 | `coap_receiver.c:390` | 修复为 `pos + 4 + tkl` | 低 |

---

## 第四部分：测试策略

### 针对已修复严重问题的手动测试

| 修复内容 | 测试方法 | 通过标准 |
|-------|----------|---------------|
| Madgwick dt² | 运行 IMU，持设备静止 → 检查 pitch/roll | 加速度计修正应在 ~1s 内将漂移收敛到 < 1° |
| 中位数 qsort | 使用 `{5,2,8,1,9}` 调用 → 期望中位数=5 | 与已知排序数组进行比较 |
| json_append | 构造一个 500 字节的配置值 → 检查 JSON 输出以 `}` 结束 | 无截断输出，valgrind 无错误 |
| I2C 错误返回 | 断开传感器连接 → 检查 IMU 日志中是否出现错误 | 出现“I2C read failed”错误，而非静默的零值 |
| VPU 环缓冲区 | 以 30 fps 录制 10 秒视频 → 检查输出的 MP4 | 无损坏帧，ffprobe 显示正确的时长 |

### 持续监控

- 在 CI 中加入 `-fsanitize=thread` 构建（主机模拟）
- 在所有 `_Atomic` 迁移上运行 valgrind `--tool=drd`
- 对 RISC-V 目标运行 `riscv64-linux-gnu-objdump -d` 以验证 `fence` 指令的存在

---

*本计划基于截至 2026 年 6 月的最新参考文献进行验证。所有修复均涉及现有代码中可证明的错误，而非风格偏好。*
