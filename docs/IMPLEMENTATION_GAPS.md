# 未实现模块清单与开发路线图

> 本文档追踪 `lingqi_tantong_c` 项目中**已设计但尚未实现**的模块，以及与落地目标平台的差距。每个模块标注了对应 Python 原型的参考位置。

---

## 分类说明

| 标记 | 含义 |
|------|------|
| 🔴 核心缺失 | 系统关键路径中完全缺失的模块，阻塞实际部署 |
| 🟠 功能降级 | 已实现占位/启发式替代，精度无法满足生产要求 |
| 🟡 工程缺口 | 模块逻辑完整但缺少平台适配/构建/测试 |
| 🟢 优化待做 | 可工作但需要性能优化 |

---

## 🔴 核心缺失模块

> **更新 (2026-05)**: 项 #2 (Mahony 互补滤波) 和项 #4 (KCP-Lite 可靠传输) 已完整实现。保留在此文档中以追踪开发历程。

### 1. VINS-Mono 视觉-惯性里程计

| 属性 | 详情 |
|------|------|
| **状态** | 🔴 完全未实现 |
| **参考** | `../lingqi tantong/算法逻辑.md` §4.1 VINS-Mono 融合 |
| **依赖** | IMU数据源 + 视觉特征提取 + Ceres/g2o 非线性优化 |
| **阻塞** | 无法实现真正的 6-DOF 位姿估计；当前空间定位完全依赖纯视觉 |

**需要实现的内容：**
- IMU 预积分器（加速度/角速度 → 两帧间相对位姿增量）
- 滑动窗口 Bundle Adjustment（窗口大小=8, Ceres Solver Jacobian 自动求导）
- 边缘化策略（Marginalization prior）
- 视觉-惯性时间戳对齐（td 估计）
- IMU 零偏估计与实时校准

**建议文件组织：**
```
src/vins/
  imu_preintegrator.c/h     — IMU 预积分节点
  marginalization.c/h       — Schur 补边缘化
  feature_tracker.c/h       — 光流特征追踪
  estimator.c/h             — 滑动窗口优化器
  initial_alignment.c/h     — 视觉-惯性初始化
```

---

### 2. Mahony 互补滤波（箭矢端姿态解算）

| 属性 | 详情 |
|------|------|
| **状态** | 🟢 已实现（`src/mahony_filter.c` + `esp32p4_firmware/main/imu_fusion.c` Madgwick 变体） |
| **参考** | `../lingqi tantong/算法逻辑.md` §(3) Mahony 互补滤波 |
| **参数** | Kp=2.0, Ki=0.005, 截止频率=0.5Hz, 200Hz |
| **备注** | 射手端 Mahony 已实现待集成；ESP32P4 固件端使用 Madgwick 算法双版本 |

**已完成内容：**
- ✅ PI 控制器修正式 Mahony 滤波器
- ✅ 加速度计归一化 → 重力方向误差 → 角速度修正 → 四元数更新
- ✅ 积分误差隔离（消除陀螺仪常值漂移）
- ✅ 输出：实时四元数 `[qw, qx, qy, qz]`

**文件位置：** `src/mahony_filter.c/h` (射手端), `esp32p4_firmware/main/imu_fusion.c/h` (箭矢端 Madgwick)

---

### 3. ATW 异步时间扭曲（AR 延迟补偿）

| 属性 | 详情 |
|------|------|
| **状态** | 🔴 未实现（当前 `ar_renderer_compensate_motion()` 仅做简单的 2D Euler 旋转，非真正 ATW） |
| **参考** | `../lingqi tantong/算法逻辑.md` §5 ATW 异步时间扭曲 |
| **阻塞** | Motion-to-Photon 延迟 > 35ms，远超 20ms 人眼感知阈值 |

**需要实现的内容：**
- 渲染线程同步读取最新 IMU 姿态四元数
- 远端时间戳预测（ΔT 内姿态外推）
- 2D 投影矩阵构建（等距投影 → 旋转 → 逆投影）
- GPU 纹理映射（OpenGL ES 3.0 片段着色器实现）
- 实现 MTP ≤ 17.8ms 目标

---

### 4. KCP-Lite 可靠传输协议

| 属性 | 详情 |
|------|------|
| **状态** | 🟢 已实现（`src/kcp_lite.c/h`，576行完整实现） |
| **参考** | `../lingqi tantong/算法逻辑.md` §6.1-6.3 KCP-Lite 协议 |
| **参数** | MTU=1100, FEC(4,5), snd_wnd=128, nodelay=1, resend=50ms |
| **备注** | 协议层完整，待集成至网络传输路径 |

**已完成内容：**
- ✅ 发送端：帧缓冲区 + FEC 编码 + 超时重传 + NACK 响应
- ✅ 接收端：FEC 解码 + NACK 请求 + 乱序重排
- ✅ 三种包类型（type=0: 控制帧, type=1: 视频帧, type=2: IMU 数据）
- ✅ 拥塞控制策略（类 KCP nodelay 模式）

**文件位置：** `src/kcp_lite.c/h`

---

### 5. MiDaS 单目深度估计

| 属性 | 详情 |
|------|------|
| **状态** | 🔴 接口预留但未实现推理 |
| **模型** | MiDaS-Small, 256×256 输入 |
| **阻塞** | 深度信息依赖单一先验（人体高度/宽度），精度极低 |

**需要实现的内容：**
- ONNX Runtime 加载 MiDaS ONNX 模型
- 预处理：归一化 → 256×256 resize
- 后处理：逆归一化 + 上采样回原分辨率
- 与空间引擎集成：深度值替代当前先验身高估计

---

### 6. TinierHAR 时序动作识别

| 属性 | 详情 |
|------|------|
| **状态** | 🔴 完全未实现 |
| **参考** | `../lingqi tantong/算法逻辑.md` §(5) TinierHAR-GRU |
| **阻塞** | 无法预测目标意图（站立/行走/奔跑/蹲下/举枪/投掷） |

**需要实现的内容：**
- 17 关键点归一化 → 34 维特征向量
- 30 帧滑动窗口 → GRU 推理 → Softmax
- 6 类动作分类 + 置信度阈值 0.6

---

### 7. ICP 点云配准（箭矢-射手空间对齐）

| 属性 | 详情 |
|------|------|
| **状态** | 🔴 完全未实现 |
| **阻塞** | 无法将箭矢端 3D 探测坐标对齐至射手端参考系 |

**需要实现的内容：**
- FPFH 特征提取（快速点特征直方图）
- SAC-IA 粗配准
- ICP 点对面精细配准
- 对齐矩阵输出（旋转+平移）

---

## 🟠 功能降级模块

### 8. 深度学习推理 — 启发式回退精度问题

| 模块 | ONNX 版本精度 | 启发式回退精度 | 差距 |
|------|-------------|--------------|------|
| YOLOv8n 检测 | mAP50≈53.6% | 边缘+对比度 ≈ <10% | 严重 |
| YOLOv8-Pose | AP≈50.2% | 纯几何推算 ≈ <5% | 严重 |
| SCRFD 人脸 | AP≈93% | 肤色+对称性 ≈ <20% | 严重 |
| ArcFace 识别 | TAR@FAR≈99% | 8×8 分块特征 ≈ <30% | 严重 |

**解决方案：** 集成 ONNX Runtime（代码中 `#ifdef HAS_ONNX_RUNTIME` 逻辑已就绪，需实际链接 ONNX Runtime 库）

---

## 🟡 工程缺口模块

### 9. 目标平台适配 (SpacemiT X60 RISC-V)

| 缺口项 | 当前状态 | 目标 |
|--------|---------|------|
| 交叉编译工具链 | 无 | riscv64-unknown-linux-gnu-gcc |
| RVV 0.7.1 向量化 | 无 | VLA intrinsics / 汇编 |
| Spacengine NPU 推理 | 无 | INT8 量化 + NPU API |
| Bianbu Linux BSP | 无 | GPIO/HW Timer/Sensor 驱动 |
| K1 内存约束适配 | 堆内存无上限 | 限制 ≤1.5GB (K1 可用) |

### 10. 测试覆盖率不足

| 指标 | 当前 | 目标 |
|------|------|------|
| 测试文件数 | 1 (`test_basic.c`) | ≥5 |
| 单元测试数 | 12 | ≥100 |
| 集成测试 | 0 | ≥10 |
| 性能基准测试 | 0 | ≥5 场景 |
| CI/CD | 无 | GitHub Actions + K1 实机 |

### 11. RISC-V 交叉编译工具链配置

CMakeLists.txt 已集成 RISC-V 多版本 RVV 向量扩展检测与编译配置：
- RVV 1.0 (`-march=rv64gcv1p0`) — 优先启用
- RVV 1.0 fallback (`-march=rv64gcv`)
- RVV 0.7.1 (`-march=rv64gcv0p7`) — X60 核心兼容
- Bianbu sysroot 路径通过 `-DBIANBU_SYSROOT` 配置

后续需要：
- 实际编译器路径配置 (`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`)
- 交叉编译 toolchain file 完善

### 12. 构建系统整合

当前存在 2 套构建方法，使用方式：
- **CMake**: 跨平台标准构建（含 RISC-V 交叉编译 preset）
- **build.py**: Python 脚本快速构建（开发便捷）

> 已移除不存在的 Makefile 和 build_zig.py 引用。

---

## 🟢 优化待做模块

### 13. IMUHandler 已接入主循环

`imu_handler_create()` 已创建并接入 `system_controller_process_video` 主循环：
- ✅ 每帧通过 `imu_handler_get_latest_pose()` 获取 IMU 姿态
- ✅ 姿态数据通过 `spatial_engine_set_camera_pose()` 参与空间计算
- ⬜ 待完成：真实 GY-87 I²C 驱动数据源（目标平台）

### 14. AR 渲染器扩展

当前 AR 渲染器仅支持 2D 矩形+文字。需要：
- OpenGL ES 3.0 渲染管线
- 骨骼动画 3D 人物渲染
- 姿态四元数驱动的渲染矩阵

### 15. 日志系统增强

需要：
- 日志文件滚动（大小/时间）
- 结构化日志格式（JSON）
- 性能计数器日志（FPS、推理时间、内存）

---

## 开发路线图 (基于实际硬件：ESP32-P4 + GY87 + MUSE Pi PRO)

### 总览

```
v1.0.0-alpha (当前: x86 C 代码 + 离线视频)
  │
  ├─ v1.1 Phase 0: 硬件验证 (1-2 周)           ← 🔴 最高优先级
  │   ├── ESP32-P4 FreeRTOS Hello World
  │   ├── OV5640 MIPI CSI JPEG 捕获测试
  │   ├── GY87 I2C 三芯片地址扫描与原始读取
  │   ├── ESP32-P4 ↔ MUSE Pi UART echo test
  │   └── USB 2.0 HS OTG 配置验证
  │
  ├─ v1.2 Phase 1: 箭矢端固件 MVP (2-3 周)      ← 🟠 核心交付
  │   ├── OV5640 QVGA@15fps JPEG 连续捕获（双缓冲+DMA）
  │   ├── Madgwick AHRS 9-DOF 融合（@200Hz, C 移植）
  │   ├── BMP180 气压高度计算 + 低通滤波
  │   ├── IMU 上电自动零偏校准
  │   ├── 精简帧协议（Magic帧同步 + CRC16 + 状态机）
  │   ├── FreeRTOS 双核任务分配（Core0:采集 Core1:发送）
  │   ├── UART 3Mbps DMA 发送
  │   └── MUSE Pi 端 协议接收 + JPEG 解码显示
  │
  ├─ v1.3 Phase 2: 核心管线迁移 (3-4 周)        ← 🟡 关键对接
  │   ├── lingqi_tantong_c 交叉编译 → RISC-V binary
  │   ├── arrow_receiver 模块新增（协议→FrameData 适配）
  │   ├── imu_handler 扩展（接收外部姿态四元数）
  │   ├── spatial_engine IMU 俯仰角修正深度估计
  │   ├── 端到端通路：ESP32-P4 → MUSE Pi 接收 → AI推理 → 显示
  │   ├── ONNX Runtime RISC-V Bianbu 编译安装
  │   ├── 模型 fp32 → INT8 量化（Spacengine 工具链）
  │   └── NPU Spacengine SDK 推理集成
  │
  └─ v2.0 Phase 3-4: 高级功能 + 优化 (8-14 周)  ← 🟢 收官
      ├── VINS-Mono 视觉-惯性里程计（IMU 预积分+滑动窗口 BA）
      ├── ATW 异步时间扭曲（IMU 姿态驱动，MTP≤17.8ms）
      ├── KCP-Lite 可靠传输（仅无线场景）
      ├── TinierHAR 时序动作识别（30帧 GRU）
      ├── MiDaS 深度估计（NPU INT8）
      ├── RVV 0.7.1 向量化加速（图像预处理关键路径）
      ├── 内存优化（K1 ≤1.5GB）
      ├── OpenGL ES 3.2 AR 3D 渲染叠加
      └── CI/CD + 硬件在环测试
```

---

### Phase 0: 硬件验证集成测试 (1-2周) — 关键硬件操作步骤

> **目标**: 确保所有硬件模块连接正确、能正常通信。
> **产出**: Hardware Validation Report (信号质量/通信稳定性/功耗测量)。

| 编号 | 任务 | 硬件 | 操作要点 | 验收标准 |
|------|------|------|---------|---------|
| H0.1 | ESP32-P4 环境搭建 | ESP32-P4 | ESP-IDF v5.4 安装，`idf.py build flash monitor` | LED Blink (FreeRTOS Task) |
| H0.2 | OV5640 拍摄首帧 | ESP32-P4+OV5640 | MIPI CSI FPC 22p 连接，`esp32-camera` example | 拍到 JPEG 照片 (USB导出) |
| H0.3 | GY87 I2C 扫描 | ESP32-P4+GY87 | VCC→3.3V, SDA/SCL→I2C, 确认上拉电阻 | 发现 0x68(MPU), 0x0D/0x1E(Mag), 0x77(BMP) |
| H0.4 | MPU6050 原始读取 | ESP32-P4+GY87 | PWR_MGMT_1=0x00 唤醒, 读 0x3B-0x48 | 6轴数据正确（重力≈1g Z轴） |
| H0.5 | 磁力计 AUX 旁路 | ESP32-P4+GY87 | INT_PIN_CFG (0x37)=0x02 启用旁路 | 直接读取 QMC5883L 寄存器 |
| H0.6 | BMP180 读取 | ESP32-P4+GY87 | 读校准系数 0xAA-0xBF, 启动温度/压力转换 | 温度±1°C, 压力 300-1100hPa |
| H0.7 | UART 基础通信 | ESP32-P4↔MUSE Pi | GPIO68(TX)↔P4_RX, GPIO69(RX)↔P4_TX, GND共地 | 115200-8N1 Echo 无误码 |
| H0.8 | UART 高速测试 | ESP32-P4↔MUSE Pi | 115200→921600→1500000→3000000 | 3Mbps 连续 1MB 数据 CRC 无误 |

#### Phase 0 实施细节

**H0.3 GY87 I2C 扫描伪代码**:
```c
// ESP-IDF I2C 扫描
for (uint8_t addr = 1; addr < 127; addr++) {
    i2c_master_write_to_device(I2C_NUM, addr, NULL, 0, 100 / portTICK_PERIOD_MS);
    if (err == ESP_OK) ESP_LOGI(TAG, "Found device at 0x%02X", addr);
}
// 期望输出: 0x68 (MPU6050), 0x0D (QMC5883L) 或 0x1E (HMC5883L), 0x77 (BMP180)
```

**H0.5 GY87 AUX I2C 旁路设置**:
```c
// 关键！GY87 的磁力计不是独立挂在 I2C 总线上，
// 而是通过 MPU6050 的 AUX I2C 接口连接。
// 必须先设置旁路模式才能直接访问磁力计:
mpu_write(0x6B, 0x00);  // 唤醒 MPU6050
mpu_write(0x37, 0x02);  // INT_PIN_CFG: BYPASS_EN = 1
mpu_write(0x6A, 0x00);  // USER_CTRL: 关闭 AUX I2C Master
// 之后就可以直接用 0x0D 或 0x1E 地址读写磁力计
```

**H0.7 K1 MUSE Pi 侧 UART 配置** (Bianbu Linux):
```bash
# 检查可用串口
ls /dev/ttyS*  # ttyS1 通常对应 40-pin GPIO 的 UART
# 配置 3Mbps (如果内核支持)
stty -F /dev/ttyS1 3000000 cs8 -cstopb -parenb
# 测试收发
cat /dev/ttyS1 &  echo "TEST" > /dev/ttyS1
```

---

### Phase 1: 箭矢端固件 MVP (2-3周) — 实现细节

| 编号 | 模块 | 文件位置 | 核心逻辑 | 验收标准 |
|------|------|---------|---------|---------|
| F1.1 | camera_capture | `esp32p4_firmware/main/camera_capture.c` | esp_camera_init + fb_count=2 + GRAB_LATEST | QVGA@15fps 无丢帧 |
| F1.2 | imu_fusion | `esp32p4_firmware/main/imu_fusion.c` | 移植 MadgwickAHRS, @200Hz 调用 | 四元数输出, Roll/Pitch ±0.5° |
| F1.3 | gy87_driver | `esp32p4_firmware/main/gy87_driver.c` | I2C 批量读取(@200Hz) → 标定 → Madgwick | 3芯片数据流稳定 |
| F1.4 | bmp180_altitude | 嵌入 `esp32p4_firmware/main/gy87_driver.c` | 读UT+UP → 补偿 → ISA公式 → 高度 | 高度分辨率 0.1m |
| F1.5 | imu_calibration | 嵌入 `esp32p4_firmware/main/main.c` + `imu_fusion.c` | 上电2s静置→零偏估计, 8字→磁力计椭球 | 校准后静止角<0.2°/s |
| F1.6 | protocol_packer | `esp32p4_firmware/main/protocol.c` | Magic包围, Type, Seq, CRC16 | Pack/Unpack 双向验证 |
| F1.7 | FreeRTOS tasks | `esp32p4_firmware/main/main.c` | xTaskCreatePinnedToCore 双核分配 | Core0采集 Core1发送 并行 |
| F1.8 | receiver (K1侧) | `src/arrow_receiver.c` | UART→环形缓冲→帧状态机→JPEG+IMU分发 | 接收延迟<5ms/帧 |

#### Phase 1 实施细节

**F1.1 OV5640 帧率优化策略**:

QVGA(320×240) JPEG 约 8-15KB/帧，目标15fps。

```c
// 帧率计算
// UART 3Mbps = 375KB/s
// 每帧 JPEG ≈ 12KB → 理论上限 ≈ 31fps
// IMU 数据 ≈ 每帧附加 50B (可忽略)
// 实际目标: 15fps (考虑协议开销和接收端解码)
```

**F1.3 GY87 驱动结构**:
```c
// gy87_driver.c 结构
typedef struct {
    i2c_port_t i2c_num;
    // MPU6050 校准
    float gyro_bias[3];       // 静止零偏
    float accel_bias[3];      // 重力校准
    float accel_scale[3];     // 6面标定
    // QMC5883L 校准
    float mag_hard_iron[3];   // 硬铁补偿
    float mag_soft_iron[9];   // 软铁矩阵
    // BMP180
    bmp180_calib_t calib;     // 校准系数
} gy87_t;
```

**F1.8 arrow_receiver 接口设计** (K1 侧):
```c
// 对齐现有 video_processor 的 create(destroy 模式
arrow_receiver_t* arrow_receiver_create(const char* uart_dev);

// 阻塞等待下一对 (frame, pose)，超时返回 -1
int arrow_receiver_get_pair(arrow_receiver_t* ar,
    jpeg_frame_t* frame, imu_pose_t* pose, int timeout_ms);

// JPEG → RGB (对接 video_processor 的 FrameData 格式)
int arrow_receiver_decode_frame(arrow_receiver_t* ar,
    const jpeg_frame_t* jpeg, FrameData* out_frame);

void arrow_receiver_destroy(arrow_receiver_t* ar);
```

---

### Phase 2: 核心管线迁移 (3-4周) — 实施细节

| 编号 | 任务 | 关键步骤 | 风险 |验收标准 |
|------|------|---------|------|------|
| M2.1 | RISC-V 交叉编译 | CMake toolchain: `rv64gcv` flags, sysroot→Bianbu | RVV 0.7.1 vs 1.0 兼容性 | 编译通过 + 运行 11 tests |
| M2.2 | arrow_receiver 对接 | receiver→FrameData→SystemController | JPEG解码延迟 | 15fps 端到端通路 |
| M2.3 | imu_handler 扩展 | 新增 `imu_handler_set_pose()` 接收外部四元数 | 时间戳对齐 | IMU数据参与空间计算 |
| M2.4 | spatial_engine 修正 | pitch/roll→深度修正公式 | 精度验证 | Z轴误差<20% (目标) |
| M2.5 | 端到端集成 | ESP32→MUSE Pi→Infer→Track→Draw→HDMI | 全链路延迟 | E2E延迟<200ms |
| M2.6 | ONNX Runtime | Bianbu apt 安装 或 源码编译(RVV) | RISC-V ORT 成熟度 | 模型加载成功 |
| M2.7 | INT8 量化 | ONNX fp32→Spacengine INT8, 500张校准图 | 精度损失 | mAP损失<3% |
| M2.8 | NPU 推理 | Spacengine API: spacengine_run() | SDK可用性 | 推理 latency < x86 fp32 |

---

### Phase 3-4: 高级功能 (8-14周) — 与现有 GAPS.md 对齐

| GAPS 编号 | 模块 | Phase | 优先级 |
|-----------|------|-------|--------|
| 1 | VINS-Mono 视觉-惯性里程计 | Phase 3 | 🔴 核心 |
| 2 | Mahony 互补滤波 | ✅ 已完成 | 🟢 已实现 |
| 3 | ATW 异步时间扭曲 | Phase 3 | 🔴 核心 |
| 4 | KCP-Lite 传输 | ✅ 已完成 | 🟢 已实现 |
| 5 | MiDaS 深度估计 (NPU) | Phase 3 | 🔴 精度 |
| 6 | TinierHAR 动作识别 | Phase 3 | 🟠 功能 |
| 7 | ICP 点云配准 | Phase 4 | 🟢 优化 |
| 13 | IMUHandler 接入数据源 | Phase 2 | ✅ 本Plan已覆盖 |
| 14 | AR 渲染器 OpenGL ES 3.0 | Phase 4 | 🟢 优化 |
| 9 | RISC-V 目标平台适配 | Phase 2 | ✅ 本Plan已覆盖 |
| 15 | 日志系统增强 | Phase 4 | 🟢 优化 |

---

### 新增硬件专用模块清单 (本改进方案特有)

| 编号 | 模块 | 平台 | 文件 | 优先级 |
|------|------|------|------|--------|
| H0 | GY87 I2C 三芯片驱动 | ESP32-P4 | `esp32p4_firmware/main/gy87_driver.c/h` | 🔴 P0 |
| H1 | Madgwick AHRS 9-DOF 融合 | ESP32-P4 | `esp32p4_firmware/main/madgwick_filter.c/h` | 🔴 P0 |
| H2 | OV5640 MIPI CSI 捕获 | ESP32-P4 | `esp32p4_firmware/main/camera_capture.c/h` | 🔴 P0 |
| H3 | BMP180 气压高度引擎 | ESP32-P4 | `esp32p4_firmware/main/bmp180_altitude.c/h` | 🟠 P1 |
| H4 | 精简帧协议(协处理器→SBC) | ESP32-P4 | `esp32p4_firmware/main/protocol.c/h` | 🔴 P0 |
| H5 | 帧协议接收解析器 | MUSE Pi | `src/arrow_receiver.c/h` | 🔴 P0 |
| H6 | IMU 自动标定流程 | ESP32-P4 | `esp32p4_firmware/main/imu_calibration.c/h` | 🟠 P1 |
| H7 | Spacengine NPU 推理适配 | MUSE Pi | `src/inference/npu_adapter.c/h` | 🟡 P2 |
| H8 | RVV 0.7.1 向量化预处理 | MUSE Pi | `src/utils/rvv_preprocess.c/h` | 🟢 P4 |

---

## 相关参考

| 文档 | 路径 |
|------|------|
| 算法逻辑原文 | `../lingqi tantong/算法逻辑.md` |
| 配置文件 | `configs/default.yaml` |
| 架构文档 | `docs/ARCHITECTURE.md` |
| 构建指南 | `docs/BUILD_GUIDE.md` |
| 系统改进方案 | `docs/IMPROVEMENT_PLAN.md` |