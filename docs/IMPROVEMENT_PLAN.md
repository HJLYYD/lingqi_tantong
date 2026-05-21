# 灵柒·探瞳 系统改进方案

> 基于当前硬件规格（ESP32-P4 + OV5640 + GY87 + MUSE Pi PRO），结合业界最佳实践制定的完整改进方案。

---

## 目录

1. [整体架构重设计](#1-整体架构重设计)
2. [子系统细化方案](#2-子系统细化方案)
3. [开发路线图](#3-开发路线图)
4. [风险与缓解](#4-风险与缓解)

---

## 1. 整体架构重设计

### 1.1 当前架构问题分析

| 问题 | 现状 | 影响 |
|------|------|------|
| **纯 x86 PC 平台** | 当前 C 代码仅在 x86 Linux/WSL 上运行，处理离线视频文件 | 无法在真实硬件上运行 |
| **IMU 数据源缺失** | `imu_handler_process()` 从未被调用，IMU 结构体只有定义 | 无姿态信息，VINS-Mono/ATW 全阻塞 |
| **视频来源为离线文件** | `video_processor` 读取 MP4 文件而非实时摄像头 | 无法实时探测 |
| **无硬件间通信** | KCP-Lite/KCP 部分完全空缺 | 箭矢端→射手端无数据通路 |
| **AI 推理无加速** | ONNX Runtime 仅在 x86 CPU 上跑，未利用 NPU | K1 2TOPS NPU 浪费 |

### 1.2 目标架构：双设备异构计算

```
┌─────────────────────────────────────────────────────────────────┐
│                        箭矢端 (Arrow End)                        │
│                    ESP32-P4 (RISC-V 400MHz)                      │
│                                                                  │
│  ┌──────────┐   MIPI CSI    ┌──────────────────────┐            │
│  │  OV5640  │──────────────▶│  ISP + JPEG Encoder  │            │
│  │  Camera  │               │  (Hardware Pipeline) │            │
│  └──────────┘               └──────────┬───────────┘            │
│                                        │ JPEG frames             │
│  ┌──────────┐   I2C(400kHz) ┌──────────▼───────────┐            │
│  │  GY87    │──────────────▶│  IMU Fusion Engine   │            │
│  │  IMU     │  MPU6050+     │  Mahony/Madgwick     │            │
│  │          │  QMC5883L+    │  → Quaternion [wxyz] │            │
│  │          │  BMP180       │  → Altitude (m)      │            │
│  └──────────┘               └──────────┬───────────┘            │
│                                        │                         │
│                               ┌────────▼───────────┐            │
│                               │  Protocol Packer   │            │
│                               │  Frame+IMU → Packets│           │
│                               └────────┬───────────┘            │
│                                        │                         │
│                          UART(3Mbps) / USB 2.0 HS(480Mbps)      │
└────────────────────────────────────────┼────────────────────────┘
                                         │
                                         │ 有线串行通信
                                         │
┌────────────────────────────────────────┼────────────────────────┐
│                        射手端 (Shooter End / Main Compute)       │
│                  MUSE Pi PRO (SpacemiT M1, 8×X60 RISC-V)        │
│                                         │                        │
│  ┌──────────────┐  SPI/UART   ┌───────▼──────────────┐         │
│  │ ESP32-C6/WiFi│◄───────────│  Protocol Receiver   │         │
│  │ (可选Hosted) │             │  Depacketizer        │         │
│  └──────────────┘             └───────┬──────────────┘         │
│                                       │                          │
│         ┌─────────────────────────────┼──────────────────┐      │
│         │                             │                  │      │
│  ┌──────▼──────┐  ┌─────────────┐  ┌──▼──────────────┐ │      │
│  │ JPEG Decode │  │ IMU Quat    │  │  Depth Engine   │ │      │
│  │ → RGB/YUV   │  │ → Pose Mat  │  │  MiDaS (NPU)    │ │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────────┘ │      │
│         │                │                │             │      │
│  ┌──────▼────────────────▼────────────────▼──────────┐ │      │
│  │            lingqi_tantong_c Core Pipeline          │ │      │
│  │  ┌──────────────────────────────────────────────┐ │ │      │
│  │  │ InferencePipeline (NPU-accelerated ONNX)      │ │ │      │
│  │  │  YOLOv8n → YOLOv8-Pose → SCRFD → ArcFace     │ │ │      │
│  │  │  (Spacengine INT8 量化, 2.0 TOPS)            │ │ │      │
│  │  └──────────────────┬───────────────────────────┘ │ │      │
│  │                     ▼                             │ │      │
│  │  ┌──────────────────────────────────────────────┐ │ │      │
│  │  │ Tracking Manager (ByteTrack + Kalman 7-state) │ │ │      │
│  │  └──────────────────┬───────────────────────────┘ │ │      │
│  │                     ▼                             │ │      │
│  │  ┌──────────────────────────────────────────────┐ │ │      │
│  │  │ Spatial Engine (Pinhole + IMU-corrected)     │ │ │      │
│  │  │ + VINS-Mono (视觉-惯性里程计)               │ │ │      │
│  │  └──────────────────┬───────────────────────────┘ │ │      │
│  │                     ▼                             │ │      │
│  │  ┌──────────────────────────────────────────────┐ │ │      │
│  │  │ Visualizer + AR Renderer + Video Writer      │ │ │      │
│  │  └──────────────────────────────────────────────┘ │ │      │
│  └──────────────────────────────────────────────────┘ │      │
│                                                       │      │
│  ┌──────────────────────────────────────────────────┐ │      │
│  │  HDMI 1080p60 / MIPI DSI Display Output          │◄┘      │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 硬件互联方案对比

| 方案 | 物理接口 | 带宽 | 延迟 | 适用场景 | 推荐度 |
|------|---------|------|------|---------|--------|
| **UART 高速** | ESP32-P4 TX/RX → K1 GPIO UART | ≤3 Mbps | <1ms | IMU数据 + 低分辨率JPEG | ⭐⭐⭐⭐⭐ |
| **USB 2.0 HS** | ESP32-P4 USB OTG → K1 USB Host | ≤480 Mbps | <2ms | 高分辨率JPEG视频流 | ⭐⭐⭐⭐ |
| **SPI Slave/Master** | ESP32-P4 SPI → K1 SPI | ≤20 Mbps | <0.5ms | 中等分辨率+IMU混合 | ⭐⭐⭐ |
| **Wi-Fi (UDP)** | ESP32-C6 → K1 Wi-Fi 6 | ≤50 Mbps | 1-5ms | 无线场景（需额外ESP32-C6） | ⭐⭐ |
| **Ethernet** | 需外接PHY芯片 | ≤100 Mbps | <1ms | 有线高带宽场景 | ⭐⭐ |

**推荐方案：UART (3Mbps) 做主通道传输 IMU + QVGA JPEG，USB 2.0 HS 做高分辨率可选通道。**

### 1.4 数据流详细设计

```
箭矢端 ESP32-P4                     射手端 MUSE Pi PRO
═══════════════                     ═══════════════════

FreeRTOS Task: camera_task          Thread: receiver_thread
  (Core 0, Priority 10)               ┌─ UART RX ISR → Ring Buffer
  ┌─ OV5640 JPEG Capture (DMA)        │
  ├─ Frame Ready Semaphore            ├─ Protocol Parser
  └─ Push to Frame Queue              │  ┌─ Magic: 0xA5 0x5A
                                      │  ├─ Type: 0x01(JPEG)/0x02(IMU)
FreeRTOS Task: imu_task               │  ├─ Seq# + Timestamp
  (Core 0, Priority 12)               │  ├─ Length + Payload
  ┌─ I2C Read MPU6050 @200Hz          │  └─ CRC16
  ├─ I2C Read QMC5883L @50Hz          │
  ├─ I2C Read BMP180 @10Hz            ├─ JPEG → Decoder → RGB Frame
  ├─ Mahony Filter → Quaternion       │
  └─ Push to IMU Queue                ├─ IMU → Quat → Pose Matrix
                                      │
FreeRTOS Task: tx_task                └─ Push to Frame+IMU Pair Queue
  (Core 1, Priority 8)
  ┌─ Dequeue Frame + IMU Pair       Thread: inference_thread
  ├─ Pack Protocol Frame              ┌─ Dequeue Frame+IMU Pair
  ├─ UART TX (DMA)                    ├─ AI Inference (NPU)
  └─ (Optional: USB TX)               ├─ ByteTrack Update
                                      ├─ Spatial Calc (IMU-corrected)
                                      ├─ Visualization Render
                                      └─ Output to HDMI/File
```

---

## 2. 子系统细化方案

### 2.1 ESP32-P4 固件架构

#### 2.1.1 开发环境

| 项目 | 选择 | 理由 |
|------|------|------|
| SDK | ESP-IDF v5.4+ | ESP32-P4 官方支持，MIPI-CSI/ISP 驱动完整 |
| 编译器 | riscv32-esp-elf-gcc (xtensa→riscv) | P4 是 RISC-V 核 |
| 组件 | esp32-camera v2.1.6+ | OV5640 驱动，支持 JPEG 硬件编码 |
| 构建 | idf.py + CMake | ESP-IDF 标准构建 |

#### 2.1.2 OV5640 摄像头配置（来自乐鑫官方最佳实践）

根据 espressif/esp32-camera 官方驱动和社区最佳实践：

```c
// ESP32-P4 的 MIPI CSI 配置（与 ESP32-S3 DVP 不同）
camera_config_t config = {
    .pin_d0 = -1,             // MIPI CSI，不需要 DVP 并行引脚
    .pin_xclk = CAM_XCLK_GPIO,
    .pin_pclk = -1,
    .pin_vsync = -1,
    .pin_href = -1,

    .xclk_freq_hz = 24000000, // 24MHz，OV5640 可支持 6-27MHz
                              // 官方推荐 20-24MHz 平衡稳定性和吞吐

    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,   // 关键：使用硬件 JPEG 编码
    .frame_size = FRAMESIZE_QVGA,     // 320×240 用于实时 AI
    // .frame_size = FRAMESIZE_VGA,   // 640×480 用于高精度
    // .frame_size = FRAMESIZE_HD,    // 1280×720 仅用于静态分析

    .jpeg_quality = 12,               // 0-63, 越小质量越高体积越大
    .fb_count = 2,                    // 双缓冲（仅 JPEG 模式有效）
    .grab_mode = CAMERA_GRAB_LATEST,  // 实时模式：丢弃旧帧取最新
};
```

**关键优化策略（来自 ESP32-CAM RTOS 项目）：**

| 策略 | 说明 |
|------|------|
| **双缓冲 (fb_count=2)** | DMA 持续传输，采集和发送并行，帧率提升 2× |
| **CAMERA_GRAB_LATEST** | 队列满时丢弃旧帧，确保低延迟 |
| **PIXFORMAT_JPEG** | 利用 OV5640 硬件 JPEG 编码器，零 CPU 开销 |
| **xclk 调优** | 从默认 20MHz 开始，根据实际布线质量逐步提升至 24-27MHz |
| **核绑定** | camera_task 绑定 Core 0，tx_task 绑定 Core 1，避免缓存抖动 |

#### 2.1.3 GY87 IMU 驱动与传感器融合

**GY87 关键注意事项（来自社区硬件经验）：**

1. **磁力计型号确认**：市售 GY87 绝大多数使用 QMC5883L (I2C 0x0D)，而非 HMC5883L (0x1E)。需先扫描 I2C 总线确认。
2. **MPU6050 AUX I2C 旁路**：QMC5883L 通过 MPU6050 的 AUX I2C 总线连接，必须先设置 MPU6050 `INT_PIN_CFG` 寄存器 bit1 (BYPASS_EN) = 1 才能直接访问磁力计。
3. **上拉电阻**：GY87 模块通常自带 I2C 上拉，如不稳定可外加 4.7kΩ。
4. **电源噪声**：IMU 对电源纹波敏感，建议加 100nF + 10μF 去耦电容。

```c
// GY87 初始化序列
void gy87_init(i2c_port_t i2c_num) {
    // === 第1步：MPU6050 唤醒与基础配置 ===
    mpu6050_write_register(0x6B, 0x00);  // PWR_MGMT_1: 退出睡眠
    mpu6050_write_register(0x1A, 0x03);  // CONFIG: DLPF 42Hz (陀螺仪)
    mpu6050_write_register(0x1B, 0x00);  // GYRO_CONFIG: ±250°/s
    mpu6050_write_register(0x1C, 0x00);  // ACCEL_CONFIG: ±2g
    mpu6050_write_register(0x19, 4);     // SMPLRT_DIV: 1kHz/(1+4)=200Hz

    // === 第2步：启用 AUX I2C 旁路以访问磁力计 ===
    mpu6050_write_register(0x37, 0x02);  // INT_PIN_CFG: BYPASS_EN=1
    mpu6050_write_register(0x6A, 0x00);  // USER_CTRL: 关闭 AUX I2C Master

    // === 第3步：QMC5883L 初始化（地址 0x0D）===
    // 0x0D → QMC5883L; 若为 HMC5883L 则 0x1E
    qmc5883l_write_register(0x09, 0x01); // CTRL1: OSR=512, Mode=Continuous, ODR=200Hz
    qmc5883l_write_register(0x0A, 0x00); // CTRL2: ±2 Gauss scale

    // === 第4步：BMP180 初始化（地址 0x77）===
    bmp180_read_calibration();           // 读取校准系数（EEPROM 0xAA-0xBF）
}
```

**Madgwick AHRS 算法（IMU+磁力计融合，9-DOF）：**

选择 Madgwick（而非 Mahony）的原因：
- Madgwick 同时融合加速度计 + 陀螺仪 + 磁力计，Yaw 角不漂
- 计算量适中（约 200 次浮点乘法/次），ESP32-P4 400MHz 完全胜任
- 单参数调优（β=0.1），远简单于 Kalman
- 开源 C 实现成熟（arduino-libraries/MadgwickAHRS），可直接移植

```c
// 移植自 Madgwick/MadgwickAHRS (GPL)，适配 ESP-IDF
typedef struct {
    float q0, q1, q2, q3;     // 四元数
    float beta;                // 增益（默认 0.1）
    float inv_sample_freq;     // 1/采样频率
    float pitch, roll, yaw;    // 欧拉角（度）
} madgwick_filter_t;

void madgwick_update_9dof(madgwick_filter_t *f,
    float gx, float gy, float gz,  // 陀螺仪 (°/s)
    float ax, float ay, float az,  // 加速度计 (g)
    float mx, float my, float mz,  // 磁力计 (μT/Gauss)
    float dt)                       // 时间步长 (s)
{
    // 1. 陀螺仪 deg/s → rad/s
    // 2. 加速度计+磁力计归一化
    // 3. 梯度下降修正步 (gradient descent corrective step)
    // 4. 四元数积分更新
    // 5. 四元数归一化
    // 6. 计算欧拉角
    // (完整实现见 MadgwickAHRS.cpp, ~200行)

    f->pitch = asinf(-2.0f*(f->q1*f->q3 - f->q0*f->q2)) * 57.2958f;
    f->roll  = atan2f(2.0f*(f->q0*f->q1 + f->q2*f->q3),
                      1.0f - 2.0f*(f->q1*f->q1 + f->q2*f->q2)) * 57.2958f;
    f->yaw   = atan2f(2.0f*(f->q0*f->q3 + f->q1*f->q2),
                      1.0f - 2.0f*(f->q2*f->q2 + f->q3*f->q3)) * 57.2958f;
}
```

**IMU 校准流程（开机必须执行）：**

| 校准项 | 方法 | 时间 |
|--------|------|------|
| 陀螺仪零偏 | 静止 2 秒，采集 400 样本求均值 | ~2s |
| 加速度计标定 | 6 面静置法（或简化：水平静置校准 Z 轴） | ~5s |
| 磁力计硬铁+软铁 | 8 字旋转采集，椭球拟合 | ~10s |

#### 2.1.4 FreeRTOS 任务架构

```
ESP32-P4 双核任务分配:

Core 0 (HP, 400MHz) — 传感器采集
├── camera_task     Priority 10, Stack 8KB
│   └── OV5640 capture → JPEG buffer → Queue
├── imu_task        Priority 12, Stack 4KB
│   └── I2C read @200Hz → Madgwick filter → IMU Queue
└── bmp180_task     Priority 5, Stack 2KB
    └── I2C read @10Hz → altitude → shared struct

Core 1 (HP, 400MHz) — 通信发送
├── tx_task         Priority 8, Stack 8KB
│   └── Dequeue frame+IMU → pack protocol → UART TX (DMA)
└── usb_task (可选)  Priority 7, Stack 6KB
    └── High-res JPEG → USB Bulk Transfer

共享资源:
├── frame_queue     QueueHandle_t, 容量=4
├── imu_queue       QueueHandle_t, 容量=8
├── altitude_shared float (atomic access)
└── tx_mutex        SemaphoreHandle_t (互斥 UART)
```

### 2.2 ESP32-P4 ↔ MUSE Pi PRO 通信协议

#### 2.2.1 精简帧协议设计

考虑到 ESP32-P4 资源有限，协议从原定的 KCP-Lite 简化。KCP 的可靠传输（FEC + ARQ）在 **有线串行** 场景下冗余度高，仅对无线链路必要。

**当前阶段采用简化方案：**

```
Packet Format (variable length):
┌───────┬──────┬──────┬──────┬──────────┬────────┬──────┬──────┐
│ Magic │ Type │ Seq# │ TS   │ Length   │ Data   │ CRC16│ Magic │
│ 2B    │ 1B   │ 2B   │ 4B   │ 2B       │ N bytes│ 2B   │ 2B    │
│ 0xA55A│      │      │ (ms) │          │        │      │ 0x5AA5│
└───────┴──────┴──────┴──────┴──────────┴────────┴──────┴──────┘

Type 定义:
  0x01 — JPEG 视频帧
  0x02 — IMU 姿态四元数 (qw,qx,qy,qz, altitude)
  0x03 — IMU 原始数据 (accel,gyro,mag 各3轴, 用于离线标定)
  0x04 — 心跳/状态 (battery, temp, FPS, 错误码)
  0x05 — 控制命令应答
```

**为什么用 Magic 头尾包围**: 在高速连续流中，单一帧头容易误匹配（JPEG 数据中含 0xA55A）。头尾双重 Magic + CRC16 校验极大提高帧同步可靠性。

**UART 配置**：3Mbps baud, 8N1, DMA 发送, 硬件流控 RTS/CTS 视情况开启。

#### 2.2.2 MUSE Pi 端接收驱动

```c
// K1 MUSE Pi 侧：通过 40-pin GPIO 的 UART 连接
// K1 GPIO68 (TX), GPIO69 (RX) → ESP32-P4 UART1 (RX/TX)
// Linux 设备节点: /dev/ttyS1 (通过 device tree 配置)

typedef struct {
    int fd;                         // UART file descriptor
    uint8_t rx_buf[4096];           // 环形接收缓冲
    uint16_t rx_head, rx_tail;
    // ... frame parsing state machine
} protocol_receiver_t;

// 帧同步状态机
typedef enum {
    STATE_IDLE,       // 等待 Magic 0xA55A
    STATE_HEADER,     // 接收 Type+Seq#+TS+Length (9 bytes)
    STATE_PAYLOAD,    // 接收 Data
    STATE_CRC,        // 校验 CRC16
    STATE_COMPLETE    // 帧完成，分发到处理线程
} frame_state_t;
```

**UART 在 K1 上的配置**（参考 MUSE Pi 用户手册 26-pin GPIO）：
- `GPIO68` → TX → 接 ESP32-P4 RX
- `GPIO69` → RX → 接 ESP32-P4 TX
- Baud: 3000000 (3Mbps)
- 需在 Bianbu Linux 上配置 device tree 或使用 `stty` 配置

#### 2.2.3 进阶：USB 2.0 HS 高带宽通道

当需要 640×480 或更高分辨率 JPEG 流时：

```c
// ESP32-P4 USB OTG 配置为 Device
// USB Video Class (UVC) 或自定义 Bulk Transfer
// 优势：480Mbps 带宽，可传 VGA@15fps JPEG (~50KB/frame)
// 劣势：需 USB 线缆，功耗增加
```

### 2.3 MUSE Pi PRO AI 推理优化

#### 2.3.1 SpacemiT K1/M1 NPU 推理链路

根据 SpacemiT 官方文档，K1 芯片：
- 支持 ONNX Runtime 推理框架
- NPU 提供 2.0 TOPS INT8 算力
- 通过 Spacengine SDK 访问 NPU
- 支持 TensorFlow Lite / ONNX Runtime 后端

**推理链路的改造路径：**

```
当前 C 代码                    目标
══════════                    ════
#ifdef HAS_ONNX_RUNTIME       #ifdef HAS_SPACENGINE_NPU
  ONNX CPU (fp32)      ───▶    Spacengine INT8 NPU
  18-25 FPS                    30 FPS (目标)
#else                         #else
  启发式回退 (<10%)     ───▶    (移除，实际部署必备 NPU)
#endif
```

**模型 INT8 量化 Pipeline：**

```
1. 导出 ONNX (fp32)
   YOLOv8n.onnx → ONNX Simplifier 优化

2. 量化校准
   ├── 收集 500+ 张代表性图像的激活分布
   ├── Spacengine 量化工具 (或使用 ONNX Runtime quantization tools)
   └── 输出: yolov8n_int8.onnx

3. NPU 部署
   ├── Spacengine SDK API 加载 INT8 模型
   ├── 配置 NPU 内存 (DDR 共享内存区域)
   └── 推理: spacengine_run(session, input, output)
```

**各模型在 K1 NPU 上的预期性能：**

| 模型 | 输入尺寸 | 参数量 | 预期 NPU 耗时 | 需求 |
|------|---------|--------|-------------|------|
| YOLOv8n | 640×640 | 3.2M | ~15ms | INT8 量化 |
| YOLOv8-Pose | 640×640 | 3.3M | ~18ms | INT8 量化 |
| SCRFD 10g | 640×640 | 0.8M | ~5ms | INT8 量化 |
| ArcFace glintr100 | 112×112 | 65M | ~25ms | INT8 量化 |

**注意**：以上为外推估计。实际性能取决于 Spacengine SDK 对每个算子的优化程度。需要实机 benchmark。

#### 2.3.2 ONNX Runtime 在 RISC-V 上的部署

SpacemiT Bianbu Linux 预置 ONNX Runtime 支持：

```bash
# Bianbu Linux 安装 ONNX Runtime
sudo apt install onnxruntime

# 或从源码编译 (启用 RVV 向量化)
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --arm64 \
    --build_shared_lib --parallel \
    --cmake_extra_defines \
    CMAKE_C_FLAGS="-march=rv64gcv" \
    CMAKE_CXX_FLAGS="-march=rv64gcv"
```

**RVV 0.7.1 向量化**（X60 核心使用的版本）：

K1 的 X60 核心使用 RVV 0.7.1（非 1.0）。与 RVV 1.0 的主要差异：
- `vsetvli` 指令语法不同
- 部分 mask 操作行为不同
- 编译器需 `-menable-experimental-extensions`

对于关键路径（图像预处理、后处理 NMS），可手写 RVV intrinsics 加速：
```c
// 图像归一化 RVV 加速示例（伪代码）
void normalize_rvv(float* dst, uint8_t* src, int n, float scale) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = vsetvl_e8m1(n - i);
        vuint8m1_t v = vle8_v_u8m1(src + i, vl);
        vfloat32m1_t vf = vfwcvt_f_xu_v_f32m1(v, vl);
        vf = vfmul_vf_f32m1(vf, scale, vl);
        vse32_v_f32m1(dst + i, vf, vl);
    }
}
```

**注意**：RVV 加速属于优化阶段（v2.0），初期可先使用标量实现。

### 2.4 空间定位改进

#### 2.4.1 IMU 辅助深度估计

当前空间引擎使用 `Z = f_y × H_avg / h_bbox` 单一先验估计深度，误差较大。

改进方案：利用 IMU 的加速度计数据辅助判断相机俯仰角，修正深度估计：

```
实际深度 Z' = Z × cos(pitch)  (当相机向下倾斜)
世界坐标 X' = (u - cx) × Z' / fx
世界坐标 Y' = (v - cy) × Z' / fy + H_camera
```

#### 2.4.2 BMP180 气压计辅助高度

```c
// BMP180 → 绝对高度 (ISA 标准大气模型)
float bmp180_altitude(float pressure_pa, float sea_level_pa) {
    return 44330.0f * (1.0f - powf(pressure_pa / sea_level_pa, 0.1903f));
}
```

与视觉高度估计进行 Kalman 融合，获得更稳定的 Z 轴。

### 2.5 与现有 C 代码库的接口对齐

#### 2.5.1 新增接口

```c
// ===== ESP32-P4 固件侧 (新增文件) =====
// src/esp32p4/main/camera_capture.h
typedef struct {
    uint8_t* jpeg_buf;
    size_t jpeg_len;
    uint32_t timestamp_ms;
} jpeg_frame_t;

// src/esp32p4/main/imu_fusion.h
typedef struct {
    float qw, qx, qy, qz;    // 姿态四元数
    float pitch, roll, yaw;  // 欧拉角 (度)
    float altitude_m;        // 气压高度 (m)
    float temperature_c;     // 温度 (°C)
    uint32_t timestamp_ms;
} imu_pose_t;

// src/esp32p4/main/protocol.h
typedef struct {
    uint8_t type;
    uint16_t seq;
    uint32_t timestamp_ms;
    uint16_t length;
    uint8_t data[];
} packet_t;

// ===== MUSE Pi 侧 (扩展现有接口) =====
// src/arrow_receiver.h  [新增]
typedef struct {
    protocol_receiver_t* receiver;
    jpeg_frame_t latest_frame;
    imu_pose_t latest_pose;
    pthread_mutex_t lock;
} arrow_receiver_t;

arrow_receiver_t* arrow_receiver_create(const char* uart_dev);
int arrow_receiver_get_pair(arrow_receiver_t* ar,
    jpeg_frame_t* frame, imu_pose_t* pose);
void arrow_receiver_destroy(arrow_receiver_t* ar);

// src/imu_handler.h  [扩展]
// 新增: 接收已融合的姿态数据，直接使用
void imu_handler_set_external_pose(imu_handler_t* h, const imu_pose_t* pose);

// src/spatial_engine.h  [扩展]
// 新增: 使用 IMU 姿态修正深度估计
void spatial_engine_set_camera_pose(spatial_engine_t* eng,
    float pitch, float roll, float yaw);
```

### 2.6 构建系统整合

项目采用 CMake + build.py 双构建模式：

```
lingqi_tantong_c/
├── CMakeLists.txt                 # 顶层 CMake (MUSE Pi 端)
├── src/                           # 现有 C 代码 (MUSE Pi 目标)
├── esp32p4_firmware/              # ESP32-P4 固件 (独立 ESP-IDF 项目)
│   ├── CMakeLists.txt
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   ├── main.c
│   │   ├── camera_capture.c/h
│   │   ├── imu_fusion.c/h
│   │   ├── protocol.c/h
│   │   └── gy87_driver.c/h
│   ├── components/
│   │   └── esp32-camera/          # 乐鑫组件
│   └── sdkconfig                  # ESP-IDF 配置
├── cmake/
│   ├── riscv64-toolchain.cmake    # RISC-V 交叉编译
│   └── esp32p4-toolchain.cmake    # ESP32-P4 交叉编译
├── configs/
│   ├── default.yaml               # 统一配置
│   └── esp32p4.yaml               # ESP32 配置
└── docs/
    ├── ARCHITECTURE.md
    ├── BUILD_GUIDE.md
    ├── IMPLEMENTATION_GAPS.md
    └── IMPROVEMENT_PLAN.md        # 本文档
```

**构建方式**：
- ❌ `Makefile` — 已移除，CMake 统一管理
- ❌ `build_zig.py` — 已移除
- ✅ `build.py` — 开发便捷脚本
- ✅ `CMakeLists.txt` — 主构建系统（含 RISC-V 交叉编译 preset）

---

## 3. 开发路线图

### Phase 0: 硬件验证集成测试 (1-2周)

```
目标: 验证所有硬件正常工作

□ ESP32-P4 固件: LED Blink (FreeRTOS Hello World)
□ OV5640 测试: 使用 esp32-camera example 拍第一张 JPEG
□ GY87 测试: I2C 扫描，验证三个芯片地址
□ GY87 测试: MPU6050 加速度计/陀螺仪原始数据读取
□ GY87 测试: 磁力计通过 AUX I2C 旁路读取
□ GY87 测试: BMP180 气压/温度读取
□ UART 通信: ESP32-P4 ↔ MUSE Pi, 115200 baud echo test
□ UART 通信: 逐步提升至 3Mbps 验证稳定性
□ USB 通信: ESP32-P4 USB OTG 配置与 MUSE Pi 识别
```

### Phase 1: 箭矢端固件 MVP (2-3周)

```
目标: ESP32-P4 能发送 JPEG + IMU 数据到 MUSE Pi

□ OV5640: QVGA@15fps JPEG 连续捕获
□ Madgwick AHRS: C 移植与 @200Hz 实时运行
□ BMP180: 高度计算 + 低通滤波
□ IMU 校准: 上电自动零偏校准
□ 精简协议: Pack/Unpack + CRC16 实现
□ 双 FreeRTOS Task: camera(Core 0) + tx(Core 1)
□ UART DMA 发送: 帧协议 → 3Mbps TX
□ MUSE Pi 端: 协议接收线程 + JPEG→SDL/OpenCV 解码显示
```

### Phase 2: 核心管线迁移与对接 (3-4周)

```
目标: 现有的 lingqi_tantong_c 代码在 MUSE Pi 上实时运行

□ 交叉编译: lingqi_tantong_c → RISC-V binary (Bianbu Linux)
□ arrow_receiver: 集成协议接收 → FrameData 接口适配
□ imu_handler: 扩展为接收外部姿态数据
□ spatial_engine: IMU 姿态修正深度估计
□ 端到端: 箭矢端 ESP32-P4 → MUSE Pi 接收 → 检测 → 显示
□ ONNX Runtime: RISC-V Bianbu 编译/安装
□ 模型转换: fp32 ONNX → INT8 量化
□ NPU 推理: Spacengine SDK 集成（如果可用）
```

### Phase 3: 高级功能实现 (4-6周)

```
目标: 补齐原路线图中核心缺失模块

□ Mahony 互补滤波: 作为 Madgwick 的备选方案
□ VINS-Mono: IMU 预积分器 + 滑动窗口 BA (简化版)
□ KCP-Lite: 无线场景下的可靠传输
□ ATW: 异步时间扭曲（利用 IMU 姿态）
□ TinierHAR: 时序动作识别（30帧滑动窗口）
□ MiDaS 深度估计: NPU INT8 推理集成
```

### Phase 4: 优化与产品化 (4-8周)

```
目标: 性能优化、稳定性、产品化

□ RVV 0.7.1 向量化: 图像预处理关键路径
□ 内存优化: K1 8GB 内 cont ≤1.5GB
□ 功耗优化: ESP32-P4 LP Core 管理
□ 系统集成: HDMI/MIPI DSI 显示输出
□ 3D 渲染: OpenGL ES 3.2 AR 叠加
□ 自动化测试: CI/CD + 硬件在环测试
□ 量产烧录方案
```

---

## 4. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **Spacengine SDK 文档/API 不完整** | 中 | 高 | 备选 ONNX Runtime CPU + RVV 加速 |
| **ESP32-P4 MIPI CSI 驱动不稳定** | 中 | 高 | 降级为 DVP 并行接口 + 外部 ISP |
| **UART 3Mbps 有线不稳** | 低 | 中 | 降为 1.5Mbps; 增加重发机制 |
| **RVV 0.7.1 编译器支持不成熟** | 高 | 中 | 先用标量，等 GCC 稳定后再向量化 |
| **NPU INT8 量化精度损失 >5%** | 中 | 中 | 混合精度 (敏感层 fp16) |
| **GY87 磁力计实际为 QMC5883L** | 高 | 低 | 代码同时支持两种，上电自动检测 |
| **MUSE Pi PRO 散热不足** | 低 | 中 | 加装主动风扇散热 (板载风扇接口) |
| **供电不足 (箭矢端)** | 低 | 中 | OV5640 + ESP32-P4 + GY87 ≈ 3W, USB PD 5V/3A 充足 |

---

## 参考文献与资源

| 资源 | 链接 | 用途 |
|------|------|------|
| esp32-camera 官方驱动 | [espressif/esp32-camera](https://components.espressif.com/components/espressif/esp32-camera) | OV5640 驱动 |
| ESP32-P4 数据手册 | [espressif.com](https://www.espressif.com.cn/sites/default/files/documentation/esp32-p4_datasheet_cn.pdf) | P4 芯片规格 |
| MadgwickAHRS | [arduino-libraries/MadgwickAHRS](https://github.com/arduino-libraries/MadgwickAHRS) | 传感器融合算法 |
| Madgwick 原始论文 | [x-io.co.uk](https://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/) | 算法理论 |
| GY-87 传感器融合教程 | [how2electronics.com](https://how2electronics.com/measure-pitch-roll-yaw-with-mpu6050-hmc5883l-esp32/) | 硬件接线参考 |
| K1 MUSE Pi 用户手册 | [spacemit-com/docs](https://github.com/spacemit-com/docs-product) | GPIO/UART/CSI 配置 |
| SpacemiT K1 芯片概述 | [banana-pi.org](https://docs.banana-pi.org/en/BPI-F3/SpacemiT_K1) | NPU/媒体特性 |
| ESP32-CAM RTOS 优化 | [Boyyt357/ESP32-CAM-RTOS](https://github.com/Boyyt357/ESP32-CAM-WIFI-RTOS-Long-Range-VTX) | FreeRTOS 双核优化 |
| MPU6050 深度教程 | [CSDN](https://blog.csdn.net/2301_80079642/article/details/147356051) | 寄存器级配置 |
| MUSE Pi Pro 实测 | [smarthomecircle.com](https://smarthomecircle.com/spacemit-muse-pi-pro-riscv-review-performance-benchmarks) | 性能基准 |