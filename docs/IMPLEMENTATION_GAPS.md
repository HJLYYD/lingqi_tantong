# Unimplemented Module Registry & Development Roadmap

> This document tracks modules in the `lingqi_tantong_c` project that are **designed but not yet implemented**, as well as gaps relative to the target deployment platform. Each module is annotated with the corresponding Python prototype reference location.

---

## Classification Legend

| Mark | Meaning |
|------|---------|
| 🔴 Core Missing | Modules completely absent from the critical system path, blocking actual deployment |
| 🟠 Degraded Functionality | Placeholder/heuristic fallback implemented; accuracy insufficient for production |
| 🟡 Engineering Gap | Module logic is complete but lacks platform adaptation/build/test |
| 🟢 Optimization Pending | Functional but requires performance optimization |

---

## 🔴 Core Missing Modules

> **Update (2026-05)**: Item #2 (Mahony Complementary Filter) and Item #4 (KCP-Lite Reliable Transport) have been fully implemented. Retained in this document to track development history.

### 1. VINS-Mono Visual-Inertial Odometry

| Attribute | Details |
|-----------|---------|
| **Status** | 🔴 Completely unimplemented |
| **Reference** | `../lingqi tantong/algorithm_logic.md` §4.1 VINS-Mono Fusion |
| **Dependencies** | IMU data source + visual feature extraction + Ceres/g2o nonlinear optimization |
| **Blocker** | Cannot achieve true 6-DOF pose estimation; current spatial localization relies entirely on pure vision |

**What needs to be implemented:**
- IMU preintegrator (acceleration/angular velocity → relative pose increment between frames)
- Sliding window Bundle Adjustment (window size=8, Ceres Solver Jacobian auto-differentiation)
- Marginalization strategy (Marginalization prior)
- Visual-inertial timestamp alignment (td estimation)
- IMU bias estimation and real-time calibration

**Suggested file organization:**
```
src/vins/
  imu_preintegrator.c/h     — IMU preintegration node
  marginalization.c/h       — Schur complement marginalization
  feature_tracker.c/h       — Optical flow feature tracking
  estimator.c/h             — Sliding window optimizer
  initial_alignment.c/h     — Visual-inertial initialization
```

---

### 2. Mahony Complementary Filter (Arrow-side Attitude Estimation)

| Attribute | Details |
|-----------|---------|
| **Status** | 🟢 Implemented (`src/mahony_filter.c` + `esp32p4_firmware/main/imu_fusion.c` Madgwick variant) |
| **Reference** | `../lingqi tantong/algorithm_logic.md` §(3) Mahony Complementary Filter |
| **Parameters** | Kp=2.0, Ki=0.005, cutoff frequency=0.5Hz, 200Hz |
| **Notes** | Shooter-side Mahony implemented, pending integration; ESP32P4 firmware uses Madgwick algorithm dual-version |

**Completed items:**
- ✅ PI controller correction Mahony filter
- ✅ Accelerometer normalization → gravity direction error → angular velocity correction → quaternion update
- ✅ Integral error isolation (eliminates gyroscope constant drift)
- ✅ Output: real-time quaternion `[qw, qx, qy, qz]`

**File locations:** `src/mahony_filter.c/h` (shooter side), `esp32p4_firmware/main/imu_fusion.c/h` (arrow side Madgwick)

---

### 3. ATW Asynchronous Time Warp (AR Latency Compensation)

| Attribute | Details |
|-----------|---------|
| **Status** | 🔴 Not implemented (current `ar_renderer_compensate_motion()` only performs simple 2D Euler rotation, not true ATW) |
| **Reference** | `../lingqi tantong/algorithm_logic.md` §5 ATW Asynchronous Time Warp |
| **Blocker** | Motion-to-Photon latency > 35ms, far exceeding the 20ms human perception threshold |

**What needs to be implemented:**
- Render thread synchronously reads latest IMU attitude quaternion
- Remote timestamp prediction (pose extrapolation within ΔT)
- 2D projection matrix construction (isometric projection → rotation → inverse projection)
- GPU texture mapping (OpenGL ES 3.0 fragment shader implementation)
- Achieve MTP ≤ 17.8ms target

---

### 4. KCP-Lite Reliable Transport Protocol

| Attribute | Details |
|-----------|---------|
| **Status** | 🟢 Implemented (`src/kcp_lite.c/h`, 576-line complete implementation) |
| **Reference** | `../lingqi tantong/algorithm_logic.md` §6.1-6.3 KCP-Lite Protocol |
| **Parameters** | MTU=1100, FEC(4,5), snd_wnd=128, nodelay=1, resend=50ms |
| **Notes** | Protocol layer complete, pending integration into network transport path |

**Completed items:**
- ✅ Sender: frame buffer + FEC encoding + timeout retransmission + NACK response
- ✅ Receiver: FEC decoding + NACK request + out-of-order reassembly
- ✅ Three packet types (type=0: control frame, type=1: video frame, type=2: IMU data)
- ✅ Congestion control strategy (KCP nodelay mode-like)

**File location:** `src/kcp_lite.c/h`

---

### 5. MiDaS Monocular Depth Estimation

| Attribute | Details |
|-----------|---------|
| **Status** | 🔴 Interface reserved but inference not implemented |
| **Model** | MiDaS-Small, 256×256 input |
| **Blocker** | Depth information relies on a single prior (human body height/width), accuracy is extremely low |

**What needs to be implemented:**
- ONNX Runtime loading of MiDaS ONNX model
- Preprocessing: normalization → 256×256 resize
- Postprocessing: inverse normalization + upsampling back to original resolution
- Integration with spatial engine: depth values replace current prior height estimation

---

### 6. ST-GCN Temporal Action Recognition

| Attribute | Details |
|-----------|---------|
| **Status** | 🟢 Implemented (`src/stgcn_action_recognizer.c`, ~445 lines) |
| **Reference** | ST-GCN (Spatial-Temporal Graph Convolutional Network) |
| **Model** | `models/Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx` |

**Implemented features:**
- ✅ Auto-detection of output class count from ONNX model shape
- ✅ 30-frame sliding window with zero-padding for cold start
- ✅ Motion (mot) channel computation from adjacent frames
- ✅ Dual-input ONNX inference (pts + mot tensors)
- ✅ Confidence-sorted top-K action classification
- ✅ Dynamic output dimension detection (7-class through 60-class models)

**Pending enhancements:**
- ⬜ INT8 quantization for SpacemiT EP acceleration (currently FP32 CPU-only)
- ⬜ Multi-person action recognition (currently single-person tracking)

---

### 7. ICP Point Cloud Registration (Arrow-Shooter Spatial Alignment)

| Attribute | Details |
|-----------|---------|
| **Status** | 🔴 Completely unimplemented |
| **Blocker** | Cannot align arrow-side 3D detection coordinates to shooter-side reference frame |

**What needs to be implemented:**
- FPFH feature extraction (Fast Point Feature Histograms)
- SAC-IA coarse registration
- ICP point-to-plane fine registration
- Alignment matrix output (rotation + translation)

---

## 🟠 Degraded Functionality Modules

### 8. Deep Learning Inference — Heuristic Fallback Accuracy Issues

| Module | ONNX Version Accuracy | Heuristic Fallback Accuracy | Gap |
|--------|----------------------|----------------------------|-----|
| YOLOv8n Detection | mAP50≈53.6% | Edge+Contrast ≈ <10% | Severe |
| YOLOv8-Pose | AP≈50.2% | Pure geometric estimation ≈ <5% | Severe |
| YOLOv5-Face Face Detection | AP≈93% | Skin color+symmetry ≈ <20% | Severe |
| ArcFace Recognition | TAR@FAR≈99% | 8×8 block features ≈ <30% | Severe |

**Solution:** Integrate ONNX Runtime (the `#ifdef HAS_ONNX_RUNTIME` logic in the code is ready; needs actual linking to the ONNX Runtime library)

---

## 🟡 Engineering Gap Modules

### 9. Target Platform Adaptation (SpacemiT X60 RISC-V)

| Gap Item | Current Status | Target |
|----------|---------------|--------|
| Cross-compilation toolchain | None | riscv64-unknown-linux-gnu-gcc |
| RVV 0.7.1 vectorization | None | VLA intrinsics / assembly |
| SpacemiT EP (Execution Provider) inference | None | INT8 quantization + IME API |
| Bianbu Linux BSP | None | GPIO/HW Timer/Sensor drivers |
| K1 memory constraint adaptation | No heap memory upper limit | Limit ≤1.5GB (K1 available) |

### 10. Insufficient Test Coverage

| Metric | Current | Target |
|--------|---------|--------|
| Test files | 1 (`test_basic.c`) | ≥5 |
| Unit tests | 12 | ≥100 |
| Integration tests | 0 | ≥10 |
| Performance benchmarks | 0 | ≥5 scenarios |
| CI/CD | None | GitHub Actions + K1 hardware |

### 11. RISC-V Cross-Compilation Toolchain Configuration

CMakeLists.txt has integrated RISC-V multi-version RVV vector extension detection and compilation configuration:
- RVV 1.0 (`-march=rv64gcv1p0`) — preferred
- RVV 1.0 fallback (`-march=rv64gcv`)
- RVV 0.7.1 (`-march=rv64gcv0p7`) — X60 core compatible
- Bianbu sysroot path configured via `-DBIANBU_SYSROOT`

Still needed:
- Actual compiler path configuration (`CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`)
- Cross-compilation toolchain file completion

### 12. Build System Integration

The project uses:
- **CMake**: Cross-platform standard build with RISC-V cross-compilation preset (RV64GCV 1.0, RV64GCV 0.7, RV64GC)
- **Toolchain files**: `cmake/riscv64-toolchain.cmake`, `cmake/esp32p4-toolchain.cmake`

---

## 🟢 Optimization Pending Modules

### 13. IMUHandler Integrated into Main Loop

`imu_handler_create()` has been created and integrated into the `system_controller_process_video` main loop:
- ✅ Each frame obtains IMU attitude via `imu_handler_get_latest_pose()`
- ✅ Attitude data participates in spatial computation via `spatial_engine_set_camera_pose()`
- ⬜ Pending: Real GY-87 I²C driver data source (target platform)

### 14. AR Renderer Extension

Current AR renderer only supports 2D rectangles + text. Needs:
- OpenGL ES 3.0 rendering pipeline
- Skeletal animation 3D character rendering
- Attitude quaternion-driven rendering matrix

### 15. Logging System Enhancement

Needs:
- Log file rotation (size/time-based)
- Structured log format (JSON)
- Performance counter logging (FPS, inference time, memory)

---

## Development Roadmap (Based on Actual Hardware: ESP32-P4 + GY87 + MUSE Pi PRO)

### Overview

```
v1.0.0-alpha (Current: x86 C code + offline video)
  │
  ├─ v1.1 Phase 0: Hardware Validation (1-2 weeks)           ← 🔴 Highest Priority
  │   ├── ESP32-P4 FreeRTOS Hello World
  │   ├── OV5640 MIPI CSI JPEG capture test
  │   ├── GY87 I2C three-chip address scan and raw readout
  │   ├── ESP32-P4 ↔ MUSE Pi UART echo test
  │   └── USB 2.0 HS OTG configuration verification
  │
  ├─ v1.2 Phase 1: Arrow-side Firmware MVP (2-3 weeks)      ← 🟠 Core Deliverable
  │   ├── OV5640 QVGA@15fps JPEG continuous capture (double-buffered+DMA)
  │   ├── Madgwick AHRS 9-DOF fusion (@200Hz, C port)
  │   ├── BMP180 barometric altitude calculation + low-pass filter
  │   ├── IMU power-on automatic bias calibration
  │   ├── Compact frame protocol (Magic frame sync + CRC16 + state machine)
  │   ├── FreeRTOS dual-core task allocation (Core0: Capture, Core1: Transmit)
  │   ├── UART 3Mbps DMA transmit
  │   └── MUSE Pi side protocol reception + JPEG decode display
  │
  ├─ v1.3 Phase 2: Core Pipeline Migration (3-4 weeks)        ← 🟡 Critical Integration
  │   ├── lingqi_tantong_c cross-compilation → RISC-V binary
  │   ├── arrow_receiver module addition (protocol→FrameData adaptation)
  │   ├── imu_handler extension (receive external attitude quaternion)
  │   ├── spatial_engine IMU pitch angle depth correction
  │   ├── End-to-end path: ESP32-P4 → MUSE Pi receive → AI inference → Display
  │   ├── ONNX Runtime RISC-V Bianbu compilation and installation
  │   ├── Model fp32 → INT8 quantization (SpacemiT toolchain)
  │   └── AI acceleration SpacemiT EP inference integration
  │
  └─ v2.0 Phase 3-4: Advanced Features + Optimization (8-14 weeks)  ← 🟢 Finalization
      ├── VINS-Mono visual-inertial odometry (IMU preintegration + sliding window BA)
      ├── ATW asynchronous time warp (IMU attitude-driven, MTP≤17.8ms)
      ├── KCP-Lite reliable transport (wireless scenarios only)
      ├── ST-GCN action recognition: ✅ already implemented (`src/stgcn_action_recognizer.c`)
      ├── MiDaS depth estimation (AI acceleration INT8)
      ├── RVV 0.7.1 vectorized acceleration (image preprocessing critical path)
      ├── Memory optimization (K1 ≤1.5GB)
      ├── OpenGL ES 3.2 AR 3D rendering overlay
      └── CI/CD + hardware-in-the-loop testing
```

---

### Phase 0: Hardware Validation Integration Test (1-2 weeks) — Key Hardware Operation Steps

> **Objective**: Ensure all hardware modules are properly connected and can communicate normally.
> **Deliverable**: Hardware Validation Report (signal quality / communication stability / power consumption measurements).

| ID | Task | Hardware | Key Operations | Acceptance Criteria |
|----|------|----------|----------------|-------------------|
| H0.1 | ESP32-P4 environment setup | ESP32-P4 | ESP-IDF v5.4 installation, `idf.py build flash monitor` | LED Blink (FreeRTOS Task) |
| H0.2 | OV5640 first frame capture | ESP32-P4+OV5640 | MIPI CSI FPC 22p connection, `esp32-camera` example | JPEG photo captured (USB export) |
| H0.3 | GY87 I2C scan | ESP32-P4+GY87 | VCC→3.3V, SDA/SCL→I2C, confirm pull-up resistors | Discover 0x68(MPU), 0x0D/0x1E(Mag), 0x77(BMP) |
| H0.4 | MPU6050 raw readout | ESP32-P4+GY87 | PWR_MGMT_1=0x00 wake, read 0x3B-0x48 | 6-axis data correct (gravity≈1g Z-axis) |
| H0.5 | Magnetometer AUX bypass | ESP32-P4+GY87 | INT_PIN_CFG (0x37)=0x02 enable bypass | Direct read of QMC5883L registers |
| H0.6 | BMP180 readout | ESP32-P4+GY87 | Read calibration coefficients 0xAA-0xBF, trigger temperature/pressure conversion | Temperature±1°C, pressure 300-1100hPa |
| H0.7 | UART basic communication | ESP32-P4↔MUSE Pi | GPIO68(TX)↔P4_RX, GPIO69(RX)↔P4_TX, GND common ground | 115200-8N1 Echo with no bit errors |
| H0.8 | UART high-speed test | ESP32-P4↔MUSE Pi | 115200→921600→1500000→3000000 | 3Mbps continuous 1MB data CRC error-free |

#### Phase 0 Implementation Details

**H0.3 GY87 I2C Scan Pseudocode**:
```c
// ESP-IDF I2C scan
for (uint8_t addr = 1; addr < 127; addr++) {
    i2c_master_write_to_device(I2C_NUM, addr, NULL, 0, 100 / portTICK_PERIOD_MS);
    if (err == ESP_OK) ESP_LOGI(TAG, "Found device at 0x%02X", addr);
}
// Expected output: 0x68 (MPU6050), 0x0D (QMC5883L) or 0x1E (HMC5883L), 0x77 (BMP180)
```

**H0.5 GY87 AUX I2C Bypass Setup**:
```c
// Critical! The GY87's magnetometer is not independently on the I2C bus,
// but connected through the MPU6050's AUX I2C interface.
// Bypass mode must be enabled first to directly access the magnetometer:
mpu_write(0x6B, 0x00);  // Wake up MPU6050
mpu_write(0x37, 0x02);  // INT_PIN_CFG: BYPASS_EN = 1
mpu_write(0x6A, 0x00);  // USER_CTRL: Disable AUX I2C Master
// After this, the magnetometer can be directly read/written at address 0x0D or 0x1E
```

**H0.7 K1 MUSE Pi Side UART Configuration** (Bianbu Linux):
```bash
# Check available serial ports
ls /dev/ttyS*  # ttyS1 typically corresponds to 40-pin GPIO UART
# Configure 3Mbps (if kernel supports it)
stty -F /dev/ttyS1 3000000 cs8 -cstopb -parenb
# Test send/receive
cat /dev/ttyS1 &  echo "TEST" > /dev/ttyS1
```

---

### Phase 1: Arrow-side Firmware MVP (2-3 weeks) — Implementation Details

| ID | Module | File Location | Core Logic | Acceptance Criteria |
|----|--------|--------------|------------|-------------------|
| F1.1 | camera_capture | `esp32p4_firmware/main/camera_capture.c` | esp_camera_init + fb_count=2 + GRAB_LATEST | QVGA@15fps no frame drops |
| F1.2 | imu_fusion | `esp32p4_firmware/main/imu_fusion.c` | Port MadgwickAHRS, @200Hz invocation | Quaternion output, Roll/Pitch ±0.5° |
| F1.3 | gy87_driver | `esp32p4_firmware/main/gy87_driver.c` | I2C batch readout (@200Hz) → calibration → Madgwick | 3-chip data stream stable |
| F1.4 | bmp180_altitude | Embedded in `esp32p4_firmware/main/gy87_driver.c` | Read UT+UP → compensation → ISA formula → altitude | Altitude resolution 0.1m |
| F1.5 | imu_calibration | Embedded in `esp32p4_firmware/main/main.c` + `imu_fusion.c` | Power-on 2s stationary → bias estimation, figure-8 → magnetometer ellipsoid | Post-calibration stationary drift <0.2°/s |
| F1.6 | protocol_packer | `esp32p4_firmware/main/protocol.c` | Magic framing, Type, Seq, CRC16 | Pack/Unpack bidirectional verification |
| F1.7 | FreeRTOS tasks | `esp32p4_firmware/main/main.c` | xTaskCreatePinnedToCore dual-core allocation | Core0 capture Core1 transmit in parallel |
| F1.8 | receiver (K1 side) | `src/arrow_receiver.c` | UART→ring buffer→frame state machine→JPEG+IMU dispatch | Receive latency <5ms/frame |

#### Phase 1 Implementation Details

**F1.1 OV5640 Frame Rate Optimization Strategy**:

QVGA(320×240) JPEG approximately 8-15KB/frame, target 15fps.

```c
// Frame rate calculation
// UART 3Mbps = 375KB/s
// Per-frame JPEG ≈ 12KB → theoretical limit ≈ 31fps
// IMU data ≈ 50B additional per frame (negligible)
// Actual target: 15fps (accounting for protocol overhead and receiver-side decoding)
```

**F1.3 GY87 Driver Structure**:
```c
// gy87_driver.c structure
typedef struct {
    i2c_port_t i2c_num;
    // MPU6050 calibration
    float gyro_bias[3];       // Stationary bias
    float accel_bias[3];      // Gravity calibration
    float accel_scale[3];     // 6-face calibration
    // QMC5883L calibration
    float mag_hard_iron[3];   // Hard iron compensation
    float mag_soft_iron[9];   // Soft iron matrix
    // BMP180
    bmp180_calib_t calib;     // Calibration coefficients
} gy87_t;
```

**F1.8 arrow_receiver Interface Design** (K1 side):
```c
// Align with existing video_processor create/destroy pattern
arrow_receiver_t* arrow_receiver_create(const char* uart_dev);

// Block until next (frame, pose) pair is available, return -1 on timeout
int arrow_receiver_get_pair(arrow_receiver_t* ar,
    jpeg_frame_t* frame, imu_pose_t* pose, int timeout_ms);

// JPEG → RGB (adapt to video_processor's FrameData format)
int arrow_receiver_decode_frame(arrow_receiver_t* ar,
    const jpeg_frame_t* jpeg, FrameData* out_frame);

void arrow_receiver_destroy(arrow_receiver_t* ar);
```

---

### Phase 2: Core Pipeline Migration (3-4 weeks) — Implementation Details

| ID | Task | Key Steps | Risk | Acceptance Criteria |
|----|------|-----------|------|-------------------|
| M2.1 | RISC-V cross-compilation | CMake toolchain: `rv64gcv` flags, sysroot→Bianbu | RVV 0.7.1 vs 1.0 compatibility | Compiles + runs 11 tests |
| M2.2 | arrow_receiver integration | receiver→FrameData→SystemController | JPEG decode latency | 15fps end-to-end path |
| M2.3 | imu_handler extension | Add `imu_handler_set_pose()` to receive external quaternion | Timestamp alignment | IMU data participates in spatial computation |
| M2.4 | spatial_engine correction | pitch/roll→depth correction formula | Accuracy verification | Z-axis error <20% (target) |
| M2.5 | End-to-end integration | ESP32→MUSE Pi→Infer→Track→Draw→HDMI | Full-chain latency | E2E latency <200ms |
| M2.6 | ONNX Runtime | Bianbu apt install or source compilation (RVV) | RISC-V ORT maturity | Model loads successfully |
| M2.7 | INT8 quantization | ONNX fp32→SpacemiT INT8, 500 calibration images | Accuracy loss | mAP loss <3% |
| M2.8 | AI acceleration inference | SpacemiT EP API | SDK availability | Inference latency < x86 fp32 |

---

### Phase 3-4: Advanced Features (8-14 weeks) — Aligned with Existing GAPS.md

| GAPS ID | Module | Phase | Priority |
|---------|--------|-------|----------|
| 1 | VINS-Mono Visual-Inertial Odometry | Phase 3 | 🔴 Core |
| 2 | Mahony Complementary Filter | ✅ Complete | 🟢 Implemented |
| 3 | ATW Asynchronous Time Warp | Phase 3 | 🔴 Core |
| 4 | KCP-Lite Transport | ✅ Complete | 🟢 Implemented |
| 5 | MiDaS Depth Estimation (AI Acceleration) | Phase 3 | 🔴 Accuracy |
| 6 | ST-GCN Action Recognition | ✅ Complete | 🟢 Implemented |
| 7 | ICP Point Cloud Registration | Phase 4 | 🟢 Optimization |
| 13 | IMUHandler Data Source Integration | Phase 2 | ✅ Covered in this plan |
| 14 | AR Renderer OpenGL ES 3.0 | Phase 4 | 🟢 Optimization |
| 9 | RISC-V Target Platform Adaptation | Phase 2 | ✅ Covered in this plan |
| 15 | Logging System Enhancement | Phase 4 | 🟢 Optimization |

---

### Hardware-Specific Module Registry (Unique to This Improvement Plan)

| ID | Module | Platform | File | Priority |
|----|--------|----------|------|----------|
| H0 | GY87 I2C Three-Chip Driver | ESP32-P4 | `esp32p4_firmware/main/gy87_driver.c/h` | 🔴 P0 |
| H1 | Madgwick AHRS 9-DOF Fusion | ESP32-P4 | `esp32p4_firmware/main/madgwick_filter.c/h` | 🔴 P0 |
| H2 | OV5640 MIPI CSI Capture | ESP32-P4 | `esp32p4_firmware/main/camera_capture.c/h` | 🔴 P0 |
| H3 | BMP180 Barometric Altitude Engine | ESP32-P4 | `esp32p4_firmware/main/bmp180_altitude.c/h` | 🟠 P1 |
| H4 | Compact Frame Protocol (Coprocessor→SBC) | ESP32-P4 | `esp32p4_firmware/main/protocol.c/h` | 🔴 P0 |
| H5 | Frame Protocol Receive Parser | MUSE Pi | `src/arrow_receiver.c/h` | 🔴 P0 |
| H6 | IMU Auto-Calibration Flow | ESP32-P4 | `esp32p4_firmware/main/imu_calibration.c/h` | 🟠 P1 |
| H7 | SpacemiT EP (Execution Provider) Inference Adapter | MUSE Pi | `src/inference/ai_accel_adapter.c/h` | 🟡 P2 |
| H8 | RVV 0.7.1 Vectorized Preprocessing | MUSE Pi | `src/utils/rvv_preprocess.c/h` | 🟢 P4 |

---

## Related References

| Document | Path |
|----------|------|
| Algorithm Logic Original | `../lingqi tantong/algorithm_logic.md` |
| Configuration File | `configs/default.yaml` |
| Architecture Document | `docs/ARCHITECTURE.md` |
| Build Guide | `docs/BUILD_GUIDE.md` |
| System Improvement Plan | `docs/IMPROVEMENT_PLAN.md` |
