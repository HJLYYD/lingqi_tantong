# K1 硬件资源充分利用 — 行动计划

> 基于硬件检查脚本 `scripts/k1_hardware_check.sh` 的输出结果
> ，按优先级逐步启用闲置硬件资源。

---

## 第一步: 在板子上运行硬件检查

```bash
cd ~/Desktop/temp/5
# 赋予执行权限
chmod +x scripts/k1_hardware_check.sh
# 运行检查
bash scripts/k1_hardware_check.sh 2>&1 | tee hardware_report.txt
```

**关键观察项**（对照输出）:

| 检查项 | 期望看到 | 如果没有 |
|--------|---------|---------|
| `/dev/tcm` | `[✓]` 可读写 | `sudo chmod 666 /dev/tcm` |
| `/dev/jpu` | `[✓]` 或 `[!]` (非必需) | 查 dmesg 确认驱动已加载 |
| `/dev/mpp_service` | `[✓]` 或 `[!]` | MPP SDK 可能需要手动编译 |
| `xquant` | `[✓]` | `pip install xquant` |
| `-mcpu=spacemit-x60` | `[✓]` | 升级 GCC 到 14+ |
| `rv64gcv1p0` | `[✓]` | K1 必有 |

---

## 第二步: SDK 下载 & 安装

### 2.1 k1x-jpu (JPEG 硬件解码)

**来源 1**: Gitee 官方仓库
```bash
git clone https://gitee.com/bianbu-linux/k1x-jpu.git
cd k1x-jpu && make && sudo make install
```

**来源 2**: 板载 APT 包管理器
```bash
apt-cache search k1x-jpu
# 如果有:
sudo apt install k1x-jpu-dev
```

**来源 3**: SpacemiT SDK 合集
```bash
git clone https://gitee.com/bianbu/spacemit-demo.git
# 查看 jpu 相关示例代码
ls spacemit-demo/jpu/
```

### 2.2 k1x-mpp (VPU 视频编解码)

```bash
# 来源 1: Gitee
git clone https://gitee.com/bianbu-linux/k1x-mpp.git
cd k1x-mpp && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install

# 来源 2: APT
apt-cache search k1x-mpp
sudo apt install k1x-mpp-dev
```

### 2.3 xquant (模型量化工具)

```bash
# 基于 PPQ v0.6.6+
pip install xquant
# 或从源码:
git clone https://gitee.com/bianbu/xquant.git
cd xquant && pip install -e .

# 验证:
python3 -c "import xquant; print('ok')"
```

### 2.4 ST-GCN 模型量化

```bash
# Step 1: 准备校准数据 (100-500 张典型场景图片的关键点序列)
mkdir -p calibration_data/stgcn

# Step 2: 写量化配置
cat > stgcn_quant_config.json << 'EOF'
{
    "model": "models/Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx",
    "output": "models/Action Prediction/Skeleton-based Action Prediction/stgcn.q.onnx",
    "calibration": "calibration_data/stgcn",
    "precision": "int8",
    "strategy": "percentile",
    "num_calibration_samples": 200
}
EOF

# Step 3: 执行量化
python3 -m xquant -c stgcn_quant_config.json
```

---

## 第三步: 逐步启用硬件 (按投入产出比排序)

### Phase A: 零代码改动（立即生效）

#### A.1 WiFi 预连接（解决 nmcli 权限 + JPEG 丢包）
```bash
sudo nmcli dev wifi connect "ESP32-Camera-AP" password "12345678"
# 然后正常运行程序（不需要 sudo 运行程序）
```

#### A.2 I2C 权限（启用板载 IMU）
```bash
sudo chmod 666 /dev/i2c-4
# 或永久: sudo vi /etc/udev/rules.d/99-imu.rules
# SUBSYSTEM=="i2c-dev", KERNEL=="i2c-4", MODE="0666"
```

#### A.3 检查摄像头
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null
v4l2-ctl -d /dev/video1 --list-formats-ext 2>/dev/null
# 如果 video1 可用: ./build/lingqi_tantong --realtime --camera /dev/video1 --coap
```

### Phase B: 安装 SDK 后重新编译（cmake 参数）

```bash
cd build

# 完整重建 (启用所有硬件加速):
cmake .. \
  -DK1_JPU_DIR=/path/to/k1x-jpu \
  -DK1_MPP_DIR=/path/to/k1x-mpp \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# 验证:
./build/lingqi_tantong --realtime --coap --save-video
# 查看日志中是否有:
#   "K1-JPU: Hardware JPEG decoder available via k1x-jpu SDK"
```

### Phase C: 模型量化（如果 xquant 可用）

```bash
# 量化 ST-GCN (FP32 → INT8):
python3 -m xquant -c stgcn_quant_config.json

# 更新 config:
# action.model_path: "models/.../stgcn.q.onnx"  (改为 .q.onnx)

# 重新编译运行 — ST-GCN 将自动使用 SpacemiT EP 硬件加速
```

### Phase D: 代码级优化（已完成部分，待验证）

已实施 (需要板上编译验证):
- ✅ RVV DFL softmax 向量化 (yolo_postprocess.c)
- ✅ RVV 图像缩放向量化 (utils.c)
- ✅ ST-GCN 异步线程 (system_controller.c — 新增 CPU 2 线程)
- ✅ JPU 硬件解码路径 (k1_jpu.c — SDK 安装后自动启用)

---

## 第四步: 预期性能提升路径

```
当前基线: 2.2 FPS (640×640, CPU EP ST-GCN, libjpeg-turbo 软解)

Phase A (WiFi + I2C + 摄像头):   2.2 → 2.5 FPS  (稳定性提升, 减少丢帧)
Phase B (JPU 硬解 + 编译优化):    2.5 → 3.2 FPS  (JPEG 解码 -15ms/帧)
Phase C (ST-GCN 量化):            3.2 → 5.0 FPS  (ST-GCN 400ms→50ms)
Phase D (RVV + 多核异步):         5.0 → 6.5 FPS  (DFL/后处理加速)

目标: 6+ FPS (满足实时人体检测 + 行为识别需求)
```

---

## 参考链接

| 资源 | 地址 |
|------|------|
| SpacemiT K1 产品页 | https://www.spacemit.com/product/k1/ |
| Bianbu OS | https://bianbu.org/ |
| 模型量化文档 | https://bianbu.spacemit.com/en/brdk/Advanced_development/7.1_Model_Quantization/ |
| 预量化模型仓库 | https://gitee.com/bianbu/spacemit-demo |
| k1x-jpu SDK | https://gitee.com/bianbu-linux/k1x-jpu (待验证) |
| k1x-mpp SDK | https://gitee.com/bianbu-linux/k1x-mpp (待验证) |
| llama.cpp K1 后端 | https://github.com/ggml-org/llama.cpp/pull/22863 |
| ONNX Runtime 嵌入式 | https://github.com/microsoft/onnxruntime/discussions/22763 |
| RISC-V Vector 规范 | https://github.com/riscv/riscv-v-spec |
