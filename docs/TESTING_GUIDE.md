# 模型识别功能测试验证指南

## 一、测试环境准备

### 1.1 硬件环境
- **开发板**：MUSE Pi Pro（SpacemiT K1 SoC）
- **CPU**：8× X60 RISC-V 64 @ 1.6GHz
- **内存**：8GB/16GB LPDDR4X
- **TCM**：512KB（Cluster0 AI核心）
- **摄像头**：USB摄像头或MIPI CSI摄像头（/dev/video0）

### 1.2 软件环境
- **操作系统**：Bianbu 2.x（K1 LTS）
- **ONNX Runtime**：SpacemiT ORT 2.0.2+ (apt install onnxruntime)
- **依赖库**：libturbojpeg, OpenCV (可选)

### 1.3 部署步骤

```bash
# 1. 拷贝项目到开发板
scp -r lingqi_tantong/ user@k1-devboard:/home/user/

# 2. 创建构建目录
cd /home/user/lingqi_tantong
mkdir build && cd build

# 3. CMake 配置（指定 ONNX Runtime 路径）
cmake .. \
  -DONNX_RUNTIME_DIR=/usr \
  -DUSE_ONNX_RUNTIME=ON \
  -DCMAKE_BUILD_TYPE=Release

# 4. 编译
make -j$(nproc)

# 5. 验证模型文件存在
ls -la ../models/Human\ Recognition/yolo11n.q.onnx
ls -la ../models/Action\ Prediction/Skeleton\ Recognition/yolov8n-pose.q.onnx
ls -la ../models/Action\ Prediction/Skeleton-based\ Action\ Prediction/stgcn.fp32.onnx
ls -la ../models/Face\ Recognition/yolov5n-face_cut.q.onnx
ls -la ../models/Face\ Recognition/arcface_mobilefacenet_cut.q.onnx
```

---

## 二、功能完整性测试

### 2.1 单元级测试：逐模块验证

每个模块独立测试，确保 create/process/destroy 生命周期完整。

#### 测试 2.1.1：YOLO11 行人检测

```bash
# 使用 benchmark 模式逐帧测试
./lingqi_tantong --mode benchmark --source image --image test_images/person.jpg
```

**检查项**：
- [ ] 模型加载日志显示 "YOLO11 model loaded" 且文件大小正确
- [ ] 日志显示 EP 类型（SpacemiT 或 CPU）
- [ ] xquant-split 格式检测成功（"detected xquant-split format with N stride groups"）
- [ ] DFL 诊断日志输出（max/min/mean peakiness）
- [ ] 检测到的人数 ≥ 0（有人场景 > 0，无人场景 = 0）
- [ ] Bounding box 坐标在图像范围内（x: 0~width, y: 0~height）
- [ ] 无内存泄漏（valgrind 或长期运行观察）

#### 测试 2.1.2：YOLOv8-Pose 姿态估计

```bash
./lingqi_tantong --mode benchmark --source image --test_image test_images/person_standing.jpg
```

**检查项**：
- [ ] 第一帧日志输出所有输出张量形状
- [ ] xquant-split 格式检测成功并缓存
- [ ] 后续帧不再重复检测格式（"cached" 日志仅出现一次）
- [ ] OKS-based NMS 正确去重（同一人不会出现多个姿态）
- [ ] 17个 COCO 关键点坐标在图像范围内或有标记为无效(-1)
- [ ] 关键点置信度正确（0.0~1.0之间）

#### 测试 2.1.3：YOLOv5-Face 人脸检测

```bash
./lingqi_tantong --mode benchmark --source image --test_image test_images/face.jpg
```

**检查项**：
- [ ] 输出格式自动检测（4D 或 5D）
- [ ] 5D xquant-split 格式正确进行 anchor decode
- [ ] 5个面部关键点坐标在图像范围内
- [ ] NMS 后无重叠人脸框

#### 测试 2.1.4：ArcFace 人脸识别

```bash
# 先注册人脸
./lingqi_tantong --mode register --identity "test_user" --image test_images/face_register.jpg

# 再识别人脸
./lingqi_tantong --mode recognize --image test_images/face_test.jpg
```

**检查项**：
- [ ] 特征向量维度 = 128
- [ ] 特征向量已 L2 归一化
- [ ] 同一人相似度 > 0.55
- [ ] 不同人相似度 < 0.55
- [ ] "unknown" 标记正确应用于未注册人脸

#### 测试 2.1.5：ST-GCN 动作识别

```bash
./lingqi_tantong --mode benchmark --source video --video_path test_videos/walking.mp4
```

**检查项**：
- [ ] 模型加载日志显示正确的类别数
- [ ] 输入名称动态检测（pts/mot）
- [ ] 30帧滑动窗口正确填充
- [ ] 动作分类结果置信度排序正确
- [ ] 空缓冲区时返回无动作（num_actions=0）

### 2.2 集成测试：多模块协同

#### 测试 2.2.1：完整 Pipeline 测试

```bash
./lingqi_tantong --mode realtime --source camera
```

**检查项**：
- [ ] 所有模型成功加载（检测 + 姿态 + 人脸 + 动作）
- [ ] SpacemiT EP 会话数 ≤ 4
- [ ] 帧率 ≥ 目标 FPS（config: target_fps=15）
- [ ] 检测 → 跟踪 → 姿态 → 动作流程完整
- [ ] 可视化输出包含所有标注（bbox、骨架、人脸框、动作标签）

### 2.3 双路径兼容性测试

分别测试 EP 加速和纯 CPU 推理两种路径：

```bash
# 测试1: EP 加速路径（默认）
./lingqi_tantong --mode benchmark --config configs/default.yaml

# 测试2: CPU-only 路径（修改配置 use_spacemit_ep: false）
./lingqi_tantong --mode benchmark --config configs/cpu_only.yaml
```

**检查项**：
- [ ] EP 路径日志显示 "ep=SpacemiT"
- [ ] CPU 路径日志显示 "ep=CPU"
- [ ] 两种路径推理结果一致（相同输入 → 相同检测）
- [ ] CPU 路径无崩溃

### 2.4 Heuristic Fallback 测试

```bash
# 删除或重命名模型文件模拟 ORT 不可用
mv models/Human\ Recognition/yolo11n.q.onnx models/Human\ Recognition/yolo11n.q.onnx.bak
./lingqi_tantong --mode benchmark
```

**检查项**：
- [ ] 优雅降级：日志提示模型不可用而非崩溃
- [ ] 系统状态中 error_count 递增
- [ ] 恢复模型后自动恢复正常

---

## 三、性能基准测试

### 3.1 单模型推理延迟

使用 `onnxruntime_perf_test` 进行独立性能测试：

```bash
# YOLO11n INT8 量化模型
onnxruntime_perf_test -e spacemit -r 1000 models/Human\ Recognition/yolo11n.q.onnx

# YOLOv8n-Pose INT8 量化模型
onnxruntime_perf_test -e spacemit -r 1000 models/Action\ Prediction/Skeleton\ Recognition/yolov8n-pose.q.onnx

# YOLOv5-Face INT8 量化模型
onnxruntime_perf_test -e spacemit -r 1000 models/Face\ Recognition/yolov5n-face_cut.q.onnx

# ArcFace INT8 量化模型
onnxruntime_perf_test -e spacemit -r 1000 models/Face\ Recognition/arcface_mobilefacenet_cut.q.onnx

# ST-GCN FP32 模型（CPU EP）
onnxruntime_perf_test -e cpu -r 1000 models/Action\ Prediction/Skeleton-based\ Action\ Prediction/stgcn.fp32.onnx
```

**目标指标**：

| 模型 | EP | 目标延迟（P50） | 目标延迟（P99） |
|---|---|---|---|
| YOLO11n | SpacemiT | < 30ms | < 50ms |
| YOLOv8n-Pose | SpacemiT | < 50ms | < 80ms |
| YOLOv5-Face | SpacemiT | < 20ms | < 35ms |
| ArcFace | SpacemiT | < 5ms | < 10ms |
| ST-GCN | CPU | < 15ms | < 25ms |

### 3.2 端到端 Pipeline 吞吐

```bash
# 运行 benchmark 模式，统计 1000 帧
./lingqi_tantong --mode benchmark --source video --video_path test_videos/scene.mp4 --max_frames 1000
```

**目标指标**：

| 指标 | 目标值 |
|---|---|
| 整体 FPS（全部模型启用） | ≥ 10 FPS |
| 整体 FPS（仅检测+姿态） | ≥ 15 FPS |
| 检测阶段延迟（YOLO11 + YOLOv8-Pose） | < 100ms |
| 后处理延迟（NMS + 坐标映射） | < 5ms |
| 跟踪延迟（ByteTrack） | < 3ms |
| 内存使用（RSS） | < 500MB |
| CPU 使用率 | < 80% |

### 3.3 内存压力测试

```bash
# 长时间运行（1小时）
./lingqi_tantong --mode realtime --source camera &
PID=$!
# 每分钟记录内存
for i in $(seq 1 60); do
  ps -o rss= -p $PID >> memory_log.txt
  sleep 60
done
```

**检查项**：
- [ ] 内存稳定，无持续增长（无内存泄漏）
- [ ] 峰值 RSS < 500MB
- [ ] 1小时后内存与启动时差异 < 50MB

### 3.4 输入张量池化性能对比

在 `ort_ctx_prepare_input` 中添加性能计数器验证池化效果：

```c
// 临时测试代码（验证后移除）
static int alloc_count = 0;
// 在 realloc 分支增加 alloc_count++
log_info("Input tensor realloc count: %d after %d frames", alloc_count, frame_count);
```

**目标**：`realloc` 次数 ≤ 1（仅在首帧或输入尺寸变化时）

---

## 四、边界条件测试

### 4.1 极端输入测试

| 测试场景 | 输入 | 预期行为 |
|---|---|---|
| 空图像 | width=0, height=0 | 返回 0，不崩溃 |
| 极小图像 | 16×16 | 返回 0 或正常处理 |
| 极大图像 | 4096×2160 | 正常处理（内存充足时）|
| 极端宽高比 | 1920×100 | yolo_preprocess 触发裁剪分支 |
| 纯黑图像 | 全零像素 | 返回 0（无检测）|
| 纯白图像 | 全255像素 | 返回 0（无检测）|
| 重复图像 | 同一个 FrameData | 结果一致（无随机性）|

### 4.2 资源边界测试

| 测试场景 | 操作 | 预期行为 |
|---|---|---|
| 多模型加载 | 6个模型同时加载 | EP 会话 ≤ 4，其余 CPU |
| TCM 不足 | /dev/tcm 不可用 | 自动回退 CPU EP |
| 磁盘空间不足 | 输出路径无空间 | 日志错误，不崩溃 |
| 模型文件损坏 | 截断的 .onnx | ort_validate_onnx_file 拒绝 |
| 线程竞争 | 多线程同时推理 | 无数据竞争（mutex 保护）|

### 4.3 长时间稳定性测试

```bash
# 24小时连续运行
./lingqi_tantong --mode realtime --source camera &
# 每小时记录：FPS、内存、检测数、错误数
```

**检查项**：
- [ ] 无崩溃
- [ ] FPS 稳定（波动 < 20%）
- [ ] 无内存泄漏
- [ ] 无文件描述符泄漏（lsof 检查）
- [ ] 检测数稳定（同场景）

---

## 五、兼容性测试

### 5.1 模型格式兼容性

| 模型变体 | 测试方法 | 预期结果 |
|---|---|---|
| FP32 标准 ONNX | 替换 .q.onnx 为 .onnx | CPU EP 推理，正确检测 |
| INT8 xquant-split | 默认配置 | SpacemiT EP，DFL 解码 |
| INT8 xquant 3-out | 部分量化模型 | 自动检测为标准格式 |
| 不同输入尺寸 | 320×320 vs 640×640 | 自动检测实际输入维度 |

### 5.2 配置兼容性

```bash
# 切换不同配置
./lingqi_tantong --config configs/default.yaml      # 全功能
./lingqi_tantong --config configs/detection_only.yaml # 仅检测
./lingqi_tantong --config configs/cpu_only.yaml      # CPU 模式
./lingqi_tantong --config configs/low_power.yaml     # 低功耗模式
```

---

## 六、回归测试脚本

### 6.1 自动化回归测试

```bash
#!/bin/bash
# regression_test.sh — 模型识别功能回归测试

PASS=0
FAIL=0
BIN="./build/lingqi_tantong"

echo "=== 模型识别功能回归测试 ==="

# Test 1: 模型加载
echo "[1/8] 模型加载测试..."
$BIN --mode benchmark --source image --test_image test_images/blank.jpg --max_frames 1 > /tmp/test1.log 2>&1
if grep -q "YOLO11 model loaded" /tmp/test1.log && \
   grep -q "YOLOv8-Pose model loaded" /tmp/test1.log; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 2: 行人检测
echo "[2/8] 行人检测测试..."
$BIN --mode benchmark --source image --test_image test_images/person.jpg --max_frames 1 > /tmp/test2.log 2>&1
if grep -q "detections after NMS" /tmp/test2.log; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 3: 姿态估计
echo "[3/8] 姿态估计测试..."
$BIN --mode benchmark --source image --test_image test_images/person_standing.jpg --max_frames 1 > /tmp/test3.log 2>&1
if grep -q "poses after" /tmp/test3.log; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 4: 空图像边界
echo "[4/8] 空图像边界测试..."
$BIN --mode benchmark --source image --test_image test_images/blank.jpg --max_frames 1 > /tmp/test4.log 2>&1
if ! grep -q "SIGSEGV\|segfault\|abort" /tmp/test4.log; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 5: CPU-only 回退
echo "[5/8] CPU EP 回退测试..."
$BIN --mode benchmark --source image --test_image test_images/person.jpg \
  --override system.use_spacemit_ep=false --max_frames 1 > /tmp/test5.log 2>&1
if grep -q "ep=CPU" /tmp/test5.log && grep -q "detections after NMS" /tmp/test5.log; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 6: 内存稳定性（5帧）
echo "[6/8] 内存稳定性测试（5帧）..."
$BIN --mode benchmark --source image --test_image test_images/person.jpg --max_frames 5 > /tmp/test6.log 2>&1
if [ $? -eq 0 ]; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 7: 格式检测缓存
echo "[7/8] 格式检测缓存测试..."
$BIN --mode benchmark --source image --test_image test_images/person.jpg --max_frames 3 > /tmp/test7.log 2>&1
CACHED_COUNT=$(grep -c "cached" /tmp/test7.log)
if [ "$CACHED_COUNT" -ge 1 ]; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

# Test 8: 配置文件加载
echo "[8/8] 配置文件加载测试..."
$BIN --mode benchmark --source image --test_image test_images/blank.jpg \
  --config configs/default.yaml --max_frames 1 > /tmp/test8.log 2>&1
if [ $? -eq 0 ]; then
  echo "  PASS"
  ((PASS++))
else
  echo "  FAIL"
  ((FAIL++))
fi

echo "=== 结果: $PASS 通过, $FAIL 失败 ==="
```

### 6.2 手动验证检查清单

| 检查项 | 状态 |
|---|---|
| [ ] 所有5个模型成功加载 | |
| [ ] SpacemiT EP 正确注册给量化模型 | |
| [ ] FP32 模型（ST-GCN）使用 CPU EP | |
| [ ] xquant-split 格式自动检测正确 | |
| [ ] DFL 解码输出合理的 peakiness 值 | |
| [ ] NMS 正确去重（无重叠框） | |
| [ ] 坐标映射正确（框不偏移） | |
| [ ] OKS-based NMS 正确去重姿态 | |
| [ ] 人脸关键点5点正确 | |
| [ ] ArcFace 特征 L2 归一化 | |
| [ ] ST-GCN 滑动窗口正确填充 | |
| [ ] 内存无泄漏 | |
| [ ] 无崩溃/段错误 | |
| [ ] 长时间运行稳定 | |

---

## 七、测试数据准备

### 7.1 测试素材清单

| 素材 | 用途 | 来源 |
|---|---|---|
| person.jpg | 单人站立 | 自拍/COCO val |
| person_standing.jpg | 姿态估计（T-pose） | 自拍/COCO val |
| multi_person.jpg | 多人检测 | COCO val |
| face.jpg | 人脸检测 | 自拍/WIDER Face |
| face_register.jpg | 人脸注册 | 自拍 |
| face_test.jpg | 人脸识别（同一人） | 自拍 |
| face_imposter.jpg | 人脸识别（不同人） | 自拍 |
| blank.jpg | 空图像边界测试 | 生成 |
| walking.mp4 | 动作识别 | NTU RGB+D / 自拍 |
| scene.mp4 | 完整 pipeline | 自拍/公开数据集 |

### 7.2 生成测试图像

```bash
# 生成纯黑图像
python3 -c "from PIL import Image; Image.new('RGB', (640,480), (0,0,0)).save('test_images/blank.jpg')"

# 生成纯白图像
python3 -c "from PIL import Image; Image.new('RGB', (640,480), (255,255,255)).save('test_images/white.jpg')"

# 生成极端宽高比图像
python3 -c "from PIL import Image; Image.new('RGB', (1920,100), (128,128,128)).save('test_images/wide.jpg')"
```