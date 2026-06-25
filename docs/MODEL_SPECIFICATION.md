# 模型规格文档 — K1 最佳性能配置

> 基于 SpacemiT 官方 BRDK Model Zoo (`archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/`)
> 对比 CV 官方 Demo 模型与项目当前模型，确定 K1 最佳配置

## 模型总览

| # | 用途 | 模型 | 来源 | 大小 | 精度 | 输入 | TCM占用 |
|---|------|------|------|------|------|------|---------|
| 1 | 人体检测+姿态 | yolov8n-pose.q.onnx | BRDK CV Demo (yolov8-pose) | 3.6MB | INT8 | 640×640 | ~170KB |
| 2 | 人脸检测 | yolov5n-face_320_cut.q.onnx | BRDK CV Demo (yolov5-face) | 2.0MB | INT8 | 320×320 | ~170KB |
| 3 | 人脸识别 | arcface_mobilefacenet_cut.q.onnx | BRDK CV Demo (arcface) | 1.2MB | INT8 | 112×112 | ~170KB |
| 4 | 行为识别 | stgcn.fp32.onnx | 第三方 (ST-GCN) | 24MB | FP32 | 30帧×14关节点 | 0 (CPU EP) |
| 5 | 目标检测(备用) | yolov11n_320x320.q.onnx | BRDK CV Demo (yolov11) | 3.0MB | INT8 | 320×320 | ~170KB |

**TCM总计**: ~510KB / 512KB (3个EP模型, 满载)  
**模型总大小**: ~30MB (4个模型)

---

## 模型详情

### 1. YOLOv8n-pose — 人体检测+姿态估计 (主模型)

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv8n-pose (Ultralytics) |
| 参数量 | 3.3M |
| 计算量 | 9.2 GFLOPs (FP32) |
| 精度 | AP 48.0 (COCO-Pose) |
| 输入 | [1, 3, 640, 640] NCHW, RGB, normalized [0,1] |
| 输出 | reg(64ch) + cls(1ch) + kpt(51ch) × 3 FPN strides, xquant-split 格式 |
| 关节点 | 17 (COCO: nose~ankles) |
| 量化方式 | xquant INT8, cls head 被量化破坏, DFL峰值度作为置信度 |
| 置信度阈值 | 0.06 (DFL-based, 因cls head不可用) |
| IoU 阈值 | 0.40 (OKS-NMS) |

**来源**: 官方 BRDK CV Demo (`CV/yolov8-pose/model/yolov8n-pose.q.onnx`)

**CV目录中的位置**: `CV/yolov8-pose/model/yolov8n-pose.q.onnx`

**官方下载** (BRDK):
```
https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/YOLOv8-pose/yolov8n-pose.q.onnx
```

---

### 2. YOLOv5n-face — 人脸检测 (辅助模型)

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv5n-face (deepcam-cn/yolov5-face) |
| 参数量 | ~1.8M |
| 输入 | [1, 3, 320, 320] NCHW, RGB, normalized [0,1] |
| 输出 | 3 FPN strides × 16ch (4bbox+1obj+1cls+10landmark), xquant 5D格式 |
| 地标 | 5 (左眼, 右眼, 鼻子, 左嘴角, 右嘴角) |
| 锚点步长 | 8 (P3/小脸), 16 (P4/中脸), 32 (P5/大脸) |
| 置信度阈值 | 0.50 |

**来源**: 官方 BRDK CV Demo (`CV/yolov5-face/model/yolov5n-face_320_cut.q.onnx`)，项目已直接使用此版本。

**官方下载**:
```
https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/YOLOv5-face/yolov5n-face_320_cut.q.onnx
```

---

### 3. ArcFace MobileFaceNet — 人脸识别 (辅助模型)

| 属性 | 值 |
|------|-----|
| 架构 | ArcFace (InsightFace) + MobileFaceNet backbone |
| 输入 | [1, 3, 112, 112] NCHW, RGB, normalized (x-127.5)/127.5 |
| 输出 | [1, 128] 特征向量 (L2归一化后) |
| 相似度 | 余弦相似度 (点积), 阈值 0.55 |
| 量化方式 | xquant INT8 |

**来源**: 官方 BRDK CV Demo (`CV/arcface/model/arcface_mobilefacenet_cut.q.onnx`)，项目已直接使用此版本。

**官方下载**:
```
https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/ArcFace/arcface_mobilefacenet_cut.q.onnx
```

---

### 4. ST-GCN — 行为识别 (独立模型)

| 属性 | 值 |
|------|-----|
| 架构 | ST-GCN (Spatial Temporal Graph Convolutional Network) |
| 输入 | [1, 30, 14, 1, 1] 或 [1, 3, 30, 14, 1] (变体依实现而定) |
| 输出 | [1, N] — N 类动作分类 (当前: 7) |
| 参数 | ~3.1M |
| 精度 | FP32 (未量化) |
| 执行提供者 | CPU EP (~400ms/推理) |
| 推理间隔 | 每 50 帧 (~12.5s @ 4FPS) |
| TCM | 0 (不在 SpacemiT EP 上) |

**为什么没有量化**: 此模型不在 CV 参考库中。需手动用 xquant 对骨骼序列校准数据做量化才能转为 INT8。

**不在 CV 目录中**: ST-GCN 是第三方模型 (NTU RGB+D 数据集), 不在 SpacemiT BRDK 模型库中。没有官方 K1 版本。

**官方下载**: 无官方 BRDK 源。当前模型来源见项目获取文档。

---

## CV Demo 对比总结

| 功能 | 项目当前模型 | CV Demo 模型 | 状态 |
|------|-------------|-------------|------|
| 人体检测+姿态 | yolov8n-pose.q.onnx (3.6MB) | yolov8n-pose.q.onnx (3.6MB) | ✅ 已对齐 |
| 纯目标检测(备用) | yolov11n_320x320.q.onnx (3.0MB) | yolov11n_320x320.q.onnx (3.0MB) | ✅ 已对齐 |
| 人脸检测 | yolov5n-face_320_cut.q.onnx (2.0MB) | yolov5n-face_320_cut.q.onnx (2.0MB) | ✅ 已对齐 |
| 人脸识别 | arcface_mobilefacenet_cut.q.onnx (1.2MB) | arcface_mobilefacenet_cut.q.onnx (1.2MB) | ✅ 已对齐 |
| 行为识别 | stgcn.fp32.onnx (24MB) | 无 | 保持 (无替代) |

---

## 当前操作

所有模型已替换为官方 BRDK CV Demo 版本:

1. ✅ **yolov8n-pose.q.onnx**: 来自 `CV/yolov8-pose/model/` (BRDK 官方)
2. ✅ **yolov11n_320x320.q.onnx**: 来自 `CV/yolov11/model/` (BRDK 官方，备用检测器)
3. ✅ **yolov5n-face_320_cut.q.onnx**: 来自 `CV/yolov5-face/model/` (BRDK 官方)
4. ✅ **arcface_mobilefacenet_cut.q.onnx**: 来自 `CV/arcface/model/` (BRDK 官方)
5. ✅ **stgcn.fp32.onnx**: 保持不变 (无 BRDK 替代模型)
6. 🔮 **stgcn 量化**: 未来用 xquant + 骨骼序列校准数据做 INT8 量化 → 可减少 350ms/推理
