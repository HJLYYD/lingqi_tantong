# 优化方案设计文档 — 灵契探瞳目标追踪系统全局优化

> 文档版本: v2.0  
> 日期: 2026-06-08  
> 设计目标: 系统性解决追踪稳定性、多人识别、遮挡处理三大功能缺陷

---

## 一、设计目标与约束

### 1.1 性能目标

| 指标 | 当前估计 | 目标 | 验证方法 |
|------|---------|------|---------|
| 30分钟ID保持率 | ~60-70% | ≥99% | 连续追踪测试，K1板上验证 |
| 最大同时追踪人数 | ~2人 | ≥10人 | 多人场景测试 |
| 多人追踪准确率 | ~60% | ≥95% | 10人场景准确率统计 |
| 30%-70%遮挡识别率 | ~0-40% | ≥85% | 遮挡场景测试 |
| 上半身/侧半身识别率 | ~15-30% | ≥85% | 部分身体测试 |

### 1.2 平台约束

- 目标平台: SpacemiT K1 (Muse Pi Pro), RISC-V 64-bit
- 推理后端: ONNX Runtime + SpacemiT EP (INT8量化模型)
- 内存限制: ≤1.5GB (K1可用)
- 计算限制: 4核Cluster0 AI + 4核Cluster1 I/O @ 1.6GHz
- **关键约束: 不新增CNN模型** — 系统已运行5个ONNX模型(EP槽位已满)

---

## 二、优化架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZED TRACKING PIPELINE (v2.0)                │
│                                                                      │
│  Frame → YOLOv8-Pose (PRIMARY)     ─┐                                │
│  Frame → YOLO11n (SEC, every 5f)   ─┤                                │
│                                      ├→ Filter w/ Partial-Body ──┐   │
│                                      │  • Full-body (Tier 1)     │   │
│  Frame → KeypointValidator          ─┤  • Upper-body (Tier 2)    │   │
│  Frame → SpatialEngine              ─┘  • Side-body (Tier 3)     │   │
│                                                                  │   │
│  ┌───────────────────────────────────────────────────────────────┘   │
│  ▼                                                                   │
│  Hungarian Algorithm (Global Optimal Assignment)                     │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Cost(i,j) = (1-α)·IoU_dist(i,j) + α·App_dist(i,j)          │    │
│  │                                                               │    │
│  │ Appearance Feature: 12-dim Pose-Keypoint Descriptor          │    │
│  │   • torso ratio, limb ratios, shoulder symmetry, etc.        │    │
│  │   • NO extra model — computed from existing YOLOv8-Pose     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ▼                                                                   │
│  Cascade Matching (Freshest-First Priority)                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ for age in 1..max_age:                                        │    │
│  │   match tracks with age=N against remaining detections        │    │
│  │   (young tracks get priority → prevents stale-track theft)    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  ▼                                                                   │
│  IoU Fallback + Occlusion Recovery (Low-IoU Threshold)               │
│  ▼                                                                   │
│  Re-Identification Pool (Deleted-Track Memory)                       │
│  ▼                                                                   │
│  Track Lifecycle:                                                    │
│    • Normal: max_lost=60 frames before deletion                      │
│    • Occluded: max_occluded=90 frames (extended lifetime)            │
│    • Partial-body: marked, tracked with lower IoU thresholds         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心优化模块设计

### 3.1 匈牙利算法 — 全局最优分配

**替换内容**: `tracking_manager.c` 中的贪婪IoU匹配循环

**算法选择**: Kuhn-Munkres (匈牙利算法) O(n³)

**实现细节**:

```c
// 代价矩阵构建
// C[i][j] = w_iou * (1 - IoU(pred_i, det_j)) + w_app * CosineDist(feat_i, feat_j)
//
// 门控: 当IoU和外观距离都超过阈值时，代价设为无穷大
// C[i][j] = 1e8  if IoU < 0.15 AND AppDist > 0.50

static float hungarian_solve(const float* cost_matrix, int rows, int cols,
                              int* assignment, float* final_cost);
```

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 矩阵最大维度 | 256×256 | 实际场景通常≤20×20 |
| 无穷代价 | 1e8 | 门控出界匹配 |
| 算法复杂度 | O(n³) | n=20时约8000次运算，<0.1ms |

### 3.2 姿态关键点外观特征 — 零额外模型

**设计原则**: 利用现有的YOLOv8-Pose输出，不增加任何CNN模型。

**12维描述符** (从COCO-17关键点提取):

| 维度 | 特征 | 计算方式 |
|------|------|---------|
| 0 | 躯干比例 | (髋-肩)_Y / 肩宽 |
| 1 | 肩宽/边界框宽 | 肩宽_像素 / bbox_w |
| 2 | 左臂比例 | 上臂(肩→肘) / 前臂(肘→腕) |
| 3 | 右臂比例 | 同上，右侧 |
| 4 | 左腿比例 | 大腿(髋→膝) / 小腿(膝→踝) |
| 5 | 右腿比例 | 同上，右侧 |
| 6 | 头肩比例 | (肩中点-鼻)_Y / bbox_h |
| 7 | 肩对称性(Y) | |左肩_Y - 右肩_Y| / bbox_h |
| 8 | 髋对称性(Y) | |左髋_Y - 右髋_Y| / bbox_h |
| 9 | 归一化身高 | (最低关键点-鼻)_Y / bbox_h |
| 10 | 质心X偏移 | (加权质心_X - bbox中心_X) / bbox_w |
| 11 | 质心Y偏移 | (加权质心_Y - bbox中心_Y) / bbox_h |

**优势**:
- 计算成本极低（~50次浮点运算）
- 对光照、服装颜色不变
- 对视角变化鲁棒（比例关系）
- 适合嵌入式部署

**外观特征库**:
- 每个轨迹保留最多50个历史特征向量
- 最新特征用于快速匹配
- 特征距离 = 0.5 × (1 - cosine_similarity)，范围[0,1]

### 3.3 级联匹配 — 新鲜轨迹优先

**设计原理** (参考 DeepSORT):

```
传统匹配：所有轨迹 vs 所有检测 → 陈旧轨迹"抢走"检测
级联匹配：按time_since_update分组，新鲜轨迹先匹配
```

**算法流程**:
```
for age in 1, 2, 3, ..., max_age:
    1. 收集 age == N 的未匹配轨迹
    2. 收集未匹配的检测
    3. 构建代价矩阵（IoU + 外观）
    4. 匈牙利算法求解
    5. 更新匹配结果
    6. 年轻轨迹(age≤3): 使用标准IoU阈值
    7. 陈旧轨迹(age>3): 使用低IoU阈值
```

**参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| cascade_max_age | 30 | 最多考虑最近30帧的轨迹 |
| cascade_min_hits | 3 | 至少命中3次才进入级联 |
| age≤3 IoU阈值 | 0.30 | 新鲜轨迹使用标准阈值 |
| age>3 IoU阈值 | 0.15 | 陈旧轨迹使用宽松阈值 |

### 3.4 部分身体检测 — 三级验证

**设计原理**: 当全身关键点验证失败时，依次尝试上半身和侧身验证。

**三级验证流程**:
```
Tier 1: 全身验证 (keypoint_validator_quick_check)
    ↓ 失败
Tier 2: 上半身验证 (keypoint_validator_upper_body_check)
    • 检查关键点0-10 (鼻→双腕)
    • 要求: ≥4个有效关键点 + 至少一侧肩膀可见
    • 检查: 头在肩上方、肩对称
    ↓ 失败
Tier 3: 侧身验证 (keypoint_validator_side_body_check)
    • 检查左侧或右侧关键点链
    • 要求: ≥3个有效关键点在同一侧
    • 检查: 肩→髋→膝垂直顺序
    ↓ 失败
REJECT: 非人体对象
```

**部分身体检测标记**:
- `detection.is_partial_body = true`
- `detection.num_visible_keypoints = N` (实际可见关键点数)
- 置信度提升 `+0.05` (补偿DFL信号损失)

### 3.5 重识别池 — 删除轨迹记忆

**设计原理**: 轨迹被删除时，保留其外观特征一段时间。新检测与池中特征匹配成功后，复用旧ID。

```
轨迹删除 → 保存 (appearance_feature, track_id) 到 reid_pool
新检测创建 → 与 reid_pool 中所有特征计算余弦距离
             → 最近距离 < appearance_max_dist → 复用旧track_id
             → 否则 → 分配新track_id

reid_pool 老化: 每 reid_pool_max_age/2 帧丢弃最旧一半条目
```

### 3.6 级联状态机修复 — 多人检测

**原问题**: TRACKING模式完全跳过YOLO11n

**修复方案**:
```c
// TRACKING模式中，每5帧运行一次YOLO11n
#define CASCADE_SECONDARY_INTERVAL 5

bool run_secondary = run_full_res ||
    (cascade_state == TRACKING && frame_counter % 5 == 0);

// 多人检测触发器: 检测数 > 轨迹数 + 1
if (num_detections > num_tracks + 1 && num_detections >= 2) {
    cascade_state = VALIDATING;  // 强制全分辨率帧
}
```

### 3.7 遮挡感知轨迹管理

| 状态 | max_lost | IoU阈值 | 说明 |
|------|----------|---------|------|
| 正常轨迹 | 60帧 | 0.30 | 标准匹配 |
| 部分身体轨迹 | 90帧 | 0.15 | 延长寿命+宽松阈值 |
| 丢失中的轨迹 | — | 0.15 | 外观特征辅助匹配 |

---

## 四、文件修改清单

### 4.1 核心修改

| 文件 | 修改类型 | 行数变化 | 关键变更 |
|------|---------|---------|---------|
| `include/tracking_manager.h` | 重写 | +180行 | 匈牙利算法结构、外观特征类型、级联配置、遮挡配置 |
| `src/tracking_manager.c` | 重写 | +600行 | 匈牙利求解器、外观特征提取、级联匹配、遮挡处理、重识别池 |
| `include/core_types.h` | 修改 | +3行 | Detection增加track_id_hint、is_partial_body、num_visible_keypoints |
| `include/keypoint_validator.h` | 修改 | +20行 | 新增upper_body_check、side_body_check接口 |
| `src/keypoint_validator.c` | 修改 | +120行 | 实现部分身体验证函数 |
| `src/inference_pipeline.c` | 修改 | +40行 | 级联状态机修复、部分身体过滤逻辑 |
| `src/system_controller.c` | 修改 | +30行 | 新增配置应用、pose_associate调用 |
| `configs/default.yaml` | 修改 | +15行 | 新增级联、遮挡、重识别配置参数 |

### 4.2 新增功能汇总

| 功能 | 模块 | 技术 |
|------|------|------|
| 匈牙利全局最优匹配 | tracking_manager | Kuhn-Munkres O(n³) |
| 姿态关键点外观特征 | tracking_manager | 12维描述符 + 余弦距离 |
| 级联匹配 | tracking_manager | DeepSORT风格年龄分组 |
| 部分身体检测 | inference_pipeline + keypoint_validator | 三级验证(全身/上半身/侧身) |
| 重识别池 | tracking_manager | 删除轨迹特征保留+复用 |
| 遮挡感知轨迹管理 | tracking_manager | 延长寿命+宽松阈值 |
| 多人检测级联修复 | inference_pipeline | TRACKING模式定期运行YOLO11n |
| 多人检测触发器 | inference_pipeline | 检测数>轨迹数→强制全检测 |

---

## 五、关键技术决策

| 决策 | 理由 |
|------|------|
| 用姿态关键点而非CNN做外观特征 | 零额外模型、零额外延迟、适合理性嵌入式部署 |
| 匈牙利算法而非贪婪匹配 | O(n³)在n≤50时可忽略，全局最优大幅减少ID交换 |
| 三级部分身体验证 | 渐进式降级：全身→上半身→侧身，最大化遮挡场景召回 |
| 重用删除轨迹ID | 保证长期ID一致性，无需持久化存储 |
| K1板上测试 | 真实硬件验证，避免x86仿真差异 |

---

## 六、风险评估与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 匈牙利算法性能 | 低 | 延迟增加<0.1ms | 早停优化：n>50时使用阈值截断 |
| 外观特征区分度不足 | 中 | 重识别准确率下降 | 特征维度可扩展(12→20)，权重可调 |
| 部分身体假阳性增加 | 中 | 非人物体被误检 | 保留几何过滤器+解剖学检查作为兜底 |
| 级联修复增加推理负载 | 低 | 延迟增加~15% | 仅每5帧运行YOLO11n(原每帧)，净节省80% |
| K1平台兼容性 | 低 | 编译/运行问题 | 纯C代码，无新增依赖 |

---

## 七、参考文献

1. Zhang Y, et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022.
2. Wojke N, et al. "Simple Online and Realtime Tracking with a Deep Association Metric." ICIP 2017.
3. Bewley A, et al. "Simple Online and Realtime Tracking." ICIP 2016.
4. Kuhn HW. "The Hungarian Method for the Assignment Problem." Naval Research Logistics, 1955.
5. Cao Z, et al. "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." CVPR 2017.
6. Ultralytics. "YOLOv8 Documentation." https://docs.ultralytics.com/, 2024.
7. Shu G, et al. "Part-based Multiple-Person Tracking with Partial Occlusion Handling." CVPR 2012.
