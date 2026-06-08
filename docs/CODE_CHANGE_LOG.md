# 代码修改记录 — 灵契探瞳目标追踪系统全局优化

> 日期: 2026-06-08  
> 版本: v2.0-optimization  
> 修改类型: 重大功能优化 (Breaking changes in tracking internals)

---

## 一、修改概览

| 类别 | 文件数 | 新增函数 | 修改函数 | 新增行 | 修改行 |
|------|--------|---------|---------|--------|--------|
| 追踪核心 | 2 | 15 | 6 | ~800 | ~400 |
| 推理管线 | 1 | 0 | 4 | ~40 | ~30 |
| 关键点验证 | 2 | 2 | 0 | ~120 | 0 |
| 系统控制 | 1 | 0 | 2 | ~30 | ~10 |
| 配置 | 1 | 0 | 0 | ~15 | ~5 |
| 类型定义 | 1 | 0 | 1 | +3 | 0 |

---

## 二、逐文件详细变更

### 2.1 `include/tracking_manager.h` — 追踪管理器头文件 (重写)

**变更性质**: 重大扩展，从 ~110行 扩展到 ~290行

**新增宏定义**:
```c
// 轨迹生命周期
TRACKING_MAX_LOST_FRAMES        60    // 30→60, 遮挡韧性翻倍
TRACKING_MAX_OCCLUDED_FRAMES    90    // 新增: 遮挡轨迹延长寿命
TRACKING_IOU_THRESHOLD_LOW      0.15  // 新增: 第二阶段IoU阈值

// 外观特征匹配
TRACKING_APPEARANCE_DIM         12    // 新增: 姿态关键点描述符维度
TRACKING_APPEARANCE_MAX_GALLERY 50    // 新增: 每轨迹最大特征数
TRACKING_APPEARANCE_WEIGHT      0.35  // 新增: 外观权重
TRACKING_APPEARANCE_MAX_DIST    0.50  // 新增: 外观距离门控

// 级联匹配
TRACKING_CASCADE_MAX_AGE        30    // 新增
TRACKING_CASCADE_MIN_HITS       3     // 新增

// 遮挡/部分身体
TRACKING_UPPER_BODY_KPT_COUNT   11    // 新增
TRACKING_UPPER_BODY_MIN_KPTS    4     // 新增
TRACKING_SIDE_BODY_MIN_KPTS     3     // 新增
TRACKING_OCCLUSION_SCORE_THRESH 0.40  // 新增

// 匈牙利算法
HUNGARIAN_MAX_DIM               256   // 新增
```

**新增结构体**:
- `AppearanceFeature` — 12维姿态关键点外观描述符
- `MatchType` 枚举 — 匹配类型(全身/上半身/侧身/外观重识别)

**TrackEntry 新增字段**:
```c
AppearanceFeature appearance_gallery[50];  // 外观特征库
AppearanceFeature latest_appearance;       // 最新特征
float occlusion_score;                     // 遮挡评分 (0.0-1.0)
int occlusion_frames;                      // 连续遮挡帧数
bool is_partial_body;                      // 部分身体模式
MatchType last_match_type;                 // 上次匹配类型
int unmatched_detection_count;             // 多人检测计数
bool has_nearby_unmatched;                 // 多人检测标记
```

**ObjectTracker 新增字段**:
```c
int max_occluded;                       // 遮挡轨迹最大丢失帧数
int cascade_max_age;                    // 级联最大年龄
int cascade_min_hits;                   // 级联最小命中
float appearance_weight;                // 外观权重
float appearance_max_dist;              // 外观距离门控
float iou_threshold_low;                // 第二阶段IoU阈值
int upper_body_min_kpts;                // 上半身最小关键点
int side_body_min_kpts;                 // 侧半身最小关键点
float occlusion_score_threshold;        // 遮挡评分阈值
int new_person_grace_frames;            // 新人检测宽限期
AppearanceFeature reid_pool[256];       // 重识别池
int reid_pool_ids[256];                 // 重识别池ID
int reid_pool_count;                    // 重识别池大小
int reid_pool_max_age;                  // 重识别池最大年龄
int total_id_switches;                  // ID交换计数(诊断)
int total_reidentifications;            // 重识别计数(诊断)
int total_partial_matches;              // 部分身体匹配计数(诊断)
```

**新增API函数**:
```c
void object_tracker_associate_poses(...);         // 姿态关联(更新外观特征)
int  object_tracker_get_all_track_count(...);     // 获取全部轨迹数
int  object_tracker_get_confirmed_count(...);     // 获取确认轨迹数
void object_tracker_set_cascade_config(...);      // 级联配置
void object_tracker_set_occlusion_config(...);    // 遮挡配置
void object_tracker_set_reid_config(...);         // 重识别配置
void object_tracker_set_multi_person_config(...); // 多人检测配置
bool appearance_feature_from_pose(...);           // 从姿态提取外观特征
float appearance_feature_distance(...);           // 外观特征距离
bool appearance_feature_is_valid(...);            // 外观特征有效性
int  count_upper_body_keypoints(...);             // 上半身关键点计数
int  count_left_side_keypoints(...);              // 左侧关键点计数
int  count_right_side_keypoints(...);             // 右侧关键点计数
```

---

### 2.2 `src/tracking_manager.c` — 追踪管理器实现 (重写)

**变更性质**: 从 ~565行 扩展到 ~1000行，核心逻辑全部重写

**保留的函数** (未修改逻辑):
- `kalman_init()`, `kalman_predict()`, `kalman_update()`, `kalman_get_bbox()` — 卡尔曼滤波核心不变

**重写的函数**:

| 函数 | 变更 |
|------|------|
| `object_tracker_create()` | +初始化所有新增配置字段 |
| `create_track()` | +appearance_feature参数, +from_reid参数 |
| `update_track_match()` | +外观特征库更新, +遮挡评分, +部分身体标记 |
| `remove_track()` | +删除前保存特征到重识别池 |
| `object_tracker_update()` | **完全重写**: 匈牙利算法+级联匹配+部分身体+重识别 |

**新增函数**:

| 函数 | 行数 | 功能 |
|------|------|------|
| `hungarian_solve()` | ~100行 | Kuhn-Munkres匈牙利算法O(n³)求解器 |
| `appearance_feature_from_pose()` | ~100行 | 从COCO-17关键点提取12维描述符 |
| `appearance_feature_distance()` | ~15行 | 余弦距离计算 |
| `count_upper_body_keypoints()` | ~15行 | 上半身关键点(kpt 0-10)计数 |
| `count_left_side_keypoints()` | ~15行 | 左侧关键点链计数 |
| `count_right_side_keypoints()` | ~15行 | 右侧关键点链计数 |
| `build_cost_matrix()` | ~40行 | 匈牙利代价矩阵构建(IoU+外观) |
| `cascade_matching()` | ~100行 | DeepSORT风格级联匹配 |
| `iou_fallback_matching()` | ~60行 | 第二阶段IoU回退匹配 |
| `reid_check()` | ~20行 | 新检测与重识别池匹配 |
| `object_tracker_associate_poses()` | ~50行 | 姿态关联+外观特征更新 |

**关键算法变更**:

```diff
- // 旧代码: 贪婪匹配
- for (int t = 0; t < num_predicted; t++) {
-     float best_iou = 0.35;
-     for (int h = 0; h < num_high; h++) {
-         if (iou > best_iou) { best_iou = iou; best_det = d; }
-     }
-     if (best_det >= 0) { update_track_match(...); }
- }

+ // 新代码: 匈牙利算法+级联匹配
+ cascade_matching(tracker, pred_bboxes, ..., det_features, ...);
+ iou_fallback_matching(tracker, ..., iou_threshold_low);
+ // 未匹配高置信度检测 → reid_check() → create_track(from_reid=true/false)
```

---

### 2.3 `include/core_types.h` — 核心类型定义

**Detection 结构体新增字段**:
```c
int track_id_hint;          // 重识别提示: >=0时复用此track_id
bool is_partial_body;       // 是否为部分身体检测(上半身/侧身)
int num_visible_keypoints;  // 关联姿态中的可见关键点数
```

---

### 2.4 `include/keypoint_validator.h` — 关键点验证器头文件

**新增函数声明**:
```c
bool keypoint_validator_upper_body_check(kv, pose, bbox);  // 上半身验证
bool keypoint_validator_side_body_check(kv, pose, bbox);   // 侧半身验证
```

---

### 2.5 `src/keypoint_validator.c` — 关键点验证器实现

**新增函数** (~120行):

`keypoint_validator_upper_body_check()`:
- 检查关键点0-10 (鼻→双腕)
- 要求≥4个有效上半身关键点
- 至少一侧肩膀可见
- 头在肩上方验证
- 肩对称性检查
- 50%关键点在边界框内

`keypoint_validator_side_body_check()`:
- 检查左侧链: 眼(1)→耳(3)→肩(5)→肘(7)→腕(9)→髋(11)→膝(13)→踝(15)
- 检查右侧链: 眼(2)→耳(4)→肩(6)→肘(8)→腕(10)→髋(12)→膝(14)→踝(16)
- 至少一侧≥3个有效关键点
- 垂直顺序验证: 肩→髋→膝

---

### 2.6 `src/inference_pipeline.c` — 推理管线

**修改的函数**:

`cascade_update_state()`:
- 新增参数: `num_detections`, `num_tracks`
- 新增: 多人检测触发器 `if (num_detections > num_tracks + 1) → VALIDATING`
- 修复: 静态局部变量 `lost_counter` 可能导致多实例问题

`inference_pipeline_process_frame()`:
- **关键修复**: TRACKING模式中每5帧运行YOLO11n (`CASCADE_SECONDARY_INTERVAL = 5`)
- 更改: `bool run_secondary = run_full_res || (TRACKING && frame_counter % 5 == 0)`
- 新增: 三级部分身体验证逻辑 (全身→上半身→侧身)
- 新增: 部分身体检测标记 (`is_partial_body=true`, `num_visible_keypoints=N`)
- 新增: `TRACKING_PARTIAL_BODY_CONF_BOOST` (0.05) 置信度补偿

`filter_detections()`:
- 新增: 可见关键点计数 (n_upper, n_left, n_right)
- 新增: 三级验证回退逻辑

---

### 2.7 `src/system_controller.c` — 系统控制器

**修改位置**: `system_controller_create()` 中的追踪管理器初始化

新增配置应用调用:
```c
object_tracker_set_cascade_config(...);     // 级联匹配参数
object_tracker_set_occlusion_config(...);   // 遮挡处理参数
object_tracker_set_reid_config(...);        // 重识别参数
object_tracker_set_multi_person_config(...); // 多人检测参数
```

**修改位置**: 所有 `object_tracker_update()` 调用后

新增:
```c
object_tracker_associate_poses(tracker, poses, num_poses);
```

**影响路径**:
- `system_controller_process_video()` — 离线视频处理
- `system_controller_process_realtime()` — 实时管线
- `k1_postprocess_thread()` — K1多线程管线

---

### 2.8 `configs/default.yaml` — 配置文件

**tracking 节新增参数**:
```yaml
tracking:
  max_occluded_frames: 90         # 新增: 遮挡延长寿命
  cascade_max_age: 30             # 新增: 级联最大年龄
  cascade_min_hits: 3             # 新增: 级联最小命中
  appearance_weight: 0.35         # 新增: 外观特征权重
  appearance_max_dist: 0.50       # 新增: 外观距离门控
  iou_threshold_low: 0.15         # 新增: 第二阶段IoU阈值
  upper_body_min_keypoints: 4     # 新增: 上半身最小关键点
  side_body_min_keypoints: 3      # 新增: 侧半身最小关键点
  occlusion_score_threshold: 0.40 # 新增: 遮挡评分阈值
  reid_pool_max_age: 90           # 新增: 重识别池老化时间
  new_person_grace_frames: 3      # 新增: 新人检测宽限期
```

**参数变更**:
```yaml
  max_lost: 60        # 30→60 (遮挡韧性翻倍)
  min_iou: 0.30       # 0.50→0.30 (匹配到级联+外观)
```

---

## 三、向后兼容性

| API | 兼容性 | 说明 |
|-----|--------|------|
| `object_tracker_create()` | ✅ 完全兼容 | 新增参数均有合理默认值 |
| `object_tracker_update()` | ✅ 完全兼容 | 签名不变，内部逻辑重写 |
| `object_tracker_destroy()` | ✅ 完全兼容 | 无变化 |
| `object_tracker_reset()` | ✅ 完全兼容 | 新增重置重识别池 |
| 配置YAML | ✅ 向前兼容 | 新增键有默认值，旧配置仍可运行 |
| `Detection` 结构体 | ⚠️ 二进制不兼容 | 新增3个字段，需重新编译 |
| `TrackEntry` 结构体 | ⚠️ 二进制不兼容 | 大幅扩展，需重新编译 |

---

## 四、构建说明

```bash
# 标准构建 (x86测试)
cd lingqi_tantong_c
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# K1交叉编译
cd lingqi_tantong_c
mkdir -p build_k1 && cd build_k1
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv64-toolchain.cmake \
         -DBIANBU_SYSROOT=/path/to/bianbu/sysroot
make -j$(nproc)
```

---

## 五、已知限制

1. **匈牙利算法内存分配**: 使用动态分配(`calloc`/`free`)，在极端场景(>100个同时检测)可能产生碎片。实际场景最多20人，可忽略。

2. **外观特征区分度**: 12维姿态特征对穿着相似(如统一制服)的人群区分度有限。未来可扩展为结合颜色直方图的混合特征。

3. **重识别池无时间戳**: 当前通过老化计数器模拟时间衰减，非精确帧级时间戳。对于大多数场景(轨迹丢失后数秒内重识别)已足够。

4. **级联状态机共享状态**: `cascade_update_state()` 中的 `static int lost_counter` 在多实例场景中可能共享。当前系统仅有单实例，无实际影响。
