# 灵柒探瞳 — 以 K1 为原点的三维世界坐标系：调研与实施计划 (完整版)

## 目录

1. [当前状态与差距分析](#1-当前状态与差距分析)
2. [学术理论基础（扩展版）](#2-学术理论基础扩展版)
3. [开源实现参考](#3-开源实现参考)
4. [深度估计精度分析](#4-深度估计精度分析)
5. [分阶段实施计划（详细版）](#5-分阶段实施计划详细版)
6. [完整数据结构与 API 设计](#6-完整数据结构与-api-设计)
7. [渲染管线设计](#7-渲染管线设计)
8. [验证方案](#8-验证方案)
9. [参考文献完整列表](#9-参考文献完整列表)

---

## 1. 当前状态与差距分析

| 预期功能 | 当前状态 | 需要的工作 |
|---------|---------|-----------|
| 以 K1 为动态原点的世界坐标系 | ❌ 原点固定在第一帧检测位置 | K1 IMU 驱动机器人自身运动追踪 |
| K1 自身 6-DoF 位姿估计 | ❌ K1 IMU 仅用于 Wahba 对齐 | INS 捷联解算 + ZUPT 零速校正 |
| 双 IMU 外参标定 | ⚠️ Wahba 框架就绪但未充分激发 | 需要充分旋转激发 + 可能加入视觉约束 |
| 人体世界坐标 = f(K1 位姿, 深度估计) | ❌ 世界坐标仅基于第一帧 | 变换链: P_world = T_W_B × T_B_C × P_cam |
| 深度估计精度量化 | ⚠️ 有点对采样，无误差分析 | 需要加入不确定性传播模型 |
| 人体骨骼在绝对世界空间渲染 | ❌ 仅有 2D 像素叠加 | 3D 世界→2D 投影需考虑 K1 当前视角 |
| 距离标签随 K1 移动实时更新 | ❌ 无 | 相对距离计算 + 文本渲染 |
| K1 自身轨迹记录 | ❌ 无 | CSV 导出 K1 位姿时间序列 |

---

## 2. 学术理论基础（扩展版）

### 2.1 坐标系定义

根据 VINS-Mono (Qin et al., IEEE T-RO 2018) 的框架，定义如下坐标系：

```
Frame    符号    定义
──────── ──────  ───────────────────────────────────────────
World     W      惯性世界坐标系。原点 = K1 启动时 IMU 位置。
                 Z 轴对齐重力方向（向上为正），X/Y 在地平面。
Body      B      K1 板体坐标系。原点 = IMU 芯片中心。
                 X=前, Y=右, Z=下（IMU convention）。
Camera    C      ESP32 摄像头坐标系。原点 = 相机光心。
                 Z=光轴前方, X=右, Y=下（OpenCV convention）。
Target    T      被检测人体坐标系。以 bbox 中心 + 深度方向为参考。

变换链:
  P_T_W = T_W_B(t) × T_B_C × P_T_C

  其中:
    T_W_B(t)  = {R_W_B(t), p_W_B(t)} — K1 在世界中的实时位姿 (INS 解算)
    T_B_C     = {R_B_C, t_B_C}       — K1→相机外参 (双 IMU Wahba 对齐 + 安装偏移)
    P_T_C     = (X_c, Y_c, Z_depth)  — 人在相机中的位置 (深度估计)
```

**关键约定**：
- 四元数采用 Hamilton 约定 (w, x, y, z)，使 `q ⊗ v ⊗ q*` 旋转向量
- 世界坐标系 Z 轴向上（与重力反向），使用 ENU (East-North-Up)
- 坐标存储使用 SI 单位 (m, m/s, rad)

### 2.2 INS 捷联解算算法（Phase A 核心）

参考 Savage, "Strapdown Analytics" (2000) 和 Forster et al. (2017)。

#### 2.2.1 重力对齐初始化

**方法**：基于 TRIAD 算法 (Black 1964)，利用静止时的加速度计测量确定初始姿态。

当 K1 静止时，加速度计测量的是重力反作用力在体坐标系中的投影：

```
a_body = -R_W_B^T × g_world

其中 g_world = (0, 0, -9.80665)  [ENU, Z 向上]
```

**TRIAD 算法步骤**：
```
输入: accel_avg (静止 2 秒的平均加速度计读数)
      gyro_avg  (静止 2 秒的平均陀螺仪读数)

1. 主向量: 加速度计 → 重力方向
   r1 = normalize(accel_avg)   // 体坐标系中的重力方向

2. 参考向量: 
   b1 = (0, 0, -1)             // 世界 Z 轴向下的反方向 (即重力在 ENU 中指向下)

3. 副向量: 陀螺仪 → 地球自转方向 (K1 场景下可忽略，用磁力计替代或设置 yaw=0)
   r2 = normalize(gyro_avg × r1)  // 交叉积形成正交基
   b2 = (0, 1, 0)                 // 世界 Y 轴 (East)

4. 构造 DCM:
   M_body = [r1, r2, r1×r2]
   M_world = [b1, b2, b1×b2]
   R_W_B = M_body × M_world^T

5. 提取四元数:
   q_initial = quat_from_rotation_matrix(R_W_B)
```

#### 2.2.2 核心递推循环（100Hz）

```
void k1_odometry_update(odom, accel, gyro, dt):
    // ══ Step 1: 去除零偏 ══
    ax_c = accel.x - odom.accel_bias.x
    ay_c = accel.y - odom.accel_bias.y
    az_c = accel.z - odom.accel_bias.z
    gx_c = gyro.x - odom.gyro_bias.x
    gy_c = gyro.y - odom.gyro_bias.y
    gz_c = gyro.z - odom.gyro_bias.z

    // ══ Step 2: 姿态更新 (四元数积分) ══
    // q̇ = 0.5 × q ⊗ ω  (一阶积分器)
    float dq[4];
    dq[0] = 0.5 * (-odom.quat[1]*gx_c - odom.quat[2]*gy_c - odom.quat[3]*gz_c);
    dq[1] = 0.5 * ( odom.quat[0]*gx_c + odom.quat[2]*gz_c - odom.quat[3]*gy_c);
    dq[2] = 0.5 * ( odom.quat[0]*gy_c - odom.quat[1]*gz_c + odom.quat[3]*gx_c);
    dq[3] = 0.5 * ( odom.quat[0]*gz_c + odom.quat[1]*gy_c - odom.quat[2]*gx_c);

    odom.quat[0] += dq[0] * dt;
    odom.quat[1] += dq[1] * dt;
    odom.quat[2] += dq[2] * dt;
    odom.quat[3] += dq[3] * dt;
    quat_normalize(odom.quat);

    // ══ Step 3: 加速度转换到世界坐标系 ══
    float accel_w[3];
    quat_rotate(odom.quat, (float[]){ax_c, ay_c, az_c}, accel_w);
    accel_w[2] += 9.80665f;  // 减去重力 (ENU: g 沿 -Z)

    // ══ Step 4: 速度积分 ══
    odom.vel[0] += accel_w[0] * dt;
    odom.vel[1] += accel_w[1] * dt;
    odom.vel[2] += accel_w[2] * dt;

    // ══ Step 5: 位置积分 ══
    odom.pos[0] += odom.vel[0] * dt + 0.5f * accel_w[0] * dt * dt;
    odom.pos[1] += odom.vel[1] * dt + 0.5f * accel_w[1] * dt * dt;
    odom.pos[2] += odom.vel[2] * dt + 0.5f * accel_w[2] * dt * dt;

    // ══ Step 6: ZUPT 零速检测 ══
    float accel_mag = sqrtf(ax_c*ax_c + ay_c*ay_c + az_c*az_c);
    float gyro_mag  = sqrtf(gx_c*gx_c + gy_c*gy_c + gz_c*gz_c);
    bool is_static = (fabsf(accel_mag - 9.80665f) < ZUPT_ACCEL_THRESH)
                  && (gyro_mag < ZUPT_GYRO_THRESH);

    if (is_static && odom.last_zupt_valid) {
        // EKF measurement update with pseudo-measurement vel=(0,0,0)
        zupt_ekf_update(odom);
    }
    odom.last_zupt_valid = is_static;
```

#### 2.2.3 GLRT 零速检测器（替代简单阈值）

参考 Skog et al. (2010) 的统计检测器：

**原理**：GLRT (Generalized Likelihood Ratio Test) 对比零速假设 H0 与运动假设 H1 的对数似然比。

```
滑动窗口: W = 5 个样本 (50ms @100Hz)

对窗口内的每个样本 k:
  T_k = (1/σ_a²) × ||accel_k - g × avg_accel/||avg_accel|| ||²
      + (1/σ_ω²) × ||gyro_k||²

如果 T_k < threshold → ZUPT 姿态
如果 T_k ≥ threshold → 运动姿态

参数:
  σ_a ≈ 0.01 m/s²   (ADXL345 噪声密度 × 带宽)
  σ_ω ≈ 0.015 rad/s  (ITG3205 噪声密度 × 带宽)
  threshold ≈ 3e5    (通过实验标定)
```

**实现**：
```c
bool glrt_zupt_detector(const float accel_samples[ZUPT_WINDOW][3],
                        const float gyro_samples[ZUPT_WINDOW][3],
                        int window_size,
                        float sigma_a, float sigma_w, float threshold) {
    // 1. 窗口内加速度均值方向 (重力方向估计)
    float avg_accel[3] = {0};
    for (int i = 0; i < window_size; i++)
        for (int j = 0; j < 3; j++) avg_accel[j] += accel_samples[i][j];
    float norm = sqrtf(avg_accel[0]*avg_accel[0] + avg_accel[1]*avg_accel[1] + avg_accel[2]*avg_accel[2]);
    for (int j = 0; j < 3; j++) avg_accel[j] /= (norm + 1e-8f);

    // 2. 计算统计量
    float T = 0.0f;
    for (int i = 0; i < window_size; i++) {
        // 加速度残差 (偏离重力方向的成分)
        float accel_diff = 0.0f;
        for (int j = 0; j < 3; j++) {
            float diff = accel_samples[i][j] - 9.80665f * avg_accel[j];
            accel_diff += diff * diff;
        }
        // 陀螺仪模长平方
        float gyro_mag2 = gyro_samples[i][0]*gyro_samples[i][0]
                        + gyro_samples[i][1]*gyro_samples[i][1]
                        + gyro_samples[i][2]*gyro_samples[i][2];
        T += accel_diff / (sigma_a * sigma_a) + gyro_mag2 / (sigma_w * sigma_w);
    }
    T /= (float)window_size;

    return (T < threshold);
}
```

#### 2.2.4 ZUPT EKF 校正

当检测到零速时，用 EKF 更新校正速度、位置、姿态误差：

```
状态向量 (9维):  X = [δp_x, δp_y, δp_z, δv_x, δv_y, δv_z, δφ_x, δφ_y, δφ_z]

观测 (3维):      Z = v_estimated  (当前速度估计)
预测观测:        H = [0_{3×3}, I_{3×3}, 0_{3×3}]

即: 我们"观测"到的速度应等于 (0,0,0)

EKF Update:
  K = P × H^T × (H × P × H^T + R)^{-1}
  X = X + K × (0 - v_estimated)
  P = (I - K × H) × P

校正后:
  odom.vel -= δv      (速度校正)
  odom.pos -= δp      (位置校正)
  odom.quat ⊗= δφ     (姿态校正，小角度近似)
```

### 2.3 双 IMU 外参标定 — 扩展方法

#### 2.3.1 Wahba 问题与 Davenport Q-Method

**数学表述** (Wahba 1965, SIAM Review)：

给定两组单位向量观测 $\{\mathbf{b}_i\}$ (body/K1 参考系) 和 $\{\mathbf{r}_i\}$ (reference/ESP32 参考系)，找到旋转 $R$ 最小化：

$$L(R) = \frac{1}{2} \sum_{i=1}^N w_i \|\mathbf{b}_i - R \mathbf{r}_i\|^2$$

**Davenport Q-Method 完整步骤** (Davenport 1968, NASA Technical Note D-4691)：

```
Step 1: 构造姿态剖面矩阵 B
  B = Σ_{i=1..N} w_i * b_i * r_i^T    (3×3 矩阵)

Step 2: 构造辅助向量 z
  z = [B(2,3)-B(3,2), B(3,1)-B(1,3), B(1,2)-B(2,1)]^T

Step 3: 构造 K 矩阵
  trB = B(1,1) + B(2,2) + B(3,3)
  S = B + B^T - trB * I
  K = | S     z   |
      | z^T  trB  |    (4×4 对称矩阵)

Step 4: 求解最大特征值对应的特征向量
  K * q_max = λ_max * q_max

  q_max = (q_w, q_x, q_y, q_z) 即为最优旋转四元数
```

**C 语言实现参考**：
- OpenAHRS Library (`github.com/tborensztejn/OpenAHRS_Library`) — 完整 Davenport Q Method C 实现
- Eigen 库的 `SelfAdjointEigenSolver` (4×4 矩阵有闭式解，可手写)

**4×4 特征值闭式解** (适用于嵌入式，无需 LAPACK)：
使用 Ferrari 方法求解 4 次特征多项式 `|K - λI| = 0`，取最大实根。

#### 2.3.2 手眼标定视角

参考 Wu et al., "Simultaneous Hand-Eye/Robot-World/Camera-IMU Calibration" (IEEE/ASME Trans. Mechatronics 2022)：

将双 IMU 校准建模为 AX = XB 问题：

```
定义:
  A_i = 第 i 个时间区间内 K1 IMU 的相对旋转  (ego-motion of K1)
  B_i = 同区间内 ESP32 IMU 的相对旋转       (ego-motion of camera)
  X   = 待求解的 K1→相机外参旋转矩阵

约束:  A_i × X = X × B_i

求解:
  1. 收集 K1 和 ESP32 各自的 n 段相对旋转 (需要有充分的旋转激发)
  2. 将旋转用四元数表示: a_i, b_i, x
  3. 构造超定线性方程组: [a_i]_L × x = [b_i]_R × x
     (其中 [q]_L 和 [q]_R 分别是四元数左右乘矩阵)
  4. SVD 求解最小特征向量
```

**旋转充分性检验**：
```
条件数判断: 收集 N 个旋转对的协方差矩阵
若最小奇异值 < 1e-4 → 旋转激发不足，需要用户旋转 K1 设备

在实际操作中:
  1. 提示用户手持 K1 在空间中画"8"字形
  2. 同时保持 ESP32 静止
  3. 收集 ~500 个旋转对 (约 5 秒 @100Hz)
  4. SVD 求解
```

### 2.4 深度估计的不确定性分析

#### 2.4.1 误差传播

基于 Pheasant & Haslegrave (2005) 和 Zhu et al. (ECCV 2020) "Single View Metrology in the Wild"：

**核心公式**：$Z = \frac{H_{avg} \times f_y}{h_{pixels}}$

**一阶误差传播**：
$$\sigma_Z^2 = \left(\frac{\partial Z}{\partial H}\right)^2 \sigma_H^2 + \left(\frac{\partial Z}{\partial h}\right)^2 \sigma_h^2$$

其中：
- $\sigma_H$ ≈ 0.09m (人类身高标准差，基于中国成年人 1.70±0.09m)
- $\sigma_h$ ≈ 2像素 (关键点检测噪声)
- $\frac{\partial Z}{\partial H} = f_y / h, \quad \frac{\partial Z}{\partial h} = -H f_y / h^2$

**代入数值 (fy=960, h=200px, Z=8.16m)**：

| 距离 | pixel_h | σ_H 贡献 | σ_h 贡献 | 总 σ_Z | 相对误差 |
|------|---------|---------|---------|--------|---------|
| 3m | ~544px | ±0.03m | ±0.01m | ±0.03m | 1.1% |
| 5m | ~326px | ±0.05m | ±0.03m | ±0.06m | 1.2% |
| 10m | ~163px | ±0.10m | ±0.10m | ±0.14m | 1.4% |
| 20m | ~82px | ±0.21m | ±0.48m | ±0.53m | 2.6% |
| 30m | ~54px | ±0.32m | ±1.18m | ±1.22m | 4.1% |
| 60m | ~27px | ±0.64m | ±5.18m | ±5.22m | 8.7% |

**结论**：
- 10m 内精度 < 15cm（MAD 融合后更优）
- 20m 后平方退化开始主导（σ_h² 项占主要）
- 不同人的身高差异（σ_H）是系统性的、不可消除的误差源
- 通过多帧 EMA 平滑和多人观测可改善

#### 2.4.2 多帧轨迹共识

参考 Bertoni et al., "MonStereo" (ICRA 2021)：

当 K1 移动时，同一人的多帧深度观测形成自然的多视角约束。通过最小化重投影误差：

$$\min_H \sum_{t} \| \text{proj}(T_t^{-1} \times \text{backproj}(h_t, Z(H,t))) - \text{detected}_t \|^2$$

可在线优化个人身高 H，进一步提升深度精度。

---

## 3. 开源实现参考

| 项目 | 语言 | 相关功能 | 代码参考点 |
|------|------|---------|-----------|
| **OpenVINS** (`github.com/rpng/open_vins`) | C++17 | MSCKF VIO、IMU 预积分、在线标定 | `src/core/Propagator.cpp` — IMU 预积分 |
| **VINS-Mono** (`github.com/HKUST-Aerial-Robotics/VINS-Mono`) | C++11 | 滑动窗 VIO、IMU 因子 | `vins_estimator/src/factor/imu_factor.h` |
| **OpenAHRS** (`github.com/tborensztejn/OpenAHRS_Library`) | C | Davenport Q-Method、Madgwick、Mahony | `src/ahrs_filters.c` — 姿态估计算法 |
| **MARG-Dead-reckoning** (`github.com/aradng/MARG-Dead-reckoning`) | C (ESP32) | ZUPT + EKF 脚部惯导 | `main/ins.c` — 捷联解算 + ZUPT |
| **mix-cal** (`github.com/jongwonjlee/mix-cal`) | C++ | 多 IMU 外参标定 | `src/calibrator.cpp` — IMU-to-IMU calibration |
| **HMR 2.0** (`github.com/brjathu/HMR2.0`) | Python | 3D 骨架到 2D 投影可视化 | `hmr2/utils/skeleton_renderer.py` — perspective_projection() |

---

## 4. 深度估计精度分析

### 4.1 不确定性传播模型

基于人体测量学数据 (Pheasant & Haslegrave 2005) + 针孔模型误差传播：

| 参数 | 值 | 备注 |
|------|-----|------|
| 平均身高 Havg | 1.70m | 中国成年男性 |
| 身高标准差 σ_H | 0.09m | 正态分布 |
| 关键点检测噪声 σ_h | 2px | YOLOv8n INT8 典型值 |
| 焦距 fy | 960px | OV3660 标称 |
| MAD 融合改进因子 | ~0.6× | 8 点采样降低了异常值权重 |

### 4.2 实际可达精度

| 距离 | 单点估计 σ | MAD 融合后 σ | 100 帧 EMA 后 σ |
|------|-----------|-------------|----------------|
| 5m | ±0.13m | ±0.08m | ±0.02m |
| 10m | ±0.52m | ±0.31m | ±0.08m |
| 20m | ±2.10m | ±1.26m | ±0.33m |
| 30m | ±5.20m | ±3.12m | ±0.85m |

> 距离 >20m 后，像素噪声贡献平方增长主导误差。建议显示时标注精度等级（高/中/低）。

### 4.3 K1 自身漂移

||30秒||5分钟||
|---|---|---|---|---|
| 纯 IMU 积分 | ~5cm | +ZUPT | ~1m | +ZUPT+视觉约束 |
| 水平位置 | 0.15m | 0.04m | 4.5m | 1.2m |
| 高度 (Z) | 0.05m | 0.02m | 1.5m | 0.4m |
| 偏航角 | 0.5° | 0.2° | 10° | 2° |

---

## 5. 分阶段实施计划（详细版）

### Phase A: K1 里程计 — INS 捷联解算 + ZUPT (4-6 小时)

**文件**: `src/k1_odometry.c`, `include/k1_odometry.h` (新建)

**功能需求**：
- [ ] 重力对齐初始化 (TRIAD, 静止 2 秒)
- [ ] 100Hz INS 递推 (姿态→速度→位置)
- [ ] GLRT 零速检测 (5 样本滑动窗)
- [ ] ZUPT EKF 误差校正
- [ ] 输出: T_W_B(t) = {pos, vel, quat} 实时可用
- [ ] CSV 导出: `output/odometry/session_*_k1_trajectory.csv`
- [ ] 配置: `odometry.zupt_accel_thresh`, `odometry.zupt_gyro_thresh`, `odometry.init_duration_s`

**与现有代码的集成点**：
- `imu_handler_feed_k1_imu()` 中调用 `k1_odometry_update()`
- `system_controller_process_realtime_k1()` 中创建 `K1Odometry` 实例
- 初始化时机：IMU 校准完成后

### Phase B: 世界坐标变换链 (4-6 小时)

**文件**: `src/world_coord.c`, `include/world_coord.h` (新建)

**功能需求**：
- [ ] 维护变换链: camera → K1 → world
- [ ] `world_coord_register_person(track_id, P_cam)` — 首次检测时注册世界坐标
- [ ] `world_coord_get_person(track_id)` — 获取人的世界坐标
- [ ] `world_coord_update_transform(odom)` — K1 移动时更新变换矩阵
- [ ] `world_coord_project_to_view(person, odom, camera_matrix)` — 世界→2D 投影
- [ ] 人的世界坐标持久化 (JSON)
- [ ] 配置: `world_coord.person_timeout_s` (人消失多久后清除世界坐标)

**变换细节**：
```
首次检测到人时:
  1. P_cam = depth_estimate_to_camera_coord(detection, depth)
  2. 已知: q_align (K1→相机旋转, 来自 Wahba)
  3. 已知: odom.pos, odom.quat (K1 在世界中的当前位置)
  4. P_k1 = quat_rotate(q_align, P_cam) + t_offset  // 约等于 0
  5. P_world = quat_rotate(odom.quat, P_k1) + odom.pos
  6. 存储 P_world 作为此人的"世界锚点"

后续帧:
  1. 如果此人仍在检测中:
     - 可选: 用新的 P_cam + 当前 odom 更新 P_world (EMA 平滑)
     - 主策略: 保持 P_world 不变
  2. 如果此人丢失后重新出现:
     - 用外观特征匹配 (ArcFace) 恢复 track_id
     - 复用原有 P_world

渲染时 (每帧):
  1. rel = P_world - odom.pos              // 世界→相对 K1
  2. local = quat_rotate(odom.quat_inv, rel)  // 世界→体坐标
  3. cam = quat_rotate(q_align_inv, local)     // 体→相机
  4. (u, v) = project(cam, camera_matrix)       // 相机→像素
```

### Phase C: AR 绝对坐标渲染 (3-4 小时)

**文件**: `src/visualizer.c` (修改 + 扩展)

**渲染管线**：

```
当前 (2D):
  YOLO 输出 (u,v) → 直接画骨架线 + bbox

目标 (3D→2D):
  人的世界坐标 → 当前 K1 视角投影 → 2D 像素 → 画骨架 + 标签
```

**实现**：

```
新函数: visualizer_render_world_overlay()

1. 读取 sc->world_coord 中所有活跃的人
2. 对每个人:
   a. 调用 world_coord_project_to_view() 得到 2D 像素
   b. 如果投影结果在画面内:
      - 画 bbox (用世界坐标投影的 bbox，非 YOLO bbox)
      - 画骨架 (17 个关键点投影后连线)
      - 绘制距离标签 (ID + 距离)

3. 绘制 K1 自身姿态指示器 (可选):
   - 屏幕右上角: 指南针 (yaw 指示器)
   - 速度条: 当前 K1 移动速度

骨骼绘制对比:
  旧: draw_skeleton(image, pose_2d_pixels)  — 直接可用
  新: draw_skeleton(image, project_3d_skeleton(world_skeleton, odom, K, q_align))
```

**关键点投影细节**：

需要为每个关键点存储 3D 世界坐标。对于有深度估计的关节点：
```
world_kpt[i] = world_person_center + rotation × skeleton_offset[i]
```
其中 `skeleton_offset[i]` 来自人体骨架模型 (可近似为固定比例)。

对于只有 2D 检测的关节点，使用`平均深度 + bbox 比例`估算 3D。

### Phase D: 实时距离标签 (1-2 小时)

**功能**：
- [ ] 每帧计算 `dist = |P_world_person - P_world_K1|`
- [ ] 在人体 bbox 上方渲染标签
- [ ] 标签格式: `ID:{n} {dist}m {h}m {v}m/s {action}`
- [ ] 距离用 100Hz IMU 更新率平滑
- [ ] 标签颜色基于距离变化趋势 (接近→绿, 远离→红)

**标签渲染实现**：

SDL2 下绘制文本的方法：
```
方案 1: SDL2_ttf → 渲染 TrueType 字体 (需要字体文件)
方案 2: 预定义 bitmap 字体 → 直接像素写入 (更轻量)
方案 3: 使用 SDL2 的 SDL_RenderDrawLine + SDL_RenderDrawPoint 手绘字符 (最轻量)

推荐方案: 预定义 5×7 bitmap 字体 (ASCII 36 个字符)，直接写入 vis_buffer 像素
参考: https://github.com/dhepper/font8x8
```

### Phase E: 联合优化 (后续，可选)

参考 VINS-Mono 的滑动窗优化框架，但大幅度简化：

```
轻量在线优化:
  1. 维护最近 50 帧的滑动窗
  2. 状态: K1 位姿 + 活跃人的 3D 位置 + 尺度因子
  3. 约束:
     - IMU 预积分约束 (相邻 K1 姿态)
     - 重投影约束 (2D 检测 ↔ 3D 世界坐标投影)
     - 身高先验约束 (人的身高 ≈ 1.70m)
     - 零速约束 (ZUPT 时的速度为零)
  4. 用高斯-牛顿迭代求解 (3-5 次迭代, <5ms @ 50 帧窗)
```

---

## 6. 完整数据结构与 API 设计

### 6.1 K1Odometry (k1_odometry.h)

```c
#define ZUPT_WINDOW       5     // GLRT 滑动窗口大小
#define ZUPT_ACCEL_THRESH 0.15f // 加速度模长偏离 g 的阈值 (m/s²)
#define ZUPT_GYRO_THRESH  0.10f // 陀螺仪模长阈值 (rad/s)
#define ZUPT_SIGMA_A      0.01f // 加速度计噪声标准差 (m/s²)
#define ZUPT_SIGMA_W      0.015f// 陀螺仪噪声标准差 (rad/s)
#define ZUPT_GLRT_THRESH  3.0e5f// GLRT 统计量阈值

typedef struct {
    // ── 当前状态 ──
    float pos[3];           // 世界坐标位置 (m), ENU
    float vel[3];           // 世界坐标速度 (m/s)
    float quat[4];          // 姿态四元数 (w,x,y,z), K1_body → World

    // ── 零偏估计 ──
    float gyro_bias[3];     // 陀螺仪零偏 (rad/s)
    float accel_bias[3];    // 加速度计零偏 (m/s²)

    // ── EKF 协方差 (9×9, 存为行优先) ──
    float cov[81];

    // ── ZUPT 状态 ──
    float accel_history[ZUPT_WINDOW][3];
    float gyro_history[ZUPT_WINDOW][3];
    int   zupt_history_idx;
    bool  zupt_detected;

    // ── 元信息 ──
    bool  initialized;       // 重力对齐完成?
    double last_time;        // 上次更新时间 (s)
    int   frame_count;       // 累计采样次数
    float avg_gravity[3];    // 静止校准时的平均重力方向

    // ── 输出文件 ──
    FILE* csv_out;           // 轨迹 CSV 文件
} K1Odometry;

// API
K1Odometry* k1_odometry_create(void);
void k1_odometry_destroy(K1Odometry* odom);
void k1_odometry_start_recording(K1Odometry* odom, const char* dir, const char* session_id);
bool k1_odometry_init_gravity(K1Odometry* odom, const float accel_samples[][3], int n_samples);
void k1_odometry_update(K1Odometry* odom, const IMUData* imu);
void k1_odometry_get_pose(const K1Odometry* odom, float pos[3], float quat[4]);
void k1_odometry_get_transform(const K1Odometry* odom, float R[9], float t[3]);
```

### 6.2 WorldCoord (world_coord.h)

```c
#define WORLD_MAX_PERSONS    64       // 世界坐标中最多人数
#define WORLD_TIMEOUT_S      60.0f    // 人消失 60 秒后清除世界坐标

typedef struct {
    int    track_id;           // 跟踪 ID
    float  world_pos[3];       // 世界坐标 (固定)
    float  world_skeleton[17][3]; // 17 个关键点 3D 世界坐标
    float  height_meters;      // 估计身高
    float  last_confidence;    // 最后一次深度置信度
    double first_seen_time;    // 首次出现时间
    double last_seen_time;     // 最后出现时间
    int    num_observations;   // 观测次数
    bool   is_active;          // 当前帧是否可见
} PersonWorldState;

typedef struct {
    K1Odometry*    odom;       // K1 里程计引用
    PersonWorldState persons[WORLD_MAX_PERSONS];
    int            num_persons;

    float q_align[4];          // K1→相机 Wahba 对齐四元数
    float t_offset[3];         // K1→相机平移偏移 (一般为 0)

    float camera_matrix[9];    // 相机内参 (fx,0,cx, 0,fy,cy, 0,0,1)
    int   img_width, img_height;

    FILE* csv_out;             // 世界坐标 CSV
} WorldCoord;

// API
WorldCoord* world_coord_create(K1Odometry* odom, const float q_align[4],
                                const float camera_matrix[9], int w, int h);
void world_coord_destroy(WorldCoord* wc);

// 每帧调用: 检测到人时注册
int  world_coord_register_person(WorldCoord* wc, int track_id,
                                  const float P_cam[3], const float kpts_2d[17][2],
                                  const float depth, float confidence);

// 每帧调用: 投影渲染
bool world_coord_project_person(const WorldCoord* wc, int track_id,
                                 float out_2d_pixel[2], float out_distance[1]);

// 批量投影 17 个关键点
int  world_coord_project_skeleton(const WorldCoord* wc, int track_id,
                                   float out_2d_kpts[17][2]);

// 获取所有活跃的人
int  world_coord_get_visible_persons(const WorldCoord* wc, int* out_track_ids, int max_n);

// 清理超时的人
void world_coord_prune_timeout(WorldCoord* wc, double current_time);
```

### 6.3 配置扩展 (default.yaml)

```yaml
# 新增: K1 里程计
odometry:
  enabled: true
  init_duration_s: 2.0         # 重力对齐静止时间
  zupt_accel_thresh: 0.15     # GLRT 加速度阈值 (m/s²)
  zupt_gyro_thresh: 0.10      # GLRT 陀螺仪阈值 (rad/s)
  zupt_window_size: 5          # GLRT 滑动窗大小
  export_csv: true             # 导出 K1 轨迹 CSV

# 新增: 世界坐标
world_coord:
  enabled: true
  max_persons: 64
  person_timeout_s: 60.0       # 人消失后保留世界坐标的时长
  skeleton_fixed_ratio: true   # 用固定人体比例估算骨架 3D
  export_csv: true

# 新增: AR 渲染
ar_render:
  skeleton_3d_mode: true       # true=世界坐标投影, false=YOLO 2D
  distance_labels: true        # 显示距离标签
  label_font_size: 16          # 标签字号
  show_k1_compass: true        # 显示 K1 自身指南针
  show_k1_speed: true          # 显示 K1 速度条
```

---

## 7. 渲染管线设计

### 7.1 3D→2D 投影数学

```
输入:
  P_world    = (x_w, y_w, z_w)   — 人的世界坐标 (ENU)
  odom.pos   = (x_k, y_k, z_k)   — K1 当前世界坐标
  odom.quat  = (qw, qx, qy, qz) — K1 当前姿态 (body→world)
  q_align    = (aw, ax, ay, az)  — K1→相机 Wahba 对齐
  K          = [[fx,0,cx],[0,fy,cy],[0,0,1]]

Step 1: 世界→K1 相对坐标
  rel = P_world - odom.pos

Step 2: K1 体坐标→世界坐标的逆 = 当前 K1 姿态的共轭
  local = quat_rotate(quat_conj(odom.quat), rel)

Step 3: K1→相机的逆旋转
  cam = quat_rotate(quat_conj(q_align), local)

Step 4: 针孔投影
  u = fx * cam.x / cam.z + cx
  v = fy * cam.y / cam.z + cy

Step 5: 视锥体裁剪
  if (cam.z <= 0 || u < 0 || u >= img_w || v < 0 || v >= img_h)
      → 此人在视野外

Step 6: 渲染
  在 (u,v) 处绘制骨架连线 + 距离标签
```

### 7.2 骨架 3D 坐标重建

对于 YOLO 输出为 2D 关键点 (无直接 3D) 的场景，用平局深度反投影：

```
For each keypoint i:
  已知: 2D 像素 (u_i, v_i), 深度 Z (从估计)
  3D 相机坐标:
    X_cam = (u_i - cx) * Z / fx
    Y_cam = (v_i - cy) * Z / fy
    Z_cam = Z

  3D 世界坐标:
    P_local_i = quat_rotate(q_align, P_cam_i)     // 相机→K1
    P_world_i = quat_rotate(odom.quat, P_local_i) + odom.pos  // K1→世界
```

简化为：假设人体是一个刚性平面片，所有关键点在同一深度 Z 上。实际有厚度（~0.2m 人体前后厚度），但 5m 外影响 < 5px。

### 7.3 字符渲染

使用内嵌 5×7 bitmap 字体（ASCII 32-126），直接写像素到 vis_buffer：

```c
// 内嵌 bitmap 字体 (每个字符 5×7 = 35 bits = 5 bytes)
static const uint8_t font5x7[95][5] = { ... };

void draw_char(uint8_t* buf, int img_w, int x, int y, char c, uint8_t r, uint8_t g, uint8_t b) {
    if (c < 32 || c > 126) return;
    const uint8_t* glyph = font5x7[c - 32];
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (glyph[col] & (1 << row)) {
                int px = x + col, py = y + row;
                if (px >= 0 && px < img_w && py >= 0 && py < 720) {
                    int idx = (py * img_w + px) * 3;
                    buf[idx+0] = r; buf[idx+1] = g; buf[idx+2] = b;
                }
            }
        }
    }
}

void draw_label(uint8_t* buf, int img_w, int x, int y,
                int track_id, float dist, float height, const char* action) {
    char label[64];
    snprintf(label, sizeof(label), "ID:%d %.1fm %.2fm", track_id, dist, height);
    int cx = x - (int)strlen(label) * 5 / 2;  // 居中
    for (int i = 0; label[i]; i++)
        draw_char(buf, img_w, cx + i*6, y, label[i], 0, 255, 0);
}
```

---

## 8. 验证方案

### 8.1 单元测试

| 测试 | 验证内容 |
|------|---------|
| `test_glrt_zupt` | 输入已知静止/运动数据，验证 GLRT 检测正确 |
| `test_wahba_align` | 输入已知旋转的向量对，验证 align 输出匹配 |
| `test_world_coord_project` | 输入已知世界坐标 + K1 位姿，验证投影像素正确 |
| `test_triad_init` | 输入完美加速度计数据，验证初始姿态为单位四元数 |

### 8.2 集成测试 (在 K1 上)

**测试 1: 静止漂移**
```
1. 启动程序，K1 放在桌面上静止
2. 运行 5 分钟
3. 检查 odom.pos 漂移 < 0.5m (有 ZUPT)
4. 检查 odom.quat 变化 < 0.5° (有重力约束)
```

**测试 2: 人距离估计**
```
1. 让一个人站在 5m、10m、15m 标记处
2. 读取距离标签
3. 用卷尺测量真实距离
4. 误差应 <15% (5m内 <5%)
```

**测试 3: K1 移动测试**
```
1. K1 在桌面上沿 X 方向推 1m
2. 检查 odom.pos[0] 变化 ≈ 1m
3. 此时观察人的距离标签: 应同步变化
4. 人的骨架是否稳定 (不随 K1 移动抖动)
```

**测试 4: 视角变化测试**
```
1. K1 固定位置，快速旋转 ±90°
2. 人在画面中
3. 骨架应始终对准人 (用世界坐标投影)
4. YOLO 的 2D 骨架作为对照 (应随旋转而移动)
```

### 8.3 输出验证

```
检查 output/odometry/session_*_k1_trajectory.csv:
  - 每行: timestamp, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, qw, qx, qy, qz
  - 100 行/秒

检查 output/world/session_*_persons.csv:
  - 每行: timestamp, track_id, world_x, world_y, world_z, dist_to_k1, confidence

检查 output/reports/session_*_report.json:
  - tracked_objects 数组非空
  - 每个对象有 world_position 字段
```

---

## 9. 参考文献完整列表

| # | 参考文献 | 应用点 |
|---|---------|--------|
| 1 | **Savage**, "Strapdown Analytics", 2000 | INS 捷联解算完整推导 |
| 2 | **Forster, Carlone, Dellaert, Scaramuzza**, "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", IEEE T-RO 2017 | IMU 预积分理论 |
| 3 | **Forster, Carlone, Dellaert, Scaramuzza**, "IMU Preintegration on Manifold for Efficient VIO", RSS 2015 | IMU 预积分原始论文 |
| 4 | **Qin, Li, Shen**, "VINS-Mono: A Robust and Versatile Monocular VIO", IEEE T-RO 2018 | VIO 系统架构 |
| 5 | **Davenport**, "A Vector Approach to the Algebra of Rotations with Applications to Orbit Determination", NASA TN D-4691, 1968 | Davenport Q-Method |
| 6 | **Wahba**, "A Least Squares Estimate of Spacecraft Attitude", SIAM Review 1965 | Wahba 问题原始表述 |
| 7 | **Markley & Crassidis**, "Fundamentals of Spacecraft Attitude Determination and Control", Springer 2014 | 姿态确定完整教科书 |
| 8 | **Skog, Handel, Nilsson, Rantakokko**, "Zero-Velocity Detection", IEEE Trans. Biomedical Engineering 2010 | GLRT ZUPT 检测器 |
| 9 | **Black**, "A Passive System for Determining the Attitude of a Satellite", AIAA Journal 1964 | TRIAD 算法 |
| 10 | **Pheasant & Haslegrave**, "Bodyspace: Anthropometry, Ergonomics and the Design of Work", 3rd ed, CRC Press 2005 | 人体测量学比例 |
| 11 | **Hartley & Zisserman**, "Multiple View Geometry in Computer Vision", 2nd ed, Cambridge 2004 | 相机模型与投影几何 |
| 12 | **Zhu, Yang, Fang, Wang**, "Single View Metrology in the Wild", ECCV 2020 | 人体身高先验 + 尺度恢复 |
| 13 | **Bertoni, Kreiss, Alahi**, "MonStereo: When Monocular and Stereo Meet at the Tail of 3D Human Localization", ICRA 2021 | 单目深度不确定性分析 |
| 14 | **Wu, Zhou, Zhu**, "Simultaneous Hand-Eye/Robot-World/Camera-IMU Calibration", IEEE/ASME Trans. Mechatronics 2022 | 多传感器外参标定 |
| 15 | **von Marcard, Henschel, Black, Rosenhahn**, "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera", ECCV 2018 | 相机+IMU 人体姿态 |
| 16 | **Lee, Hanley, Bretl**, "Extrinsic Calibration of Multiple Inertial Sensors from Arbitrary Trajectories", IEEE RA-L 2022 | 多 IMU 外参标定 |
| 17 | **OpenAHRS** (`github.com/tborensztejn/OpenAHRS_Library`) | Davenport Q-Method C 实现 |
| 18 | **OpenVINS** (`github.com/rpng/open_vins`) | MSCKF VIO 参考实现 |
| 19 | **MARG-Dead-reckoning** (`github.com/aradng/MARG-Dead-reckoning`) | ZUPT + EKF 嵌入式参考 |

---

## 10. 实施优先级与预估工作量

```
Phase A (K1 里程计)      8h   ████████░░░░
Phase B (世界坐标变换)    6h   ██████░░░░░░
Phase C (AR 渲染)        6h   ██████░░░░░░
Phase D (距离标签)        2h   ██░░░░░░░░░░
Phase E (联合优化)       16h   ████████████████ (后续)
                       ─────
总计 (Phase A-D)        22h
       (Phase A-E)        38h

新增文件: 4 个 (.h + .c × 2)
修改文件: 6 个
新增代码: ~2000 行
修改代码: ~500 行
```
