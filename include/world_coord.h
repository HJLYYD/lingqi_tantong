#ifndef WORLD_COORD_H
#define WORLD_COORD_H

#include "core_types.h"
#include "k1_odometry.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WORLD_MAX_PERSONS       64      /* 世界坐标中最多人数 */
#define WORLD_TIMEOUT_S         60.0f   /* 人消失 60 秒后清除世界坐标 */
#define WORLD_MAX_SKELETON_KPTS 17      /* COCO 17-point skeleton */
#define WORLD_BACKFACE_CULL_M   0.1f    /* 后向面剔除阈值 (m): 参考 OpenGL near-plane
                                         * 惯例 0.1-1.0, 避免 1cm 处数值发散 */

/*
 * PersonWorldState — 人在世界坐标系中的持久状态
 *
 * 世界锚点策略 (参考 SEVIS-3D Schmidt-EKF, DuToit et al.):
 *   - 首次检测时固定在当前投影的世界位置 (锚点)
 *   - 后续帧: 自适应更新 — 只有当观测位移超过阈值时才更新锚点
 *   - 静止时锁定锚点不动, 防止 K1 里程计漂移污染世界坐标
 *   - 运动时使用慢速动量 (alpha=0.15) 平滑更新
 */
typedef struct {
    int    track_id;                              /* 跟踪 ID */
    float  world_pos[3];                          /* 人体中心世界坐标 (ENU: x,y,z 向上) */
    float  world_skeleton[WORLD_MAX_SKELETON_KPTS][3]; /* 17 个关键点 3D 世界坐标 */
    float  height_meters;                         /* 估计身高 (m) */
    float  last_depth;                            /* 最近一次深度估计 (m) */
    float  last_confidence;                       /* 最近一次深度置信度 */
    double first_seen_time;                       /* 首次出现时间 (s) */
    double last_seen_time;                        /* 最后出现时间 (s) */
    int    num_observations;                      /* 观测次数 */
    bool   is_active;                             /* 当前帧是否可见 */
    bool   has_skeleton;                          /* 是否有 3D 骨架数据 */
} PersonWorldState;

/*
 * WorldCoord — 以 K1 为原点的动态世界坐标系统
 *
 * 变换链:
 *   P_world = T_W_B(t) × T_B_C × P_cam
 *
 *   其中:
 *     T_W_B(t) = K1Odometry 输出的实时位姿 (R_W_B, p_W_B)
 *     T_B_C    = K1→相机外参 (Wahba 对齐四元数 + 安装偏移)
 *     P_cam    = 深度估计 + 逆投影得到的相机坐标
 */
typedef struct {
    K1Odometry*       odom;                /* K1 里程计引用 (只读) */
    PersonWorldState  persons[WORLD_MAX_PERSONS];
    int               num_persons;

    /* ── 外参: K1→相机变换 ── */
    float q_align[4];                      /* K1→相机 Wahba 对齐四元数 (w,x,y,z) */
    float t_offset[3];                     /* K1→相机平移偏移 (m, 一般为 0) */
    bool  align_valid;                     /* 外参是否有效 */

    /* ── 相机内参 ── */
    float camera_matrix[9];                /* 行优先: [fx,0,cx, 0,fy,cy, 0,0,1] */
    int   img_width;
    int   img_height;

    /* ── 时间 ── */
    double last_prune_time;                /* 上次清理超时人员的时间 */

    /* ── 线程安全 ──
     * register_person()/prune_timeout() 在 postprocess 线程中写入,
     * project_person()/get_visible_persons() 在 viz 线程中读取.
     * 此锁保护 persons[] 数组. */
    pthread_mutex_t persons_mutex;

    /* ── CSV 输出 ── */
    FILE* csv_out;                         /* 世界坐标 CSV */
} WorldCoord;

/* ── 生命周期 ── */
WorldCoord* world_coord_create(K1Odometry* odom,
                                const float q_align[4],
                                const float camera_matrix[9],
                                int img_w, int img_h);
void world_coord_destroy(WorldCoord* wc);

/* ── 设置外参 ── */
void world_coord_set_alignment(WorldCoord* wc, const float q_align[4], const float t_offset[3]);

/* ── 每帧调用: 检测到人时注册/更新世界坐标 ──
 * @param k1_pos_at_capture  帧捕获时刻的 K1 位置 (NULL=内部查询当前位姿)
 * @param k1_quat_at_capture 帧捕获时刻的 K1 姿态四元数 (NULL=内部查询)
 * @param detection_bbox     检测框 (用于 per-keypoint 深度偏移, NULL=使用平面假设)
 * 返回 track_id (新注册或已存在的) */
int  world_coord_register_person(WorldCoord* wc, int track_id,
                                  const float P_cam[3],
                                  const float kpts_2d[17][2], int num_kpts,
                                  float depth, float confidence,
                                  double current_time,
                                  const float k1_pos_at_capture[3],
                                  const float k1_quat_at_capture[4],
                                  const BBox* detection_bbox);

/* ── 标记某人在当前帧可见 (用于清理超时判断) ── */
void world_coord_mark_visible(WorldCoord* wc, int track_id, double current_time);

/* ── 每帧调用: 将人的世界坐标投影到当前 K1 视角 ──
 * 返回 true 如果投影在画面内, false 表示在视野外或无效.
 * out_distance: K1→人距离 (m), out_height: 估计身高 (m), 均可为 NULL. */
bool world_coord_project_person(const WorldCoord* wc, int track_id,
                                 float out_2d_pixel[2],
                                 float out_distance[1],
                                 float out_height[1]);

/* ── 批量投影 17 个关键点到当前视角 ──
 * 返回成功投影的关键点数量 */
int  world_coord_project_skeleton(const WorldCoord* wc, int track_id,
                                   float out_2d_kpts[WORLD_MAX_SKELETON_KPTS][2]);

/* ── 获取所有活跃的人 (用于渲染循环) ── */
int  world_coord_get_visible_persons(const WorldCoord* wc, int* out_track_ids, int max_n);

/* ── 获取人的世界坐标 (只读) ── */
const PersonWorldState* world_coord_get_person(const WorldCoord* wc, int track_id);

/* ── 清理超时的人 ── */
void world_coord_prune_timeout(WorldCoord* wc, double current_time);

/* ── CSV 记录 ── */
void world_coord_start_recording(WorldCoord* wc, const char* dir, const char* session_id);
void world_coord_stop_recording(WorldCoord* wc);

#ifdef __cplusplus
}
#endif

#endif /* WORLD_COORD_H */
