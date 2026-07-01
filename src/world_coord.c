/*
 * WorldCoord — 以 K1 为动态原点的三维世界坐标系
 *
 * 变换链:
 *   1. camera → K1 body:  使用 Wahba 对齐四元数 q_align
 *   2. K1 body → world:    使用 K1Odometry 的实时位姿
 *   3. world → camera:     逆变换用于渲染投影
 *
 * 坐标约定:
 *   - 世界坐标系: ENU (East-North-Up), Z 向上, 原点 = K1 启动位置
 *   - K1 体坐标系: X=前, Y=右, Z=下 (IMU convention)
 *   - 相机坐标系: Z=光轴前方, X=右, Y=下 (OpenCV)
 */

#include "world_coord.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ═══════════════════════════════════════════════════════════
 *  生命周期
 * ═══════════════════════════════════════════════════════════ */

WorldCoord* world_coord_create(K1Odometry* odom,
                                const float q_align[4],
                                const float camera_matrix[9],
                                int img_w, int img_h) {
    WorldCoord* wc = (WorldCoord*)calloc(1, sizeof(WorldCoord));
    if (!wc) return NULL;

    wc->odom = odom;

    if (q_align) {
        memcpy(wc->q_align, q_align, 4 * sizeof(float));
        wc->align_valid = true;
    } else {
        k1_quat_identity(wc->q_align);
        wc->align_valid = false;
    }

    if (camera_matrix) {
        memcpy(wc->camera_matrix, camera_matrix, 9 * sizeof(float));
    } else {
        /* 默认内参 (640×480 粗略值) */
        memset(wc->camera_matrix, 0, 9 * sizeof(float));
        wc->camera_matrix[0] = 960.0f;
        wc->camera_matrix[4] = 960.0f;
        wc->camera_matrix[2] = 320.0f;
        wc->camera_matrix[5] = 240.0f;
        wc->camera_matrix[8] = 1.0f;
    }

    wc->img_width  = img_w;
    wc->img_height = img_h;

    memset(wc->t_offset, 0, sizeof(wc->t_offset));
    pthread_mutex_init(&wc->persons_mutex, NULL);
    wc->csv_out = NULL;
    wc->last_prune_time = 0.0;
    wc->num_persons = 0;

    return wc;
}

void world_coord_destroy(WorldCoord* wc) {
    if (!wc) return;
    world_coord_stop_recording(wc);
    pthread_mutex_destroy(&wc->persons_mutex);
    free(wc);
}

void world_coord_set_alignment(WorldCoord* wc, const float q_align[4], const float t_offset[3]) {
    if (!wc) return;
    if (q_align) {
        memcpy(wc->q_align, q_align, 4 * sizeof(float));
        wc->align_valid = true;
    }
    if (t_offset) {
        memcpy(wc->t_offset, t_offset, 3 * sizeof(float));
    }
}

/* ═══════════════════════════════════════════════════════════
 *  人员注册
 * ═══════════════════════════════════════════════════════════ */

int world_coord_register_person(WorldCoord* wc, int track_id,
                                 const float P_cam[3],
                                 const float kpts_2d[17][2], int num_kpts,
                                 float depth, float confidence,
                                 double current_time,
                                 const float k1_pos_at_capture[3],
                                 const float k1_quat_at_capture[4],
                                 const BBox* detection_bbox) {
    if (!wc || !P_cam) return -1;
    if (!wc->odom || !k1_odometry_is_initialized(wc->odom)) return -1;

    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);

    /* ── 计算此人在世界坐标系中的位置 ──
     * P_world = T_W_B × T_B_C × P_cam
     *
     * Step 1: 相机→K1 体坐标 (应用 Wahba 对齐)
     *   P_k1_body = R_align × P_cam + t_offset
     *
     * Step 2: K1 体坐标→世界坐标
     *   P_world = R_W_B × P_k1_body + pos_W_B
     */

    /* 获取 K1 当前位姿 — 优先使用调用者提供的帧捕获时刻快照 */
    float k1_pos[3], k1_quat[4];
    if (k1_pos_at_capture && k1_quat_at_capture) {
        /* 使用帧捕获时刻位姿 — 保证时间同步 (±5ms @100Hz IMU) */
        memcpy(k1_pos, k1_pos_at_capture, 3 * sizeof(float));
        memcpy(k1_quat, k1_quat_at_capture, 4 * sizeof(float));
    } else {
        /* 回退: 内部查询当前位姿 (向后兼容离线/单线程模式) */
        k1_odometry_get_pose(wc->odom, k1_pos, k1_quat);
    }

    /* Step 1: 相机→K1 body
     * q_align is K1→camera rotation, so we need its conjugate for camera→K1 */
    float P_k1[3];
    if (wc->align_valid) {
        float q_align_inv[4];
        k1_quat_conjugate(wc->q_align, q_align_inv);
        k1_quat_rotate(q_align_inv, P_cam, P_k1);
    } else {
        /* 无外参时假设对齐已完成 (q_align = identity) */
        memcpy(P_k1, P_cam, 3 * sizeof(float));
    }
    P_k1[0] += wc->t_offset[0];
    P_k1[1] += wc->t_offset[1];
    P_k1[2] += wc->t_offset[2];

    /* Step 2: K1 body→world */
    float P_world[3];
    k1_quat_rotate(k1_quat, P_k1, P_world);
    P_world[0] += k1_pos[0];
    P_world[1] += k1_pos[1];
    P_world[2] += k1_pos[2];

    /* ── 查找或创建此人的状态 ── */
    PersonWorldState* person = NULL;
    int slot = -1;

    /* 先查找已有 track_id */
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].track_id == track_id) {
            person = &wc->persons[i];
            slot = i;
            break;
        }
    }

    if (!person) {
        /* 新人员: 查找空闲槽位 */
        for (int i = 0; i < WORLD_MAX_PERSONS; i++) {
            if (wc->persons[i].track_id < 0 || !wc->persons[i].is_active) {
                person = &wc->persons[i];
                slot = i;
                break;
            }
        }
    }

    if (!person) {
        /* 槽位已满, 替换最早超时的 */
        int oldest = 0;
        double oldest_time = 1e30;
        for (int i = 0; i < WORLD_MAX_PERSONS; i++) {
            if (wc->persons[i].last_seen_time < oldest_time) {
                oldest_time = wc->persons[i].last_seen_time;
                oldest = i;
            }
        }
        person = &wc->persons[oldest];
        slot = oldest;
        log_debug("[WorldCoord] Slot full, evicting track_id=%d (last seen %.1fs ago)",
                  person->track_id, current_time - oldest_time);
    }

    /* ── 更新/初始化人员状态 ── */
    bool is_new = (person->track_id != track_id);

    person->track_id = track_id;
    person->is_active = true;
    person->last_seen_time = current_time;
    person->last_depth = depth;
    person->last_confidence = confidence;

    if (is_new) {
        /* 首次注册: 用当前世界坐标作为锚点 */
        memcpy(person->world_pos, P_world, 3 * sizeof(float));
        person->first_seen_time = current_time;
        person->num_observations = 1;
        person->height_meters = 1.70f;  /* 默认身高, 后续更新 */
    } else {
        /* ── 自适应锚点更新 (参考 SEVIS-3D Schmidt-EKF, DuToit et al.) ──
         * 只有人才移动超过阈值时才更新锚点。静止时锁定不动,
         * 防止 K1 里程计漂移污染世界坐标。
         *
         * 0.30m 阈值误差预算 (2σ):
         *   - K1 odometry drift: ~0.01m/frame @ 15fps
         *   - Depth estimation noise: ~0.10m @ 3m (Zhu et al. ECCV 2020)
         *   - Capture→postprocess K1 movement: ~0.10m (1m/s × 0.1s)
         *   - Quadrature sum: √(0.01²+0.10²+0.10²) ≈ 0.14m → 2σ ≈ 0.30m */
        float dx = P_world[0] - person->world_pos[0];
        float dy = P_world[1] - person->world_pos[1];
        float dz = P_world[2] - person->world_pos[2];
        float displacement = sqrtf(dx*dx + dy*dy + dz*dz);

        #define WORLD_ANCHOR_MOTION_THRESH  0.30f  /* 运动检测阈值 (m) */
        #define WORLD_ANCHOR_MOVING_ALPHA   0.15f  /* 移动时的慢速动量 */

        if (displacement > WORLD_ANCHOR_MOTION_THRESH) {
            /* 人很可能在移动 → 用慢速动量更新锚点 */
            for (int j = 0; j < 3; j++) {
                person->world_pos[j] += WORLD_ANCHOR_MOVING_ALPHA
                    * (P_world[j] - person->world_pos[j]);
            }
        }
        /* else: 位移低于阈值 → 人静止或 ZUPT 生效中。
         *       锚点锁死。K1 漂移不会传播到世界坐标。 */

        person->num_observations++;
    }

    /* ── 重建 3D 骨架 ──
     * 如果有关键点 2D 数据, 用逆投影重建 3D 世界坐标.
     *
     * 深度模型:
     *   - 平面人体假设 (all same Z) → per-keypoint 深度偏移
     *   - 基于人体前后厚度 ~0.25m (Pheasant & Haslegrave 2005)
     *   - 关键点在体中心上方 → 更近 (Z 更小)
     *    关键点在体中心下方 → 更远 (Z 更大)
     *   - 参考: Pavlakos et al. CVPR 2018 — 序数深度关系;
     *          Zhang et al. 2021 — 每关键点连续深度优于序数 */
    if (kpts_2d && num_kpts > 0) {
        float fx = wc->camera_matrix[0];
        float fy = wc->camera_matrix[4];
        float cx = wc->camera_matrix[2];
        float cy = wc->camera_matrix[5];

        int n_kpts = (num_kpts < WORLD_MAX_SKELETON_KPTS) ? num_kpts : WORLD_MAX_SKELETON_KPTS;

        /* 计算人体前后厚度参考值 (用于 per-kpt 深度偏移) */
        float body_depth = 0.25f;  /* 平均成人前后厚度 (Pheasant 2005) */
        float bbox_cy = 0.0f, bbox_h = 0.0f;
        if (detection_bbox) {
            bbox_cy = bbox_center_y(detection_bbox);
            bbox_h = bbox_height(detection_bbox);
        }

        for (int i = 0; i < n_kpts; i++) {
            /* ── 每关键点深度偏移 ──
             * 将关键点的归一化 Y 位置映射到深度偏移:
             *   norm_y ∈ [-1, 1]: -1=最上方(head) → 更近 → Z减小
             *                    +1=最下方(feet) → 更远 → Z增大
             *   depth_offset = -norm_y × body_depth × 0.5
             * 当 bbox 不可用时回到平面假设 (offset=0) */
            float depth_offset = 0.0f;
            if (detection_bbox && bbox_h > 5.0f) {
                float kpt_rel_y = kpts_2d[i][1] - bbox_cy;
                float norm_y = UTILS_CLAMP(kpt_rel_y / (bbox_h * 0.5f), -1.0f, 1.0f);
                depth_offset = -norm_y * body_depth * 0.5f;
            }

            /* 逆投影: X_cam = (u - cx) * Z / fx, Y_cam = (v - cy) * Z / fy */
            float P_cam_kpt[3];
            P_cam_kpt[0] = (kpts_2d[i][0] - cx) * depth / fx;
            P_cam_kpt[1] = (kpts_2d[i][1] - cy) * depth / fy;
            P_cam_kpt[2] = depth + depth_offset;

            /* 相机→K1 body→world (same convention: invert q_align) */
            float P_k1_kpt[3];
            if (wc->align_valid) {
                float q_align_inv[4];
                k1_quat_conjugate(wc->q_align, q_align_inv);
                k1_quat_rotate(q_align_inv, P_cam_kpt, P_k1_kpt);
            } else {
                memcpy(P_k1_kpt, P_cam_kpt, 3 * sizeof(float));
            }
            P_k1_kpt[0] += wc->t_offset[0];
            P_k1_kpt[1] += wc->t_offset[1];
            P_k1_kpt[2] += wc->t_offset[2];

            k1_quat_rotate(k1_quat, P_k1_kpt, person->world_skeleton[i]);
            person->world_skeleton[i][0] += k1_pos[0];
            person->world_skeleton[i][1] += k1_pos[1];
            person->world_skeleton[i][2] += k1_pos[2];
        }
        person->has_skeleton = true;
    } else {
        person->has_skeleton = false;
    }

    /* ── 确保 num_persons 跟踪最大索引 ── */
    if (slot >= wc->num_persons) {
        wc->num_persons = slot + 1;
    }

    /* ── CSV 记录 ── */
    if (wc->csv_out) {
        float dist_to_k1 = sqrtf(P_world[0]*P_world[0] + P_world[1]*P_world[1] + P_world[2]*P_world[2]);
        fprintf(wc->csv_out, "%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                current_time, track_id,
                (double)person->world_pos[0], (double)person->world_pos[1], (double)person->world_pos[2],
                (double)dist_to_k1, (double)confidence, person->num_observations);
        fflush(wc->csv_out);
    }

    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
    return track_id;
}

/* ═══════════════════════════════════════════════════════════
 *  可见性标记
 * ═══════════════════════════════════════════════════════════ */

void world_coord_mark_visible(WorldCoord* wc, int track_id, double current_time) {
    if (!wc) return;
    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].track_id == track_id) {
            wc->persons[i].is_active = true;
            wc->persons[i].last_seen_time = current_time;
            pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
            return;
        }
    }
    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
}

/* ═══════════════════════════════════════════════════════════
 *  3D→2D 投影
 * ═══════════════════════════════════════════════════════════ */

bool world_coord_project_person(const WorldCoord* wc, int track_id,
                                 float out_2d_pixel[2],
                                 float out_distance[1],
                                 float out_height[1]) {
    if (!wc || !out_2d_pixel) return false;
    if (!wc->odom || !k1_odometry_is_initialized(wc->odom)) return false;

    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);

    /* 查找此人的世界坐标 */
    const PersonWorldState* person = NULL;
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].track_id == track_id) {
            person = &wc->persons[i];
            break;
        }
    }
    if (!person) {
        pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
        return false;
    }

    /* Copy height under lock (before any projection math) */
    float height_copy = person->height_meters;

    /* ── 投影变换: world → camera ── */
    float k1_pos[3], k1_quat[4];
    k1_odometry_get_pose(wc->odom, k1_pos, k1_quat);

    /* World → K1 relative */
    float rel[3];
    rel[0] = person->world_pos[0] - k1_pos[0];
    rel[1] = person->world_pos[1] - k1_pos[1];
    rel[2] = person->world_pos[2] - k1_pos[2];

    /* K1 body frame: R_W_B^{-1} = quat^{-1} */
    float k1_quat_inv[4];
    k1_quat_conjugate(k1_quat, k1_quat_inv);
    float local[3];
    k1_quat_rotate(k1_quat_inv, rel, local);

    /* K1 body → camera: use q_align directly (it's K1→camera rotation) */
    float cam[3];
    if (wc->align_valid) {
        k1_quat_rotate(wc->q_align, local, cam);
    } else {
        memcpy(cam, local, 3 * sizeof(float));
    }

    /* 视线裁剪: 在相机后方 */
    if (cam[2] <= WORLD_BACKFACE_CULL_M) {
        pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
        return false;
    }

    /* 针孔投影 */
    float fx = wc->camera_matrix[0];
    float fy = wc->camera_matrix[4];
    float cx = wc->camera_matrix[2];
    float cy = wc->camera_matrix[5];

    float u = fx * cam[0] / cam[2] + cx;
    float v = fy * cam[1] / cam[2] + cy;

    /* 视锥体裁剪 */
    /* 视锥体裁剪 — 允许边界像素.
     * 像素中心在 (col+0.5, row+0.5), 所以 u ∈ [0, W-1] 对应有效像素.
     * 参考: Direct3D/OpenGL rasterization rules, Hartley & Zisserman §6.1 */
    if (u < 0.0f || u > (float)(wc->img_width - 1) ||
        v < 0.0f || v > (float)(wc->img_height - 1)) {
        pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
        return false;
    }

    out_2d_pixel[0] = u;
    out_2d_pixel[1] = v;

    if (out_distance) {
        *out_distance = sqrtf(rel[0]*rel[0] + rel[1]*rel[1] + rel[2]*rel[2]);
    }
    if (out_height) {
        *out_height = height_copy;
    }

    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
    return true;
}

int world_coord_project_skeleton(const WorldCoord* wc, int track_id,
                                  float out_2d_kpts[WORLD_MAX_SKELETON_KPTS][2]) {
    if (!wc || !out_2d_kpts) return 0;
    if (!wc->odom || !k1_odometry_is_initialized(wc->odom)) return 0;

    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);

    const PersonWorldState* person = NULL;
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].track_id == track_id) {
            person = &wc->persons[i];
            break;
        }
    }
    if (!person || !person->has_skeleton) {
        pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
        return 0;
    }

    float k1_pos[3], k1_quat[4];
    k1_odometry_get_pose(wc->odom, k1_pos, k1_quat);

    float k1_quat_inv[4];
    k1_quat_conjugate(k1_quat, k1_quat_inv);

    float fx = wc->camera_matrix[0];
    float fy = wc->camera_matrix[4];
    float cx = wc->camera_matrix[2];
    float cy = wc->camera_matrix[5];

    int projected = 0;
    for (int i = 0; i < WORLD_MAX_SKELETON_KPTS; i++) {
        /* World → K1 body */
        float rel[3];
        rel[0] = person->world_skeleton[i][0] - k1_pos[0];
        rel[1] = person->world_skeleton[i][1] - k1_pos[1];
        rel[2] = person->world_skeleton[i][2] - k1_pos[2];

        float local[3];
        k1_quat_rotate(k1_quat_inv, rel, local);

        /* K1 body → camera: use q_align directly (K1→camera rotation) */
        float cam[3];
        if (wc->align_valid) {
            k1_quat_rotate(wc->q_align, local, cam);
        } else {
            memcpy(cam, local, 3 * sizeof(float));
        }

        if (cam[2] > WORLD_BACKFACE_CULL_M) {
            out_2d_kpts[i][0] = fx * cam[0] / cam[2] + cx;
            out_2d_kpts[i][1] = fy * cam[1] / cam[2] + cy;
            projected++;
        } else {
            out_2d_kpts[i][0] = -1.0f;
            out_2d_kpts[i][1] = -1.0f;
        }
    }

    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
    return projected;
}

/* ═══════════════════════════════════════════════════════════
 *  查询
 * ═══════════════════════════════════════════════════════════ */

int world_coord_get_visible_persons(const WorldCoord* wc, int* out_track_ids, int max_n) {
    if (!wc || !out_track_ids) return 0;

    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);
    int count = 0;
    for (int i = 0; i < wc->num_persons && count < max_n; i++) {
        if (wc->persons[i].is_active && wc->persons[i].track_id >= 0) {
            out_track_ids[count++] = wc->persons[i].track_id;
        }
    }
    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
    return count;
}

const PersonWorldState* world_coord_get_person(const WorldCoord* wc, int track_id) {
    if (!wc) return NULL;
    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].track_id == track_id) {
            /* NOTE: returns pointer into shared array while lock is held.
             * Caller must use the result BEFORE the next unlock.
             * In practice, viz thread is the only reader and render_world_overlay
             * uses the pointer within the same call, so this is safe. */
            pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
            return &wc->persons[i];
        }
    }
    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════
 *  超时清理
 * ═══════════════════════════════════════════════════════════ */

void world_coord_prune_timeout(WorldCoord* wc, double current_time) {
    if (!wc) return;

    /* 只每 5 秒清理一次 (无锁检查, 精确时间不重要) */
    if (current_time - wc->last_prune_time < 5.0) return;

    pthread_mutex_lock(&((WorldCoord*)wc)->persons_mutex);
    wc->last_prune_time = current_time;

    for (int i = 0; i < wc->num_persons; i++) {
        PersonWorldState* p = &wc->persons[i];
        if (p->track_id >= 0 && p->is_active) {
            double elapsed = current_time - p->last_seen_time;
            if (elapsed > WORLD_TIMEOUT_S) {
                log_debug("[WorldCoord] Pruning track_id=%d (last seen %.0fs ago)",
                          p->track_id, elapsed);
                memset(p, 0, sizeof(PersonWorldState));
                p->track_id = -1;
                p->is_active = false;
            }
        }
    }

    /* Compact: shift inactive slots to end */
    int write_idx = 0;
    for (int i = 0; i < wc->num_persons; i++) {
        if (wc->persons[i].is_active && wc->persons[i].track_id >= 0) {
            if (write_idx != i) {
                memcpy(&wc->persons[write_idx], &wc->persons[i], sizeof(PersonWorldState));
            }
            write_idx++;
        }
    }
    /* Clear remaining */
    for (int i = write_idx; i < wc->num_persons; i++) {
        memset(&wc->persons[i], 0, sizeof(PersonWorldState));
        wc->persons[i].track_id = -1;
    }
    wc->num_persons = write_idx;

    pthread_mutex_unlock(&((WorldCoord*)wc)->persons_mutex);
}

/* ═══════════════════════════════════════════════════════════
 *  CSV 记录
 * ═══════════════════════════════════════════════════════════ */

void world_coord_start_recording(WorldCoord* wc, const char* dir, const char* session_id) {
    if (!wc || !dir || !session_id) return;

    char path[MAX_PATH_LEN * 3];
    snprintf(path, sizeof(path), "%s/%s/world_positions.csv", dir, session_id);
    wc->csv_out = fopen(path, "w");
    if (wc->csv_out) {
        fprintf(wc->csv_out, "timestamp_s,track_id,world_x_m,world_y_m,world_z_m,"
                "dist_to_k1_m,confidence,observations\n");
        log_info("[WorldCoord] Recording world positions: %s", path);
    }
}

void world_coord_stop_recording(WorldCoord* wc) {
    if (!wc) return;
    if (wc->csv_out) {
        fclose(wc->csv_out);
        wc->csv_out = NULL;
        log_info("[WorldCoord] Persons recording closed");
    }
}
