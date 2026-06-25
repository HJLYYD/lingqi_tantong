#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "system_controller.h"
#include "logger.h"
#include "utils.h"
#include "k1_platform.h"
#include "k1_imu.h"
#include "video_processor.h"
#include "ort_common.h"
#include "spacemit_ort_bridge.h"
#include "stgcn_action_recognizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <errno.h>

static void associate_poses_with_objects(TrackedObject* objects, int num_objects,
                                          PoseEstimation* poses, int num_poses);
static void associate_faces_with_objects(TrackedObject* objects, int num_objects,
                                          FaceIdentity* faces, int num_faces);
static float get_current_fps(const SystemController* sc);

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#define K1_RING_SIZE        4
#define K1_MAX_PIPELINE_THREADS 8
#define K1_PIPELINE_NUM_HEARTS   6   /* capture, inference, stgcn, postprocess, viz, output */

typedef struct {
    uint8_t* rgb_data;
    size_t data_size;
    int width;
    int height;
    int64_t timestamp_us;
    int frame_index;
    bool has_frame;

    InferenceResult inference;
    bool has_inference;

    TrackingResult tracking;
    SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
    int num_positions;
    bool has_tracking;

    int ref_count;
    pthread_mutex_t mutex;
    pthread_cond_t avail_cond;
} K1PipelineSlot;

typedef struct {
    K1Platform* platform;
    K1PipelineSlot slots[K1_RING_SIZE];
    int capture_idx;
    int infer_idx;
    int post_idx;
    int active_slots;

    pthread_mutex_t ring_mutex;
    pthread_cond_t capture_done_cond;
    pthread_cond_t infer_done_cond;
    pthread_cond_t track_done_cond;
    pthread_cond_t post_done_cond;

    pthread_t threads[K1_MAX_PIPELINE_THREADS];
    int num_threads;
    volatile bool running;
    volatile bool shutdown;

    /* ── Heartbeat monitoring ──
     * Each worker thread bumps its heartbeat every loop iteration.
     * Main thread checks hearts are still beating; if a thread dies,
     * it shuts down the pipeline to prevent deadlocks. */
    volatile int thread_heartbeats[K1_PIPELINE_NUM_HEARTS];  /* [0]=capture, [1]=inference, [2]=stgcn, [3]=postprocess, [4]=viz */
    volatile bool thread_alive[K1_PIPELINE_NUM_HEARTS];

    SystemController* controller;
} K1Pipeline;

typedef struct {
    K1Pipeline* pipeline;
    int thread_id;
    int cpu_core;
    const char* name;
} K1ThreadArg;

static bool k1_ring_init_slots(K1Pipeline* pl, int width, int height) {
    size_t sz = (size_t)width * height * 3;
    for (int i = 0; i < K1_RING_SIZE; i++) {
        K1PipelineSlot* slot = &pl->slots[i];
        slot->rgb_data = (uint8_t*)calloc(1, sz);
        if (!slot->rgb_data) return false;
        slot->data_size = sz;
        slot->width = width;
        slot->height = height;
        slot->has_frame = false;
        slot->has_inference = false;
        slot->has_tracking = false;
        slot->ref_count = 0;
        pthread_mutex_init(&slot->mutex, NULL);
        pthread_cond_init(&slot->avail_cond, NULL);
        inference_result_init(&slot->inference);
        memset(&slot->tracking, 0, sizeof(slot->tracking));
        memset(slot->positions, 0, sizeof(slot->positions));
        slot->num_positions = 0;
    }
    return true;
}

static void k1_ring_destroy_slots(K1Pipeline* pl) {
    for (int i = 0; i < K1_RING_SIZE; i++) {
        K1PipelineSlot* slot = &pl->slots[i];
        free(slot->rgb_data);
        slot->rgb_data = NULL;
        pthread_mutex_destroy(&slot->mutex);
        pthread_cond_destroy(&slot->avail_cond);
    }
}

static void k1_pipeline_init(K1Pipeline* pl, SystemController* sc) {
    memset(pl, 0, sizeof(*pl));
    pl->platform = k1_platform_init();
    pl->controller = sc;
    pl->capture_idx = 0;
    pl->infer_idx = 0;
    pl->post_idx = 0;
    pl->active_slots = 0;
    pl->running = true;
    pl->shutdown = false;
    pthread_mutex_init(&pl->ring_mutex, NULL);
    pthread_cond_init(&pl->capture_done_cond, NULL);
    pthread_cond_init(&pl->infer_done_cond, NULL);
    pthread_cond_init(&pl->track_done_cond, NULL);
    pthread_cond_init(&pl->post_done_cond, NULL);
}

static void k1_pipeline_destroy(K1Pipeline* pl) {
    /* Set shutdown UNDER the mutex so waiting threads atomically see the new
     * predicate.  Broadcast after unlock is safe; the mutex provides the
     * memory barrier for the flag write. */
    pthread_mutex_lock(&pl->ring_mutex);
    pl->running = false;
    pl->shutdown = true;
    pthread_cond_broadcast(&pl->capture_done_cond);
    pthread_cond_broadcast(&pl->infer_done_cond);
    pthread_cond_broadcast(&pl->track_done_cond);
    pthread_cond_broadcast(&pl->post_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);

    for (int i = 0; i < pl->num_threads; i++) {
        pthread_join(pl->threads[i], NULL);
    }

    k1_ring_destroy_slots(pl);
    pthread_mutex_destroy(&pl->ring_mutex);
    pthread_cond_destroy(&pl->capture_done_cond);
    pthread_cond_destroy(&pl->infer_done_cond);
    pthread_cond_destroy(&pl->track_done_cond);
    pthread_cond_destroy(&pl->post_done_cond);
}

static int k1_pipeline_slot_acquire(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
    while (pl->active_slots >= K1_RING_SIZE && !pl->shutdown) {
        pthread_cond_wait(&pl->post_done_cond, &pl->ring_mutex);
    }
    if (pl->shutdown) {
        pthread_mutex_unlock(&pl->ring_mutex);
        return -1;
    }
    int slot = pl->capture_idx;
    pl->capture_idx = (pl->capture_idx + 1) % K1_RING_SIZE;
    pl->active_slots++;
    pthread_mutex_unlock(&pl->ring_mutex);
    return slot;
}

static void k1_pipeline_slot_capture_done(K1Pipeline* pl, int slot) {
    K1PipelineSlot* s = &pl->slots[slot];
    pthread_mutex_lock(&s->mutex);
    s->has_frame = true;
    pthread_cond_signal(&s->avail_cond);
    pthread_mutex_unlock(&s->mutex);

    /* Full memory barrier: ensures has_frame is visible to threads
     * scanning under ring_mutex on RISC-V (weak memory ordering). */
    __sync_synchronize();

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_signal(&pl->capture_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static int k1_pipeline_slot_wait_captured(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
    while (pl->active_slots == 0 && !pl->shutdown) {
        pthread_cond_wait(&pl->capture_done_cond, &pl->ring_mutex);
    }
    if (pl->shutdown) {
        pthread_mutex_unlock(&pl->ring_mutex);
        return -1;
    }

    __sync_synchronize();  /* RISC-V: see has_frame stores from capture_done */

    for (int i = 0; i < K1_RING_SIZE; i++) {
        int idx = (pl->infer_idx + i) % K1_RING_SIZE;
        K1PipelineSlot* s = &pl->slots[idx];
        if (s->has_frame && !s->has_inference) {
            pl->infer_idx = idx;
            pthread_mutex_unlock(&pl->ring_mutex);
            return idx;
        }
    }
    pthread_mutex_unlock(&pl->ring_mutex);
    return -1;
}

static void k1_pipeline_slot_infer_done(K1Pipeline* pl, int slot) {
    K1PipelineSlot* s = &pl->slots[slot];
    pthread_mutex_lock(&s->mutex);
    s->has_inference = true;
    pthread_cond_signal(&s->avail_cond);
    pthread_mutex_unlock(&s->mutex);

    __sync_synchronize();  /* RISC-V weak-memory barrier */

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_signal(&pl->infer_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static int k1_pipeline_slot_wait_inferred(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
    __sync_synchronize();  /* RISC-V: see has_inference stores from infer_done */
    while (!pl->shutdown) {
        for (int i = 0; i < K1_RING_SIZE; i++) {
            int idx = (pl->post_idx + i) % K1_RING_SIZE;
            K1PipelineSlot* s = &pl->slots[idx];
            if (s->has_inference && !s->has_tracking) {
                pl->post_idx = idx;
                pthread_mutex_unlock(&pl->ring_mutex);
                return idx;
            }
        }
        pthread_cond_wait(&pl->infer_done_cond, &pl->ring_mutex);
    }
    pthread_mutex_unlock(&pl->ring_mutex);
    return -1;
}

static void k1_pipeline_slot_tracking_done(K1Pipeline* pl, int slot) {
    K1PipelineSlot* s = &pl->slots[slot];
    pthread_mutex_lock(&s->mutex);
    s->has_tracking = true;
    pthread_cond_signal(&s->avail_cond);
    pthread_mutex_unlock(&s->mutex);

    __sync_synchronize();  /* RISC-V weak-memory barrier */

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_signal(&pl->track_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static int k1_pipeline_slot_wait_tracked(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
    __sync_synchronize();  /* RISC-V: see has_tracking stores from tracking_done */
    while (!pl->shutdown) {
        for (int i = 0; i < K1_RING_SIZE; i++) {
            int idx = (pl->post_idx + i) % K1_RING_SIZE;
            K1PipelineSlot* s = &pl->slots[idx];
            if (s->has_tracking) {
                pl->post_idx = idx;
                pthread_mutex_unlock(&pl->ring_mutex);
                return idx;
            }
        }
        pthread_cond_wait(&pl->track_done_cond, &pl->ring_mutex);
    }
    pthread_mutex_unlock(&pl->ring_mutex);
    return -1;
}

static void k1_pipeline_slot_release(K1Pipeline* pl, int slot) {
    K1PipelineSlot* s = &pl->slots[slot];
    pthread_mutex_lock(&s->mutex);
    s->has_frame = false;
    s->has_inference = false;
    s->has_tracking = false;
    inference_result_init(&s->inference);
    memset(&s->tracking, 0, sizeof(s->tracking));
    s->num_positions = 0;
    pthread_mutex_unlock(&s->mutex);

    pthread_mutex_lock(&pl->ring_mutex);
    pl->active_slots--;
    pthread_cond_signal(&pl->post_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static void k1_apply_rt_priority(const char* role) {
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = 50;
    int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    if (rc == 0) {
        log_info("[K1 Pipeline] %s thread elevated to SCHED_FIFO prio 50", role);
    } else if (rc == EPERM) {
        log_debug("[K1 Pipeline] %s: SCHED_FIFO unavailable (no CAP_SYS_NICE), using SCHED_OTHER", role);
    } else {
        log_warning("[K1 Pipeline] %s: pthread_setschedparam failed: %d", role, rc);
    }
}

static void* k1_capture_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("Capture");
    pl->thread_alive[0] = true;
    log_info("[K1 Pipeline] Capture thread on CPU %d", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;

    bool use_v4l2 = (sc->video_processor != NULL &&
                    video_processor_get_source_type(sc->video_processor) == VP_SOURCE_CAMERA &&
                    video_processor_is_opened(sc->video_processor));
    bool use_coap = (sc->coap_receiver != NULL);
    if (!use_v4l2 && !use_coap) {
        log_error("[K1 Pipeline] Capture: no source available (no V4L2 camera or CoAP receiver)");
        return NULL;
    }

    double last_frame_time = k1_get_time_ms() / 1000.0;
    int cap_attempts = 0, cap_coap_ok = 0;
    int consecutive_decode_fails = 0;
    int consecutive_frame_misses = 0;
    (void)cap_coap_ok; (void)consecutive_frame_misses;

    while (pl->running) {
        pl->thread_heartbeats[0]++;

        /* ── Check shutdown BEFORE any blocking call ── */
        if (pl->shutdown) break;

        int slot = k1_pipeline_slot_acquire(pl);
        if (slot < 0) break;

        K1PipelineSlot* s = &pl->slots[slot];
        bool got_frame = false;
        cap_attempts++;

        if (use_v4l2) {
            FrameData* fd = video_processor_read_frame_raw(sc->video_processor);
            if (fd && fd->data && fd->width == cam_w && fd->height == cam_h) {
                memcpy(s->rgb_data, fd->data, (size_t)cam_w * cam_h * 3);
                s->timestamp_us = (int64_t)(fd->timestamp * 1e6);
                s->frame_index = fd->frame_index;
                got_frame = true;
                frame_data_destroy(fd);
            } else if (fd) {
                if (fd->data && fd->width > 0 && fd->height > 0) {
                    utils_resize_image(fd->data, fd->width, fd->height,
                                       s->rgb_data, cam_w, cam_h, 3);
                    s->timestamp_us = (int64_t)(fd->timestamp * 1e6);
                    s->frame_index = fd->frame_index;
                    got_frame = true;
                }
                frame_data_destroy(fd);
            }
        }

        /* ── K1 本地 IMU: 读取并馈送到 Madgwick 滤波器 ── */
        {
            IMUHandler* imuh = sc->imu_handler;
            if (imuh && imuh->k1_imu) {
                K1Imu* ki = (K1Imu*)imuh->k1_imu;
                IMUData k1_raw;
                if (k1_imu_read_sample(ki, &k1_raw)) {
                    imu_handler_feed_k1_imu(imuh, &k1_raw);
                }
            }
        }

        if (!got_frame && use_coap) {
            ArrowSourceFrame coap_frame;
            if (coap_receiver_get_latest_frame(sc->coap_receiver, &coap_frame)) {
                cap_coap_ok++;
                if (soft_jpeg_decode_to_rgb(coap_frame.jpeg_data, coap_frame.jpeg_len,
                                            s->rgb_data, cam_w, cam_h) == 0) {
                    got_frame = true;
                    s->timestamp_us = (int64_t)coap_frame.timestamp;
                    s->frame_index = coap_frame.frame_index;
                    consecutive_decode_fails = 0;
                } else {
                    consecutive_decode_fails++;
                    if (consecutive_decode_fails <= 3 || consecutive_decode_fails % 10 == 0) {
                        log_warning("[K1 Pipeline] CoAP decode failed frame#%d: %zu bytes "
                                    "(%d consecutive fails — check WiFi signal / ESP32 stability)",
                                    coap_frame.frame_index, coap_frame.jpeg_len,
                                    consecutive_decode_fails);
                    }
                }

                if (coap_frame.has_pose) {
                    imu_handler_set_external_pose(sc->imu_handler,
                        coap_frame.pose.qw, coap_frame.pose.qx,
                        coap_frame.pose.qy, coap_frame.pose.qz,
                        coap_frame.pose.pitch, coap_frame.pose.roll,
                        coap_frame.pose.yaw, coap_frame.pose.altitude_m,
                        coap_frame.pose.temperature_c,
                        coap_frame.pose.timestamp_ms);
                }
            }
        }

        if (!got_frame) {
            consecutive_frame_misses++;
            if (cap_attempts == 1) {
                log_info("[Pipeline] Waiting for first frame (CoAP=%d V4L2=%d)...",
                         use_coap, use_v4l2);
            } else if (cap_attempts <= 3 || cap_attempts % 500 == 0) {
                double idle_s = (k1_get_time_ms() / 1000.0) - last_frame_time;
                log_info("[Pipeline] No frame: %d attempts, %.0fs idle",
                         cap_attempts, idle_s);
            }
            /* ── Frame timeout check ── */
            double now_s = k1_get_time_ms() / 1000.0;
            double idle_s = now_s - last_frame_time;
            if (idle_s > sc->frame_timeout_s) {
                log_info("[K1 Pipeline] No frames received for %.0fs (timeout=%ds), shutting down",
                         idle_s, sc->frame_timeout_s);
                sc->running = 0;
                pthread_mutex_lock(&pl->ring_mutex);
                pl->running = false;
                pl->shutdown = true;
                pthread_cond_broadcast(&pl->capture_done_cond);
                pthread_cond_broadcast(&pl->infer_done_cond);
                pthread_cond_broadcast(&pl->track_done_cond);
                pthread_cond_broadcast(&pl->post_done_cond);
                pl->active_slots--;
                pl->capture_idx = (pl->capture_idx - 1 + K1_RING_SIZE) % K1_RING_SIZE;
                pthread_cond_signal(&pl->post_done_cond);
                pthread_mutex_unlock(&pl->ring_mutex);
                break;
            }
            pthread_mutex_lock(&pl->ring_mutex);
            pl->active_slots--;
            pl->capture_idx = (pl->capture_idx - 1 + K1_RING_SIZE) % K1_RING_SIZE;
            pthread_cond_signal(&pl->post_done_cond);
            pthread_mutex_unlock(&pl->ring_mutex);
            struct timespec ts = {0, 5000000L};
            nanosleep(&ts, NULL);
            continue;
        }

        last_frame_time = k1_get_time_ms() / 1000.0;
        consecutive_frame_misses = 0;
        k1_pipeline_slot_capture_done(pl, slot);
    }

    return NULL;
}

static void* k1_inference_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("Inference");
    pl->thread_alive[1] = true;
    log_info("[K1 Pipeline] Inference thread on CPU %d (Cluster0)", ta->cpu_core);

    while (pl->running) {
        pl->thread_heartbeats[1]++;
        int slot = k1_pipeline_slot_wait_captured(pl);
        if (slot < 0) {
            if (pl->shutdown) break;
            usleep(1000);
            continue;
        }

        K1PipelineSlot* s = &pl->slots[slot];
        inference_pipeline_process_frame(
            sc->inference_pipeline, s->rgb_data, s->width, s->height, &s->inference);

        k1_pipeline_slot_infer_done(pl, slot);
    }

    return NULL;
}

/*
 * ── K1 ST-GCN async thread (CPU 2, Cluster0) ──
 *
 * ST-GCN 使用 CPU EP (FP32 模型, 未量化), 不占用 TCM。
 * 在独立核心上异步运行, 与 YOLO/Face 推理 (CPU 1) 并行:
 *   CPU 1: pose + face detect + face recog (SpacemiT EP, ~200ms)
 *   CPU 2: ST-GCN action recognition (CPU EP, FP32, ~400ms every 50 frames)
 *
 * 唤醒间隔 = action_inference_interval 帧 (config: 50 frames ≈ 12s at 4 FPS)
 * 单次推理 ~400ms, 在间隔内安全完成。
 */
static void* k1_stgcn_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("ST-GCN");
    pl->thread_alive[2] = true;
    log_info("[K1 Pipeline] ST-GCN thread on CPU %d (Cluster0)", ta->cpu_core);

    int interval = 10; /* default, updated below */
    {
        AIInferencePipeline* ip = sc->inference_pipeline;
        if (ip && ip->action_recognizer) {
            interval = ip->action_inference_interval;
        }
    }
    log_info("[K1 Pipeline] ST-GCN async interval: %d frames", interval);

    int last_run_frame = 0;
    while (pl->running) {
        pl->thread_heartbeats[2]++;

        /* Check if it's time to run (based on frame count) */
        int current_frame = sc->frame_count;
        if (current_frame > 0 &&
            current_frame - last_run_frame >= interval &&
            sc->inference_pipeline &&
            sc->inference_pipeline->action_recognizer) {
            last_run_frame = current_frame;
            stgcn_action_recognizer_run_async(
                sc->inference_pipeline->action_recognizer);
        }

        /* Sleep 5ms between checks — ST-GCN runs infrequently,
         * no need to spin tight. */
        usleep(5000);
    }

    return NULL;
}

static void* k1_postprocess_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("PostProcess");
    pl->thread_alive[3] = true;
    log_info("[K1 Pipeline] Post-process thread on CPU %d (Cluster0)", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;
    float cam_fps = 15.0f;

    while (pl->running) {
        pl->thread_heartbeats[3]++;
        int slot = k1_pipeline_slot_wait_inferred(pl);
        if (slot < 0) break;

        K1PipelineSlot* s = &pl->slots[slot];
        sc->frame_count++;

        InferenceResult* inference = &s->inference;

        if (sc->frame_count == 1 && inference->num_detections > 0) {
            spatial_engine_initialize_coordinate_system(
                sc->spatial_engine, cam_h, cam_w, &inference->detections[0]);
        }

        int num_positions = 0;
        for (int i = 0; i < inference->num_detections && num_positions < MAX_DETECTIONS_PER_FRAME; i++) {
            SpatialResult sr = spatial_engine_calculate_position(
                sc->spatial_engine, &inference->detections[i], cam_w, cam_h, NULL, 0, 0, NULL, -1);
            s->positions[num_positions++] = sr.position;
        }
        s->num_positions = num_positions;

        TrackingResult tracking = object_tracker_update(
            sc->tracking_manager, inference->detections, inference->num_detections,
            s->positions, num_positions, sc->frame_count);

        object_tracker_associate_poses(sc->tracking_manager, inference->poses, inference->num_poses);

        /* ── Sync confirmed track counts to inference pipeline for cascade state machine ── */
        if (sc->inference_pipeline) {
            sc->inference_pipeline->confirmed_track_count =
                object_tracker_get_confirmed_count(sc->tracking_manager);
            sc->inference_pipeline->total_track_count =
                object_tracker_get_all_track_count(sc->tracking_manager);
        }

        associate_poses_with_objects(tracking.tracked_objects, tracking.num_tracked,
                                      inference->poses, inference->num_poses);
        associate_faces_with_objects(tracking.tracked_objects, tracking.num_tracked,
                                      inference->faces, inference->num_faces);

        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            spatial_engine_update_trajectory(sc->spatial_engine, obj->track_id, &obj->spatial_pos);

            float velocity[3];
            if (spatial_engine_get_velocity(sc->spatial_engine, obj->track_id, 1.0f / cam_fps, velocity)) {
                memcpy(obj->velocity, velocity, sizeof(velocity));
                obj->has_velocity = true;
            }

            float height = spatial_engine_calculate_height(sc->spatial_engine, &obj->detection,
                                                            obj->has_pose ? &obj->pose : NULL);
            obj->height_meters = height;
            obj->has_height = true;
        }

        /* ── Save tracked objects to result manager ── */
        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            if (obj->frames_seen >= 3) {
                result_manager_add_tracked_object(sc->result_manager,
                    obj->track_id, obj->height_meters,
                    obj->frames_seen, obj->has_pose ? 1 : 0);
            }
        }

        IMUExternalPose imu_pose;
        if (imu_handler_get_latest_pose(sc->imu_handler, &imu_pose)) {
            spatial_engine_set_camera_pose(sc->spatial_engine, imu_pose.pitch, imu_pose.roll, imu_pose.yaw);
        }

        /* ── 双 IMU 状态 (每 30 帧) ── */
        if (sc->frame_count % 30 == 0 && sc->imu_handler) {
            if (!imu_handler_is_alignment_done(sc->imu_handler)) {
                log_info("[IMU] Aligning frames... (%d/200 samples)",
                         sc->imu_handler->align_ctx.samples_collected);
            }
        }

        sc->detection_count += tracking.num_tracked;

        /* ── Per-frame pipeline timing ── */
        double proc_start_ms = k1_get_time_ms();
        double frame_time = (proc_start_ms - (s->timestamp_us / 1000.0));
        if (sc->proc_times_count < SC_MAX_PROC_TIMES) {
            sc->processing_times[sc->proc_times_count++] = (float)(frame_time / 1000.0);
        }

        /* ── Real FPS from elapsed wall-clock time ── */
        static double last_fps_time_ms = 0;
        double now_ms = k1_get_time_ms();
        float current_fps = 1.0f;
        if (last_fps_time_ms > 0) {
            double elapsed_s = (now_ms - last_fps_time_ms) / 1000.0;
            if (elapsed_s > 0.001) current_fps = 1.0f / (float)elapsed_s;
        }
        last_fps_time_ms = now_ms;
        if (sc->fps_history_count < SC_MAX_FPS_HISTORY) {
            sc->fps_history[sc->fps_history_count++] = current_fps;
        }

        /* Store tracking result in slot for viz thread */
        s->tracking = tracking;

        /* Max frames check */
        if (sc->max_frames > 0 && sc->frame_count >= sc->max_frames) {
            log_info("[K1 Pipeline] Reached max_frames=%d, shutting down", sc->max_frames);
            sc->running = 0;
            pthread_mutex_lock(&pl->ring_mutex);
            pl->running = false;
            pl->shutdown = true;
            pthread_cond_broadcast(&pl->capture_done_cond);
            pthread_cond_broadcast(&pl->infer_done_cond);
            pthread_cond_broadcast(&pl->track_done_cond);
            pthread_cond_broadcast(&pl->post_done_cond);
            pthread_mutex_unlock(&pl->ring_mutex);
        }

        k1_pipeline_slot_tracking_done(pl, slot);
    }

    return NULL;
}

/*
 * ── K1 Viz/Render thread (CPU 6, Cluster1) ──
 *
 * Offloads visualization rendering and display output from Cluster0
 * to Cluster1.  Renders bounding-box overlay + skeleton onto vis_buffer
 * and pushes to display_output (framebuffer / RTSP / file).
 */
static void* k1_viz_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    pl->thread_alive[4] = true;
    log_info("[K1 Pipeline] Viz thread on CPU %d (Cluster1)", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;
    uint8_t* vis_buffer = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!vis_buffer) return NULL;

    while (pl->running) {
        pl->thread_heartbeats[4]++;
        int slot = k1_pipeline_slot_wait_tracked(pl);
        if (slot < 0) break;

        K1PipelineSlot* s = &pl->slots[slot];
        TrackingResult* tracking = &s->tracking;
        float avg_fps = get_current_fps(sc);

        /* Render detection view */
        visualizer_render_detection_view(sc->visualizer,
                                          s->rgb_data, cam_w, cam_h,
                                          tracking->tracked_objects, tracking->num_tracked,
                                          sc->frame_count, avg_fps,
                                          vis_buffer);

        /* Push to display / stream / video file (non-blocking, async internal threads) */
        if (sc->display_output) {
            display_output_write_frame(sc->display_output, vis_buffer);
        }

        /* Periodic status log */
        if (sc->frame_count % 30 == 0) {
            log_info("[STATUS] frame=%d | objs=%d | %.1f FPS",
                     sc->frame_count, tracking->num_tracked, avg_fps);
        }

        /* ── Save per-frame metadata (JSON lines) ── */
        {
            const InferenceResult* ir = &s->inference;
            int has_action = (ir->has_action && ir->action.num_actions > 0) ? 1 : 0;
            result_manager_save_frame_metadata(sc->result_manager,
                "realtime", sc->frame_count,
                ir->num_detections, ir->num_poses,
                ir->num_faces, tracking->num_tracked, has_action);
        }

        k1_pipeline_slot_release(pl, slot);
    }

    free(vis_buffer);
    return NULL;
}

SystemStatus system_controller_process_realtime_k1(SystemController* sc) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));

    if (!sc) {
        strncpy(status.message, "Invalid system controller", sizeof(status.message) - 1);
        return status;
    }

    sc->mode = PIPELINE_MODE_REALTIME;
    log_info("============================================================");
    log_info("Starting K1 PIPELINE realtime mode on Muse Pi Pro");
    log_info("  Dual-Cluster: Cluster0(AI) cores 0-3, Cluster1(I/O) cores 4-7");
    log_info("============================================================");

    K1Platform* plat = k1_platform_init();
    if (!plat || !k1_platform_is_k1()) {
        log_critical("K1 platform NOT detected — this build requires K1 hardware");
        strncpy(status.message, "K1 platform required", sizeof(status.message) - 1);
        return status;
    }

    int cam_w = config_get_int(sc->config, "video.camera_width", 640);
    int cam_h = config_get_int(sc->config, "video.camera_height", 480);
    float cam_fps = (float)config_get_float(sc->config, "video.camera_fps", 15.0f);

    const char* camera_dev = config_get_string(sc->config, "video.camera_device", "/dev/video0");
    const char* camera_format = config_get_string(sc->config, "video.camera_format", "MJPEG");

    if (camera_dev) {
        sc->video_processor = video_processor_create_from_camera(camera_dev, cam_w, cam_h, cam_fps, camera_format);
        if (sc->video_processor) {
            log_info("[K1 Pipeline] Camera device opened: %s (%dx%d @ %.1f FPS)", camera_dev, cam_w, cam_h, cam_fps);
        } else {
            log_warning("[K1 Pipeline] V4L2 camera %s unavailable; will rely on CoAP receiver", camera_dev);
        }
    }

    /* ── CoAP/UDP receiver (WiFi from ESP32) ── */
    if (sc->coap_enabled && sc->coap_esp_ip[0] != '\0') {
        sc->coap_receiver = coap_receiver_create(
            sc->coap_esp_ip, sc->coap_esp_port,
            sc->coap_wifi_ssid[0] != '\0' ? sc->coap_wifi_ssid : NULL,
            sc->coap_wifi_password[0] != '\0' ? sc->coap_wifi_password : NULL);
        if (sc->coap_receiver) {
            log_info("[K1 Pipeline] CoAP receiver: %s:%d (CoAP/UDP)",
                     sc->coap_esp_ip, sc->coap_esp_port);
            /* 注册原始 IMU 数据回调 → Madgwick 相机滤波器 */
            coap_receiver_set_imu_raw_callback(sc->coap_receiver,
                (CoapImuRawCallback)imu_handler_feed_external_raw, sc->imu_handler);
        }
    }

    /* ── K1 本地 I2C IMU (GY85) ── */
    int k1_imu_bus = config_get_int(sc->config, "k1_imu.i2c_bus", 4);
    float k1_imu_rate = (float)config_get_float(sc->config, "k1_imu.sample_rate_hz", 100.0);
    K1Imu* k1_imu = k1_imu_create(k1_imu_bus, k1_imu_rate);
    if (k1_imu && k1_imu_get_state(k1_imu) != K1_IMU_STATE_ERROR) {
        imu_handler_set_k1_imu(sc->imu_handler, k1_imu);
        log_info("[K1 Pipeline] K1 local IMU: /dev/i2c-%d @ %.0fHz", k1_imu_bus, k1_imu_rate);
    } else {
        log_warning("[K1 Pipeline] K1 local IMU unavailable — using ESP32 IMU only");
    }

    /* ── Create display output ── */
    {
        uint32_t disp_channels = DISPLAY_CHANNEL_NONE;
        if (sc->display_enabled && sc->display_device[0] != '\0')
            disp_channels |= DISPLAY_CHANNEL_FRAMEBUFFER;
        if (sc->stream_enabled && sc->stream_url[0] != '\0')
            disp_channels |= DISPLAY_CHANNEL_RTSP;
        if (sc->save_video_enabled && sc->save_video_path[0] != '\0')
            disp_channels |= DISPLAY_CHANNEL_VIDEO_FILE;

        if (disp_channels != DISPLAY_CHANNEL_NONE) {
            sc->display_output = display_output_create(
                cam_w, cam_h, cam_fps,
                disp_channels,
                sc->display_device,
                sc->stream_url,
                sc->save_video_path);
            if (sc->display_output) {
                log_info("[K1 Pipeline] Display output active (channels=0x%x)", disp_channels);
            } else {
                log_warning("[K1 Pipeline] Display output creation failed, continuing without display");
            }
        } else {
            log_info("[K1 Pipeline] No display/output channels enabled (headless mode)");
        }
    }

    K1Pipeline pl;
    k1_pipeline_init(&pl, sc);
    if (!k1_ring_init_slots(&pl, cam_w, cam_h)) {
        strncpy(status.message, "Failed to init ring buffer", sizeof(status.message) - 1);
        k1_pipeline_destroy(&pl);
        goto k1_cleanup;
    }

    const char* session_id = result_manager_start_session(sc->result_manager, "realtime_k1");
    (void)session_id;

    /* Start IMU recording (K1 local GY85 + ESP32 remote via CoAP) */
    if (sc->imu_handler) {
        imu_handler_start_recording(sc->imu_handler, "output", session_id);
    }

    K1ThreadArg thread_args[K1_MAX_PIPELINE_THREADS];
    int t = 0;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER1_CAPTURE, "Capture"};
    pthread_create(&pl.threads[t], NULL, k1_capture_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER0_INFERENCE, "Inference"};
    pthread_create(&pl.threads[t], NULL, k1_inference_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER0_DETECT, "ST-GCN"};
    pthread_create(&pl.threads[t], NULL, k1_stgcn_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER0_AI, "PostProcess"};
    pthread_create(&pl.threads[t], NULL, k1_postprocess_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER1_VIZ, "Viz"};
    pthread_create(&pl.threads[t], NULL, k1_viz_thread, &thread_args[t]);
    t++;

    pl.num_threads = t;

    log_info("[K1 Pipeline] %d threads: Capture(CPU%d)→Inference(CPU%d)→ST-GCN(CPU%d)→Post(CPU%d)→Viz(CPU%d)",
             pl.num_threads,
             K1_CPU_CLUSTER1_CAPTURE, K1_CPU_CLUSTER0_INFERENCE,
             K1_CPU_CLUSTER0_DETECT, K1_CPU_CLUSTER0_AI, K1_CPU_CLUSTER1_VIZ);

    sc->running = 1;
    sc->start_time = k1_get_time_ms() / 1000.0;

    log_info("[K1 Pipeline] === Pipeline alive ===");
    log_info("[K1 Pipeline] Source: %s | Display: %s | Video: %s",
             sc->coap_receiver ? "CoAP/UDP" :
                (sc->video_processor ? "V4L2 Camera" : "None"),
             sc->display_output ? "active" : "none",
             sc->save_video_enabled ? sc->save_video_path : "disabled");
    log_info("[K1 Pipeline] Waiting for frames. Idle timeout: %ds. Press Ctrl+C to stop.",
             sc->frame_timeout_s);

    /* ── Main thread: startup wait + heartbeat monitor ──
     * Elevate to SCHED_FIFO prio 51 (1 tick above pipeline threads at 50)
     * so shutdown broadcasts + pthread_join can preempt worker threads. */
    {
        struct sched_param sp;
        memset(&sp, 0, sizeof(sp));
        sp.sched_priority = 51;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0) {
            log_info("[K1 Pipeline] Monitor thread elevated to SCHED_FIFO prio 51");
        }

        int prev_hearts[K1_PIPELINE_NUM_HEARTS] = {0};  /* all 6 hearts zero-initialized */
        int stuck_count = 0;
        int startup_ticks = 0;
        while (sc->running) {
            /* 500ms poll interval, split into 10ms slices for responsive shutdown */
            for (int p = 0; p < 50 && sc->running; p++) {
                usleep(10000);
            }
            if (!sc->running) break;

            /* ── Startup grace: wait up to 30s for all threads to come alive ── */
            int all_alive = 1;
            for (int i = 0; i < pl.num_threads && i < K1_PIPELINE_NUM_HEARTS; i++) {
                if (!pl.thread_alive[i]) { all_alive = 0; break; }
            }
            if (!all_alive) {
                startup_ticks++;
                if (startup_ticks > 60) {  /* 30s timeout */
                    log_error("[K1 Pipeline] Worker threads failed to start within 30s");
                    sc->running = 0;
                    pthread_mutex_lock(&pl.ring_mutex);
                    pl.running = false; pl.shutdown = true;
                    pthread_cond_broadcast(&pl.capture_done_cond);
                    pthread_cond_broadcast(&pl.infer_done_cond);
                    pthread_cond_broadcast(&pl.track_done_cond);
                    pthread_cond_broadcast(&pl.post_done_cond);
                    pthread_mutex_unlock(&pl.ring_mutex);
                }
                continue;
            }

            /* ── Check capture thread liveness (driver of the pipeline).
             * Inference/postprocess legitimately wait for frames —
             * only the capture thread must keep running. 30 checks
             * × 500ms = 15s timeout. ── */
            if (pl.thread_heartbeats[0] == prev_hearts[0]) stuck_count++;
            else                                          stuck_count = 0;
            for (int i = 0; i < pl.num_threads; i++)
                prev_hearts[i] = pl.thread_heartbeats[i];

            if (stuck_count >= 30) {
                log_error("[K1 Pipeline] Capture thread stuck (%d checks), shutting down",
                          stuck_count);
                sc->running = 0;
                pthread_mutex_lock(&pl.ring_mutex);
                pl.running = false; pl.shutdown = true;
                pthread_cond_broadcast(&pl.capture_done_cond);
                pthread_cond_broadcast(&pl.infer_done_cond);
                pthread_cond_broadcast(&pl.track_done_cond);
                pthread_cond_broadcast(&pl.post_done_cond);
                pthread_mutex_unlock(&pl.ring_mutex);
            }
        }
    }

    k1_pipeline_destroy(&pl);

k1_cleanup:
    ;
    float avg_fps = 0.0f;
    if (sc->fps_history_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->fps_history_count; i++) sum += sc->fps_history[i];
        avg_fps = sum / sc->fps_history_count;
    }

    float avg_time_ms = 0.0f;
    if (sc->proc_times_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->proc_times_count; i++) sum += sc->processing_times[i];
        avg_time_ms = (sum / sc->proc_times_count) * 1000.0f;
    }

    result_manager_update_session_stats(sc->result_manager, sc->frame_count, sc->detection_count, avg_fps, avg_time_ms);
    result_manager_end_session(sc->result_manager, sc->spatial_engine);

    if (sc->coap_receiver) {
        coap_receiver_destroy(sc->coap_receiver);
        sc->coap_receiver = NULL;
    }
    if (sc->display_output) {
        display_output_destroy(sc->display_output);
        sc->display_output = NULL;
    }

    sc->running = 0;
    return system_controller_get_final_status(sc);
}

SystemController* system_controller_create(const char* config_path) {
    log_info("============================================================");
    log_info("LingQi TanTong System Controller Initializing...");
    log_info("============================================================");

    SystemController* sc = (SystemController*)calloc(1, sizeof(SystemController));
    if (!sc) return NULL;

    sc->config = config_manager_create(config_path);
    {
        bool ep_pref = config_get_bool(sc->config, "system.use_spacemit_ep", true);
        ort_set_ep_enabled(ep_pref);
        int ep_threads = config_get_int(sc->config, "system.spacemit_ep_intra_threads", 1);
        spacemit_ort_set_ep_intra_threads(ep_threads);
        log_info("SpacemiT EP config: enabled=%d, intra_threads=%d", ep_pref, ep_threads);
    }
    sc->model_store = model_store_create("models");
    if (!sc->model_store) {
        log_warning("Model store creation failed");
    }
    sc->video_processor = NULL;
    sc->imu_handler = imu_handler_create(10, 0.01f, 0.1f);

    sc->inference_pipeline = inference_pipeline_create();
    if (!sc->inference_pipeline) {
        log_warning("Inference pipeline creation failed");
    }

    if (sc->inference_pipeline) {
        int ret = inference_pipeline_load_models(sc->inference_pipeline, "models", sc->config);
        if (ret < 0) {
            log_critical("Model loading failed — PRIMARY pose model is required");
            system_controller_destroy(sc);
            return NULL;
        }
        log_info("Models loaded: %d/%d", ret, 5);
    }

    sc->tracking_manager = object_tracker_create(
        config_get_int(sc->config, "tracking.max_lost", 30),
        config_get_float(sc->config, "tracking.min_iou", 0.3f),
        config_get_float(sc->config, "tracking.max_distance", 5.0f),
        config_get_int(sc->config, "tracking.max_track_history", 300)
    );

    /* ── Apply enhanced tracking confirmation config ── */
    object_tracker_set_enhanced_config(
        sc->tracking_manager,
        config_get_int(sc->config, "tracking.confirmation_frames", TRACKING_CONFIRMATION_FRAMES),
        config_get_int(sc->config, "tracking.min_keypoints_for_confirm", TRACKING_MIN_KEYPOINTS_FOR_CONFIRM),
        config_get_int(sc->config, "tracking.min_keypoints_strong", TRACKING_MIN_KEYPOINTS_STRONG),
        config_get_float(sc->config, "tracking.spatial_jump_max_m", TRACKING_SPATIAL_JUMP_MAX_M)
    );

    /* ── NEW: Apply cascade matching config ── */
    object_tracker_set_cascade_config(
        sc->tracking_manager,
        config_get_int(sc->config, "tracking.cascade_max_age", 30),
        config_get_int(sc->config, "tracking.cascade_min_hits", 3),
        config_get_float(sc->config, "tracking.appearance_weight", 0.35f),
        config_get_float(sc->config, "tracking.appearance_max_dist", 0.50f),
        config_get_float(sc->config, "tracking.iou_threshold_low", 0.15f)
    );

    /* ── NEW: Apply occlusion handling config ── */
    object_tracker_set_occlusion_config(
        sc->tracking_manager,
        config_get_int(sc->config, "tracking.max_occluded_frames", 90),
        config_get_int(sc->config, "tracking.upper_body_min_keypoints", 4),
        config_get_int(sc->config, "tracking.side_body_min_keypoints", 3),
        config_get_float(sc->config, "tracking.occlusion_score_threshold", 0.40f)
    );

    /* ── NEW: Apply re-identification config ── */
    object_tracker_set_reid_config(
        sc->tracking_manager,
        config_get_int(sc->config, "tracking.reid_pool_max_age", 90)
    );

    /* ── NEW: Apply multi-person detection config ── */
    object_tracker_set_multi_person_config(
        sc->tracking_manager,
        config_get_int(sc->config, "tracking.new_person_grace_frames", 3)
    );

    float fx = config_get_float(sc->config, "spatial.fx", 960.0f);
    float fy = config_get_float(sc->config, "spatial.fy", 960.0f);
    float cx = config_get_float(sc->config, "spatial.cx", 960.0f);
    float cy = config_get_float(sc->config, "spatial.cy", 540.0f);
    float avg_height = config_get_float(sc->config, "spatial.avg_human_height", 1.7f);
    float cam_mat[9] = {fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f};
    sc->spatial_engine = spatial_engine_create(cam_mat, NULL, fx, avg_height);

    /* ── Depth estimation tuning (2024-2026 academic refinements) ── */
    if (sc->spatial_engine) {
        sc->spatial_engine->depth_ema_alpha =
            config_get_float(sc->config, "spatial.depth_ema_alpha", DEPTH_EMA_ALPHA);
        sc->spatial_engine->depth_outlier_mad_mult =
            config_get_float(sc->config, "spatial.depth_outlier_mad_mult", DEPTH_OUTLIER_MAD_MULT);
    }

    sc->visualizer = visualizer_create(
        config_get_bool(sc->config, "visualization.show_info_bar", true),
        config_get_bool(sc->config, "visualization.corner_markers", true),
        config_get_bool(sc->config, "visualization.crosshair", true),
        true
    );

    int render_w = config_get_int(sc->config, "visualization.render_size.0", 1920);
    int render_h = config_get_int(sc->config, "visualization.render_size.1", 1080);
    sc->ar_renderer = ar_renderer_create(render_w, render_h, true);
    sc->result_manager = result_manager_create("output");

    sc->coap_receiver = NULL;
    sc->display_output = NULL;
    sc->mode = PIPELINE_MODE_OFFLINE;

    sc->frame_count = 0;
    sc->max_frames = config_get_int(sc->config, "system.max_frames", 0);
    sc->frame_timeout_s = config_get_int(sc->config, "coap.frame_timeout_s", 10);
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    sc->fps_history_count = 0;
    sc->proc_times_count = 0;
    sc->detection_count = 0;
    sc->running = 0;

    /* Display / streaming config */
    sc->display_enabled = config_get_bool(sc->config, "visualization.display_enabled", false);
    {
        const char* fb_dev = config_get_string(sc->config, "visualization.display_device", "/dev/fb0");
        if (fb_dev) {
            strncpy(sc->display_device, fb_dev, sizeof(sc->display_device) - 1);
        } else {
            sc->display_device[0] = '\0';
        }
    }
    sc->stream_enabled = config_get_bool(sc->config, "visualization.rtsp_enabled", false);
    {
        const char* rtsp_u = config_get_string(sc->config, "visualization.rtsp_url", "");
        if (rtsp_u) {
            strncpy(sc->stream_url, rtsp_u, sizeof(sc->stream_url) - 1);
        } else {
            sc->stream_url[0] = '\0';
        }
    }
    sc->save_video_enabled = config_get_bool(sc->config, "visualization.record_to_video", false);
    {
        const char* vp = config_get_string(sc->config, "visualization.video_output_path", "output/realtime_output.mp4");
        if (vp) {
            strncpy(sc->save_video_path, vp, sizeof(sc->save_video_path) - 1);
        } else {
            sc->save_video_path[0] = '\0';
        }
    }

    /* CoAP receiver config (ESP32 CoAP/UDP) */
    sc->coap_receiver = NULL;
    sc->coap_enabled = config_get_bool(sc->config, "coap.enabled", true);
    {
        const char* ip = config_get_string(sc->config, "coap.esp_ip", "192.168.4.1");
        strncpy(sc->coap_esp_ip, ip ? ip : "192.168.4.1", sizeof(sc->coap_esp_ip) - 1);
    }
    sc->coap_esp_port = config_get_int(sc->config, "coap.esp_port", 5683);
    {
        const char* ssid = config_get_string(sc->config, "coap.wifi_ssid", "ESP32-Camera-AP");
        strncpy(sc->coap_wifi_ssid, ssid ? ssid : "", sizeof(sc->coap_wifi_ssid) - 1);
    }
    {
        const char* pw = config_get_string(sc->config, "coap.wifi_password", "12345678");
        strncpy(sc->coap_wifi_password, pw ? pw : "", sizeof(sc->coap_wifi_password) - 1);
    }

    log_info("System Controller initialized");
    return sc;
}

void system_controller_destroy(SystemController* sc) {
    if (!sc) return;

    if (sc->config) config_manager_destroy(sc->config);
    if (sc->model_store) model_store_destroy(sc->model_store);
    if (sc->video_processor) video_processor_destroy(sc->video_processor);
    if (sc->imu_handler) imu_handler_destroy(sc->imu_handler);
    if (sc->inference_pipeline) inference_pipeline_destroy(sc->inference_pipeline);
    if (sc->tracking_manager) object_tracker_destroy(sc->tracking_manager);
    if (sc->spatial_engine) spatial_engine_destroy(sc->spatial_engine);
    if (sc->visualizer) visualizer_destroy(sc->visualizer);
    if (sc->ar_renderer) ar_renderer_destroy(sc->ar_renderer);
    if (sc->result_manager) result_manager_destroy(sc->result_manager);
    if (sc->coap_receiver) coap_receiver_destroy(sc->coap_receiver);
    if (sc->display_output) display_output_destroy(sc->display_output);

    /* ── Global resource cleanup (once at process exit) ── */
    {
        K1Platform* plat = k1_platform_init();   /* obtain singleton for destroy */
        if (plat) k1_platform_destroy(plat);
    }
    ort_global_shutdown();       /* ONNX Runtime global environment */

    free(sc);
}

static void associate_poses_with_objects(TrackedObject* objects, int num_objects,
                                          PoseEstimation* poses, int num_poses) {
    if (!objects || !poses || num_objects <= 0 || num_poses <= 0) return;

    bool pose_used[MAX_DETECTIONS_PER_FRAME];
    memset(pose_used, 0, sizeof(pose_used));

    for (int i = 0; i < num_objects; i++) {
        TrackedObject* obj = &objects[i];
        float best_iou = 0.0f;
        int best_pose = -1;

        for (int j = 0; j < num_poses; j++) {
            if (pose_used[j] || !poses[j].has_bbox) continue;
            float iou = bbox_iou(&obj->detection.bbox, &poses[j].bbox);
            if (iou > best_iou && iou > 0.3f) {
                best_iou = iou;
                best_pose = j;
            }
        }

        if (best_pose >= 0) {
            obj->pose = poses[best_pose];
            obj->has_pose = true;
            pose_used[best_pose] = true;
        }
    }
}

static void associate_faces_with_objects(TrackedObject* objects, int num_objects,
                                          FaceIdentity* faces, int num_faces) {
    if (!objects || !faces || num_objects <= 0 || num_faces <= 0) return;

    bool face_used[MAX_DETECTIONS_PER_FRAME];
    memset(face_used, 0, sizeof(face_used));

    for (int i = 0; i < num_objects; i++) {
        TrackedObject* obj = &objects[i];
        float best_iou = 0.0f;
        int best_face = -1;

        for (int j = 0; j < num_faces; j++) {
            if (face_used[j]) continue;
            float iou = bbox_iou(&obj->detection.bbox, &faces[j].bbox);
            if (iou > best_iou && iou > 0.2f) {
                best_iou = iou;
                best_face = j;
            }
        }

        if (best_face >= 0) {
            obj->face = faces[best_face];
            obj->has_face = true;
            face_used[best_face] = true;
        }
    }
}

static float get_current_fps(const SystemController* sc) {
    if (sc->fps_history_count >= 10) {
        float sum = 0.0f;
        for (int i = sc->fps_history_count - 10; i < sc->fps_history_count; i++) {
            sum += sc->fps_history[i];
        }
        return sum / 10.0f;
    } else if (sc->fps_history_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->fps_history_count; i++) {
            sum += sc->fps_history[i];
        }
        return sum / sc->fps_history_count;
    }
    return 0.0f;
}

SystemStatus system_controller_process_video(SystemController* sc,
                                              const char* video_path,
                                              const char* output_path,
                                              int max_frames,
                                              bool show_windows,
                                              int save_frame_interval) {
    (void)show_windows;
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));

    if (!sc || !video_path) {
        strncpy(status.message, "Invalid parameters", sizeof(status.message) - 1);
        status.message[sizeof(status.message) - 1] = '\0';
        return status;
    }

    log_info("Processing video: %s", video_path);

    char output_video_path[MAX_PATH_LEN];
    const char* output_dir = output_path ? output_path : "output";
    snprintf(output_video_path, sizeof(output_video_path), "%s/output.mp4", output_dir);

    const char* session_id = result_manager_start_session(sc->result_manager, video_path);
    (void)session_id;

    /* Start IMU recording if handler is available (offline mode may not have IMU) */
    if (sc->imu_handler) {
        imu_handler_start_recording(sc->imu_handler, "output", session_id);
    }

    VideoProcessor* processor = video_processor_create(video_path, 0, 0, false);
    if (!processor || video_processor_open(processor, video_path) != VP_OK) {
        strncpy(status.message, "Failed to open video", sizeof(status.message) - 1);
        status.message[sizeof(status.message) - 1] = '\0';
        result_manager_add_error(sc->result_manager, "VideoOpenError", "Failed to open video");
        result_manager_end_session(sc->result_manager, sc->spatial_engine);
        if (processor) video_processor_destroy(processor);
        return status;
    }

    int video_w = video_processor_get_width(processor);
    int video_h = video_processor_get_height(processor);
    float video_fps = video_processor_get_fps(processor);

    uint8_t* vis_buffer = (uint8_t*)malloc(video_w * video_h * 3);
    if (!vis_buffer) {
        video_processor_destroy(processor);
        strncpy(status.message, "Failed to allocate vis buffer", sizeof(status.message) - 1);
        return status;
    }

    VideoWriter* video_writer = NULL;
    /* CLI override takes priority; fall back to config value */
    if (save_frame_interval <= 0) {
        save_frame_interval = config_get_int(sc->config, "video.save_frame_interval", 10);
    }
    if (save_frame_interval > 0) {
        video_writer = video_writer_create(output_video_path, video_w, video_h, video_fps > 0.0f ? video_fps : 30.0f);
        if (video_writer) {
            log_info("Video output enabled: %s", output_video_path);
        }
    }

    sc->running = 1;
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    double prev_time = sc->start_time;

    FrameData* frame;
    while (sc->running && (frame = video_processor_read_frame_raw(processor)) != NULL) {
        double frame_start = (double)utils_get_time_ms() / 1000.0;
        sc->frame_count++;

        if (max_frames > 0 && sc->frame_count > max_frames) {
            frame_data_destroy(frame);
            break;
        }

        InferenceResult inference;
        inference_pipeline_process_frame(
            sc->inference_pipeline, frame->data, frame->width, frame->height, &inference);

        if (sc->frame_count == 1 && inference.num_detections > 0) {
            spatial_engine_initialize_coordinate_system(
                sc->spatial_engine, frame->height, frame->width, &inference.detections[0]);
        }

        SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
        int num_positions = 0;
        for (int i = 0; i < inference.num_detections && num_positions < MAX_DETECTIONS_PER_FRAME; i++) {
            SpatialResult sr = spatial_engine_calculate_position(
                sc->spatial_engine, &inference.detections[i], frame->width, frame->height, NULL, 0, 0, NULL, -1);
            positions[num_positions++] = sr.position;
        }

        TrackingResult tracking = object_tracker_update(
            sc->tracking_manager, inference.detections, inference.num_detections,
            positions, num_positions, sc->frame_count);

        /* ── NEW: Pose association via tracker (updates appearance features) ── */
        object_tracker_associate_poses(sc->tracking_manager, inference.poses, inference.num_poses);

        /* ── Sync confirmed track counts to inference pipeline ── */
        if (sc->inference_pipeline) {
            sc->inference_pipeline->confirmed_track_count =
                object_tracker_get_confirmed_count(sc->tracking_manager);
            sc->inference_pipeline->total_track_count =
                object_tracker_get_all_track_count(sc->tracking_manager);
        }

        associate_poses_with_objects(tracking.tracked_objects, tracking.num_tracked,
                                      inference.poses, inference.num_poses);
        associate_faces_with_objects(tracking.tracked_objects, tracking.num_tracked,
                                      inference.faces, inference.num_faces);

        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            spatial_engine_update_trajectory(sc->spatial_engine, obj->track_id, &obj->spatial_pos);

            float velocity[3];
            if (spatial_engine_get_velocity(sc->spatial_engine, obj->track_id, 1.0f / 30.0f, velocity)) {
                memcpy(obj->velocity, velocity, sizeof(velocity));
                obj->has_velocity = true;
            }

            float height = spatial_engine_calculate_height(sc->spatial_engine, &obj->detection,
                                                            obj->has_pose ? &obj->pose : NULL);
            obj->height_meters = height;
            obj->has_height = true;
        }

        /* ── Save tracked objects to result manager ── */
        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            if (obj->frames_seen >= 3) {
                result_manager_add_tracked_object(sc->result_manager,
                    obj->track_id, obj->height_meters,
                    obj->frames_seen, obj->has_pose ? 1 : 0);
            }
        }

        IMUExternalPose imu_pose;
        if (imu_handler_get_latest_pose(sc->imu_handler, &imu_pose)) {
            spatial_engine_set_camera_pose(sc->spatial_engine, imu_pose.pitch, imu_pose.roll, imu_pose.yaw);
        }

        sc->detection_count += tracking.num_tracked;

        double frame_time = (double)utils_get_time_ms() / 1000.0 - frame_start;
        if (sc->proc_times_count < SC_MAX_PROC_TIMES) {
            sc->processing_times[sc->proc_times_count++] = (float)frame_time;
        }

        double elapsed = (double)utils_get_time_ms() / 1000.0 - prev_time;
        float current_fps = elapsed > 0.0f ? 1.0f / (float)elapsed : 0.0f;
        if (sc->fps_history_count < SC_MAX_FPS_HISTORY) {
            sc->fps_history[sc->fps_history_count++] = current_fps;
        }
        prev_time = (double)utils_get_time_ms() / 1000.0;

        float avg_fps = get_current_fps(sc);

        if (video_writer && save_frame_interval > 0) {
            visualizer_render_detection_view(sc->visualizer,
                                              frame->data, frame->width, frame->height,
                                              tracking.tracked_objects, tracking.num_tracked,
                                              sc->frame_count, avg_fps,
                                              vis_buffer);

            video_writer_write_frame(video_writer, vis_buffer);

            if (sc->frame_count % save_frame_interval == 0) {
                log_info("Frame %d processed: %d objects, FPS=%.1f", sc->frame_count, tracking.num_tracked, avg_fps);
            }
        }

        if (sc->frame_count % 50 == 0 && (!video_writer || save_frame_interval == 0)) {
            log_info("Processed %d frames: avg_fps=%.1f", sc->frame_count, avg_fps);
        }

        frame_data_destroy(frame);
    }

    if (video_writer) {
        video_writer_destroy(video_writer);
        log_info("Output video saved to: %s", output_video_path);
    }

    free(vis_buffer);
    video_processor_destroy(processor);

    float avg_fps = 0.0f;
    if (sc->fps_history_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->fps_history_count; i++) sum += sc->fps_history[i];
        avg_fps = sum / sc->fps_history_count;
    }

    float avg_time_ms = 0.0f;
    if (sc->proc_times_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->proc_times_count; i++) sum += sc->processing_times[i];
        avg_time_ms = (sum / sc->proc_times_count) * 1000.0f;
    }

    result_manager_update_session_stats(sc->result_manager, sc->frame_count, sc->detection_count, avg_fps, avg_time_ms);
    result_manager_end_session(sc->result_manager, sc->spatial_engine);

    sc->running = 0;
    return system_controller_get_final_status(sc);
}

SystemStatus system_controller_get_status(const SystemController* sc) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));

    if (!sc) return status;

    status.is_running = sc->running;
    status.frame_count = sc->frame_count;
    status.fps = get_current_fps(sc);

    if (sc->proc_times_count >= 10) {
        float sum = 0.0f;
        for (int i = sc->proc_times_count - 10; i < sc->proc_times_count; i++) {
            sum += sc->processing_times[i];
        }
        status.processing_time_ms = (sum / 10.0f) * 1000.0f;
    }

    status.active_tracks = sc->tracking_manager ? sc->tracking_manager->num_tracks : 0;

    return status;
}

SystemStatus system_controller_get_final_status(const SystemController* sc) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));

    if (!sc) {
        strncpy(status.message, "System controller is NULL", sizeof(status.message) - 1);
        status.message[sizeof(status.message) - 1] = '\0';
        return status;
    }

    double total_time = k1_get_time_ms() / 1000.0 - sc->start_time;

    float avg_fps = 0.0f;
    if (sc->fps_history_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->fps_history_count; i++) sum += sc->fps_history[i];
        avg_fps = sum / sc->fps_history_count;
    }

    float avg_time_ms = 0.0f;
    if (sc->proc_times_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->proc_times_count; i++) sum += sc->processing_times[i];
        avg_time_ms = (sum / sc->proc_times_count) * 1000.0f;
    }

    status.is_running = false;
    status.frame_count = sc->frame_count;
    status.fps = avg_fps;
    status.processing_time_ms = avg_time_ms;
    status.active_tracks = sc->tracking_manager ? sc->tracking_manager->num_tracks : 0;

    snprintf(status.message, sizeof(status.message),
             "Processed %d frames in %.2fs | Detected %d objects | Avg FPS: %.1f | Avg Time: %.1fms",
             sc->frame_count, total_time, sc->detection_count, avg_fps, avg_time_ms);

    return status;
}

void system_controller_reset(SystemController* sc) {
    if (!sc) return;

    log_info("Resetting system...");
    sc->frame_count = 0;
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    sc->fps_history_count = 0;
    sc->proc_times_count = 0;
    sc->detection_count = 0;

    if (sc->tracking_manager) object_tracker_reset(sc->tracking_manager);
    if (sc->spatial_engine) spatial_engine_clear_trajectories(sc->spatial_engine);
    if (sc->inference_pipeline) inference_pipeline_reset(sc->inference_pipeline);

    log_info("System reset complete");
}
