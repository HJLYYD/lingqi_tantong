#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "system_controller.h"
#include "logger.h"
#include "utils.h"
#include "k1_platform.h"
#include "video_processor.h"
#ifdef HAS_ONNX_RUNTIME
#include "ort_common.h"
#ifdef HAS_SPACEMIT_EP
#include "spacemit_ort_bridge.h"
#endif
#endif
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

#ifdef HAS_K1_PIPELINE
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#define K1_RING_SIZE        4
#define K1_MAX_PIPELINE_THREADS 6

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
    pthread_cond_t post_done_cond;

    pthread_t threads[K1_MAX_PIPELINE_THREADS];
    int num_threads;
    volatile bool running;
    volatile bool shutdown;

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
    pthread_cond_init(&pl->post_done_cond, NULL);
}

static void k1_pipeline_destroy(K1Pipeline* pl) {
    pl->running = false;
    pl->shutdown = true;

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_broadcast(&pl->capture_done_cond);
    pthread_cond_broadcast(&pl->infer_done_cond);
    pthread_cond_broadcast(&pl->post_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);

    for (int i = 0; i < pl->num_threads; i++) {
        pthread_join(pl->threads[i], NULL);
    }

    k1_ring_destroy_slots(pl);
    pthread_mutex_destroy(&pl->ring_mutex);
    pthread_cond_destroy(&pl->capture_done_cond);
    pthread_cond_destroy(&pl->infer_done_cond);
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

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_signal(&pl->infer_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static int k1_pipeline_slot_wait_inferred(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
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
    log_info("[K1 Pipeline] Capture thread on CPU %d", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;

    bool use_v4l2 = (sc->video_processor != NULL &&
                    video_processor_get_source_type(sc->video_processor) == VP_SOURCE_CAMERA &&
                    video_processor_is_opened(sc->video_processor));
    bool use_arrow = (sc->arrow_receiver != NULL);
    bool use_mjpeg = (sc->mjpeg_receiver != NULL);
    if (!use_v4l2 && !use_arrow && !use_mjpeg) {
        log_error("[K1 Pipeline] Capture: no source available (no V4L2 camera, Arrow UART, or MJPEG)");
        return NULL;
    }

    double last_frame_time = k1_get_time_ms() / 1000.0;
    int cap_attempts = 0, cap_mjpeg_ok = 0;

    while (pl->running) {
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

        if (!got_frame && use_arrow) {
            arrow_receiver_update(sc->arrow_receiver);
            ArrowSourceFrame arrow_frame;
            if (arrow_receiver_get_latest_frame(sc->arrow_receiver, &arrow_frame)) {
                if (soft_jpeg_decode_to_rgb(arrow_frame.jpeg_data, arrow_frame.jpeg_len,
                                            s->rgb_data, cam_w, cam_h) == 0) {
                    got_frame = true;
                    s->timestamp_us = (int64_t)(arrow_frame.timestamp * 1000.0);
                    s->frame_index = arrow_frame.frame_index;
                }

                if (arrow_frame.has_pose) {
                    imu_handler_set_external_pose(sc->imu_handler,
                        arrow_frame.pose.qw, arrow_frame.pose.qx,
                        arrow_frame.pose.qy, arrow_frame.pose.qz,
                        arrow_frame.pose.pitch, arrow_frame.pose.roll,
                        arrow_frame.pose.yaw, arrow_frame.pose.altitude_m,
                        arrow_frame.pose.temperature_c,
                        arrow_frame.pose.timestamp_ms);
                }
            }
        }

        if (!got_frame && use_mjpeg) {
            ArrowSourceFrame mjpeg_frame;
            if (mjpeg_receiver_get_latest_frame(sc->mjpeg_receiver, &mjpeg_frame)) {
                cap_mjpeg_ok++;
                if (soft_jpeg_decode_to_rgb(mjpeg_frame.jpeg_data, mjpeg_frame.jpeg_len,
                                            s->rgb_data, cam_w, cam_h) == 0) {
                    got_frame = true;
                    s->timestamp_us = (int64_t)(mjpeg_frame.timestamp * 1000.0);
                    s->frame_index = mjpeg_frame.frame_index;
                    if (mjpeg_frame.frame_index <= 3) {
                        log_info("[K1 Pipeline] Capture got MJPEG frame#%d: %zu bytes (ok=%d/%d attempts)",
                                 mjpeg_frame.frame_index, mjpeg_frame.jpeg_len,
                                 cap_mjpeg_ok, cap_attempts);
                    }
                } else {
                    log_warning("[K1 Pipeline] MJPEG decode failed frame#%d: %zu bytes",
                                mjpeg_frame.frame_index, mjpeg_frame.jpeg_len);
                }

                if (mjpeg_frame.has_pose) {
                    imu_handler_set_external_pose(sc->imu_handler,
                        mjpeg_frame.pose.qw, mjpeg_frame.pose.qx,
                        mjpeg_frame.pose.qy, mjpeg_frame.pose.qz,
                        mjpeg_frame.pose.pitch, mjpeg_frame.pose.roll,
                        mjpeg_frame.pose.yaw, mjpeg_frame.pose.altitude_m,
                        mjpeg_frame.pose.temperature_c,
                        mjpeg_frame.pose.timestamp_ms);
                }
            }
        }

        if (!got_frame) {
            if (cap_attempts <= 5 || cap_attempts % 100 == 0) {
                log_debug("[K1 Pipeline] Capture attempt#%d: no frame (V4L2=%d arrow=%d mjpeg=%d, got=%d)",
                          cap_attempts, use_v4l2, use_arrow, use_mjpeg, cap_mjpeg_ok);
            }
            /* ── Frame timeout check ── */
            double now_s = k1_get_time_ms() / 1000.0;
            double idle_s = now_s - last_frame_time;
            if (idle_s > sc->frame_timeout_s) {
                log_info("[K1 Pipeline] No frames received for %.0fs (timeout=%ds), shutting down",
                         idle_s, sc->frame_timeout_s);
                sc->running = false;
                pl->running = false;
                pthread_mutex_lock(&pl->ring_mutex);
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
    log_info("[K1 Pipeline] Inference thread on CPU %d (Cluster0)", ta->cpu_core);

    while (pl->running) {
        int slot = k1_pipeline_slot_wait_captured(pl);
        if (slot < 0) {
            if (pl->shutdown) break;
            usleep(1000);
            continue;
        }

        K1PipelineSlot* s = &pl->slots[slot];
        InferenceResult inference = inference_pipeline_process_frame(
            sc->inference_pipeline, s->rgb_data, s->width, s->height);

        pthread_mutex_lock(&s->mutex);
        s->inference = inference;
        pthread_mutex_unlock(&s->mutex);

        k1_pipeline_slot_infer_done(pl, slot);
    }

    return NULL;
}

static void* k1_postprocess_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("PostProcess");
    log_info("[K1 Pipeline] Post-process thread on CPU %d (Cluster0)", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;
    float cam_fps = 15.0f;

    uint8_t* vis_buffer = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!vis_buffer) return NULL;

    while (pl->running) {
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
                sc->spatial_engine, &inference->detections[i], cam_w, cam_h, NULL, 0, 0);
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

        IMUExternalPose imu_pose;
        if (imu_handler_get_latest_pose(sc->imu_handler, &imu_pose)) {
            spatial_engine_set_camera_pose(sc->spatial_engine, imu_pose.pitch, imu_pose.roll, imu_pose.yaw);
        }

        sc->detection_count += tracking.num_tracked;

        double frame_time = (k1_get_time_ms() - (s->timestamp_us / 1000.0));
        if (sc->proc_times_count < SC_MAX_PROC_TIMES) {
            sc->processing_times[sc->proc_times_count++] = (float)(frame_time / 1000.0);
        }

        float current_fps = 1.0f;
        if (sc->fps_history_count < SC_MAX_FPS_HISTORY) {
            sc->fps_history[sc->fps_history_count++] = current_fps;
        }

        float avg_fps = get_current_fps(sc);

        visualizer_render_detection_view(sc->visualizer,
                                          s->rgb_data, cam_w, cam_h,
                                          tracking.tracked_objects, tracking.num_tracked,
                                          sc->frame_count, avg_fps,
                                          vis_buffer);

        /* ── Output to display / stream / video file ── */
        if (sc->display_output) {
            display_output_write_frame(sc->display_output, vis_buffer);
        }

        if (sc->frame_count % 30 == 0) {
            log_info("[K1 Pipeline] Frame %d: %d objects, %.1f FPS",
                     sc->frame_count, tracking.num_tracked, avg_fps);
        }

        /* ── Max frames check (signal shutdown via controller) ── */
        if (sc->max_frames > 0 && sc->frame_count >= sc->max_frames) {
            log_info("[K1 Pipeline] Reached max_frames=%d, shutting down", sc->max_frames);
            sc->running = false;
            pl->running = false;
        }

        k1_pipeline_slot_release(pl, slot);
    }

    free(vis_buffer);
    return NULL;
}

SystemStatus system_controller_process_realtime_k1(SystemController* sc,
                                                    const char* uart_device_A,
                                                    const char* uart_device_C,
                                                    int baudrate) {
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
        log_warning("K1 platform not available, falling back to single-threaded mode");
        return system_controller_process_realtime(sc, uart_device_A, uart_device_C, baudrate);
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
            log_warning("[K1 Pipeline] V4L2 camera %s unavailable; will rely on Arrow UART", camera_dev);
        }
    }

    if (uart_device_A) {
        sc->arrow_receiver = arrow_receiver_create(uart_device_A, baudrate);
        if (sc->arrow_receiver) {
            log_info("[K1 Pipeline] Arrow UART: %s @ %d bps", uart_device_A, baudrate);
            if (uart_device_C) {
                arrow_receiver_add_secondary_link(sc->arrow_receiver, uart_device_C, baudrate);
                log_info("[K1 Pipeline] Dual-link: secondary UART %s", uart_device_C);
            }
        }
    }

    /* ── MJPEG HTTP receiver (WiFi from ESP32) ── */
    if (sc->mjpeg_enabled && sc->mjpeg_esp_ip[0] != '\0') {
        sc->mjpeg_receiver = mjpeg_receiver_create(
            sc->mjpeg_esp_ip, sc->mjpeg_esp_port,
            sc->mjpeg_wifi_ssid[0] != '\0' ? sc->mjpeg_wifi_ssid : NULL,
            sc->mjpeg_wifi_password[0] != '\0' ? sc->mjpeg_wifi_password : NULL);
        if (sc->mjpeg_receiver) {
            log_info("[K1 Pipeline] MJPEG receiver: %s:%d",
                     sc->mjpeg_esp_ip, sc->mjpeg_esp_port);
        }
    }

    sc->ai_context = ai_accel_context_create();
    if (sc->ai_context) {
        log_info("[K1 Pipeline] %s", ai_accel_describe(sc->ai_context));
    } else {
        log_warning("[K1 Pipeline] AI accel context unavailable, ONNX CPU path only");
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
                log_info("[K1 Pipeline] Display output active");
            }
        }
    }

    K1Pipeline pl;
    k1_pipeline_init(&pl, sc);
    if (!k1_ring_init_slots(&pl, cam_w, cam_h)) {
        strncpy(status.message, "Failed to init ring buffer", sizeof(status.message) - 1);
        k1_pipeline_destroy(&pl);
        return status;
    }

    const char* session_id = result_manager_start_session(sc->result_manager, "realtime_k1");
    (void)session_id;

    K1ThreadArg thread_args[K1_MAX_PIPELINE_THREADS];
    int t = 0;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER1_CAPTURE, "Capture"};
    pthread_create(&pl.threads[t], NULL, k1_capture_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER0_INFERENCE, "Inference"};
    pthread_create(&pl.threads[t], NULL, k1_inference_thread, &thread_args[t]);
    t++;

    thread_args[t] = (K1ThreadArg){&pl, t, K1_CPU_CLUSTER0_AI, "PostProcess"};
    pthread_create(&pl.threads[t], NULL, k1_postprocess_thread, &thread_args[t]);
    t++;

    pl.num_threads = t;

    log_info("[K1 Pipeline] %d threads running: Capture(CPU%d) → Inference(CPU%d) → PostProcess(CPU%d)",
             pl.num_threads,
             K1_CPU_CLUSTER1_CAPTURE, K1_CPU_CLUSTER0_INFERENCE, K1_CPU_CLUSTER0_AI);

    sc->running = true;
    sc->start_time = k1_get_time_ms() / 1000.0;

    while (sc->running) {
        usleep(100000);
    }

    k1_pipeline_destroy(&pl);

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
    result_manager_end_session(sc->result_manager);

    if (sc->ai_context) {
        ai_accel_context_destroy(sc->ai_context);
        sc->ai_context = NULL;
    }
    if (sc->arrow_receiver) {
        arrow_receiver_destroy(sc->arrow_receiver);
        sc->arrow_receiver = NULL;
    }
    if (sc->mjpeg_receiver) {
        mjpeg_receiver_destroy(sc->mjpeg_receiver);
        sc->mjpeg_receiver = NULL;
    }
    if (sc->display_output) {
        display_output_destroy(sc->display_output);
        sc->display_output = NULL;
    }

    sc->running = false;
    return system_controller_get_final_status(sc);
}
#endif

SystemController* system_controller_create(const char* config_path) {
    log_info("============================================================");
    log_info("LingQi TanTong System Controller Initializing...");
    log_info("============================================================");

    SystemController* sc = (SystemController*)calloc(1, sizeof(SystemController));
    if (!sc) return NULL;

    sc->config = config_manager_create(config_path);
#ifdef HAS_ONNX_RUNTIME
    {
        bool ep_pref = config_get_bool(sc->config, "system.use_spacemit_ep", true);
        ort_set_ep_enabled(ep_pref);
#ifdef HAS_SPACEMIT_EP
        int ep_threads = config_get_int(sc->config, "system.spacemit_ep_intra_threads", 1);
        spacemit_ort_set_ep_intra_threads(ep_threads);
        log_info("SpacemiT EP config: enabled=%d, intra_threads=%d", ep_pref, ep_threads);
#endif
    }
#endif
    sc->model_store = model_store_create("models");
    if (!sc->model_store) {
        log_warning("Model store creation failed");
    }
    sc->video_processor = NULL;
    sc->imu_handler = imu_handler_create(10, 0.01f, 0.1f);

    const char* detection_backend = config_get_string(sc->config, "detection.backend", "cpu");
    bool use_ai_accel = (detection_backend != NULL && strcmp(detection_backend, "ai_accel") == 0);
    (void)use_ai_accel;
    sc->inference_pipeline = inference_pipeline_create();
    if (!sc->inference_pipeline) {
        log_warning("Inference pipeline creation failed");
    }

    if (sc->inference_pipeline) {
        inference_pipeline_load_models(sc->inference_pipeline, "models", sc->config);
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

    sc->arrow_receiver = NULL;
    sc->ai_context = NULL;
    sc->display_output = NULL;
    sc->mode = PIPELINE_MODE_OFFLINE;

    sc->frame_count = 0;
    sc->max_frames = config_get_int(sc->config, "system.max_frames", 0);
    sc->frame_timeout_s = config_get_int(sc->config, "arrow.frame_timeout_s", 10);
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    sc->fps_history_count = 0;
    sc->proc_times_count = 0;
    sc->detection_count = 0;
    sc->running = false;

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

    /* MJPEG receiver config */
    sc->mjpeg_receiver = NULL;
    sc->mjpeg_enabled = config_get_bool(sc->config, "mjpeg.enabled", false);
    {
        const char* ip = config_get_string(sc->config, "mjpeg.esp_ip", "192.168.4.1");
        strncpy(sc->mjpeg_esp_ip, ip ? ip : "192.168.4.1", sizeof(sc->mjpeg_esp_ip) - 1);
    }
    sc->mjpeg_esp_port = config_get_int(sc->config, "mjpeg.esp_port", 80);
    {
        const char* ssid = config_get_string(sc->config, "mjpeg.wifi_ssid", "");
        strncpy(sc->mjpeg_wifi_ssid, ssid ? ssid : "", sizeof(sc->mjpeg_wifi_ssid) - 1);
    }
    {
        const char* pw = config_get_string(sc->config, "mjpeg.wifi_password", "");
        strncpy(sc->mjpeg_wifi_password, pw ? pw : "", sizeof(sc->mjpeg_wifi_password) - 1);
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
    if (sc->arrow_receiver) arrow_receiver_destroy(sc->arrow_receiver);
    if (sc->mjpeg_receiver) mjpeg_receiver_destroy(sc->mjpeg_receiver);
    if (sc->ai_context) ai_accel_context_destroy(sc->ai_context);
    if (sc->display_output) display_output_destroy(sc->display_output);

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

    VideoProcessor* processor = video_processor_create(video_path, 0, 0, false);
    if (!processor || video_processor_open(processor, video_path) != VP_OK) {
        strncpy(status.message, "Failed to open video", sizeof(status.message) - 1);
        status.message[sizeof(status.message) - 1] = '\0';
        result_manager_add_error(sc->result_manager, "VideoOpenError", "Failed to open video");
        result_manager_end_session(sc->result_manager);
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

    sc->running = true;
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

        InferenceResult inference = inference_pipeline_process_frame(
            sc->inference_pipeline, frame->data, frame->width, frame->height);

        if (sc->frame_count == 1 && inference.num_detections > 0) {
            spatial_engine_initialize_coordinate_system(
                sc->spatial_engine, frame->height, frame->width, &inference.detections[0]);
        }

        SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
        int num_positions = 0;
        for (int i = 0; i < inference.num_detections && num_positions < MAX_DETECTIONS_PER_FRAME; i++) {
            SpatialResult sr = spatial_engine_calculate_position(
                sc->spatial_engine, &inference.detections[i], frame->width, frame->height, NULL, 0, 0);
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
    result_manager_end_session(sc->result_manager);

    sc->running = false;
    return system_controller_get_final_status(sc);
}

SystemStatus system_controller_process_realtime(SystemController* sc,
                                                 const char* uart_device_A,
                                                 const char* uart_device_C,
                                                 int baudrate) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));

    if (!sc) {
        strncpy(status.message, "Invalid system controller", sizeof(status.message) - 1);
        return status;
    }

    sc->mode = PIPELINE_MODE_REALTIME;
    log_info("============================================================");
    log_info("Starting REAL-TIME pipeline on Muse Pi Pro");
    log_info("============================================================");

    sc->ai_context = ai_accel_context_create();
    if (sc->ai_context) {
        log_info("[realtime] %s", ai_accel_describe(sc->ai_context));
    } else {
        log_warning("AI acceleration not available, using CPU inference via ONNX Runtime");
    }

    int cam_w = config_get_int(sc->config, "video.camera_width", 640);
    int cam_h = config_get_int(sc->config, "video.camera_height", 480);
    float cam_fps = (float)config_get_float(sc->config, "video.camera_fps", 15.0f);

    const char* camera_dev = config_get_string(sc->config, "video.camera_device", "/dev/video0");
    const char* camera_format = config_get_string(sc->config, "video.camera_format", "MJPEG");

    if (camera_dev) {
        sc->video_processor = video_processor_create_from_camera(camera_dev, cam_w, cam_h, cam_fps, camera_format);
        if (sc->video_processor) {
            log_info("Camera device opened: %s (%dx%d @ %.1f FPS)", camera_dev, cam_w, cam_h, cam_fps);
        } else {
            log_warning("Camera device %s unavailable, will rely on Arrow UART for frames", camera_dev);
        }
    }

    if (uart_device_A) {
        sc->arrow_receiver = arrow_receiver_create(uart_device_A, baudrate);
        if (sc->arrow_receiver) {
            log_info("Arrow UART receiver initialized: %s @ %d bps", uart_device_A, baudrate);

            if (uart_device_C) {
                arrow_receiver_add_secondary_link(sc->arrow_receiver, uart_device_C, baudrate);
                log_info("Dual-link mode: secondary UART %s", uart_device_C);
            }
        } else {
            log_warning("Arrow UART init failed, continuing with camera-only mode");
        }
    }

    /* ── MJPEG HTTP receiver (WiFi from ESP32) ── */
    if (sc->mjpeg_enabled && sc->mjpeg_esp_ip[0] != '\0') {
        sc->mjpeg_receiver = mjpeg_receiver_create(
            sc->mjpeg_esp_ip, sc->mjpeg_esp_port,
            sc->mjpeg_wifi_ssid[0] != '\0' ? sc->mjpeg_wifi_ssid : NULL,
            sc->mjpeg_wifi_password[0] != '\0' ? sc->mjpeg_wifi_password : NULL);
        if (sc->mjpeg_receiver) {
            log_info("MJPEG receiver initialized: %s:%d",
                     sc->mjpeg_esp_ip, sc->mjpeg_esp_port);
        } else {
            log_warning("MJPEG receiver init failed for %s:%d",
                        sc->mjpeg_esp_ip, sc->mjpeg_esp_port);
        }
    }

    uint8_t* vis_buffer = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!vis_buffer) {
        strncpy(status.message, "Failed to allocate vis buffer", sizeof(status.message) - 1);
        result_manager_end_session(sc->result_manager);
        return status;
    }

    const char* session_id2 = result_manager_start_session(sc->result_manager, "realtime_session");
    (void)session_id2;

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
                log_info("[realtime] Display output active");
            }
        }
    }

    sc->running = true;
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    double prev_time = sc->start_time;
    double last_imu_log = sc->start_time;
    double last_frame_time = sc->start_time;

    uint8_t* frame_rgb = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!frame_rgb) {
        free(vis_buffer);
        strncpy(status.message, "Failed to allocate frame buffer", sizeof(status.message) - 1);
        result_manager_end_session(sc->result_manager);
        return status;
    }

    log_info("Realtime pipeline running. Press Ctrl+C to stop.");
    log_info("Resolution: %dx%d, Target FPS: %.1f", cam_w, cam_h, cam_fps);

    while (sc->running) {
        double frame_start = (double)utils_get_time_ms() / 1000.0;

        if (sc->arrow_receiver) {
            arrow_receiver_update(sc->arrow_receiver);
        }
        if (sc->mjpeg_receiver) {
            mjpeg_receiver_update(sc->mjpeg_receiver);
        }

        bool got_frame = false;

        if (sc->video_processor &&
            video_processor_get_source_type(sc->video_processor) == VP_SOURCE_CAMERA &&
            video_processor_is_opened(sc->video_processor)) {
            FrameData* fd = video_processor_read_frame_raw(sc->video_processor);
            if (fd && fd->data) {
                if (fd->width == cam_w && fd->height == cam_h) {
                    memcpy(frame_rgb, fd->data, (size_t)cam_w * cam_h * 3);
                } else {
                    utils_resize_image(fd->data, fd->width, fd->height,
                                       frame_rgb, cam_w, cam_h, 3);
                }
                got_frame = true;
                frame_data_destroy(fd);
            } else if (fd) {
                frame_data_destroy(fd);
            }
        }

        if (!got_frame && sc->arrow_receiver) {
            ArrowSourceFrame arrow_frame;
            if (arrow_receiver_get_latest_frame(sc->arrow_receiver, &arrow_frame)) {
                if (soft_jpeg_decode_to_rgb(arrow_frame.jpeg_data, arrow_frame.jpeg_len,
                                            frame_rgb, cam_w, cam_h) == 0) {
                    got_frame = true;
                }

                if (arrow_frame.has_pose) {
                    imu_handler_set_external_pose(sc->imu_handler,
                        arrow_frame.pose.qw, arrow_frame.pose.qx,
                        arrow_frame.pose.qy, arrow_frame.pose.qz,
                        arrow_frame.pose.pitch, arrow_frame.pose.roll,
                        arrow_frame.pose.yaw, arrow_frame.pose.altitude_m,
                        arrow_frame.pose.temperature_c,
                        arrow_frame.pose.timestamp_ms);
                }
            }
        }

        if (!got_frame && sc->mjpeg_receiver) {
            ArrowSourceFrame mjpeg_frame;
            if (mjpeg_receiver_get_latest_frame(sc->mjpeg_receiver, &mjpeg_frame)) {
                if (soft_jpeg_decode_to_rgb(mjpeg_frame.jpeg_data, mjpeg_frame.jpeg_len,
                                            frame_rgb, cam_w, cam_h) == 0) {
                    got_frame = true;
                }

                if (mjpeg_frame.has_pose) {
                    imu_handler_set_external_pose(sc->imu_handler,
                        mjpeg_frame.pose.qw, mjpeg_frame.pose.qx,
                        mjpeg_frame.pose.qy, mjpeg_frame.pose.qz,
                        mjpeg_frame.pose.pitch, mjpeg_frame.pose.roll,
                        mjpeg_frame.pose.yaw, mjpeg_frame.pose.altitude_m,
                        mjpeg_frame.pose.temperature_c,
                        mjpeg_frame.pose.timestamp_ms);
                }
            }
        }

        if (!got_frame) {
            /* ── Frame timeout: exit if no frames for too long ── */
            double now_s = (double)utils_get_time_ms() / 1000.0;
            double idle_s = now_s - last_frame_time;
            if (idle_s > sc->frame_timeout_s) {
                log_info("[realtime] No frames received for %.0fs (timeout=%ds), exiting",
                         idle_s, sc->frame_timeout_s);
                sc->running = false;
                break;
            }
            struct timespec ts = {0, 10000000L};
            nanosleep(&ts, NULL);
            continue;
        }
        last_frame_time = (double)utils_get_time_ms() / 1000.0;

        sc->frame_count++;

        InferenceResult inference = inference_pipeline_process_frame(
            sc->inference_pipeline, frame_rgb, cam_w, cam_h);

        if (sc->frame_count == 1 && inference.num_detections > 0) {
            spatial_engine_initialize_coordinate_system(
                sc->spatial_engine, cam_h, cam_w, &inference.detections[0]);
        }

        SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
        int num_positions = 0;
        for (int i = 0; i < inference.num_detections && num_positions < MAX_DETECTIONS_PER_FRAME; i++) {
            SpatialResult sr = spatial_engine_calculate_position(
                sc->spatial_engine, &inference.detections[i], cam_w, cam_h, NULL, 0, 0);
            positions[num_positions++] = sr.position;
        }

        TrackingResult tracking = object_tracker_update(
            sc->tracking_manager, inference.detections, inference.num_detections,
            positions, num_positions, sc->frame_count);

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
            if (spatial_engine_get_velocity(sc->spatial_engine, obj->track_id, 1.0f / cam_fps, velocity)) {
                memcpy(obj->velocity, velocity, sizeof(velocity));
                obj->has_velocity = true;
            }

            float height = spatial_engine_calculate_height(sc->spatial_engine, &obj->detection,
                                                            obj->has_pose ? &obj->pose : NULL);
            obj->height_meters = height;
            obj->has_height = true;
        }

        IMUExternalPose imu_pose;
        if (imu_handler_get_latest_pose(sc->imu_handler, &imu_pose)) {
            spatial_engine_set_camera_pose(sc->spatial_engine, imu_pose.pitch, imu_pose.roll, imu_pose.yaw);

            double now = (double)utils_get_time_ms() / 1000.0;
            if (now - last_imu_log > 2.0) {
                log_debug("IMU: pitch=%.1f roll=%.1f yaw=%.1f alt=%.1fm",
                          imu_pose.pitch, imu_pose.roll, imu_pose.yaw, imu_pose.altitude_m);
                last_imu_log = now;
            }
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

        visualizer_render_detection_view(sc->visualizer,
                                          frame_rgb, cam_w, cam_h,
                                          tracking.tracked_objects, tracking.num_tracked,
                                          sc->frame_count, avg_fps,
                                          vis_buffer);

        /* ── Output to display / stream / video file ── */
        if (sc->display_output) {
            display_output_write_frame(sc->display_output, vis_buffer);
        }

        if (sc->frame_count % 30 == 0) {
            log_info("Realtime frame %d: %d objects, %.1f FPS, %.1fms/frame",
                     sc->frame_count, tracking.num_tracked, avg_fps, frame_time * 1000.0f);
        }

        /* ── Max frames check ── */
        if (sc->max_frames > 0 && sc->frame_count >= sc->max_frames) {
            log_info("[realtime] Reached max_frames=%d, exiting", sc->max_frames);
            sc->running = false;
            break;
        }
    }

    free(vis_buffer);
    free(frame_rgb);

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
    result_manager_end_session(sc->result_manager);

    if (sc->ai_context) {
        ai_accel_context_destroy(sc->ai_context);
        sc->ai_context = NULL;
    }

    if (sc->arrow_receiver) {
        arrow_receiver_destroy(sc->arrow_receiver);
        sc->arrow_receiver = NULL;
    }
    if (sc->mjpeg_receiver) {
        mjpeg_receiver_destroy(sc->mjpeg_receiver);
        sc->mjpeg_receiver = NULL;
    }
    if (sc->display_output) {
        display_output_destroy(sc->display_output);
        sc->display_output = NULL;
    }

    sc->running = false;
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

    double total_time = (double)utils_get_time_ms() / 1000.0 - sc->start_time;

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
