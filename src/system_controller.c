#include "system_controller.h"
#include "logger.h"
#include "utils.h"
#include "k1_platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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

static void* k1_capture_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    log_info("[K1 Pipeline] Capture thread on CPU %d", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;

    uint8_t* frame_rgb = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!frame_rgb) {
        log_error("[K1 Pipeline] Failed to allocate capture buffer");
        return NULL;
    }

    while (pl->running) {
        int slot = k1_pipeline_slot_acquire(pl);
        if (slot < 0) break;

        K1PipelineSlot* s = &pl->slots[slot];
        bool got_frame = false;

        if (sc->arrow_receiver) {
            arrow_receiver_update(sc->arrow_receiver);
            ArrowSourceFrame arrow_frame;
            if (arrow_receiver_get_latest_frame(sc->arrow_receiver, &arrow_frame)) {
#ifdef HAS_K1_JPU
                if (k1_platform_has_cap(K1_CAP_JPU)) {
                    soft_jpeg_decode_to_rgb(arrow_frame.jpeg_data, arrow_frame.jpeg_len,
                                            s->rgb_data, cam_w, cam_h);
                } else
#endif
                {
                    soft_jpeg_decode_to_rgb(arrow_frame.jpeg_data, arrow_frame.jpeg_len,
                                            s->rgb_data, cam_w, cam_h);
                }
                got_frame = true;
                s->timestamp_us = arrow_frame.timestamp * 1000.0;
                s->frame_index = arrow_frame.frame_index;

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

        if (!got_frame) {
            memset(s->rgb_data, 64, (size_t)cam_w * cam_h * 3);
            s->timestamp_us = k1_get_time_us();
            s->frame_index = sc->frame_count;
        }

        k1_pipeline_slot_capture_done(pl, slot);
    }

    free(frame_rgb);
    return NULL;
}

static void* k1_inference_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
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

        if (sc->frame_count % 30 == 0) {
            log_info("[K1 Pipeline] Frame %d: %d objects, %.1f FPS",
                     sc->frame_count, tracking.num_tracked, avg_fps);
        }

        k1_pipeline_slot_release(pl, slot);
    }

    free(vis_buffer);
    return NULL;
}

static SystemStatus system_controller_process_realtime_k1(SystemController* sc,
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
    if (!plat || !plat->is_k1) {
        log_warning("K1 platform not available, falling back to single-threaded mode");
        return system_controller_process_realtime(sc, uart_device_A, uart_device_C, baudrate);
    }

    int cam_w = config_get_int(sc->config, "video.camera_width", 640);
    int cam_h = config_get_int(sc->config, "video.camera_height", 480);
    float cam_fps = (float)config_get_float(sc->config, "video.camera_fps", 15.0f);

    const char* camera_dev = config_get_string(sc->config, "video.camera_device", "/dev/video0");
    const char* camera_format = config_get_string(sc->config, "video.camera_format", "MJPEG");

    if (camera_dev && uart_device_A) {
        sc->video_processor = video_processor_create_from_camera(camera_dev, cam_w, cam_h, cam_fps, camera_format);
        if (sc->video_processor) {
            log_info("[K1 Pipeline] Camera device opened: %s", camera_dev);
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

    sc->ai_context = ai_accel_context_create();
    if (sc->ai_context) {
        const char* model_path = config_get_string(sc->config, "detection.model",
            "models/Human Recognition/yolov8n.onnx");
        int input_sz = config_get_int(sc->config, "detection.input_size", 640);
        ai_accel_load_model(sc->ai_context, AI_MODEL_YOLOV8N, model_path, input_sz, input_sz);
        ai_accel_warmup(sc->ai_context, AI_MODEL_YOLOV8N, input_sz, input_sz);
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
    sc->model_store = model_store_create("models", config_get_bool(sc->config, "system.use_onnx", false));
    if (!sc->model_store) {
        log_warning("Model store creation failed");
    }
    sc->video_processor = NULL;
    sc->imu_handler = imu_handler_create(10, 0.01f, 0.1f);

    const char* detection_backend = config_get_string(sc->config, "detection.backend", "cpu");
    bool use_ai_accel = (detection_backend != NULL && strcmp(detection_backend, "ai_accel") == 0);
    (void)use_ai_accel;
    sc->inference_pipeline = inference_pipeline_create(config_get_bool(sc->config, "system.use_onnx", false));
    if (!sc->inference_pipeline) {
        log_warning("Inference pipeline creation failed");
    }

    if (sc->inference_pipeline) {
        inference_pipeline_load_models(sc->inference_pipeline, "models");
    }

    sc->tracking_manager = object_tracker_create(
        config_get_int(sc->config, "tracking.max_lost", 30),
        config_get_float(sc->config, "tracking.min_iou", 0.3f),
        config_get_float(sc->config, "tracking.max_distance", 5.0f),
        config_get_int(sc->config, "tracking.max_track_history", 300)
    );

    float focal = config_get_float(sc->config, "spatial.fx", 500.0f);
    float avg_height = config_get_float(sc->config, "spatial.avg_human_height", 1.7f);
    sc->spatial_engine = spatial_engine_create(NULL, NULL, focal, avg_height);

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
    sc->mode = PIPELINE_MODE_OFFLINE;

    sc->frame_count = 0;
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    sc->fps_history_count = 0;
    sc->proc_times_count = 0;
    sc->detection_count = 0;
    sc->running = false;

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
    if (sc->ai_context) ai_accel_context_destroy(sc->ai_context);

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
    snprintf(output_video_path, sizeof(output_video_path), "%s/output.avi", output_dir);

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
    while ((frame = video_processor_read_frame_raw(processor)) != NULL) {
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
        const char* yolov8_model = config_get_string(sc->config, "detection.model",
            "models/Human Recognition/yolov8n.onnx");
        if (yolov8_model && ai_accel_is_available()) {
            int input_sz = config_get_int(sc->config, "detection.input_size", 640);
            ai_accel_load_model(sc->ai_context, AI_MODEL_YOLOV8N, yolov8_model, input_sz, input_sz);
            ai_accel_warmup(sc->ai_context, AI_MODEL_YOLOV8N, input_sz, input_sz);
        }
    } else {
        log_warning("AI acceleration not available, using CPU inference via ONNX Runtime");
    }

    int cam_w = config_get_int(sc->config, "video.camera_width", 640);
    int cam_h = config_get_int(sc->config, "video.camera_height", 480);
    float cam_fps = (float)config_get_float(sc->config, "video.camera_fps", 15.0f);

    const char* camera_dev = config_get_string(sc->config, "video.camera_device", "/dev/video0");
    const char* camera_format = config_get_string(sc->config, "video.camera_format", "MJPEG");

    if (camera_dev && uart_device_A) {
        sc->video_processor = video_processor_create_from_camera(camera_dev, cam_w, cam_h, cam_fps, camera_format);
        if (sc->video_processor) {
            log_info("Camera device opened: %s (%dx%d @ %.1f FPS)", camera_dev, cam_w, cam_h, cam_fps);
        } else {
            log_warning("Camera device %s unavailable, using Arrow-only mode", camera_dev);
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

    uint8_t* vis_buffer = (uint8_t*)malloc((size_t)cam_w * cam_h * 3);
    if (!vis_buffer) {
        strncpy(status.message, "Failed to allocate vis buffer", sizeof(status.message) - 1);
        result_manager_end_session(sc->result_manager);
        return status;
    }

    const char* session_id2 = result_manager_start_session(sc->result_manager, "realtime_session");
    (void)session_id2;

    sc->running = true;
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    double prev_time = sc->start_time;
    double last_imu_log = sc->start_time;

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

        bool got_frame = false;

        if (sc->arrow_receiver) {
            ArrowSourceFrame arrow_frame;
            if (arrow_receiver_get_latest_frame(sc->arrow_receiver, &arrow_frame)) {
                soft_jpeg_decode_to_rgb(arrow_frame.jpeg_data, arrow_frame.jpeg_len,
                                        frame_rgb, cam_w, cam_h);
                got_frame = true;

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

        if (!got_frame) {
            memset(frame_rgb, 64, (size_t)cam_w * cam_h * 3);
            log_debug("No frame available, waiting...");
#ifdef PLATFORM_RISCV64
            struct timespec ts = {0, 33333333L};
            nanosleep(&ts, NULL);
#endif
            continue;
        }

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

        if (sc->frame_count % 30 == 0) {
            log_info("Realtime frame %d: %d objects, %.1f FPS, %.1fms/frame",
                     sc->frame_count, tracking.num_tracked, avg_fps, frame_time * 1000.0f);
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
