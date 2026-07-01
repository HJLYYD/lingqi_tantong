#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "system_controller.h"
#include "pipeline_state.h"
#include "video_writer.h"
#include "logger.h"
#include "terminal_ui.h"
#include "utils.h"
#include "k1_platform.h"
#include "k1_imu.h"
#include "k1_odometry.h"
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
#include <signal.h>
#include <ucontext.h>
#include <turbojpeg.h>

/* ── SIGILL diagnostic: print faulting instruction address ── */
static void sigill_handler(int sig, siginfo_t* si, void* ctx) {
    ucontext_t* uc = (ucontext_t*)ctx;
    /* On RISC-V, the PC is in uc->uc_mcontext.__gregs[0] (REG_PC = 0) */
    uintptr_t fault_addr = (uintptr_t)si->si_addr;
    log_critical("!!! SIGILL at PC=0x%lx, fault_addr=0x%lx !!!",
                 (unsigned long)uc->uc_mcontext.__gregs[0],
                 (unsigned long)fault_addr);
    log_critical(">>> Run: addr2line -e build/lingqi_tantong 0x%lx",
                 (unsigned long)uc->uc_mcontext.__gregs[0]);
    log_flush();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void install_sigill_handler(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = sigill_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, NULL);
}

static void associate_poses_with_objects(TrackedObject* objects, int num_objects,
                                          PoseEstimation* poses, int num_poses);
static void associate_faces_with_objects(TrackedObject* objects, int num_objects,
                                          FaceIdentity* faces, int num_faces);
static float get_current_fps(const SystemController* sc);
static void render_overlay_bgr(uint8_t* rgb, int img_w, int img_h,
                                const InferenceResult* inference,
                                const TrackingResult* tracking,
                                float fps, int frame_num);

/* ── Push frame data to Web UI via WebSocket ──
 * Throttled to ~5 FPS (200ms).  Embeds a small JPEG thumbnail as base64
 * directly in the JSON so detection data and image stay in sync.
 * The frontend renders the thumbnail immediately — no separate HTTP fetch. */
static void push_frame_to_web(SystemController* sc, const InferenceResult* ir,
                               const TrackingResult* tr, int frame_num, float fps,
                               const uint8_t* rgb_data, int cam_w, int cam_h) {
    if (!sc || !sc->web_server) return;

    static int64_t last_push_ms = 0;
    int64_t now_ms = utils_get_time_ms();
    if (now_ms - last_push_ms < 200) return;
    last_push_ms = now_ms;

    /* ── Compress small JPEG thumbnail from RGB data ──
     * 320×240 @ quality 22 = ~2-5 KB. Throttled: every 2nd frame. */
    #define MAX_JPEG_SZ 8192
    char jpeg_b64[12288];
    jpeg_b64[0] = '\0';
    int jpeg_b64_len = 0;
    uint8_t raw_jpeg[8192];
    int raw_jpeg_len = 0;
    int tw = 320, th = 240;  /* thumbnail size for bbox scaling */

    static int frame_skip_counter = 1;
    int encode_this_frame = (++frame_skip_counter % 2 == 0);

    if (rgb_data && cam_w > 0 && cam_h > 0 && encode_this_frame) {
        /* Quick nearest-neighbour downsample to thumbnail size */
        uint8_t* thumb = (uint8_t*)malloc((size_t)tw * th * 3);
        if (thumb) {
            for (int y = 0; y < th; y++) {
                int sy = y * cam_h / th;
                for (int x = 0; x < tw; x++) {
                    int sx = x * cam_w / tw;
                    uint8_t* dst = thumb + (y * tw + x) * 3;
                    const uint8_t* src = rgb_data + (sy * cam_w + sx) * 3;
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
                }
            }
            /* libjpeg-turbo compress */
            unsigned long jpeg_sz = 0;
            uint8_t* jpeg_buf = NULL;
            tjhandle tj = tjInitCompress();
            if (tj) {
                int rc = tjCompress2(tj, thumb, tw, 0, th, TJPF_RGB,
                                     &jpeg_buf, &jpeg_sz,
                                     TJSAMP_420, 22, TJFLAG_FASTDCT);
                /* JPEG must be < MAX_JPEG_SZ to fit in base64 buffer */
                if (rc == 0 && jpeg_buf && jpeg_sz > 0 && jpeg_sz < MAX_JPEG_SZ) {
                    /* Save raw JPEG for binary WebSocket channel */
                    memcpy(raw_jpeg, jpeg_buf, (size_t)jpeg_sz);
                    raw_jpeg_len = (int)jpeg_sz;
                    /* Base64 encode (standard alphabet) */
                    static const char b64t[64] =
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    for (unsigned long i = 0; i < jpeg_sz; i += 3) {
                        uint32_t tri = ((uint32_t)jpeg_buf[i]) << 16;
                        if (i + 1 < jpeg_sz) tri |= ((uint32_t)jpeg_buf[i+1]) << 8;
                        if (i + 2 < jpeg_sz) tri |= ((uint32_t)jpeg_buf[i+2]);
                        int remain = (int)(jpeg_sz - i);
                        jpeg_b64[jpeg_b64_len++] = b64t[(tri >> 18) & 63];
                        jpeg_b64[jpeg_b64_len++] = b64t[(tri >> 12) & 63];
                        jpeg_b64[jpeg_b64_len++] = (remain >= 2) ? b64t[(tri >> 6) & 63] : '=';
                        jpeg_b64[jpeg_b64_len++] = (remain >= 3) ? b64t[tri & 63] : '=';
                    }
                } else if (rc == 0 && jpeg_sz >= MAX_JPEG_SZ) {
                    log_warn("[Web] JPEG thumbnail too large for base64 buffer: %lu bytes (max %d)", jpeg_sz, MAX_JPEG_SZ-1);
                }
                if (jpeg_buf) tjFree(jpeg_buf);
                tjDestroy(tj);
            }
            free(thumb);
        }
    }
    jpeg_b64[jpeg_b64_len] = '\0';

    char buf[WS_MAX_FRAME_JSON_LEN];
    int w = 0;
    int len = sizeof(buf);

    #define APPEND(...) do { \
        int n = snprintf(buf + w, (size_t)(len - w > 0 ? len - w : 0), __VA_ARGS__); \
        if (n > 0) w += n; if (w >= len) w = len; \
    } while(0)

    APPEND("{\"type\":\"f\",\"n\":%d,\"f\":%.1f", frame_num, (double)fps);

    /* Embedded JPEG thumbnail — detection + image sync guaranteed */
    if (jpeg_b64_len > 0) {
        APPEND(",\"j\":\"%s\"", jpeg_b64);
    }

    /* Scale bbox coords from camera resolution to thumbnail resolution */
    float sx = (cam_w > 0) ? (float)tw / (float)cam_w : 1.0f;
    float sy = (cam_h > 0) ? (float)th / (float)cam_h : 1.0f;

    /* Detections — top 10 by confidence, scaled to thumbnail coords */
    APPEND(",\"d\":[");
    int dcount = ir->num_detections > 10 ? 10 : ir->num_detections;
    for (int i = 0; i < dcount; i++) {
        if (i > 0) APPEND(",");
        const Detection* d = &ir->detections[i];
        APPEND("[%.0f,%.0f,%.0f,%.0f,%.1f]",
               (double)(d->bbox.x_min * sx), (double)(d->bbox.y_min * sy),
               (double)(d->bbox.x_max * sx), (double)(d->bbox.y_max * sy),
               (double)d->confidence);
    }
    APPEND("]");

    /* Poses — top 5, keypoints scaled to thumbnail coords */
    APPEND(",\"p\":[");
    int pcount = ir->num_poses > 5 ? 5 : ir->num_poses;
    for (int i = 0; i < pcount; i++) {
        if (i > 0) APPEND(",");
        const Pose* p = &ir->poses[i];
        APPEND("{\"k\":[");
        int first = 1;
        for (int k = 0; k < p->num_keypoints && k < MAX_KPTS; k++) {
            if (p->keypoints[k].confidence < 0.2f) continue;
            if (!first) APPEND(","); first = 0;
            APPEND("[%.0f,%.0f]", (double)(p->keypoints[k].x * sx), (double)(p->keypoints[k].y * sy));
        }
        APPEND("],\"b\":[%.0f,%.0f,%.0f,%.0f]}",
               (double)(p->bbox.x_min * sx), (double)(p->bbox.y_min * sy),
               (double)(p->bbox.x_max * sx), (double)(p->bbox.y_max * sy));
    }
    APPEND("]");

    /* Tracks — bbox scaled to thumbnail coords, spatial pos unchanged */
    APPEND(",\"t\":[");
    for (int i = 0; i < tr->num_tracked && i < MAX_TRACKS; i++) {
        if (i > 0) APPEND(",");
        const TrackedObj* t = &tr->tracked_objects[i];
        APPEND("{\"i\":%d,\"a\":%d,\"b\":[%.0f,%.0f,%.0f,%.0f]",
               t->track_id, t->is_active ? 1 : 0,
               (double)(t->detection.bbox.x_min * sx), (double)(t->detection.bbox.y_min * sy),
               (double)(t->detection.bbox.x_max * sx), (double)(t->detection.bbox.y_max * sy));
        if (t->spatial_pos.is_valid) {
            APPEND(",\"p\":[%.1f,%.1f,%.1f]",
                   (double)t->spatial_pos.x, (double)t->spatial_pos.y,
                   (double)t->spatial_pos.z);
        }
        APPEND("}");
    }
    APPEND("]");

    /* Action — short keys */
    if (ir->has_action && ir->action.num_actions > 0) {
        APPEND(",\"a\":[\"%s\",%.0f]",
               ir->action.actions[0].action_name,
               (double)(ir->action.actions[0].confidence * 100.0f));
    }

    APPEND("}");

    #undef APPEND

    if (w > 0 && w < len) {
        web_server_push_frame_jpeg(sc->web_server, buf, frame_num,
                                   raw_jpeg, raw_jpeg_len);
    }
}

/* ── Broadcast pipeline state change to Web UI ── */
static void broadcast_pipeline_state(SystemController* sc, const char* event) {
    if (!sc || !sc->web_server || !sc->web_server->running) return;
    PipelineState st = psm_get(&sc->state_machine);
    char data[256];
    snprintf(data, sizeof(data),
             "{\"state\":\"%s\",\"frame_count\":%d}",
             psm_state_name(st), sc->frame_count);
    web_server_broadcast_event(sc->web_server, event, data);
}

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
    volatile bool has_frame;       /* volatile + __sync_synchronize for RISC-V weak memory */

    InferenceResult inference;
    volatile bool has_inference;   /* volatile + __sync_synchronize */

    TrackingResult tracking;
    SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
    int num_positions;
    volatile bool has_tracking;    /* volatile + __sync_synchronize */

    /* ── K1 pose at frame capture time ──
     * Snapshot by capture thread immediately after IMU feed.
     * Used by postprocess thread for time-synchronized world-coord
     * registration (P_world = T_W_B(t_capture) × T_B_C × P_cam). */
    float k1_pos_at_capture[3];
    float k1_quat_at_capture[4];
    bool  has_k1_pose;

    int ref_count;
    pthread_mutex_t mutex;
    pthread_cond_t avail_cond;
} K1PipelineSlot;

typedef struct {
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

    /* ── Video output (--save-video or config visualization.record_to_video) ── */
    VideoWriter* video_writer;

    /* ── Capture throttle (set by main thread before thread creation) ── */
    int cap_min_interval_ms;   /* 0=disabled; use ~200 for 5 FPS decimation */
} K1Pipeline;

typedef struct {
    K1Pipeline* pipeline;
    int thread_id;
    int cpu_core;
    const char* name;
    bool started;              /* true if pthread_create succeeded */
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
    /* k1_platform_init() is already called by system_controller_process_realtime_k1()
     * before entering this function — it's a singleton, no need to re-init. */
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
    log_info("[Pipeline] k1_pipeline_destroy: broadcasting shutdown to %d threads", pl->num_threads);
    log_flush();
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
        log_info("[Pipeline] k1_pipeline_destroy: joining thread %d/%d", i, pl->num_threads);
        log_flush();
        pthread_join(pl->threads[i], NULL);
    }
    log_info("[Pipeline] k1_pipeline_destroy: all %d threads joined", pl->num_threads);
    log_flush();

    /* ── Finalize video output ── */
    if (pl->video_writer) {
        log_info("[Pipeline] k1_pipeline_destroy: destroying video_writer");
        log_flush();
        video_writer_destroy(pl->video_writer);
        pl->video_writer = NULL;
        log_info("[Pipeline] Video output finalized");
    }

    log_info("[Pipeline] k1_pipeline_destroy: destroying ring slots");
    log_flush();
    k1_ring_destroy_slots(pl);
    pthread_mutex_destroy(&pl->ring_mutex);
    pthread_cond_destroy(&pl->capture_done_cond);
    pthread_cond_destroy(&pl->infer_done_cond);
    pthread_cond_destroy(&pl->track_done_cond);
    pthread_cond_destroy(&pl->post_done_cond);
    log_info("[Pipeline] k1_pipeline_destroy: complete");
    log_flush();
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
    /* C11 release-store: all prior writes (frame data, inference init) happen-before
     * any thread that sees has_frame == true via acquire-load.  Replaces the
     * heavier __sync_synchronize() full barrier. */
    __sync_synchronize(); s->has_frame = true;
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

    /* C11 acquire-load: paired with release-store in capture_done.
     * On RISC-V, pthread_mutex_lock already provides acquire semantics,
     * so this is a safety double-check — no full barrier overhead. */
    for (int i = 0; i < K1_RING_SIZE; i++) {
        int idx = (pl->infer_idx + i) % K1_RING_SIZE;
        K1PipelineSlot* s = &pl->slots[idx];
        if (s->has_frame &&
            !s->has_inference) {
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
    __sync_synchronize(); s->has_inference = true;
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
            if (s->has_inference &&
                !s->has_tracking) {
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
    __sync_synchronize(); s->has_tracking = true;
    pthread_cond_signal(&s->avail_cond);
    pthread_mutex_unlock(&s->mutex);

    pthread_mutex_lock(&pl->ring_mutex);
    pthread_cond_signal(&pl->track_done_cond);
    pthread_mutex_unlock(&pl->ring_mutex);
}

static int k1_pipeline_slot_wait_tracked(K1Pipeline* pl) {
    pthread_mutex_lock(&pl->ring_mutex);
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
    /* C11 relaxed: the mutex provides the necessary ordering for slot reset */
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
        log_info("[Pipeline] %s thread elevated to SCHED_FIFO prio 50", role);
    } else if (rc == EPERM) {
        log_debug("[Pipeline] %s: SCHED_FIFO unavailable (no CAP_SYS_NICE), using SCHED_OTHER", role);
    } else {
        log_warning("[Pipeline] %s: pthread_setschedparam failed: %d", role, rc);
    }
}

static void* k1_capture_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    k1_apply_rt_priority("Capture");
    pl->thread_alive[0] = true;
    log_set_thread_name("capture");
    log_info("[Pipeline] Capture thread on CPU %d", ta->cpu_core);

    int cam_w = pl->slots[0].width;
    int cam_h = pl->slots[0].height;

    bool use_v4l2 = (sc->video_processor != NULL &&
                    video_processor_get_source_type(sc->video_processor) == VP_SOURCE_CAMERA &&
                    video_processor_is_opened(sc->video_processor));
    bool use_coap = (sc->coap_receiver != NULL);
    if (!use_v4l2 && !use_coap) {
        log_error("[Pipeline] Capture: no source available (no V4L2 camera or CoAP receiver)");
        return NULL;
    }

    double last_frame_time = k1_get_time_ms() / 1000.0;
    int cap_attempts = 0;
    int consecutive_decode_fails = 0;

    /* ── Frame decimation / skip tracking (CoAP source only) ──
     * The CoAP receiver gets frames at ~25 FPS from the ESP32, but the
     * inference pipeline can only process ~2-5 FPS.  Track how many CoAP
     * frames we skip between captures so the user can tune the decimation
     * rate or diagnose WiFi/ESP32 issues. */
    double   last_capture_time_ms = 0.0;
    int      last_coap_frame_idx  = -1;
    int      coap_total_skipped   = 0;
    int      coap_report_interval = 50;   /* log skip stats every N captured frames */
    int      coap_captures_since_report = 0;
    /* ── Minimum capture interval (set by main thread via K1Pipeline) ── */
    int      cap_min_interval_ms = pl->cap_min_interval_ms;

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

        /* ── K1 本地 IMU: 读取并馈送到 Madgwick 滤波器 ──
         * NOTE: k1_imu_read_sample() returns ALREADY bias-corrected data
         * (gyro bias subtracted at source).  The odometry layer uses zero
         * biases (default from calloc) to avoid double-correction.
         * The GLRT ZUPT window now stores bias-corrected values from
         * k1_odometry_update's own bias subtraction (which is a no-op
         * with zero biases — the data passes through unchanged). */
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

        /* ── K1 位姿快照: 在帧捕获时刻记录 K1 姿态 ──
         * 解决时间同步问题: world_coord_register_person() 原本在 postprocess
         * 线程中调用 k1_odometry_get_pose() (~100ms 延迟), 导致变换链使用
         * 过期位姿。此快照将 K1 姿态锁定在帧捕获时刻 (±5ms @100Hz IMU),
         * 确保 P_world = T_W_B(t_capture) × T_B_C × P_cam 的一致性。
         * 参考: VINS-Mono (Qin & Shen, IEEE T-RO 2018) — IMU 预积分必须
         * 与视觉帧时间戳精确配对。 */
        s->has_k1_pose = false;
        {
            K1Odometry* odom = imu_handler_get_odometry(sc->imu_handler);
            if (odom && k1_odometry_is_initialized(odom)) {
                k1_odometry_get_pose(odom, s->k1_pos_at_capture, s->k1_quat_at_capture);
                s->has_k1_pose = true;
            }
        }

        if (!got_frame && use_coap) {
            ArrowSourceFrame coap_frame;
            if (coap_receiver_get_latest_frame(sc->coap_receiver, &coap_frame)) {
                /* ── JPEG decode: libjpeg-turbo software ── */
                int dec_rc = soft_jpeg_decode_to_rgb(coap_frame.jpeg_data, coap_frame.jpeg_len,
                                                     s->rgb_data, cam_w, cam_h);
                if (dec_rc == 0) {
                    got_frame = true;
                    s->timestamp_us = (int64_t)coap_frame.timestamp;
                    s->frame_index = coap_frame.frame_index;
                    consecutive_decode_fails = 0;

                    /* Store JPEG for /api/frame.jpg web endpoint */
                    pthread_mutex_lock(&sc->jpeg_mutex);
                    free(sc->latest_jpeg);
                    sc->latest_jpeg = (uint8_t*)malloc(coap_frame.jpeg_len);
                    if (sc->latest_jpeg) {
                        memcpy(sc->latest_jpeg, coap_frame.jpeg_data, coap_frame.jpeg_len);
                        sc->latest_jpeg_len = coap_frame.jpeg_len;
                    }
                    pthread_mutex_unlock(&sc->jpeg_mutex);

                    /* ── CoAP frame skip tracking ── */
                    if (last_coap_frame_idx >= 0) {
                        int skipped = coap_frame.frame_index - last_coap_frame_idx - 1;
                        if (skipped > 0) {
                            coap_total_skipped += skipped;
                        }
                    }
                    last_coap_frame_idx = coap_frame.frame_index;
                    coap_captures_since_report++;

                    /* Periodic skip report (cumulative stats for the whole session) */
                    if (coap_captures_since_report >= coap_report_interval) {
                        int total_received = coap_frame.frame_index;
                        int total_captured = coap_captures_since_report;
                        int total_skipped  = coap_total_skipped;
                        float skip_pct = total_received > 0
                            ? (float)total_skipped * 100.0f / (float)(total_received) : 0.0f;
                        log_info("[Capture] CoAP decimation: %d captured (recent), "
                                 "%d received total, %.1f%% skipped cumulative",
                                 total_captured, total_received, (double)skip_pct);
                        coap_captures_since_report = 0;
                    }
                } else {
                    consecutive_decode_fails++;
                    if (consecutive_decode_fails <= 3 || consecutive_decode_fails % 10 == 0) {
                        log_warning("[Pipeline] CoAP decode failed frame#%d: %zu bytes "
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
                log_info("[Pipeline] No frames received for %.0fs (timeout=%ds), shutting down",
                         idle_s, sc->frame_timeout_s);
                log_flush();  /* flush BEFORE shutdown ops — crash may follow */
                /* ── Coordinated shutdown through pl->shutdown only ──
                 * sc->running is owned by the main monitor thread.
                 * Setting pl->shutdown under the ring mutex ensures all
                 * waiting threads atomically see the flag and break. */
                pthread_mutex_lock(&pl->ring_mutex);
                pl->running = false;
                pl->shutdown = true;
                pthread_cond_broadcast(&pl->capture_done_cond);
                pthread_cond_broadcast(&pl->infer_done_cond);
                pthread_cond_broadcast(&pl->track_done_cond);
                pthread_cond_broadcast(&pl->post_done_cond);
                /* Return the acquired slot (active_slots was bumped by acquire) */
                pl->active_slots--;
                pl->capture_idx = (pl->capture_idx - 1 + K1_RING_SIZE) % K1_RING_SIZE;
                pthread_cond_signal(&pl->post_done_cond);
                pthread_mutex_unlock(&pl->ring_mutex);
                log_info("[Pipeline] Capture thread shutdown broadcast complete, exiting");
                log_flush();
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

        /* ── Minimum capture interval throttle ──
         * Prevents burst captures when the pipeline starts (ring buffer
         * initially empty) or when inference suddenly speeds up.  This
         * ensures consistent frame spacing in the output video even at
         * low effective FPS.  Default 0 = disabled (no artificial delay). */
        if (cap_min_interval_ms > 0 && last_capture_time_ms > 0.0) {
            double now_ms = k1_get_time_ms();
            double elapsed_ms = now_ms - last_capture_time_ms;
            if (elapsed_ms < (double)cap_min_interval_ms) {
                double sleep_ms = (double)cap_min_interval_ms - elapsed_ms;
                struct timespec ts;
                ts.tv_sec  = (time_t)(sleep_ms / 1000.0);
                ts.tv_nsec = (long)((sleep_ms - ts.tv_sec * 1000.0) * 1e6);
                nanosleep(&ts, NULL);
            }
        }
        last_capture_time_ms = k1_get_time_ms();

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
    log_set_thread_name("inference");
    log_info("[Pipeline] Inference thread on CPU %d (Cluster0)", ta->cpu_core);

    while (pl->running) {
        pl->thread_heartbeats[1]++;
        int slot = k1_pipeline_slot_wait_captured(pl);
        if (slot < 0) {
            if (pl->shutdown) break;
            usleep(1000);
            continue;
        }

        K1PipelineSlot* s = &pl->slots[slot];

        if (!sc->inference_pipeline) {
            log_error("[Pipeline] Inference pipeline is NULL — aborting inference thread");
            k1_pipeline_slot_infer_done(pl, slot);
            break;
        }

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
    log_set_thread_name("stgcn");
    log_info("[Pipeline] ST-GCN thread on CPU %d (Cluster0)", ta->cpu_core);

    int interval = 10; /* default, updated below */
    {
        AIInferencePipeline* ip = sc->inference_pipeline;
        if (ip && ip->action_recognizer) {
            interval = ip->action_inference_interval;
        }
    }
    log_info("[Pipeline] ST-GCN async interval: %d frames", interval);

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
    log_set_thread_name("postproc");
    log_info("[Pipeline] Post-process thread on CPU %d (Cluster0)", ta->cpu_core);

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

        /* Write tracking result directly into slot (avoids stack copy of 1.3MB) */
        TrackingResult* tracking = &s->tracking;
        object_tracker_update(
            sc->tracking_manager, inference->detections, inference->num_detections,
            s->positions, num_positions, sc->frame_count, tracking);

        object_tracker_associate_poses(sc->tracking_manager, inference->poses, inference->num_poses);

        /* ── Sync confirmed track counts to inference pipeline for cascade state machine ── */
        if (sc->inference_pipeline) {
            sc->inference_pipeline->confirmed_track_count =
                object_tracker_get_confirmed_count(sc->tracking_manager);
            sc->inference_pipeline->total_track_count =
                object_tracker_get_all_track_count(sc->tracking_manager);
        }

        associate_poses_with_objects(tracking->tracked_objects, tracking->num_tracked,
                                      inference->poses, inference->num_poses);
        associate_faces_with_objects(tracking->tracked_objects, tracking->num_tracked,
                                      inference->faces, inference->num_faces);

        for (int i = 0; i < tracking->num_tracked; i++) {
            TrackedObject* obj = &tracking->tracked_objects[i];
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
        for (int i = 0; i < tracking->num_tracked; i++) {
            TrackedObject* obj = &tracking->tracked_objects[i];
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

        /* ── Phase B: 更新 WorldCoord 外参 (Wahba 对齐) ── */
        if (sc->world_coord && sc->imu_handler) {
            DualImuPose dual;
            if (imu_handler_get_dual_pose(sc->imu_handler, &dual) && dual.align_valid) {
                float q_align[4] = {dual.align_qw, dual.align_qx, dual.align_qy, dual.align_qz};
                float t_zero[3] = {0.0f, 0.0f, 0.0f};
                world_coord_set_alignment(sc->world_coord, q_align, t_zero);
            }

            /* 为每个检测到的人注册世界坐标 */
            for (int i = 0; i < tracking->num_tracked; i++) {
                TrackedObject* obj = &tracking->tracked_objects[i];
                if (!obj->spatial_pos.is_valid) continue;

                /* ── 从当前帧 2D 检测 + 深度逆投影得到相机坐标 P_cam ──
                 * FIX: 统一使用当前 bbox 中心 + 当前平滑深度,
                 * 而非混合 world_x (旧 moment) + 单独反推 Y (新 moment).
                 * 参考: Hartley & Zisserman §6.2 — K 定义像素↔射线双射,
                 * 混合不同深度的分量导致不一致的射线. */
                float depth = obj->spatial_pos.world_z;
                float fx = sc->spatial_engine->camera_matrix[0][0];
                float fy = sc->spatial_engine->camera_matrix[1][1];
                float cx = sc->spatial_engine->camera_matrix[0][2];
                float cy = sc->spatial_engine->camera_matrix[1][2];
                float u = bbox_center_x(&obj->detection.bbox);
                float v = bbox_center_y(&obj->detection.bbox);

                float P_cam[3];
                P_cam[0] = (u - cx) * depth / fx;
                P_cam[1] = (v - cy) * depth / fy;
                P_cam[2] = depth;

                /* 提取 2D 关键点 (用于 3D 骨架重建) */
                float kpts_2d[17][2];
                int num_kpts = 0;
                if (obj->has_pose) {
                    num_kpts = (obj->pose.num_keypoints < 17) ? obj->pose.num_keypoints : 17;
                    for (int k = 0; k < num_kpts; k++) {
                        kpts_2d[k][0] = obj->pose.keypoints[k].x;
                        kpts_2d[k][1] = obj->pose.keypoints[k].y;
                    }
                }

                double now_s = (double)s->timestamp_us / 1e6;
                world_coord_register_person(sc->world_coord, obj->track_id,
                                             P_cam, kpts_2d, num_kpts,
                                             obj->spatial_pos.world_z,
                                             obj->spatial_pos.confidence,
                                             now_s,
                                             s->has_k1_pose ? s->k1_pos_at_capture : NULL,
                                             s->has_k1_pose ? s->k1_quat_at_capture : NULL,
                                             &obj->detection.bbox);
            }

            /* 清理超时人员 */
            double now_s = (double)s->timestamp_us / 1e6;
            world_coord_prune_timeout(sc->world_coord, now_s);
        }

        /* ── 双 IMU 状态 (每 30 帧) ── */
        if (sc->frame_count % 30 == 0 && sc->imu_handler) {
            if (!imu_handler_is_alignment_done(sc->imu_handler)) {
                log_info("[IMU] Aligning frames... (%d/200 samples)",
                         sc->imu_handler->align_ctx.samples_collected);
            }
        }

        sc->detection_count += tracking->num_tracked;

        /* ── Per-frame pipeline timing ── */
        double proc_start_ms = k1_get_time_ms();
        double frame_time = (proc_start_ms - (s->timestamp_us / 1000.0));
        if (sc->proc_times_count < SC_MAX_PROC_TIMES) {
            sc->processing_times[sc->proc_times_count++] = (float)(frame_time / 1000.0);
        }

        /* ── Record frame wall-clock timestamp for accurate FPS calculation ──
         * Store the absolute timestamp (seconds) rather than an instantaneous
         * FPS rate.  get_current_fps() computes (N-1) / Δt from these. */
        if (sc->fps_history_count < SC_MAX_FPS_HISTORY) {
            sc->fps_history[sc->fps_history_count++] = (float)(k1_get_time_ms() / 1000.0);
        }

        /* tracking already points to s->tracking — data is in-place */

        /* Max frames check */
        if (sc->max_frames > 0 && sc->frame_count >= sc->max_frames) {
            log_info("[Pipeline] Reached max_frames=%d, shutting down", sc->max_frames);
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
 * to Cluster1.  Renders bounding-box overlay + skeleton onto frame data
 * and writes to video output when --save-video is enabled.
 */
static void* k1_viz_thread(void* arg) {
    K1ThreadArg* ta = (K1ThreadArg*)arg;
    K1Pipeline* pl = ta->pipeline;
    SystemController* sc = pl->controller;

    k1_pin_current_to_cpu(ta->cpu_core);
    pl->thread_alive[4] = true;
    log_set_thread_name("viz");
    log_info("[Pipeline] Viz thread on CPU %d (Cluster1)", ta->cpu_core);

    while (pl->running) {
        pl->thread_heartbeats[4]++;

        int slot = k1_pipeline_slot_wait_tracked(pl);
        if (slot < 0) break;

        K1PipelineSlot* s = &pl->slots[slot];
        TrackingResult* tracking = &s->tracking;
        float avg_fps = get_current_fps(sc);

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

        /* ── Push frame data to Web UI ── */
        push_frame_to_web(sc, &s->inference, tracking, sc->frame_count, avg_fps,
                          s->rgb_data, s->width, s->height);

        /* ── Render overlay + video output ── */
        if (pl->video_writer) {
            render_overlay_bgr(s->rgb_data, s->width, s->height,
                               &s->inference, tracking, avg_fps, sc->frame_count);
            video_writer_write_frame(pl->video_writer, s->rgb_data);
        }

        k1_pipeline_slot_release(pl, slot);
    }

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
    install_sigill_handler();  /* diagnostic: catch illegal instruction + print PC */

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

    /* CLI --camera overrides config value; stored in sc->camera_device */
    const char* camera_dev = (sc->camera_device[0] != '\0')
        ? sc->camera_device
        : config_get_string(sc->config, "video.camera_device", "/dev/video0");
    const char* camera_format = config_get_string(sc->config, "video.camera_format", "MJPEG");

    if (camera_dev) {
        sc->video_processor = video_processor_create_from_camera(camera_dev, cam_w, cam_h, cam_fps, camera_format);
        if (sc->video_processor) {
            log_info("[Pipeline] Camera device opened: %s (%dx%d @ %.1f FPS)", camera_dev, cam_w, cam_h, cam_fps);
        } else {
            log_warning("[Pipeline] V4L2 camera %s unavailable; will rely on CoAP receiver", camera_dev);
        }
    }
    log_flush();  /* ensure camera init result is visible before CoAP init */

    /* ── CoAP/UDP receiver (WiFi from ESP32) ── */
    if (sc->coap_enabled && sc->coap_esp_ip[0] != '\0') {
        log_info("[Pipeline] Creating CoAP receiver for %s:%d (SSID=%s)...",
                 sc->coap_esp_ip, sc->coap_esp_port,
                 sc->coap_wifi_ssid[0] ? sc->coap_wifi_ssid : "(none)");
        log_flush();  /* flush BEFORE coap_receiver_create — which may block on wifi_connect() */

        sc->coap_receiver = coap_receiver_create(
            sc->coap_esp_ip, sc->coap_esp_port,
            sc->coap_wifi_ssid[0] != '\0' ? sc->coap_wifi_ssid : NULL,
            sc->coap_wifi_password[0] != '\0' ? sc->coap_wifi_password : NULL);
        if (sc->coap_receiver) {
            log_info("[Pipeline] CoAP receiver ready: %s:%d (CoAP/UDP)",
                     sc->coap_esp_ip, sc->coap_esp_port);
            /* 注册原始 IMU 数据回调 → Madgwick 相机滤波器 */
            coap_receiver_set_imu_raw_callback(sc->coap_receiver,
                (CoapImuRawCallback)imu_handler_feed_external_raw, sc->imu_handler);
        } else {
            log_warning("[Pipeline] CoAP receiver creation FAILED — "
                        "will retry or use fallback source");
        }
    } else {
        log_info("[Pipeline] CoAP receiver DISABLED (coap_enabled=%d, esp_ip=%s)",
                 sc->coap_enabled,
                 sc->coap_esp_ip[0] ? sc->coap_esp_ip : "(empty)");
    }
    log_flush();  /* ensure CoAP init result is visible */

    /* ── K1 本地 I2C IMU (GY85) ── */
    int k1_imu_bus = config_get_int(sc->config, "k1_imu.i2c_bus", 4);
    float k1_imu_rate = (float)config_get_float(sc->config, "k1_imu.sample_rate_hz", 100.0);
    K1Imu* k1_imu = k1_imu_create(k1_imu_bus, k1_imu_rate);
    if (k1_imu && k1_imu_get_state(k1_imu) != K1_IMU_STATE_ERROR) {
        imu_handler_set_k1_imu(sc->imu_handler, k1_imu);
        log_info("[Pipeline] K1 local IMU: /dev/i2c-%d @ %.0fHz", k1_imu_bus, k1_imu_rate);
    } else {
        if (k1_imu) {
            k1_imu_destroy(k1_imu);  /* prevent leak when state is ERROR */
            k1_imu = NULL;
        }
        log_warning("[Pipeline] K1 local IMU unavailable — using ESP32 IMU only");
    }

    K1Pipeline pl;
    k1_pipeline_init(&pl, sc);
    if (!k1_ring_init_slots(&pl, cam_w, cam_h)) {
        strncpy(status.message, "Failed to init ring buffer", sizeof(status.message) - 1);
        k1_pipeline_destroy(&pl);
        goto k1_cleanup;
    }

    const char* session_id = result_manager_start_session(sc->result_manager, "realtime_k1");

    /* ── Create video output (--save-video CLI or config visualization.record_to_video) ── */
    {
        bool record_enabled = sc->save_video ||
            config_get_bool(sc->config, "visualization.record_to_video", false);
        if (record_enabled) {
            char video_out[MAX_PATH_LEN];
            result_manager_get_video_path(sc->result_manager, session_id,
                                          video_out, sizeof(video_out));
            float out_fps = (float)config_get_float(sc->config, "video.camera_fps", 15.0f);
            pl.video_writer = video_writer_create(video_out, cam_w, cam_h, out_fps > 0.0f ? out_fps : 15.0f);
            if (pl.video_writer) {
                log_info("[Pipeline] Video output enabled: %s (%dx%d @ %.1f FPS)",
                         video_out, cam_w, cam_h, out_fps);
            } else {
                log_warning("[Pipeline] Failed to create video writer — video output disabled");
            }
        } else {
            log_info("[Pipeline] Headless mode (no video output)");
            pl.video_writer = NULL;
        }
    }

    /* Start IMU recording (K1 local GY85 + ESP32 remote via CoAP) */
    if (sc->imu_handler) {
        imu_handler_start_recording(sc->imu_handler, "output", session_id);
    }

    /* Start WorldCoord recording (person world coordinates CSV) */
    if (sc->world_coord) {
        world_coord_start_recording(sc->world_coord, "output", session_id);
    }

    /* ── Capture throttle config (read in main thread, safe for worker access) ── */
    pl.cap_min_interval_ms = config_get_int(sc->config, "video.capture_min_interval_ms", 0);

    K1ThreadArg thread_args[K1_MAX_PIPELINE_THREADS];
    int t = 0;
    bool all_threads_ok = true;
    #define CREATE_THREAD(core, name, func) do {                                           \
        thread_args[t] = (K1ThreadArg){&pl, t, core, name, false};                        \
        int ret = pthread_create(&pl.threads[t], NULL, func, &thread_args[t]);            \
        if (ret == 0) {                                                                   \
            thread_args[t].started = true;                                                 \
            t++;                                                                          \
        } else {                                                                          \
            log_error("[Pipeline] Failed to create %s thread: %s", name, strerror(ret)); \
            all_threads_ok = false;                                                       \
        }                                                                                 \
    } while(0)

    CREATE_THREAD(K1_CPU_CLUSTER1_CAPTURE, "Capture", k1_capture_thread);
    CREATE_THREAD(K1_CPU_CLUSTER0_INFERENCE, "Inference", k1_inference_thread);
    CREATE_THREAD(K1_CPU_CLUSTER0_STGCN, "ST-GCN", k1_stgcn_thread);
    CREATE_THREAD(K1_CPU_CLUSTER0_AI, "PostProcess", k1_postprocess_thread);
    CREATE_THREAD(K1_CPU_CLUSTER1_VIZ, "Viz", k1_viz_thread);
    #undef CREATE_THREAD

    if (!all_threads_ok) {
        log_warning("[Pipeline] Some threads failed to start — pipeline may be degraded");
    }

    pl.num_threads = t;

    log_info("[Pipeline] %d threads: Capture(CPU%d)→Inference(CPU%d)→ST-GCN(CPU%d)→Post(CPU%d)→Viz(CPU%d)",
             pl.num_threads,
             K1_CPU_CLUSTER1_CAPTURE, K1_CPU_CLUSTER0_INFERENCE,
             K1_CPU_CLUSTER0_STGCN, K1_CPU_CLUSTER0_AI, K1_CPU_CLUSTER1_VIZ);

    sc->running = 1;
    sc->start_time = k1_get_time_ms() / 1000.0;

    log_info("[Pipeline] === Pipeline alive ===");
    log_info("[Pipeline] Source: %s",
             sc->coap_receiver ? "CoAP/UDP" :
                (sc->video_processor ? "V4L2 Camera" : "None"));
    log_info("[Pipeline] Waiting for frames. Idle timeout: %ds. Press Ctrl+C to stop.",
             sc->frame_timeout_s);
    log_flush();  /* flush all init logs before entering monitor loop */

    /* ── Main thread: startup wait + heartbeat monitor ──
     * Elevate to SCHED_FIFO prio 51 (1 tick above pipeline threads at 50)
     * so shutdown broadcasts + pthread_join can preempt worker threads. */
    {
        struct sched_param sp;
        memset(&sp, 0, sizeof(sp));
        sp.sched_priority = 51;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0) {
            log_info("[Pipeline] Monitor thread elevated to SCHED_FIFO prio 51");
        }

        int prev_hearts[K1_PIPELINE_NUM_HEARTS] = {0};  /* all 6 hearts zero-initialized */
        int stuck_count = 0;
        int startup_ticks = 0;
        int monitor_ticks = 0;  /* for periodic status line updates */
        while (sc->running && !pl.shutdown) {  /* shutdown flag from capture timeout or SIGINT */
            /* 500ms poll interval, split into 10ms slices for responsive shutdown */
            for (int p = 0; p < 50 && sc->running && !pl.shutdown; p++) {
                usleep(10000);
            }
            if (!sc->running || pl.shutdown) break;
            monitor_ticks++;

            /* ── Realtime status bar (every second = 2 × 500ms ticks) ──
             * Multi-segment Claude Code-style layout:
             *   Frames N │ Tracks N │ Dets N │ FPS 11.0 │ TRCK │ Up 34s
             * Segments auto-drop from right when terminal < 80 cols. */
            if (tui_is_interactive() && monitor_ticks % 2 == 0) {
                double uptime_s = (k1_get_time_ms() / 1000.0) - sc->start_time;
                int active_tracks = sc->tracking_manager
                    ? sc->tracking_manager->num_tracks : 0;
                int fc = sc->frame_count;
                /* Rolling 10-sample FPS (computed from frame timestamps) */
                float rolling_fps = 0.0f;
                if (sc->fps_history_count >= 10) {
                    int oldest = sc->fps_history_count - 10;
                    float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[oldest];
                    rolling_fps = (dt > 0.001f) ? 9.0f / dt : 0.0f;
                } else if (sc->fps_history_count >= 2) {
                    float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[0];
                    rolling_fps = (dt > 0.001f) ? (float)(sc->fps_history_count - 1) / dt : 0.0f;
                }
                int dets = sc->detection_count;
                /* Cascade state with color coding */
                const char* cascade_label = "SRCH";
                TUIColorSlot cascade_color = TUI_CLR_STATUS_WARNING;  /* searching = amber */
                if (sc->inference_pipeline) {
                    switch (sc->inference_pipeline->cascade_state) {
                    case PIPELINE_CASCADE_TRACKING:
                        cascade_label = "TRCK";
                        cascade_color = TUI_CLR_STATUS_SUCCESS;  /* tracking = green */
                        break;
                    case PIPELINE_CASCADE_VALIDATING:
                        cascade_label = "VALD";
                        cascade_color = TUI_CLR_STATUS_INFO;     /* validating = cyan */
                        break;
                    default: break;
                    }
                }

                /* FPS color: green >10, amber >5, red <5 */
                TUIColorSlot fps_color = TUI_CLR_STATUS_ERROR;
                if (rolling_fps > 10.0f)      fps_color = TUI_CLR_STATUS_SUCCESS;
                else if (rolling_fps > 5.0f)  fps_color = TUI_CLR_STATUS_WARNING;

                /* Track count color */
                TUIColorSlot trk_color = TUI_CLR_FG_MUTED;
                if (active_tracks > 0)        trk_color = TUI_CLR_STATUS_INFO;

                tui_status_bar_begin();
                tui_status_bar_segment("Frames",  "%d",   fc);
                tui_status_bar_segment_colored(trk_color, "Tracks", "%d", active_tracks);
                tui_status_bar_segment("Dets",    "%d",   dets);
                tui_status_bar_segment_colored(fps_color, "FPS", "%.1f", (double)rolling_fps);
                tui_status_bar_segment_colored(cascade_color, "", "%s", cascade_label);
                {
                    int m = (int)uptime_s / 60, s = (int)uptime_s % 60;
                    tui_status_bar_segment("Up", "%dm%02ds", m, s);
                }
                tui_status_bar_end();
            }

            /* ── Startup grace: wait up to 30s for all threads to come alive ── */
            int all_alive = 1;
            for (int i = 0; i < pl.num_threads && i < K1_PIPELINE_NUM_HEARTS; i++) {
                if (!pl.thread_alive[i]) { all_alive = 0; break; }
            }
            if (!all_alive) {
                startup_ticks++;
                if (startup_ticks > 60) {  /* 30s timeout */
                    log_error("[Pipeline] Worker threads failed to start within 30s");
                    sc->running = 0;
                    psm_transition(&sc->state_machine, PIPELINE_STATE_ERROR,
                                   "Worker threads failed to start within 30s");
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

            /* ── Transition to RUNNING once all threads are alive ── */
            if (psm_get(&sc->state_machine) == PIPELINE_STATE_STARTING) {
                psm_transition(&sc->state_machine, PIPELINE_STATE_RUNNING,
                               "All worker threads alive");
                broadcast_pipeline_state(sc, "pipeline_running");
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
                log_error("[Pipeline] Capture thread stuck (%d checks), shutting down",
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

    /* Clear the status bar on exit */
    if (tui_is_interactive()) {
        tui_status_bar_clear();
    }

    log_info("[Pipeline] Monitor loop exited (shutdown=%d running=%d), entering k1_pipeline_destroy",
             (int)pl.shutdown, (int)pl.running);
    log_flush();

    /* ── Transition: RUNNING → STOPPING (if not already) ── */
    if (psm_get(&sc->state_machine) == PIPELINE_STATE_RUNNING) {
        psm_transition(&sc->state_machine, PIPELINE_STATE_STOPPING,
                       "Monitor loop exited, shutting down pipeline");
        broadcast_pipeline_state(sc, "pipeline_stopping");
    }

    k1_pipeline_destroy(&pl);

k1_cleanup:
    ;
    log_info("[Pipeline] k1_cleanup: calculating session stats");
    log_flush();

    float avg_fps = 0.0f;
    if (sc->fps_history_count >= 2) {
        float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[0];
        avg_fps = (dt > 0.001f) ? (float)(sc->fps_history_count - 1) / dt : 0.0f;
    }

    float avg_time_ms = 0.0f;
    if (sc->proc_times_count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < sc->proc_times_count; i++) sum += sc->processing_times[i];
        avg_time_ms = (sum / sc->proc_times_count) * 1000.0f;
    }

    log_info("[Pipeline] k1_cleanup: ending session (frames=%d dets=%d fps=%.1f)",
             sc->frame_count, sc->detection_count, (double)avg_fps);
    log_flush();

    result_manager_update_session_stats(sc->result_manager, sc->frame_count, sc->detection_count, avg_fps, avg_time_ms);
    result_manager_end_session(sc->result_manager, sc->spatial_engine);

    log_info("[Pipeline] k1_cleanup: session ended, destroying CoAP receiver");
    log_flush();

    if (sc->coap_receiver) {
        coap_receiver_destroy(sc->coap_receiver);
        sc->coap_receiver = NULL;
    }

    log_info("[Pipeline] k1_cleanup: CoAP destroyed, stopping world_coord recording");
    log_flush();

    if (sc->world_coord) {
        world_coord_stop_recording(sc->world_coord);
    }

    log_info("[Pipeline] k1_cleanup: complete, returning final status");
    log_flush();

    sc->running = 0;
    return system_controller_get_final_status(sc);
}

SystemController* system_controller_create(const char* config_path,
                                            const char* pose_model_variant) {
    log_info("============================================================");
    log_info("LingQi TanTong System Controller Initializing...");
    log_info("============================================================");

    SystemController* sc = (SystemController*)calloc(1, sizeof(SystemController));
    if (!sc) return NULL;

    if (pose_model_variant && pose_model_variant[0] != '\0') {
        strncpy(sc->pose_model_variant, pose_model_variant,
                sizeof(sc->pose_model_variant) - 1);
    }

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

    /* ── Phase A: Configure K1 Odometry from config ── */
    {
        float zupt_accel = config_get_float(sc->config, "odometry.zupt_accel_thresh", 0.15f);
        float zupt_gyro  = config_get_float(sc->config, "odometry.zupt_gyro_thresh", 0.10f);
        float sigma_a    = config_get_float(sc->config, "odometry.sigma_accel", 0.01f);
        float sigma_w    = config_get_float(sc->config, "odometry.sigma_gyro", 0.015f);
        float glrt_thr   = config_get_float(sc->config, "odometry.glrt_threshold", 3.0e5f);
        int   init_samp  = config_get_int(sc->config, "odometry.init_duration_s", 2);
        init_samp = init_samp * 100;  /* seconds → samples @100Hz */
        imu_handler_set_odometry_params(sc->imu_handler, zupt_accel, zupt_gyro,
                                         sigma_a, sigma_w, glrt_thr, init_samp);
        log_info("[Odom] Configured: zupt=(%.2f,%.2f) sigma=(%.3f,%.3f) glrt=%.0f init=%d",
                 zupt_accel, zupt_gyro, sigma_a, sigma_w, glrt_thr, init_samp);
    }

    /* ── CLI override: inject pose model variant into config before model load ── */
    if (sc->pose_model_variant[0] != '\0') {
        config_set_string(sc->config, "pose.model_variant", sc->pose_model_variant);
        log_info("Pose model variant override: %s", sc->pose_model_variant);
    }

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

    /* ── Phase B: World Coordinate System (dynamic K1-centric) ── */
    {
        float default_align[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float default_cam[9] = {fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f};
        int cam_w = config_get_int(sc->config, "video.camera_width", 640);
        int cam_h = config_get_int(sc->config, "video.camera_height", 480);
        sc->world_coord = world_coord_create(
            imu_handler_get_odometry(sc->imu_handler),
            default_align,
            default_cam,
            cam_w, cam_h);
    }

    sc->result_manager = result_manager_create("output");

    sc->coap_receiver = NULL;
    sc->mode = PIPELINE_MODE_OFFLINE;

    sc->frame_count = 0;
    sc->max_frames = config_get_int(sc->config, "system.max_frames", 0);
    sc->frame_timeout_s = config_get_int(sc->config, "coap.frame_timeout_s", 10);
    sc->start_time = (double)utils_get_time_ms() / 1000.0;
    sc->fps_history_count = 0;
    sc->proc_times_count = 0;
    sc->detection_count = 0;
    sc->running = 0;

    /* ── GUI-driven pipeline state machine ── */
    psm_init(&sc->state_machine);
    sc->pipeline_thread = 0;

    /* ── Web UI JPEG frame buffer ── */
    pthread_mutex_init(&sc->jpeg_mutex, NULL);
    sc->latest_jpeg = NULL;
    sc->latest_jpeg_len = 0;

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

    /* ── BUGFIX: Stop threads BEFORE freeing resources ──
     * CoAP receiver thread writes JPEG frames into buffers owned by
     * video_processor/inference_pipeline.  If we free those buffers
     * while the thread is running → use-after-free → segfault.
     *
     * Order: 1) Signal stop  2) Destroy CoAP (joins thread)
     *        3) Wait for pipeline  4) Free remaining resources */
    sc->running = 0;

    /* Step 1: Destroy CoAP receiver FIRST (joins its thread) */
    if (sc->coap_receiver) {
        coap_receiver_destroy(sc->coap_receiver);
        sc->coap_receiver = NULL;
    }

    /* Step 2: Force pipeline stop and wait briefly */
    if (psm_is_running(&sc->state_machine)) {
        psm_transition(&sc->state_machine, PIPELINE_STATE_STOPPING,
                       "Force-stop from destroy");
        for (int wait = 0; wait < 30 && psm_get(&sc->state_machine) != PIPELINE_STATE_IDLE; wait++) {
            usleep(200000);  /* 200ms × 30 = 6s max */
        }
    }

    /* Force IDLE so cleanup can proceed safely */
    if (psm_get(&sc->state_machine) != PIPELINE_STATE_IDLE) {
        PipelineStateMachine* psm = (PipelineStateMachine*)&sc->state_machine;
        pthread_mutex_lock(&psm->state_mutex);
        psm->state = PIPELINE_STATE_IDLE;
        pthread_mutex_unlock(&psm->state_mutex);
    }

    log_info("[Destroy] Threads stopped, freeing resources...");
    log_flush();

    /* ── Now safe to free resources ──
     * Each step has its own log_flush to pinpoint segfault location.
     * Order matters: world_coord → imu_handler (odometry held by world_coord
     * is destroyed by imu_handler, so world_coord must go first). */

    if (sc->config) {
        config_manager_destroy(sc->config);
        sc->config = NULL;
        log_debug("[Destroy] config_manager done"); log_flush();
    }
    if (sc->model_store) {
        model_store_destroy(sc->model_store);
        sc->model_store = NULL;
        log_debug("[Destroy] model_store done"); log_flush();
    }
    if (sc->video_processor) {
        video_processor_destroy(sc->video_processor);
        sc->video_processor = NULL;
        log_debug("[Destroy] video_processor done"); log_flush();
    }

    /*
     * CRITICAL: world_coord holds a pointer to imu_handler->odometry.
     * Destroy world_coord BEFORE imu_handler to avoid use-after-free.
     * world_coord_destroy only closes CSV + frees wc, it does NOT
     * access odometry, so this is safe.
     */
    if (sc->world_coord) {
        world_coord_destroy(sc->world_coord);
        sc->world_coord = NULL;
        log_debug("[Destroy] world_coord done"); log_flush();
    }

    if (sc->imu_handler) {
        imu_handler_destroy(sc->imu_handler);
        sc->imu_handler = NULL;
        log_debug("[Destroy] imu_handler done"); log_flush();
    }

    /* ── Inference pipeline: release all ONNX sessions BEFORE ort_global_shutdown ── */
    if (sc->inference_pipeline) {
        /* Guard: inference_pipeline may hold ONNX sessions that rely on
         * the global ORT environment.  If ORT was already partially torn
         * down by a failed k1_pipeline_init, skip session release to
         * avoid double-free inside ONNX Runtime. */
        inference_pipeline_destroy(sc->inference_pipeline);
        sc->inference_pipeline = NULL;
        log_debug("[Destroy] inference_pipeline done"); log_flush();
    }

    if (sc->tracking_manager) {
        object_tracker_destroy(sc->tracking_manager);
        sc->tracking_manager = NULL;
        log_debug("[Destroy] tracking_manager done"); log_flush();
    }

    if (sc->spatial_engine) {
        spatial_engine_destroy(sc->spatial_engine);
        sc->spatial_engine = NULL;
        log_debug("[Destroy] spatial_engine done"); log_flush();
    }

    if (sc->result_manager) {
        result_manager_destroy(sc->result_manager);
        sc->result_manager = NULL;
        log_debug("[Destroy] result_manager done"); log_flush();
    }
    /* coap_receiver already destroyed at top (before resource cleanup) */

    /* ── GUI pipeline state machine ── */
    psm_destroy(&sc->state_machine);
    log_debug("[Destroy] psm done"); log_flush();

    /* ── Web UI JPEG buffer ── */
    pthread_mutex_destroy(&sc->jpeg_mutex);
    free(sc->latest_jpeg);
    sc->latest_jpeg = NULL;
    log_debug("[Destroy] jpeg buffer done"); log_flush();

    /* ── Global resource cleanup (once at process exit) ── */
    {
        K1Platform* plat = k1_platform_init();   /* obtain singleton for destroy */
        if (plat) {
            k1_platform_destroy(plat);
            log_debug("[Destroy] k1_platform done"); log_flush();
        }
    }

    /* Release the global ONNX Runtime environment LAST, after all
     * per-model sessions have been destroyed by inference_pipeline_destroy.
     * Guard: if ort_get_api() returns NULL, the env was never created
     * or was already released — skip to avoid double-release crash. */
    {
        const OrtApi* api_check = ort_get_api();
        if (api_check) {
            ort_global_shutdown();
            log_debug("[Destroy] ort_global_shutdown done"); log_flush();
        } else {
            log_debug("[Destroy] ort already down, skipping global shutdown");
            log_flush();
        }
    }

    free(sc);
    log_debug("[Destroy] system_controller freed"); log_flush();
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
    /* fps_history[] now stores wall-clock timestamps (seconds).
     * Compute the rolling FPS as (window_frames - 1) / (newest - oldest). */
    if (sc->fps_history_count >= 10) {
        int oldest = sc->fps_history_count - 10;
        float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[oldest];
        return (dt > 0.001f) ? 9.0f / dt : 0.0f;
    } else if (sc->fps_history_count >= 2) {
        float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[0];
        return (dt > 0.001f) ? (float)(sc->fps_history_count - 1) / dt : 0.0f;
    }
    return 0.0f;
}

/*
 * ── RGB24 Pixel Rendering Utilities (no OpenCV dependency) ──
 *
 * Draws bounding boxes, COCO-17 skeleton, track IDs, and depth info
 * directly into the raw RGB24 frame buffer before video encoding.
 */

/* COCO-17 skeleton connections (pairs of keypoint indices) */
static const int SKELETON_EDGES[][2] = {
    {15,13}, {13,11}, {16,14}, {14,12}, {11,12},  /* legs + hips */
    {5,11},  {6,12},  {5,6},   {5,7},   {6,8},    /* torso */
    {7,9},   {8,10},  {1,2},   {0,1},   {0,2},    /* arms + face */
    {1,3},   {2,4},   {3,5},   {4,6}              /* ears→shoulders */
};
#define NUM_SKELETON_EDGES (sizeof(SKELETON_EDGES) / sizeof(SKELETON_EDGES[0]))

/* Skeleton edge colors (BGR order for RGB24 buffer) */
static const uint8_t SKELETON_COLORS[][3] = {
    {255, 0, 255},   /* magenta: legs */
    {255, 0, 255},
    {255, 0, 255},
    {255, 0, 255},
    {0, 255, 0},     /* green: hips */
    {0, 255, 0},     /* green: torso */
    {0, 255, 0},
    {0, 255, 255},   /* yellow: shoulders→arms */
    {0, 255, 255},
    {0, 255, 255},
    {0, 255, 255},
    {255, 128, 0},   /* orange: face */
    {255, 128, 0},
    {255, 128, 0},
    {255, 128, 0},   /* ears→shoulders */
    {255, 128, 0},
    {255, 128, 0},
    {255, 128, 0},
};

/* Guard against NaN/Inf from model output.
 *
 * CRITICAL: With -ffast-math, isnan()/isinf() are UB on Clang — the compiler
 * optimizes v!=v to always false (per -ffinite-math-only).  Use IEEE 754
 * bit-level inspection via type-punning (memcpy to uint32_t).  The compiler
 * cannot optimize away integer bitwise operations, and the optimizer barrier
 * (the memcpy through volatile) prevents constant-propagation.
 *
 * IEEE 754 binary32:
 *   NaN:  exponent=0xFF, mantissa≠0   → raw32 > 0x7F800000
 *   +Inf: 0x7F800000, -Inf: 0xFF800000
 *   Model output: always within [-1e6,1e6] = [-0x49742400, 0x49742400],
 *   so any |val| > 1e30f → definitely Inf or broken. */
static inline bool bad_float(float v) {
    uint32_t raw;
    memcpy(&raw, &v, sizeof(raw));
    /* Infinity: exponent all-1s, mantissa zero */
    if (raw == 0x7F800000u || raw == 0xFF800000u) return true;
    /* NaN: exponent all-1s, mantissa non-zero */
    if ((raw & 0x7F800000u) == 0x7F800000u && (raw & 0x007FFFFFu) != 0) return true;
    /* Extreme out-of-range values (model output beyond physical limits) */
    if (v > 1e30f || v < -1e30f) return true;
    return false;
}

static inline void put_pixel(uint8_t* rgb, int w, int h, int x, int y,
                              uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    size_t off = ((size_t)y * (size_t)w + (size_t)x) * 3;
    rgb[off + 0] = r;  /* R */
    rgb[off + 1] = g;  /* G */
    rgb[off + 2] = b;  /* B */
}

static void draw_rect(uint8_t* rgb, int w, int h,
                       int x1, int y1, int x2, int y2,
                       uint8_t r, uint8_t g, uint8_t b, int thickness) {
    /* Clamp to frame to avoid integer overflow in loop bounds */
    if (x1 < 0) x1 = 0; if (x2 >= w) x2 = w - 1;
    if (y1 < 0) y1 = 0; if (y2 >= h) y2 = h - 1;
    if (x1 > x2 || y1 > y2) return;
    for (int t = 0; t < thickness; t++) {
        int lx = x1 + t, rx = x2 - t, ty = y1 + t, by = y2 - t;
        for (int x = lx; x <= rx; x++) {
            put_pixel(rgb, w, h, x, ty, r, g, b);
            put_pixel(rgb, w, h, x, by, r, g, b);
        }
        for (int y = ty; y <= by; y++) {
            put_pixel(rgb, w, h, lx, y, r, g, b);
            put_pixel(rgb, w, h, rx, y, r, g, b);
        }
    }
}

static void draw_line(uint8_t* rgb, int w, int h,
                       float x1, float y1, float x2, float y2,
                       uint8_t r, uint8_t g, uint8_t b, int thickness) {
    /* NaN/Inf guard: model output can produce invalid floats.
     * Without this, (int)NaN is UB and can overflow the loop. */
    if (bad_float(x1) || bad_float(y1) || bad_float(x2) || bad_float(y2))
        return;
    float dx = x2 - x1, dy = y2 - y1;
    float len = sqrtf(dx*dx + dy*dy);
    if (len < 0.5f) return;
    int steps = (int)len;
    if (steps > 5000) return;  /* safety cap: longest diagonal < 2500px */
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        int cx = (int)(x1 + dx * t + 0.5f);
        int cy = (int)(y1 + dy * t + 0.5f);
        for (int dt = -thickness/2; dt <= thickness/2; dt++) {
            put_pixel(rgb, w, h, cx + dt, cy, r, g, b);
            put_pixel(rgb, w, h, cx, cy + dt, r, g, b);
        }
    }
}

/* Minimal 5x7 bitmap font for frame numbers / IDs */
static void draw_char(uint8_t* rgb, int img_w, int img_h,
                       int x, int y, char c, uint8_t r, uint8_t g, uint8_t b) {
    /* Simple stroke-based digit rendering (0-9, A-Z, space, dot, colon) */
    if (c < ' ' || c > 'z') return;
    const char* glyph = NULL;
    switch (c) {
    case '0': glyph = ".##." "#..#" "#..#" "#..#" "#..#" "#..#" ".##."; break;
    case '1': glyph = "..#." ".##." "..#." "..#." "..#." "..#." ".###"; break;
    case '2': glyph = ".##." "#..#" "...#" "..#." ".#.." "#..." ".###"; break;
    case '3': glyph = ".##." "#..#" "...#" ".##." "...#" "#..#" ".##."; break;
    case '4': glyph = "...#" "..##" ".#.#" "#..#" ".###" "...#" "...#"; break;
    case '5': glyph = ".###" "#..." "#..." ".##." "...#" "#..#" ".##."; break;
    case '6': glyph = ".##." "#..." "#..." "###." "#..#" "#..#" ".##."; break;
    case '7': glyph = ".###" "...#" "..#." ".#.." ".#.." ".#.." ".#.."; break;
    case '8': glyph = ".##." "#..#" "#..#" ".##." "#..#" "#..#" ".##."; break;
    case '9': glyph = ".##." "#..#" "#..#" ".###" "...#" "...#" ".##."; break;
    case 'F': glyph = ".###" "#..." "#..." ".##." "#..." "#..." "#..."; break;
    case 'P': glyph = "###." "#..#" "#..#" "###." "#..." "#..." "#..."; break;
    case 'S': glyph = ".##." "#..#" "#..." ".##." "...#" "#..#" ".##."; break;
    case 'm': glyph = "...." "...." "##.#" "#.#." "#.#." "#.#." "#.#."; break;
    case 's': glyph = "...." "...." ".##." "#..." ".##." "...#" ".##."; break;
    case ' ': glyph = "...." "...." "...." "...." "...." "...." "...."; break;
    case '.': glyph = "...." "...." "...." "...." "...." "...." "..#."; break;
    case ':': glyph = "...." "..#." "...." "...." "...." "..#." "...."; break;
    default:  glyph = "...." "...." "...." "...." "...." "...." "...."; break;
    }
    if (!glyph) return;
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 4; col++) {
            if (glyph[row * 4 + col] == '#') {
                put_pixel(rgb, img_w, img_h, x + col, y + row, r, g, b);
            }
        }
    }
}

static void draw_text(uint8_t* rgb, int img_w, int img_h,
                       int x, int y, const char* text,
                       uint8_t r, uint8_t g, uint8_t b) {
    int cx = x;
    for (const char* p = text; *p && cx < img_w - 4; p++) {
        draw_char(rgb, img_w, img_h, cx, y, *p, r, g, b);
        cx += 5;  /* 4px char + 1px spacing */
    }
}

/*
 * Render detection overlays onto a raw RGB24 frame.
 *
 * Draws:
 *   - Green bounding boxes for confirmed tracks (frames_seen >= 3)
 *   - Yellow bounding boxes for tentative tracks (1-2 frames)
 *   - COCO-17 skeleton keypoints + connections
 *   - Track ID label + depth/height info (top-left corner of bbox)
 *   - FPS + frame counter overlay (top-right corner)
 */
static void render_overlay_bgr(uint8_t* rgb, int img_w, int img_h,
                                const InferenceResult* inference,
                                const TrackingResult* tracking,
                                float fps, int frame_num) {
    (void)inference;  /* reserved for face/action overlay expansion */
    if (!rgb || !tracking) return;

    /* ── Defensive: clamp num_tracked to array bounds ── */
    int num_tracked = tracking->num_tracked;
    if (num_tracked < 0 || num_tracked > MAX_TRACKED_OBJECTS) {
        log_warning("[Viz] tracking->num_tracked=%d out of bounds, clamping to 0", num_tracked);
        num_tracked = 0;
    }

    /* ── Draw each tracked object ── */
    for (int i = 0; i < num_tracked; i++) {
        const TrackedObject* obj = &tracking->tracked_objects[i];
        const BBox* b = &obj->detection.bbox;

        int x1 = (int)b->x_min, y1 = (int)b->y_min;
        int x2 = (int)b->x_max, y2 = (int)b->y_max;

        /* Skip if bbox coords are NaN/Inf (model output artifact) */
        if (bad_float(b->x_min) || bad_float(b->y_min) ||
            bad_float(b->x_max) || bad_float(b->y_max)) continue;

        /* Choose color: green=confirmed, yellow=tentative, red=occluded */
        uint8_t r, g, b_col;
        if (obj->is_occluded) {
            r = 0; g = 0; b_col = 255;       /* red */
        } else if (obj->frames_seen >= 3) {
            r = 0; g = 255; b_col = 0;       /* green */
        } else {
            r = 0; g = 255; b_col = 255;     /* yellow/cyan */
        }

        /* Bounding box */
        draw_rect(rgb, img_w, img_h, x1, y1, x2, y2, r, g, b_col, 2);

        /* Track ID + depth label */
        char label[64];
        if (obj->has_height) {
            snprintf(label, sizeof(label), "ID%d %.1fm",
                     obj->track_id, (double)obj->height_meters);
        } else {
            snprintf(label, sizeof(label), "ID%d", obj->track_id);
        }
        int label_y = y1 - 10;
        if (label_y < 2) label_y = y1 + 14;
        draw_text(rgb, img_w, img_h, x1 + 2, label_y, label,
                  255, 255, 255);

        /* Skeleton */
        if (obj->has_pose && obj->pose.num_keypoints >= 4) {
            const PoseEstimation* pose = &obj->pose;
            int nk = (pose->num_keypoints < 17) ? pose->num_keypoints : 17;

            /* Keypoints (circles approximated as 3×3 crosses) */
            for (int k = 0; k < nk; k++) {
                if (pose->keypoints[k].confidence < 0.15f) continue;
                if (bad_float(pose->keypoints[k].x) || bad_float(pose->keypoints[k].y))
                    continue;
                int kx = (int)(pose->keypoints[k].x + 0.5f);
                int ky = (int)(pose->keypoints[k].y + 0.5f);
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        put_pixel(rgb, img_w, img_h, kx + dx, ky + dy,
                                  0, 255, 255);
                    }
                }
            }

            /* Skeleton edges */
            for (int e = 0; e < (int)NUM_SKELETON_EDGES; e++) {
                int k1 = SKELETON_EDGES[e][0];
                int k2 = SKELETON_EDGES[e][1];
                if (k1 >= nk || k2 >= nk) continue;
                if (pose->keypoints[k1].confidence < 0.15f ||
                    pose->keypoints[k2].confidence < 0.15f) continue;
                draw_line(rgb, img_w, img_h,
                          pose->keypoints[k1].x, pose->keypoints[k1].y,
                          pose->keypoints[k2].x, pose->keypoints[k2].y,
                          SKELETON_COLORS[e][0], SKELETON_COLORS[e][1], SKELETON_COLORS[e][2],
                          1);
            }
        }
    }

    /* ── Top-right FPS + frame info overlay ── */
    {
        char info[64];
        snprintf(info, sizeof(info), "FPS:%.1f F:%d T:%d",
                 (double)fps, frame_num, tracking->num_tracked);
        /* Draw semi-transparent background bar (darken top 20px strip).
         * Clamp start_x to 0: img_w < 220 would underflow int → UB. */
        int bar_start = (img_w > 220) ? (img_w - 220) : 0;
        for (int y = 0; y < 20; y++) {
            for (int x = bar_start; x < img_w; x++) {
                size_t off = ((size_t)y * (size_t)img_w + (size_t)x) * 3;
                rgb[off]     = (uint8_t)(rgb[off]     / 3);  /* R */
                rgb[off + 1] = (uint8_t)(rgb[off + 1] / 3);  /* G */
                rgb[off + 2] = (uint8_t)(rgb[off + 2] / 3);  /* B */
            }
        }
        draw_text(rgb, img_w, img_h, img_w - 210, 4, info, 255, 255, 255);
    }
}

SystemStatus system_controller_process_video(SystemController* sc,
                                              const char* video_path,
                                              const char* output_path,
                                              int max_frames,
                                              int save_frame_interval) {
    SystemStatus status;
    memset(&status, 0, sizeof(SystemStatus));
    (void)output_path;  /* used by result_manager via sc internally */

    if (!sc || !video_path) {
        strncpy(status.message, "Invalid parameters", sizeof(status.message) - 1);
        status.message[sizeof(status.message) - 1] = '\0';
        return status;
    }

    log_info("Processing video: %s", video_path);

    const char* session_id = result_manager_start_session(sc->result_manager, video_path);

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

    char output_video_path[MAX_PATH_LEN];
    result_manager_get_video_path(sc->result_manager, session_id,
                                  output_video_path, sizeof(output_video_path));

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

    /*
     * ── Per-frame heap allocations (avoids 1.3MB stack overflow from
     *     TrackingResult + InferenceResult on K1's limited stack) ──
     */
    InferenceResult* inference = (InferenceResult*)calloc(1, sizeof(InferenceResult));
    TrackingResult* tracking = (TrackingResult*)calloc(1, sizeof(TrackingResult));
    if (!inference || !tracking) {
        log_critical("Failed to allocate per-frame inference/tracking buffers");
        free(inference);
        free(tracking);
        video_processor_destroy(processor);
        if (video_writer) video_writer_destroy(video_writer);
        strncpy(status.message, "Out of memory", sizeof(status.message) - 1);
        return status;
    }

    FrameData* frame;
    while (sc->running && (frame = video_processor_read_frame_raw(processor)) != NULL) {
        double frame_start = (double)utils_get_time_ms() / 1000.0;
        sc->frame_count++;

        if (max_frames > 0 && sc->frame_count > max_frames) {
            frame_data_destroy(frame);
            break;
        }

        inference_result_init(inference);

        if (!sc->inference_pipeline) {
            log_error("Inference pipeline is NULL — cannot process frames");
            frame_data_destroy(frame);
            break;
        }

        inference_pipeline_process_frame(
            sc->inference_pipeline, frame->data, frame->width, frame->height, inference);
        if (sc->frame_count == 1 && inference->num_detections > 0) {
            spatial_engine_initialize_coordinate_system(
                sc->spatial_engine, frame->height, frame->width, &inference->detections[0]);
        }

        SpatialPosition positions[MAX_DETECTIONS_PER_FRAME];
        int num_positions = 0;
        for (int i = 0; i < inference->num_detections && num_positions < MAX_DETECTIONS_PER_FRAME; i++) {
            SpatialResult sr = spatial_engine_calculate_position(
                sc->spatial_engine, &inference->detections[i], frame->width, frame->height, NULL, 0, 0, NULL, -1);
            positions[num_positions++] = sr.position;
        }

        object_tracker_update(
            sc->tracking_manager, inference->detections, inference->num_detections,
            positions, num_positions, sc->frame_count, tracking);

        object_tracker_associate_poses(sc->tracking_manager, inference->poses, inference->num_poses);

        /* ── Detailed progress with detection count ── */
        if (sc->progress_cb && sc->frame_count % 5 == 0) {
            float cur_fps = get_current_fps(sc);
            int dets = inference->num_detections;
            sc->progress_cb(sc->frame_count, max_frames, cur_fps,
                            (float)(utils_get_time_ms() - (int64_t)(frame_start * 1000.0)),
                            "proc", dets);
        }

        /* ── Sync confirmed track counts to inference pipeline ── */
        if (sc->inference_pipeline) {
            sc->inference_pipeline->confirmed_track_count =
                object_tracker_get_confirmed_count(sc->tracking_manager);
            sc->inference_pipeline->total_track_count =
                object_tracker_get_all_track_count(sc->tracking_manager);
        }

        associate_poses_with_objects(tracking->tracked_objects, tracking->num_tracked,
                                      inference->poses, inference->num_poses);
        associate_faces_with_objects(tracking->tracked_objects, tracking->num_tracked,
                                      inference->faces, inference->num_faces);

        for (int i = 0; i < tracking->num_tracked; i++) {
            TrackedObject* obj = &tracking->tracked_objects[i];
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
        for (int i = 0; i < tracking->num_tracked; i++) {
            TrackedObject* obj = &tracking->tracked_objects[i];
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

        /* ── Phase B: WorldCoord registration (offline mode) ── */
        if (sc->world_coord && sc->imu_handler) {
            DualImuPose dual;
            if (imu_handler_get_dual_pose(sc->imu_handler, &dual) && dual.align_valid) {
                float q_align[4] = {dual.align_qw, dual.align_qx, dual.align_qy, dual.align_qz};
                float t_zero[3] = {0.0f, 0.0f, 0.0f};
                world_coord_set_alignment(sc->world_coord, q_align, t_zero);
            }

            for (int i = 0; i < tracking->num_tracked; i++) {
                TrackedObject* obj = &tracking->tracked_objects[i];
                if (!obj->spatial_pos.is_valid) continue;

                /* ── 统一 P_cam 构造 (离线模式) ──
                 * 与实时路径一致的修复: 使用统一 bbox+深度, 而非混合来源. */
                float depth = obj->spatial_pos.world_z;
                float fx = sc->spatial_engine->camera_matrix[0][0];
                float fy = sc->spatial_engine->camera_matrix[1][1];
                float cx = sc->spatial_engine->camera_matrix[0][2];
                float cy = sc->spatial_engine->camera_matrix[1][2];
                float u = bbox_center_x(&obj->detection.bbox);
                float v = bbox_center_y(&obj->detection.bbox);

                float P_cam[3];
                P_cam[0] = (u - cx) * depth / fx;
                P_cam[1] = (v - cy) * depth / fy;
                P_cam[2] = depth;

                float kpts_2d[17][2];
                int num_kpts = 0;
                if (obj->has_pose) {
                    num_kpts = (obj->pose.num_keypoints < 17) ? obj->pose.num_keypoints : 17;
                    for (int k = 0; k < num_kpts; k++) {
                        kpts_2d[k][0] = obj->pose.keypoints[k].x;
                        kpts_2d[k][1] = obj->pose.keypoints[k].y;
                    }
                }

                double now_s = (double)utils_get_time_ms() / 1000.0;
                /* 离线模式: 单线程运行, K1 位姿是实时的, 传递 NULL 使用内部查询回退 */
                world_coord_register_person(sc->world_coord, obj->track_id,
                                             P_cam, kpts_2d, num_kpts,
                                             obj->spatial_pos.world_z,
                                             obj->spatial_pos.confidence,
                                             now_s,
                                             NULL, NULL,  /* 离线模式: 回退到内部查询 K1 位姿 */
                                             &obj->detection.bbox);
            }
            {
                double now_s = (double)utils_get_time_ms() / 1000.0;
                world_coord_prune_timeout(sc->world_coord, now_s);
            }
        }

        sc->detection_count += tracking->num_tracked;

        double frame_time = (double)utils_get_time_ms() / 1000.0 - frame_start;
        if (sc->proc_times_count < SC_MAX_PROC_TIMES) {
            sc->processing_times[sc->proc_times_count++] = (float)frame_time;
        }

        if (sc->fps_history_count < SC_MAX_FPS_HISTORY) {
            sc->fps_history[sc->fps_history_count++] = (float)(utils_get_time_ms() / 1000.0);
        }

        float avg_fps = get_current_fps(sc);

        /*
         * ── Render detection overlays onto frame BEFORE writing to video ──
         * This draws bounding boxes, skeleton keypoints, track IDs, and
         * depth/height info directly into frame->data (raw RGB24).
         */
        /* ── Push frame data to Web UI ── */
        push_frame_to_web(sc, inference, tracking, sc->frame_count, avg_fps,
                          frame->data, frame->width, frame->height);

        render_overlay_bgr(frame->data, frame->width, frame->height,
                           inference, tracking, avg_fps, sc->frame_count);

        /* Write annotated frame to video output */
        if (video_writer && save_frame_interval > 0) {
            video_writer_write_frame(video_writer, frame->data);

            if (sc->frame_count % save_frame_interval == 0) {
                log_info("Frame %d processed: %d objects, FPS=%.1f", sc->frame_count, tracking->num_tracked, avg_fps);
            }
        }

        /* Heartbeat — always log every 50 frames so we know the pipeline is alive */
        if (sc->frame_count % 50 == 0) {
            log_info("Heartbeat: %d frames processed, %.1f FPS, %d tracks",
                     sc->frame_count, avg_fps, tracking->num_tracked);
        }

        frame_data_destroy(frame);
    }

    if (video_writer) {
        video_writer_destroy(video_writer);
        log_info("Output video saved to: %s", output_video_path);
    }

    free(inference);
    free(tracking);
    video_processor_destroy(processor);

    float avg_fps = 0.0f;
    if (sc->fps_history_count >= 2) {
        float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[0];
        avg_fps = (dt > 0.001f) ? (float)(sc->fps_history_count - 1) / dt : 0.0f;
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
    if (sc->fps_history_count >= 2) {
        float dt = sc->fps_history[sc->fps_history_count - 1] - sc->fps_history[0];
        avg_fps = (dt > 0.001f) ? (float)(sc->fps_history_count - 1) / dt : 0.0f;
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

/* ═══════════════════════════════════════════════════════════════════════
 * GUI-driven async pipeline control (Phase A)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Wrapper: runs process_realtime_k1 in a pthread, managing state transitions */
static void* k1_realtime_thread_func(void* arg) {
    SystemController* sc = (SystemController*)arg;
    log_info("[Async] Pipeline monitor thread started");
    system_controller_process_realtime_k1(sc);

    /* After pipeline exits, transition back to IDLE (unless ERROR) */
    PipelineState final = psm_get(&sc->state_machine);
    if (final == PIPELINE_STATE_STOPPING) {
        psm_transition(&sc->state_machine, PIPELINE_STATE_IDLE,
                       "Pipeline stopped cleanly");
        broadcast_pipeline_state(sc, "pipeline_idle");
    } else if (final != PIPELINE_STATE_ERROR && final != PIPELINE_STATE_IDLE) {
        psm_transition(&sc->state_machine, PIPELINE_STATE_IDLE,
                       "Pipeline thread exited");
        broadcast_pipeline_state(sc, "pipeline_idle");
    }
    log_info("[Async] Pipeline monitor thread finished, state=%s",
             psm_state_name(psm_get(&sc->state_machine)));
    return NULL;
}

int system_controller_start_async(SystemController* sc, PipelineMode mode) {
    if (!sc) return -1;

    PipelineState current = psm_get(&sc->state_machine);
    if (current != PIPELINE_STATE_IDLE) {
        log_error("[Async] Cannot start: pipeline is in state %s (must be idle)",
                  psm_state_name(current));
        return -1;
    }

    sc->mode = mode;
    sc->running = 1;

    if (!psm_transition(&sc->state_machine, PIPELINE_STATE_STARTING,
                        "Async start requested")) {
        sc->running = 0;
        return -1;
    }
    broadcast_pipeline_state(sc, "pipeline_starting");

    int rc = pthread_create(&sc->pipeline_thread, NULL,
                            k1_realtime_thread_func, sc);
    if (rc != 0) {
        log_error("[Async] pthread_create failed: %d", rc);
        sc->running = 0;
        psm_transition(&sc->state_machine, PIPELINE_STATE_ERROR,
                       "Failed to create pipeline thread");
        broadcast_pipeline_state(sc, "pipeline_error");
        return -1;
    }

    /* Detach: the thread will manage its own lifecycle.
     * stop_async() will join it. */
    pthread_detach(sc->pipeline_thread);

    log_info("[Async] Pipeline starting (mode=%d), thread created", (int)mode);
    return 0;
}

int system_controller_stop_async(SystemController* sc) {
    if (!sc) return -1;

    PipelineState current = psm_get(&sc->state_machine);
    if (!psm_is_running(&sc->state_machine)) {
        log_warn("[Async] Cannot stop: pipeline is in state %s",
                 psm_state_name(current));
        return -1;
    }

    psm_transition(&sc->state_machine, PIPELINE_STATE_STOPPING,
                   "Async stop requested");

    /* Signal the monitor loop to exit */
    sc->running = 0;

    log_info("[Async] Stop signal sent, waiting for pipeline to finish...");

    /* Wait up to 15s for the pipeline to stop */
    bool stopped = psm_wait_for(&sc->state_machine, PIPELINE_STATE_IDLE, 15000);
    if (!stopped) {
        log_error("[Async] Pipeline did not stop within 15s timeout");
        return -1;
    }

    log_info("[Async] Pipeline stopped");
    return 0;
}

int system_controller_is_running(const SystemController* sc) {
    if (!sc) return 0;
    return psm_is_running(&sc->state_machine) ? 1 : 0;
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

void system_controller_reset_world_origin(SystemController* sc) {
    if (!sc || !sc->spatial_engine) return;
    spatial_engine_reset_origin(sc->spatial_engine, NULL);
    log_info("[System] World origin reset triggered (lazy re-init on next detection)");
}
