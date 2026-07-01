#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include <signal.h>
#include <stdatomic.h>
#include "core_types.h"
#include "pipeline_state.h"
#include "config_manager.h"
#include "model_store.h"
#include "video_processor.h"
#include "imu_handler.h"
#include "inference_pipeline.h"
#include "tracking_manager.h"
#include "spatial_engine.h"
#include "world_coord.h"
#include "result_manager.h"
#include "coap_receiver.h"
#include "web_server.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SC_MAX_FPS_HISTORY      256
#define SC_MAX_PROC_TIMES       256

/* ── Progress callback for CLI live display ── */
typedef void (*SCProgressFn)(int frame, int total, float fps, float proc_ms,
                              const char* state, int dets);

typedef struct SystemController {
    ConfigManager* config;
    ModelStore* model_store;
    VideoProcessor* video_processor;
    IMUHandler* imu_handler;
    AIInferencePipeline* inference_pipeline;
    ObjectTracker* tracking_manager;
    SpatialLocalizationEngine* spatial_engine;
    WorldCoord* world_coord;
    ResultManager* result_manager;
    CoapReceiver* coap_receiver;

    PipelineMode mode;
    _Atomic int frame_count;      /* BUGFIX: _Atomic for RISC-V weak memory (was volatile).
                                     postproc writes, stgcn/viz read; acquire/release semantics
                                     needed for cross-hart visibility on X60 OoO cores. */
    int max_frames;               /* 0 = unlimited */
    int frame_timeout_s;          /* auto-exit after N seconds of no frames */
    double start_time;

    /* ── CoAP receiver config (ESP32 CoAP/UDP protocol) ── */
    bool coap_enabled;
    char coap_esp_ip[64];
    int  coap_esp_port;
    char coap_wifi_ssid[64];
    char coap_wifi_password[64];

    /* ── Camera device (CLI --camera override, "" = use config) ── */
    char camera_device[64];

    /* ── Video output (CLI --save-video or config visualization.record_to_video) ── */
    bool save_video;

    /* ── Progress callback (CLI live display) ── */
    SCProgressFn progress_cb;

    /* ── Model variant CLI override ("" = use config) ── */
    char pose_model_variant[32];

    float fps_history[SC_MAX_FPS_HISTORY];  /* frame wall-clock timestamps (s), not instantaneous FPS */
    _Atomic int fps_history_count;   /* BUGFIX: _Atomic for RISC-V (was volatile) */
    float processing_times[SC_MAX_PROC_TIMES];
    _Atomic int proc_times_count;    /* BUGFIX: _Atomic for RISC-V (was volatile) */
    _Atomic int detection_count;     /* BUGFIX: _Atomic for RISC-V (was volatile) */
    volatile sig_atomic_t running;   /* sig_atomic_t: safe for signal handler + cross-thread */

    /* ── GUI-driven pipeline lifecycle (Phase A) ── */
    PipelineStateMachine state_machine;  /* five-state FSM for external control */
    pthread_t            pipeline_thread; /* monitor thread for async start/stop */

    /* ── Web UI integration ── */
    WebServer*           web_server;     /* set by main() after web_server_create */
    uint8_t*             latest_jpeg;    /* latest JPEG frame for /api/frame.jpg */
    size_t               latest_jpeg_len;
    pthread_mutex_t      jpeg_mutex;     /* protects latest_jpeg */
} SystemController;

SystemController* system_controller_create(const char* config_path,
                                            const char* pose_model_variant);
void system_controller_destroy(SystemController* sc);

SystemStatus system_controller_process_video(SystemController* sc,
                                              const char* video_path,
                                              const char* output_path,
                                              int max_frames,
                                              int save_frame_interval);

SystemStatus system_controller_process_realtime_k1(SystemController* sc);

/* ── GUI-driven async pipeline control (Phase A) ── */

/**
 * Start the pipeline asynchronously.
 * Creates the monitor thread which spawns K1 pipeline worker threads.
 * Returns immediately. Pipeline state transitions:
 *   IDLE → STARTING → RUNNING (on success) or ERROR (on failure)
 */
int  system_controller_start_async(SystemController* sc, PipelineMode mode);

/**
 * Stop the running pipeline gracefully.
 * Sets stopping signal, joins monitor thread, transitions back to IDLE.
 */
int  system_controller_stop_async(SystemController* sc);

/** Check if pipeline is currently running. */
int  system_controller_is_running(const SystemController* sc);

SystemStatus system_controller_get_status(const SystemController* sc);
SystemStatus system_controller_get_final_status(const SystemController* sc);

void system_controller_reset(SystemController* sc);
void system_controller_reset_world_origin(SystemController* sc);

#ifdef __cplusplus
}
#endif

#endif
