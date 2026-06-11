#ifndef SYSTEM_CONTROLLER_H
#define SYSTEM_CONTROLLER_H

#include "core_types.h"
#include "config_manager.h"
#include "model_store.h"
#include "video_processor.h"
#include "imu_handler.h"
#include "inference_pipeline.h"
#include "tracking_manager.h"
#include "spatial_engine.h"
#include "visualizer.h"
#include "ar_renderer.h"
#include "result_manager.h"
#include "video_writer.h"
#include "arrow_receiver.h"
#include "ai_accel_adapter.h"
#include "display_output.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SC_MAX_FPS_HISTORY      256
#define SC_MAX_PROC_TIMES       256

typedef struct {
    ConfigManager* config;
    ModelStore* model_store;
    VideoProcessor* video_processor;
    IMUHandler* imu_handler;
    AIInferencePipeline* inference_pipeline;
    ObjectTracker* tracking_manager;
    SpatialLocalizationEngine* spatial_engine;
    Visualizer* visualizer;
    ARRenderer* ar_renderer;
    ResultManager* result_manager;
    ArrowReceiver* arrow_receiver;
    AIAcclContext* ai_context;
    DisplayOutput* display_output;

    PipelineMode mode;
    int frame_count;
    int max_frames;               /* 0 = unlimited */
    int frame_timeout_s;          /* auto-exit after N seconds of no frames */
    double start_time;

    /* ── Display / streaming config ── */
    bool display_enabled;         /* framebuffer output */
    char display_device[MAX_PATH_LEN];
    bool stream_enabled;          /* RTSP / UDP-TS output */
    char stream_url[MAX_PATH_LEN];
    bool save_video_enabled;      /* MP4 file output */
    char save_video_path[MAX_PATH_LEN];

    float fps_history[SC_MAX_FPS_HISTORY];
    int fps_history_count;
    float processing_times[SC_MAX_PROC_TIMES];
    int proc_times_count;
    int detection_count;
    bool running;
} SystemController;

SystemController* system_controller_create(const char* config_path);
void system_controller_destroy(SystemController* sc);

SystemStatus system_controller_process_video(SystemController* sc,
                                              const char* video_path,
                                              const char* output_path,
                                              int max_frames,
                                              bool show_windows,
                                              int save_frame_interval);

SystemStatus system_controller_process_realtime(SystemController* sc,
                                                 const char* uart_device_A,
                                                 const char* uart_device_C,
                                                 int baudrate);

#ifdef HAS_K1_PIPELINE
SystemStatus system_controller_process_realtime_k1(SystemController* sc,
                                                    const char* uart_device_A,
                                                    const char* uart_device_C,
                                                    int baudrate);
#endif

SystemStatus system_controller_get_status(const SystemController* sc);
SystemStatus system_controller_get_final_status(const SystemController* sc);

void system_controller_reset(SystemController* sc);

#ifdef __cplusplus
}
#endif

#endif
