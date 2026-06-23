#include "system_controller.h"
#include "logger.h"
#include "k1_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

static volatile sig_atomic_t g_running_flag = 1;
static SystemController* g_sc = NULL;

/*
 * Signal handler with two-stage shutdown.
 *
 * First  signal (Ctrl+C):  set g_running_flag=0 for graceful shutdown.
 *                          Main loop checks sc->running at top of each frame
 *                          and will exit cleanly after current inference completes.
 * Second signal:           _exit(1) immediately — no cleanup, no flushing.
 *                          Use when ORT inference is stuck for 4+ seconds.
 *
 * All functions used are async-signal-safe: write(), _exit().
 */
static void signal_handler(int sig) {
    (void)sig;
    static volatile sig_atomic_t count = 0;
    count++;

    if (count == 1) {
        const char msg[] = "\nShutting down... (Ctrl+C again to force quit)\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        g_running_flag = 0;
        if (g_sc) g_sc->running = 0;
    } else {
        _exit(1);
    }
}

static void print_usage(const char* program_name) {
    printf("LingQi TanTong - SpacemiT K1 Muse Pi Pro\n\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --video_path <path>         Input video file (offline mode)\n");
    printf("  --output_path <path>        Output directory (default: output)\n");
    printf("  --max-frames <N>            Max frames to process (0=unlimited, works for all modes)\n");
    printf("  --save_frame_interval <N>   Save frame every N frames (default: 10)\n");
    printf("  --config <path>             Configuration YAML file\n");
    printf("  --realtime                  Real-time pipeline mode (Muse Pi Pro)\n");
    printf("  --camera <path>             Camera device (default: /dev/video0)\n");
    printf("  --display                   Enable framebuffer display (/dev/fb0)\n");
    printf("  --display-device <path>     Framebuffer device (default: /dev/fb0)\n");
    printf("  --rtsp <url>                RTSP streaming (e.g. rtsp://0.0.0.0:8554/live)\n");
    printf("  --udp-stream <addr>         UDP MPEG-TS streaming (e.g. udp://192.168.1.100:1234)\n");
    printf("  --rtmp <url>                RTMP streaming\n");
    printf("  --save-video                Save output to MP4 video file\n");
    printf("  --frame-timeout <S>         Auto-exit after S seconds of no frames (default: 10)\n");
    printf("  --coap                      Enable CoAP/UDP receiver (ESP32 WiFi)\n");
    printf("  --coap-ip <ip>              ESP32 IP address (default: 192.168.4.1)\n");
    printf("  --coap-port <port>          CoAP/UDP port (default: 5683)\n");
    printf("  --wifi-ssid <ssid>          WiFi SSID to connect (default: ESP32-Camera-AP)\n");
    printf("  --wifi-password <pw>        WiFi password (default: 12345678)\n");
    printf("  --help                      Show this help\n");
    printf("\nExamples:\n");
    printf("  # CoAP/UDP WiFi receiver (ESP32 camera):\n");
    printf("  %s --realtime --coap --display\n", program_name);
    printf("  %s --realtime --coap --rtsp rtsp://0.0.0.0:8554/live --display\n", program_name);
    printf("  # Offline video processing:\n");
    printf("  %s --video_path test.mp4 --save_frame_interval 1\n", program_name);
    printf("  %s --video_path test.mp4 --max-frames 100\n", program_name);
}

int main(int argc, char* argv[]) {
    const char* video_path = "test.mp4";
    const char* output_path = "output/results";
    const char* config_path = "configs/default.yaml";
    const char* camera_dev = "/dev/video0";
    int save_frame_interval = 0;  /* 0 = use config value */
    bool realtime_mode = false;

    /* Display / streaming options (override config) */
    bool cli_display_enabled = false;
    const char* cli_display_device = NULL;
    bool cli_stream_enabled = false;
    const char* cli_stream_url = NULL;
    bool cli_save_video = false;
    int cli_max_frames = 0;
    int cli_frame_timeout = 0;

    /* CoAP receiver CLI overrides */
    bool cli_coap_enabled = false;
    const char* cli_coap_ip = NULL;
    int  cli_coap_port = 0;
    const char* cli_wifi_ssid = NULL;
    const char* cli_wifi_password = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--realtime") == 0) {
            realtime_mode = true;
        } else if (strcmp(argv[i], "--video_path") == 0 && i + 1 < argc) {
            video_path = argv[++i];
        } else if (strcmp(argv[i], "--output_path") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--max_frames") == 0 && i + 1 < argc) {
            cli_max_frames = atoi(argv[++i]);  /* also accept underscore variant */
        } else if (strcmp(argv[i], "--save_frame_interval") == 0 && i + 1 < argc) {
            save_frame_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--camera") == 0 && i + 1 < argc) {
            camera_dev = argv[++i];
        } else if (strcmp(argv[i], "--display") == 0) {
            cli_display_enabled = true;
        } else if (strcmp(argv[i], "--display-device") == 0 && i + 1 < argc) {
            cli_display_device = argv[++i];
        } else if (strcmp(argv[i], "--rtsp") == 0 && i + 1 < argc) {
            cli_stream_enabled = true;
            cli_stream_url = argv[++i];
        } else if (strcmp(argv[i], "--udp-stream") == 0 && i + 1 < argc) {
            cli_stream_enabled = true;
            cli_stream_url = argv[++i];
        } else if (strcmp(argv[i], "--rtmp") == 0 && i + 1 < argc) {
            cli_stream_enabled = true;
            cli_stream_url = argv[++i];
        } else if (strcmp(argv[i], "--save-video") == 0) {
            cli_save_video = true;
        } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            cli_max_frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--frame-timeout") == 0 && i + 1 < argc) {
            cli_frame_timeout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--coap") == 0) {
            cli_coap_enabled = true;
        } else if (strcmp(argv[i], "--coap-ip") == 0 && i + 1 < argc) {
            cli_coap_ip = argv[++i];
        } else if (strcmp(argv[i], "--coap-port") == 0 && i + 1 < argc) {
            cli_coap_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--wifi-ssid") == 0 && i + 1 < argc) {
            cli_wifi_ssid = argv[++i];
        } else if (strcmp(argv[i], "--wifi-password") == 0 && i + 1 < argc) {
            cli_wifi_password = argv[++i];
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Ensure logs/ directory exists (ignore failure — logger falls back to stderr) */
    mkdir("logs", 0755);
    logger_init("logs/system.log", LOG_LEVEL_INFO);

    log_info("============================================================");
    log_info("LingQi TanTong - SpacemiT K1 Muse Pi Pro");
    log_info("============================================================");

    if (realtime_mode) {
        log_info("Mode: REALTIME");
        log_info("Camera: %s", camera_dev);
    } else {
        log_info("Mode: OFFLINE");
        log_info("Video: %s", video_path);
        log_info("Output: %s", output_path);
        log_info("Max frames: %d", cli_max_frames);
        log_info("Save interval: %d", save_frame_interval);
    }
    log_info("Config: %s", config_path ? config_path : "defaults");
    log_info("============================================================");

    SystemController* sc = system_controller_create(config_path);
    if (!sc) {
        log_critical("Failed to create system controller");
        logger_close();
        return 1;
    }
    g_sc = sc;

    /* ── CLI overrides for display / streaming / frame limits ── */
    if (cli_display_enabled) {
        sc->display_enabled = true;
        log_info("CLI override: display enabled");
    }
    if (cli_display_device) {
        strncpy(sc->display_device, cli_display_device, sizeof(sc->display_device) - 1);
        sc->display_enabled = true;
        log_info("CLI override: display device = %s", cli_display_device);
    }
    if (cli_stream_enabled && cli_stream_url) {
        sc->stream_enabled = true;
        strncpy(sc->stream_url, cli_stream_url, sizeof(sc->stream_url) - 1);
        log_info("CLI override: stream = %s", cli_stream_url);
    }
    if (cli_save_video) {
        sc->save_video_enabled = true;
        log_info("CLI override: save video enabled");
    }
    if (cli_max_frames > 0) {
        sc->max_frames = cli_max_frames;
        log_info("CLI override: max_frames = %d", cli_max_frames);
    }
    if (cli_frame_timeout > 0) {
        sc->frame_timeout_s = cli_frame_timeout;
        log_info("CLI override: frame_timeout = %ds", cli_frame_timeout);
    }
    if (cli_coap_enabled) {
        sc->coap_enabled = true;
        log_info("CLI override: CoAP receiver enabled");
    }
    if (cli_coap_ip) {
        strncpy(sc->coap_esp_ip, cli_coap_ip, sizeof(sc->coap_esp_ip) - 1);
        sc->coap_enabled = true;
        log_info("CLI override: CoAP IP = %s", cli_coap_ip);
    }
    if (cli_coap_port > 0) {
        sc->coap_esp_port = cli_coap_port;
        log_info("CLI override: CoAP port = %d", cli_coap_port);
    }
    if (cli_wifi_ssid) {
        strncpy(sc->coap_wifi_ssid, cli_wifi_ssid, sizeof(sc->coap_wifi_ssid) - 1);
        log_info("CLI override: WiFi SSID = %s", cli_wifi_ssid);
    }
    if (cli_wifi_password) {
        strncpy(sc->coap_wifi_password, cli_wifi_password, sizeof(sc->coap_wifi_password) - 1);
        log_info("CLI override: WiFi password set");
    }

    if (realtime_mode) {
        sc->running = 1;
        log_info("K1 dual-cluster pipeline mode activated");
        SystemStatus status = system_controller_process_realtime_k1(sc);
        log_info("============================================================");
        log_info("K1 Pipeline Session Complete");
        log_info("============================================================");
        log_info("Frames processed: %d", status.frame_count);
        log_info("Average FPS: %.1f", status.fps);
        log_info("Average proc time: %.1fms", status.processing_time_ms);
        log_info("Message: %s", status.message);
        log_info("============================================================");
    } else {
        SystemStatus status = system_controller_process_video(
            sc, video_path, output_path, cli_max_frames, false, save_frame_interval);

        log_info("============================================================");
        log_info("Offline Processing Complete");
        log_info("============================================================");
        log_info("Frames processed: %d", status.frame_count);
        log_info("Average FPS: %.1f", status.fps);
        log_info("Average proc time: %.1fms", status.processing_time_ms);
        log_info("Message: %s", status.message);
        log_info("============================================================");
    }

    system_controller_destroy(sc);
    logger_close();

    return 0;
}