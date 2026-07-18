/*
 * main.c — LingQi TanTong CLI
 *
 * CLI UI is built on terminal_ui.h semantic primitives:
 *   - Automatic TTY/CI/pipe detection (NO_COLOR, CI env, isatty)
 *   - 3 output modes: HUMAN (color+Unicode), PLAIN (ASCII), MACHINE (JSON Lines)
 *   - Checklist spinner for model loading / pipeline init
 *   - Status line for realtime mode
 *   - Progress bar for offline video processing
 */
#include "system_controller.h"
#include "pipeline_state.h"
#include "web_server.h"
#include "logger.h"
#include "terminal_ui.h"
#include "k1_platform.h"
#include "benchmark.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef _WIN32
#define strcasecmp _stricmp
#endif

static volatile sig_atomic_t g_run = 1;
static SystemController* g_sc = NULL;
static WebServer*        g_ws = NULL;    /* Web UI server (--web) */

/* ── Progress callback (offline video mode) ───────────────────────
 * Wraps the old per-frame callback into tui_progress.
 * Created lazily on first frame, destroyed on completion. */
static TuiProgress* g_prog = NULL;

static void progress_cb(int frame, int total, float fps, float proc_ms,
                         const char* state, int dets) {
    if (!g_prog && total > 0) {
        g_prog = tui_progress_start("Processing video", total);
    }
    if (g_prog) {
        /* Tick the delta from last frame.  tui_progress_tick clamps internally. */
        static int last_frame = 0;
        int delta = frame - last_frame;
        if (delta > 0) tui_progress_tick(g_prog, delta);
        last_frame = frame;
    }
    (void)fps; (void)proc_ms; (void)state; (void)dets;
}

/* ── Hardware info display ── */
static void show_hardware(void) {
    K1Platform* p = k1_platform_init();
    if (!p) return;

    tui_section("Hardware");
    tui_keyval("Platform", "%s", k1_platform_is_k1() ? "K1 (SpacemiT)" : "Generic");
    tui_keyval("CPU Cores", "%d (Cluster0:4 AI + Cluster1:4 I/O)", k1_platform_cpu_count());
    tui_keyval("RVV 1.0", "%s", k1_platform_has_cap(K1_CAP_RVV_1_0) ? "YES" : "NO");
    tui_keyval("SpacemiT EP", "%s", k1_platform_has_cap(K1_CAP_SPACEMIT_EP) ? "INT8 NPU" : "CPU only");
    tui_keyval("VPU (H.265)", "%s", k1_platform_has_cap(K1_CAP_VPU) ? "YES" : "NO");
    tui_keyval("JPU (JPEG)", "%s", k1_platform_has_cap(K1_CAP_JPU) ? "YES" : "NO");
    tui_keyval("TCM", "%d KB shared", k1_platform_get_tcm_size());
    tui_blank();
}

/* ── Signal handler ── */
static void on_signal(int sig) {
    (void)sig;
    static volatile sig_atomic_t n = 0;
    n++;
    if (n == 1) {
        write(STDERR_FILENO, "\nShutting down... (Ctrl+C again to force quit)\n", 50);
        g_run = 0;
        if (g_sc) {
            g_sc->running = 0;
            /* In GUI mode, also transition state machine to stop */
            psm_transition(&g_sc->state_machine, PIPELINE_STATE_STOPPING,
                           "SIGINT received");
        }
    } else {
        _exit(1);
    }
}

static LogLevel parse_lv(const char* s) {
    if (!s) return LOG_LV_INFO;
    if (strcasecmp(s, "trace") == 0) return LOG_LV_TRACE;
    if (strcasecmp(s, "debug") == 0) return LOG_LV_DEBUG;
    if (strcasecmp(s, "info")  == 0) return LOG_LV_INFO;
    if (strcasecmp(s, "warn")  == 0) return LOG_LV_WARN;
    if (strcasecmp(s, "error") == 0) return LOG_LV_ERROR;
    return LOG_LV_INFO;
}

/* ── Usage ── */
static void usage(const char* p) {
    printf("LingQi TanTong -- Edge AI Inference Pipeline\n\n");
    printf("Usage: %s [OPTIONS]\n\n", p);
    printf("Without arguments, starts in GUI mode with embedded Web UI.\n");
    printf("Open http://localhost:8080 in a browser to control the pipeline.\n\n");
    printf("Modes:\n");
    printf("  --realtime           K1 dual-cluster realtime pipeline (CLI mode)\n");
    printf("  --video PATH         Offline video file processing (CLI mode)\n");
    printf("  --benchmark          Model inference benchmark (paper-ready tables)\n\n");
    printf("Benchmark options:\n");
    printf("  --benchmark-model N  Filter: yolo|pose|face|arcface|stgcn|all\n");
    printf("  --benchmark-runs N   Timed iterations per model (default: 30)\n");
    printf("  --benchmark-video P  Video file for pipeline E2E profiling\n\n");
    printf("Options:\n");
    printf("  --web [PORT]         Web UI port (default: 8080, implied in GUI mode)\n");
    printf("  --config PATH        YAML config (default: configs/default.yaml)\n");
    printf("  --output PATH        Output directory (default: output)\n");
    printf("  --max-frames N       Frame limit (0 = unlimited)\n");
    printf("  --save-interval N    Save frame every N frames\n");
    printf("  --pose-model NAME    yolov8n-pose | yolo11n-pose\n");
    printf("  --camera DEV         Camera device (default: /dev/video1)\n");
    printf("  --log-level LVL      trace | debug | info | warn | error\n");
    printf("  --json               Machine-readable JSON Lines output\n");
    printf("  --quiet              Minimal output (equivalent to NO_COLOR=1)\n\n");
    printf("CoAP/UDP (ESP32 WiFi):\n");
    printf("  --coap               Enable CoAP/UDP receiver\n");
    printf("  --coap-ip IP         ESP32 IP (default: 192.168.4.1)\n");
    printf("  --coap-port PORT     CoAP port (default: 5683)\n");
    printf("  --wifi-ssid SSID     WiFi SSID to connect\n");
    printf("  --wifi-password PW   WiFi password\n");
    printf("  --save-video         Save output to MP4 video file\n");
    printf("Examples:\n");
    printf("  %s                               (GUI mode — browser control)\n", p);
    printf("  %s --web 9000                    (GUI mode on custom port)\n", p);
    printf("  %s --realtime --coap             (CLI mode)\n", p);
    printf("  %s --video test.mp4 --max-frames 100  (CLI mode)\n", p);
    printf("  %s --benchmark                          (model timing)\n", p);
    printf("  %s --benchmark --benchmark-runs 50      (50 runs per model)\n", p);
    printf("  %s --benchmark --benchmark-video test_video.mp4  (+ pipeline profile)\n", p);
}

int main(int argc, char* argv[]) {
    const char* video    = NULL;
    const char* output   = "output";
    const char* config   = "configs/default.yaml";
    const char* camera   = "/dev/video1";
    const char* lv_str   = "info";
    int max_f = 0, save_i = 0;
    bool rt = false, coap = false;
    const char* coap_ip  = NULL;
    int coap_port = 0;
    const char* wifi_ss = NULL, *wifi_pw = NULL, *pm = NULL;
    bool save_video = false;
    bool json_mode = false;
    bool quiet_mode = false;
    int  web_port  = 0;        /* --web [PORT]  */
    bool benchmark = false;
    const char* bm_model = NULL;
    int  bm_runs   = 0;
    const char* bm_video = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--realtime")) rt = true;
        else if (!strcmp(argv[i], "--video") && i+1<argc) video = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1<argc) output = argv[++i];
        else if (!strcmp(argv[i], "--config") && i+1<argc) config = argv[++i];
        else if (!strcmp(argv[i], "--camera") && i+1<argc) camera = argv[++i];
        else if (!strcmp(argv[i], "--max-frames") && i+1<argc) max_f = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save-interval") && i+1<argc) save_i = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log-level") && i+1<argc) lv_str = argv[++i];
        else if (!strcmp(argv[i], "--pose-model") && i+1<argc) pm = argv[++i];
        else if (!strcmp(argv[i], "--coap")) coap = true;
        else if (!strcmp(argv[i], "--coap-ip") && i+1<argc) coap_ip = argv[++i];
        else if (!strcmp(argv[i], "--coap-port") && i+1<argc) coap_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wifi-ssid") && i+1<argc) wifi_ss = argv[++i];
        else if (!strcmp(argv[i], "--wifi-password") && i+1<argc) wifi_pw = argv[++i];
        else if (!strcmp(argv[i], "--save-video")) save_video = true;
        else if (!strcmp(argv[i], "--json"))  json_mode = true;
        else if (!strcmp(argv[i], "--quiet")) quiet_mode = true;
        else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
        else if (!strcmp(argv[i], "--benchmark-model") && i+1<argc) bm_model = argv[++i];
        else if (!strcmp(argv[i], "--benchmark-runs") && i+1<argc) bm_runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--benchmark-video") && i+1<argc) bm_video = argv[++i];
        else if (!strcmp(argv[i], "--web")) {
            web_port = 8080;  /* default */
            if (i + 1 < argc && argv[i+1][0] >= '0' && argv[i+1][0] <= '9')
                web_port = atoi(argv[++i]);
        }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    /* ── Init terminal UI (auto-detect mode, respect --json / --quiet) ── */
    if (json_mode) {
        tui_init(TUI_MACHINE);
    } else if (quiet_mode) {
        tui_init(TUI_PLAIN);
    } else {
        tui_init(TUI_AUTO);
    }

    mkdir("logs", 0755);
    log_init("logs/system.log", parse_lv(lv_str));
    log_set_format(LOG_FMT_JSON);
    log_install_signal_handlers();

    /* ── Banner ── */
    tui_banner("LingQi TanTong", "Edge AI Pipeline — SpacemiT K1 Muse Pi Pro");

    /* ── Benchmark mode: early exit after running benchmarks ── */
    if (benchmark) {
        int bm_ret = benchmark_run(config, bm_model, bm_runs, bm_video);
        log_shutdown();
        tui_shutdown();
        return bm_ret;
    }

    /* ── Hardware ── */
    show_hardware();

    /* ── Configuration ── */
    tui_section("Configuration");
    bool gui_mode = (!rt && !video);  /* default: GUI mode with web UI */
    if (gui_mode) {
        tui_ok("Mode: GUI (Web UI)");
        if (web_port <= 0) web_port = 8080;
        tui_keyval("Web UI port", "%d", web_port);
    } else if (rt) {
        tui_ok("Mode: REALTIME (CLI)");
        tui_keyval("Camera", "%s", camera);
    } else {
        tui_ok("Mode: OFFLINE (CLI)");
        tui_keyval("Input", "%s", video);
        tui_keyval("Max frames", "%d", max_f);
    }
    tui_keyval("Config", "%s", config);
    tui_keyval("Output", "%s", output);
    tui_keyval("Log level", "%s", lv_str);
    tui_keyval("Pose model", "%s", pm ? pm : "yolov8n-pose");
    if (save_video) tui_keyval("Save video", "YES (%s/<session>/annotated.mp4)", output);
    tui_blank();

    /* ── Create system controller ── */
    g_sc = system_controller_create(config, pm);
    if (!g_sc) {
        tui_fail("Failed to create system controller");
        log_shutdown();
        tui_shutdown();
        return 1;
    }

    if (max_f > 0) g_sc->max_frames = max_f;
    if (coap) g_sc->coap_enabled = true;
    if (coap_ip) { strncpy(g_sc->coap_esp_ip, coap_ip, sizeof(g_sc->coap_esp_ip)-1); g_sc->coap_enabled = true; }
    if (coap_port > 0) g_sc->coap_esp_port = coap_port;
    if (wifi_ss) strncpy(g_sc->coap_wifi_ssid, wifi_ss, sizeof(g_sc->coap_wifi_ssid)-1);
    if (wifi_pw) strncpy(g_sc->coap_wifi_password, wifi_pw, sizeof(g_sc->coap_wifi_password)-1);
    if (save_video) g_sc->save_video = true;
    if (camera && camera[0]) {
        strncpy(g_sc->camera_device, camera, sizeof(g_sc->camera_device) - 1);
    }

    /* ── Start embedded web UI server ──
     * In GUI mode, always start. In CLI mode, only with --web flag. */
    if (gui_mode || web_port > 0) {
        g_ws = web_server_create_default(web_port > 0 ? web_port : 8080);
        if (g_ws) {
            if (gui_mode) {
                tui_ok("Web UI: http://localhost:%d (GUI mode — open in browser)", web_port);
            } else {
                tui_ok("Web UI: http://localhost:%d", web_port);
            }
            /* Attach SystemController for WS command handling (bidirectional) */
            web_server_set_controller(g_ws, g_sc);
            g_sc->web_server = g_ws;
        } else {
            if (gui_mode) {
                tui_fail("Web UI: failed to start on port %d", web_port > 0 ? web_port : 8080);
                log_shutdown();
                tui_shutdown();
                return 1;
            }
            tui_warn("Web UI: failed to start on port %d", web_port);
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    g_sc->progress_cb = progress_cb;

    int64_t t0 = utils_get_time_ms();
    SystemStatus st; memset(&st, 0, sizeof(st));

    if (gui_mode) {
        /* ── GUI mode: web server running, wait for user commands ── */
        tui_intro("gui");
        tui_info("Pipeline is IDLE — use the Web UI to start.");
        tui_info("Press Ctrl+C to exit.");
        tui_blank();

        /* Idle wait loop — SIGINT sets g_run = 0 */
        while (g_run) {
            /* If pipeline was started and stopped, keep waiting */
            pause();
        }

        /* If pipeline is running, stop it gracefully */
        if (psm_is_running(&g_sc->state_machine)) {
            tui_muted("Stopping pipeline...");
            system_controller_stop_async(g_sc);
        }

        /* Final status from the state machine */
        st.frame_count = g_sc->frame_count;
    } else if (rt) {
        g_sc->running = 1;
        tui_intro("realtime");
        st = system_controller_process_realtime_k1(g_sc);
    } else {
        tui_intro("offline");
        st = system_controller_process_video(g_sc, video, output, max_f, save_i);
    }

    /* ── Clean up progress bar (offline mode) ── */
    if (g_prog) {
        tui_progress_done(g_prog);
        g_prog = NULL;
    }

    /* ── Session outro ── */
    int64_t elapsed_ms = utils_get_time_ms() - t0;
    double elapsed_s = (double)elapsed_ms / 1000.0;
    tui_outro(st.frame_count, st.fps, st.processing_time_ms,
              st.total_tracks, st.error_count,
              gui_mode ? "gui" : (rt ? "realtime" : "offline"), elapsed_s);
    tui_keyval("Log", "logs/system.log");
    tui_blank();

    /* ── Stop web server ── */
    if (g_ws) {
        tui_muted("Stopping Web UI server...");
        web_server_destroy_default(g_ws);
        g_ws = NULL;
    }

    /* Reset signal handlers before destroy — prevent SEGV from stale g_sc pointer */
    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);

    system_controller_destroy(g_sc);
    g_sc = NULL;
    log_shutdown();
    tui_shutdown();
    return st.error_count > 0 ? 1 : 0;
}
