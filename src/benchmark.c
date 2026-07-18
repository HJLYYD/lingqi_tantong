/**
 * benchmark.c — LingQi TanTong Performance Benchmark Module
 *
 * Usage:
 *   ./lingqi_tantong --benchmark                        (all models, 30 runs each)
 *   ./lingqi_tantong --benchmark --benchmark-runs 50    (50 timed runs each)
 *   ./lingqi_tantong --benchmark --benchmark-model yolo  (pose model only)
 *
 * Output: formatted ASCII tables suitable for paper screenshots.
 */

#include "benchmark.h"
#include "yolov8_pose_estimator.h"
#include "yolov5_face_detector.h"
#include "arcface_recognizer.h"
#include "stgcn_action_recognizer.h"
#include "inference_pipeline.h"
#include "tracking_manager.h"
#include "keypoint_validator.h"
#include "video_processor.h"
#include "config_manager.h"
#include "k1_platform.h"
#include "utils.h"
#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Sorting & Statistics
 * ═══════════════════════════════════════════════════════════════════════════ */

static int cmp_double_asc(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

void bm_sort_double(double* arr, int n) {
    if (n > 1) qsort(arr, (size_t)n, sizeof(double), cmp_double_asc);
}

void bm_compute_stats(double* latencies, int n,
                      double* out_min, double* out_max,
                      double* out_mean, double* out_median,
                      double* out_stddev, double* out_p95) {
    if (n <= 0) {
        *out_min = *out_max = *out_mean = *out_median = *out_stddev = *out_p95 = 0.0;
        return;
    }
    bm_sort_double(latencies, n);

    *out_min = latencies[0];
    *out_max = latencies[n - 1];

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += latencies[i];
    *out_mean = sum / (double)n;

    if (n % 2 == 1)
        *out_median = latencies[n / 2];
    else
        *out_median = (latencies[n / 2 - 1] + latencies[n / 2]) * 0.5;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = latencies[i] - *out_mean;
        var += d * d;
    }
    *out_stddev = sqrt(var / (double)n);

    int p95_idx = (int)((double)n * 0.95 + 0.5);
    if (p95_idx >= n) p95_idx = n - 1;
    if (p95_idx < 0) p95_idx = 0;
    *out_p95 = latencies[p95_idx];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Synthetic Test Data Generation
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Generate a dummy RGB frame (random pixel values) for model warmup/timing.
 *  Using deterministic seed for reproducibility. */
static uint8_t* synthetic_rgb_frame(int width, int height) {
    size_t size = (size_t)width * height * 3;
    uint8_t* data = (uint8_t*)malloc(size);
    if (!data) return NULL;
    srand(42);  /* deterministic seed for reproducible benchmarks */
    for (size_t i = 0; i < size; i++) {
        data[i] = (uint8_t)(rand() % 256);
    }
    return data;
}

/** Generate a dummy 112×112 face crop for ArcFace benchmark. */
static uint8_t* synthetic_face_crop(void) {
    return synthetic_rgb_frame(112, 112);
}

/** Push synthetic skeleton data into ST-GCN until buffer is full. */
static void stgcn_fill_buffer(STGCNActionRecognizer* rec, int img_w, int img_h) {
    PoseEstimation dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.num_keypoints = 17;
    /* Generate a simple standing pose pattern */
    float cx = (float)(img_w / 2);
    float top_y = (float)(img_h * 0.15f);
    float bot_y = (float)(img_h * 0.95f);
    float sh_y = (float)(img_h * 0.25f);
    float hi_y = (float)(img_h * 0.55f);
    float kn_y = (float)(img_h * 0.75f);

    /* COCO-17: assign plausible standing-pose coordinates */
    float kx[17] = {cx, cx-4, cx+4, cx-8, cx+8,  /* nose, eyes, ears */
                    cx-15, cx+15,                 /* shoulders */
                    cx-25, cx+25,                 /* elbows */
                    cx-30, cx+30,                 /* wrists */
                    cx-12, cx+12,                 /* hips */
                    cx-14, cx+14,                 /* knees */
                    cx-16, cx+16};                /* ankles */
    float ky[17] = {top_y, top_y-3, top_y-3, top_y-2, top_y-2,
                    sh_y, sh_y,
                    sh_y+30, sh_y+30,
                    sh_y+60, sh_y+60,
                    hi_y, hi_y,
                    kn_y, kn_y,
                    bot_y, bot_y};
    float kc[17] = {0.9f, 0.8f, 0.8f, 0.7f, 0.7f,
                    0.9f, 0.9f, 0.8f, 0.8f, 0.7f, 0.7f,
                    0.9f, 0.9f, 0.8f, 0.8f, 0.7f, 0.7f};

    for (int i = 0; i < 17; i++) {
        dummy.keypoints[i].x = kx[i];
        dummy.keypoints[i].y = ky[i];
        dummy.keypoints[i].confidence = kc[i];
    }

    /* Push enough frames to fill the ST-GCN buffer */
    for (int f = 0; f < rec->num_frames + 5; f++) {
        stgcn_action_recognizer_push_pose(rec, &dummy, img_w, img_h);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * High-precision timing
 * ═══════════════════════════════════════════════════════════════════════════ */

static double bm_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Individual Model Benchmarks
 * ═══════════════════════════════════════════════════════════════════════════ */

BMModelResult benchmark_yolo_pose(const char* model_path, int input_w, int input_h,
                                   float conf_thresh, float iou_thresh,
                                   int warmup_runs, int timed_runs) {
    BMModelResult result;
    memset(&result, 0, sizeof(result));
    snprintf(result.model_name, BM_MAX_NAME_LEN, "YOLO-Pose");
    snprintf(result.precision, sizeof(result.precision), "%s",
             strstr(model_path, ".q.onnx") ? "INT8+EP" : "FP32");
    snprintf(result.input_shape, sizeof(result.input_shape), "%d×%d×3", input_w, input_h);
    result.warmup_runs = warmup_runs;
    result.timed_runs = timed_runs;

    YOLOv8PoseEstimator* est = yolov8_pose_estimator_create(
        model_path, input_w, input_h, conf_thresh, iou_thresh);
    if (!est) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create estimator for: %s", model_path);
        result.valid = false;
        return result;
    }

    uint8_t* frame = synthetic_rgb_frame(input_w, input_h);
    if (!frame) {
        yolov8_pose_estimator_destroy(est);
        snprintf(result.error_msg, sizeof(result.error_msg), "OOM allocating test frame");
        result.valid = false;
        return result;
    }

    /* ── Warmup ── */
    PoseEstimation dummy_poses[12];
    for (int i = 0; i < warmup_runs; i++) {
        yolov8_pose_estimator_estimate(est, frame, input_w, input_h, dummy_poses, 12);
    }

    /* ── Timed runs ── */
    int valid_runs = 0;
    for (int i = 0; i < timed_runs && valid_runs < BM_MAX_RUNS; i++) {
        double t0 = bm_get_time_us();
        yolov8_pose_estimator_estimate(est, frame, input_w, input_h, dummy_poses, 12);
        double t1 = bm_get_time_us();
        result.latencies_ms[valid_runs++] = (t1 - t0) * 0.001;
    }

    free(frame);
    yolov8_pose_estimator_destroy(est);

    if (valid_runs > 0) {
        bm_compute_stats(result.latencies_ms, valid_runs,
                         &result.min_ms, &result.max_ms, &result.mean_ms,
                         &result.median_ms, &result.stddev_ms, &result.p95_ms);
        result.timed_runs = valid_runs;
        result.valid = true;
    } else {
        snprintf(result.error_msg, sizeof(result.error_msg), "All timed runs failed");
        result.valid = false;
    }

    return result;
}

BMModelResult benchmark_yolov5_face(const char* model_path, int input_w, int input_h,
                                     float conf_thresh, float nms_thresh,
                                     int warmup_runs, int timed_runs) {
    BMModelResult result;
    memset(&result, 0, sizeof(result));
    snprintf(result.model_name, BM_MAX_NAME_LEN, "YOLOv5-Face");
    snprintf(result.precision, sizeof(result.precision), "%s",
             strstr(model_path, ".q.onnx") ? "INT8+EP" : "FP32");
    snprintf(result.input_shape, sizeof(result.input_shape), "%d×%d×3", input_w, input_h);
    result.warmup_runs = warmup_runs;
    result.timed_runs = timed_runs;

    YOLOv5FaceDetector* det = yolov5_face_detector_create(
        model_path, input_w, input_h, conf_thresh, nms_thresh, false);
    if (!det) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create detector for: %s", model_path);
        result.valid = false;
        return result;
    }

    uint8_t* frame = synthetic_rgb_frame(input_w, input_h);
    if (!frame) {
        yolov5_face_detector_destroy(det);
        snprintf(result.error_msg, sizeof(result.error_msg), "OOM allocating test frame");
        result.valid = false;
        return result;
    }

    /* ── Warmup ── */
    FaceIdentity dummy_faces[10];
    for (int i = 0; i < warmup_runs; i++) {
        yolov5_face_detector_detect_faces(det, frame, input_w, input_h, dummy_faces, 10);
    }

    /* ── Timed runs ── */
    int valid_runs = 0;
    for (int i = 0; i < timed_runs && valid_runs < BM_MAX_RUNS; i++) {
        double t0 = bm_get_time_us();
        yolov5_face_detector_detect_faces(det, frame, input_w, input_h, dummy_faces, 10);
        double t1 = bm_get_time_us();
        result.latencies_ms[valid_runs++] = (t1 - t0) * 0.001;
    }

    free(frame);
    yolov5_face_detector_destroy(det);

    if (valid_runs > 0) {
        bm_compute_stats(result.latencies_ms, valid_runs,
                         &result.min_ms, &result.max_ms, &result.mean_ms,
                         &result.median_ms, &result.stddev_ms, &result.p95_ms);
        result.timed_runs = valid_runs;
        result.valid = true;
    } else {
        snprintf(result.error_msg, sizeof(result.error_msg), "All timed runs failed");
        result.valid = false;
    }

    return result;
}

BMModelResult benchmark_arcface(const char* model_path, int warmup_runs, int timed_runs) {
    BMModelResult result;
    memset(&result, 0, sizeof(result));
    snprintf(result.model_name, BM_MAX_NAME_LEN, "ArcFace");
    snprintf(result.precision, sizeof(result.precision), "%s",
             strstr(model_path, ".q.onnx") ? "INT8+EP" : "FP32");
    snprintf(result.input_shape, sizeof(result.input_shape), "112×112×3");
    result.warmup_runs = warmup_runs;
    result.timed_runs = timed_runs;

    ArcFaceRecognizer* rec = arcface_recognizer_create(model_path, 112, 112, 0.45f);
    if (!rec) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create recognizer for: %s", model_path);
        result.valid = false;
        return result;
    }

    uint8_t* face = synthetic_face_crop();
    if (!face) {
        arcface_recognizer_destroy(rec);
        snprintf(result.error_msg, sizeof(result.error_msg), "OOM allocating face crop");
        result.valid = false;
        return result;
    }

    /* ── Warmup ── */
    float dummy_feat[128];
    for (int i = 0; i < warmup_runs; i++) {
        arcface_recognizer_extract_feature(rec, face, 112, 112, dummy_feat);
    }

    /* ── Timed runs ── */
    int valid_runs = 0;
    for (int i = 0; i < timed_runs && valid_runs < BM_MAX_RUNS; i++) {
        double t0 = bm_get_time_us();
        arcface_recognizer_extract_feature(rec, face, 112, 112, dummy_feat);
        double t1 = bm_get_time_us();
        result.latencies_ms[valid_runs++] = (t1 - t0) * 0.001;
    }

    free(face);
    arcface_recognizer_destroy(rec);

    if (valid_runs > 0) {
        bm_compute_stats(result.latencies_ms, valid_runs,
                         &result.min_ms, &result.max_ms, &result.mean_ms,
                         &result.median_ms, &result.stddev_ms, &result.p95_ms);
        result.timed_runs = valid_runs;
        result.valid = true;
    } else {
        snprintf(result.error_msg, sizeof(result.error_msg), "All timed runs failed");
        result.valid = false;
    }

    return result;
}

BMModelResult benchmark_stgcn(const char* model_path, int num_frames, int num_kpts,
                               int num_persons, int num_classes, float conf_thresh,
                               int warmup_runs, int timed_runs) {
    BMModelResult result;
    memset(&result, 0, sizeof(result));
    snprintf(result.model_name, BM_MAX_NAME_LEN, "ST-GCN");
    snprintf(result.precision, sizeof(result.precision), "%s",
             strstr(model_path, ".q.onnx") ? "INT8+EP" : "FP32 (CPU EP)");
    snprintf(result.input_shape, sizeof(result.input_shape),
             "(1,3,%d,%d,%d)", num_frames, num_kpts, num_persons);
    result.warmup_runs = warmup_runs;
    result.timed_runs = timed_runs;

    STGCNActionRecognizer* rec = stgcn_action_recognizer_create(
        model_path, num_frames, num_kpts, num_persons, num_classes, conf_thresh);
    if (!rec) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create recognizer for: %s", model_path);
        result.valid = false;
        return result;
    }

    /* Fill skeleton buffer with synthetic data */
    stgcn_fill_buffer(rec, 640, 480);

    /* ── Warmup ── */
    for (int i = 0; i < warmup_runs; i++) {
        stgcn_action_recognizer_recognize(rec);
    }

    /* ── Timed runs ── */
    int valid_runs = 0;
    for (int i = 0; i < timed_runs && valid_runs < BM_MAX_RUNS; i++) {
        double t0 = bm_get_time_us();
        stgcn_action_recognizer_recognize(rec);
        double t1 = bm_get_time_us();
        result.latencies_ms[valid_runs++] = (t1 - t0) * 0.001;
    }

    stgcn_action_recognizer_destroy(rec);

    if (valid_runs > 0) {
        bm_compute_stats(result.latencies_ms, valid_runs,
                         &result.min_ms, &result.max_ms, &result.mean_ms,
                         &result.median_ms, &result.stddev_ms, &result.p95_ms);
        result.timed_runs = valid_runs;
        result.valid = true;
    } else {
        snprintf(result.error_msg, sizeof(result.error_msg), "All timed runs failed");
        result.valid = false;
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline Profiling (end-to-end stage breakdown)
 * ═══════════════════════════════════════════════════════════════════════════ */

BMPipelineProfile benchmark_pipeline_profile(const char* config_path,
                                              const char* video_path,
                                              int max_frames) {
    BMPipelineProfile profile;
    memset(&profile, 0, sizeof(profile));

    if (!video_path || video_path[0] == '\0') {
        profile.valid = false;
        return profile;
    }

    ConfigManager* config = config_manager_create(config_path);
    if (!config) {
        profile.valid = false;
        return profile;
    }

    /* Create inference pipeline */
    AIInferencePipeline* pipeline = inference_pipeline_create();
    if (!pipeline) {
        config_manager_destroy(config);
        profile.valid = false;
        return profile;
    }

    if (inference_pipeline_load_models(pipeline, "models", config) != 0) {
        inference_pipeline_destroy(pipeline);
        config_manager_destroy(config);
        profile.valid = false;
        return profile;
    }

    /* Open video */
    VideoProcessor* vp = video_processor_create(video_path, 640, 480, false);
    if (!vp || video_processor_open(vp, video_path) != VP_OK) {
        if (vp) video_processor_destroy(vp);
        inference_pipeline_destroy(pipeline);
        config_manager_destroy(config);
        profile.valid = false;
        return profile;
    }

    int w = video_processor_get_width(vp);
    int h = video_processor_get_height(vp);
    if (w <= 0) w = 640;
    if (h <= 0) h = 480;

    /* Warmup: 3 frames */
    for (int i = 0; i < 3; i++) {
        FrameData* fd = video_processor_read_frame(vp);
        if (!fd) break;
        InferenceResult ir_result;
        inference_pipeline_process_frame(pipeline, fd->data, fd->width, fd->height, &ir_result);
        frame_data_destroy(fd);
    }

    /* Timed frames */
    int frame_count = 0;
    for (int f = 0; f < max_frames; f++) {
        double t_total_0 = bm_get_time_us();

        double t_cap_0 = bm_get_time_us();
        FrameData* fd = video_processor_read_frame(vp);
        double t_cap_1 = bm_get_time_us();
        if (!fd) break;

        InferenceResult ir_result;
        inference_pipeline_process_frame(pipeline, fd->data, fd->width, fd->height, &ir_result);

        double t_total_1 = bm_get_time_us();

        profile.capture_ms    += (t_cap_1 - t_cap_0) * 0.001;
        profile.inference_pose_ms += ir_result.processing_time_ms; /* total as proxy */
        profile.total_ms      += (t_total_1 - t_total_0) * 0.001;

        frame_data_destroy(fd);
        frame_count++;
    }
    profile.num_samples = frame_count;

    /* Average */
    if (frame_count > 0) {
        profile.capture_ms    /= (double)frame_count;
        profile.inference_pose_ms /= (double)frame_count;
        profile.total_ms      /= (double)frame_count;
    }

    video_processor_destroy(vp);
    inference_pipeline_destroy(pipeline);
    config_manager_destroy(config);

    profile.valid = (frame_count > 0);
    return profile;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Formatted Output (screenshot-ready ASCII tables)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define BM_LINE_LEN 120

static void bm_hr(void) {
    printf("+");
    for (int i = 0; i < BM_LINE_LEN - 2; i++) putchar('=');
    printf("+\n");
}

static void bm_hr_thin(void) {
    printf("+");
    for (int i = 0; i < BM_LINE_LEN - 2; i++) putchar('-');
    printf("+\n");
}

static void bm_title(const char* title) {
    bm_hr();
    int pad = (int)((BM_LINE_LEN - 2 - strlen(title)) / 2);
    printf("|");
    for (int i = 0; i < pad; i++) putchar(' ');
    printf("%s", title);
    for (int i = pad + (int)strlen(title); i < BM_LINE_LEN - 2; i++) putchar(' ');
    printf("|\n");
    bm_hr();
}

static void bm_labeled_value(const char* label, const char* fmt, ...) {
    printf("|  %-40s ", label);
    char buf[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    printf("%-70s |\n", buf);
}

/* Table rendering for model benchmark results */
static void bm_table_header(void) {
    bm_hr_thin();
    printf("| %-18s | %6s | %12s | %5s | %8s | %8s | %8s | %8s | %8s | %6s |\n",
           "Model", "Prec", "Input Shape", "Runs",
           "Min(ms)", "Max(ms)", "Mean(ms)", "Med(ms)", "Std(ms)", "P95(ms)");
    bm_hr_thin();
}

static void bm_table_row(const BMModelResult* r) {
    if (!r->valid) {
        printf("| %-18s | %6s | %12s | %5s | %8s | %8s | %8s | %8s | %8s | %6s |\n",
               r->model_name, "-", "-", "-", "-", "-", "-", "-", "-", "-");
        printf("|   ^-- ERROR: %-97s |\n", r->error_msg);
        return;
    }
    printf("| %-18s | %6s | %12s | %5d | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %6.2f |\n",
           r->model_name, r->precision, r->input_shape, r->timed_runs,
           r->min_ms, r->max_ms, r->mean_ms, r->median_ms, r->stddev_ms, r->p95_ms);
}

void bm_print_separator(int cols, const int* widths) {
    (void)cols; (void)widths;
    bm_hr_thin();
}

void bm_print_result_row(const BMModelResult* r) {
    bm_table_row(r);
}

void bm_print_report(const BMBenchmarkReport* report) {
    if (!report) return;

    /* ══════════════════════════════════════════════════════════════
     * Header
     * ══════════════════════════════════════════════════════════════ */
    bm_title("LingQi TanTong — Performance Benchmark Report");

    printf("|  %-40s %-75s |\n", "Platform:", report->platform);
    printf("|  %-40s %-75s |\n", "CPU:", "");
    {
        char cpu_info[128];
        snprintf(cpu_info, sizeof(cpu_info), "%d cores, RVV %s, TCM %d KB, SpacemiT EP: %s",
                 report->cpu_cores, report->rvv_version, report->tcm_size_kb,
                 report->spacemit_ep_available ? "YES" : "NO");
        printf("|  %-40s %-75s |\n", "", cpu_info);
    }

    /* ══════════════════════════════════════════════════════════════
     * Table 1: Model Inference Latency
     * ══════════════════════════════════════════════════════════════ */
    printf("\n");
    bm_title("Table 1: Per-Model Inference Latency (single forward pass)");

    printf("|  Benchmark configuration:                                           |\n");
    printf("|    Warmup runs per model: %-40d                         |\n",
           report->num_models > 0 ? report->models[0].warmup_runs : BM_DEFAULT_WARMUP_RUNS);
    printf("|    Timed runs per model:  %-40d                         |\n",
           report->num_models > 0 ? report->models[0].timed_runs : BM_DEFAULT_TIMED_RUNS);
    printf("|    Input: synthetic frame (deterministic seed 42)                  |\n");
    printf("|    Measurement: clock_gettime(CLOCK_MONOTONIC) microsecond precision|\n");
    bm_hr_thin();

    bm_table_header();

    for (int i = 0; i < report->num_models; i++) {
        bm_table_row(&report->models[i]);
    }
    bm_hr_thin();

    /* ── Speedup summary ── */
    printf("\n");
    bm_title("Table 1b: INT8+EP Speedup vs FP32 (if both available)");
    bm_table_header();
    /* Find pairs of FP32 and INT8 results for same model */
    for (int i = 0; i < report->num_models; i++) {
        if (!report->models[i].valid) continue;
        /* Look for a matching FP32 variant */
        for (int j = i + 1; j < report->num_models; j++) {
            if (!report->models[j].valid) continue;
            if (strcmp(report->models[i].model_name, report->models[j].model_name) == 0) {
                const BMModelResult* fp32 = NULL, *int8 = NULL;
                if (strcmp(report->models[i].precision, "FP32") == 0 &&
                    strstr(report->models[j].precision, "INT8")) {
                    fp32 = &report->models[i];
                    int8 = &report->models[j];
                } else if (strcmp(report->models[j].precision, "FP32") == 0 &&
                           strstr(report->models[i].precision, "INT8")) {
                    fp32 = &report->models[j];
                    int8 = &report->models[i];
                }
                if (fp32 && int8 && fp32->valid && int8->valid) {
                    double speedup = fp32->mean_ms / (int8->mean_ms > 0.001 ? int8->mean_ms : 0.001);
                    printf("| %-18s | %6s | %12s | %5d | %8.2f | %8s | %8s | %8s | %8s | %5.1fx |\n",
                           int8->model_name, "INT8", int8->input_shape, int8->timed_runs,
                           int8->mean_ms, "-", "-", "-", "-", speedup);
                    printf("| %-18s | %6s | %12s | %5d | %8.2f | %8s | %8s | %8s | %8s | %6s |\n",
                           fp32->model_name, "FP32", fp32->input_shape, fp32->timed_runs,
                           fp32->mean_ms, "-", "-", "-", "-", "1.0x");
                    break;
                }
            }
        }
    }
    bm_hr_thin();

    /* ══════════════════════════════════════════════════════════════
     * Table 2: Latency Stability Distribution
     * ══════════════════════════════════════════════════════════════ */
    printf("\n");
    bm_title("Table 2: Per-Model Latency Distribution (raw samples in ms)");

    for (int i = 0; i < report->num_models; i++) {
        const BMModelResult* r = &report->models[i];
        if (!r->valid) continue;
        printf("|  %-18s (%s, %s): ", r->model_name, r->precision, r->input_shape);
        /* Print first 15 values inline */
        int show = r->timed_runs < 15 ? r->timed_runs : 15;
        for (int k = 0; k < show; k++) {
            printf("%.1f", r->latencies_ms[k]);
            if (k < show - 1) printf(", ");
        }
        if (r->timed_runs > 15) printf(", ... (%d total)", r->timed_runs);
        printf("\n");
        printf("|  %-40s", "");
        printf("  Range: [%.1f, %.1f]ms  Mean: %.1fms  Median: %.1fms  P95: %.1fms  CV: %.1f%%\n",
               r->min_ms, r->max_ms, r->mean_ms, r->median_ms, r->p95_ms,
               r->mean_ms > 0.01 ? (r->stddev_ms / r->mean_ms * 100.0) : 0.0);
    }
    bm_hr_thin();

    /* ══════════════════════════════════════════════════════════════
     * Table 3: Pipeline E2E Profile
     * ══════════════════════════════════════════════════════════════ */
    printf("\n");
    bm_title("Table 3: End-to-End Pipeline Profile (requires --benchmark-video)");
    if (report->pipeline_valid) {
        printf("|  %-40s %-75s |\n", "Frames profiled:", "");
        char buf[64]; snprintf(buf, sizeof(buf), "%d frames", report->total_frames_processed);
        printf("|  %-40s %-75s |\n", "", buf);

        bm_hr_thin();
        printf("| %-30s | %10s | %10s | %-57s |\n",
               "Pipeline Stage", "Time(ms)", "Percent", "Notes");
        bm_hr_thin();

        double total = report->pipeline.total_ms;
        if (total < 0.01) total = 1.0;

        #define BM_STAGE_ROW(name, val, note) do { \
            printf("| %-30s | %10.2f | %9.1f%% | %-57s |\n", \
                   (name), (val), ((val) / total * 100.0), (note)); \
        } while(0)

        BM_STAGE_ROW("1. Frame Capture", report->pipeline.capture_ms, "read + JPEG decode");
        BM_STAGE_ROW("2. AI Inference (all models)", report->pipeline.inference_pose_ms,
                     report->pipeline.inference_pose_ms > 0 ?
                     "pose+face+arcface+stgcn combined" : "see per-model Table 1");
        bm_hr_thin();
        printf("| %-30s | %10.2f | %9.1f%% | %-57s |\n",
               "TOTAL (capture+inference+output)", total, 100.0,
               "end-to-end per frame");
        bm_hr_thin();

        printf("|  Note: For per-model breakdown, see Table 1.                              |\n");
        printf("|  Use ./lingqi_tantong --benchmark --benchmark-video test_video.mp4        |\n");
        printf("|  to profile with an actual video file (not synthetic data).               |\n");

        #undef BM_STAGE_ROW
    } else {
        printf("|                                                                           |\n");
        printf("|  NOT RUN — pipeline profiling requires a video file.                      |\n");
        printf("|                                                                           |\n");
        printf("|  Usage:                                                                    |\n");
        printf("|    ./lingqi_tantong --benchmark --benchmark-video test_video.mp4           |\n");
        printf("|                                                                           |\n");
        printf("|  This profiles the full capture→inference→postprocess→render pipeline.    |\n");
        printf("|  For per-model single-forward-pass latency, see Table 1 above.            |\n");
    }
    bm_hr_thin();

    /* ══════════════════════════════════════════════════════════════
     * Legend / Methodology
     * ══════════════════════════════════════════════════════════════ */
    printf("\n");
    bm_title("Methodology Notes");
    printf("|  * All measurements use clock_gettime(CLOCK_MONOTONIC)                |\n");
    printf("|  * Warmup runs excluded from statistics                               |\n");
    printf("|  * Synthetic input data (deterministic seed) for reproducibility      |\n");
    printf("|  * CV = Coefficient of Variation = stddev/mean * 100%%                 |\n");
    printf("|  * INT8+EP = INT8 quantized model on SpacemiT EP (RVV+IME)           |\n");
    printf("|  * FP32 = FP32 model on CPU EP (no hardware acceleration)             |\n");
    printf("|  * P95 = 95th percentile latency (worst-case excluding outliers)      |\n");
    bm_hr();

    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main Benchmark Entry Point
 * ═══════════════════════════════════════════════════════════════════════════ */

int benchmark_run(const char* config_path, const char* model_filter, int timed_runs,
                  const char* video_path) {
    if (timed_runs <= 0) timed_runs = BM_DEFAULT_TIMED_RUNS;
    if (timed_runs > BM_MAX_RUNS) timed_runs = BM_MAX_RUNS;
    int warmup_runs = BM_DEFAULT_WARMUP_RUNS;

    ConfigManager* config = config_manager_create(config_path);
    if (!config) {
        fprintf(stderr, "BENCHMARK ERROR: Cannot load config: %s\n", config_path);
        return -1;
    }

    BMBenchmarkReport report;
    memset(&report, 0, sizeof(report));
    report.num_models = 0;

    /* ── System info ── */
    K1Platform* plat = k1_platform_init();
    if (plat) {
        snprintf(report.platform, sizeof(report.platform), "SpacemiT K1 Muse Pi Pro");
        report.cpu_cores = k1_platform_cpu_count();
        snprintf(report.rvv_version, sizeof(report.rvv_version), "%s",
                 k1_platform_has_cap(K1_CAP_RVV_1_0) ? "1.0" : "NONE");
        report.spacemit_ep_available = k1_platform_has_cap(K1_CAP_SPACEMIT_EP);
        report.tcm_size_kb = k1_platform_get_tcm_size();
        k1_platform_destroy(plat);
    } else {
        snprintf(report.platform, sizeof(report.platform), "Generic (non-K1)");
        report.cpu_cores = 4;
        snprintf(report.rvv_version, sizeof(report.rvv_version), "NONE");
        report.spacemit_ep_available = false;
        report.tcm_size_kb = 0;
    }

    /* ── Determine model paths from config ── */
    bool run_all = (model_filter == NULL || model_filter[0] == '\0');

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  LingQi TanTong — Model Inference Benchmark                                                           ║\n");
    printf("║  Platform: %-90s ║\n", report.platform);
    printf("║  Config:   %-90s ║\n", config_path);
    printf("║  Warmup:   %d runs per model  |  Timed: %d runs per model  |  Precision: clock_gettime(MONOTONIC) ║\n",
           warmup_runs, timed_runs);
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /* ══════════════════════════════════════════════════════════════
     * 1. YOLOv8-Pose (INT8)
     * ══════════════════════════════════════════════════════════════ */
    if (run_all || strstr(model_filter, "pose") || strstr(model_filter, "yolo")) {
        printf("[1/5] Benchmarking YOLOv8-Pose (INT8+EP)...");
        fflush(stdout);

        const char* variant = config_get_string(config, "pose.model_variant", "yolov8n-pose");
        const char* mp = NULL;
        if (strcmp(variant, "yolo11n-pose") == 0)
            mp = "models/Action Prediction/Skeleton Recognition/yolo11n-pose.q.onnx";
        else
            mp = "models/Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx";

        int w = config_get_int(config, "pose.input_size.0", 640);
        int h = config_get_int(config, "pose.input_size.1", 640);
        float conf = config_get_float(config, "pose.confidence_threshold", 0.04f);
        float iou  = config_get_float(config, "pose.iou_threshold", 0.55f);

        /* Override model name if yolo11n variant */
        BMModelResult r = benchmark_yolo_pose(mp, w, h, conf, iou, warmup_runs, timed_runs);
        if (strcmp(variant, "yolo11n-pose") == 0)
            snprintf(r.model_name, BM_MAX_NAME_LEN, "YOLO11n-Pose");
        else
            snprintf(r.model_name, BM_MAX_NAME_LEN, "YOLOv8-Pose");

        printf(" %s (%.1f ms avg, %d runs)\n",
               r.valid ? "OK" : "FAIL", r.valid ? r.mean_ms : 0.0, r.timed_runs);
        report.models[report.num_models++] = r;
    }

    /* ══════════════════════════════════════════════════════════════
     * 2. YOLO11n Person Detection (INT8) — if separate model exists
     * ══════════════════════════════════════════════════════════════ */
    if (run_all || strstr(model_filter, "yolo11")) {
        printf("[2/5] Benchmarking YOLO11n Detection (INT8+EP)...");
        fflush(stdout);

        /* YOLO11n as standalone person detector (not pose) */
        const char* mp = "models/Action Prediction/Object Detection/yolov11n_320x320.q.onnx";
        int w = 320, h = 320;
        float conf = config_get_float(config, "detection.confidence_threshold", 0.12f);
        float iou  = config_get_float(config, "detection.iou_threshold", 0.40f);

        /* Use YOLO-Pose estimator API — handles both pose and detection models */
        BMModelResult r = benchmark_yolo_pose(mp, w, h, conf, iou, warmup_runs, timed_runs);
        snprintf(r.model_name, BM_MAX_NAME_LEN, "YOLO11n");

        printf(" %s (%.1f ms avg, %d runs)\n",
               r.valid ? "OK" : "FAIL", r.valid ? r.mean_ms : 0.0, r.timed_runs);
        report.models[report.num_models++] = r;
    }

    /* ══════════════════════════════════════════════════════════════
     * 3. YOLOv5-Face (INT8)
     * ══════════════════════════════════════════════════════════════ */
    if (run_all || strstr(model_filter, "face")) {
        printf("[3/5] Benchmarking YOLOv5-Face (INT8+EP)...");
        fflush(stdout);

        const char* mp = config_get_string(config, "face.detection_model_path", NULL);
        if (!mp || mp[0] == '\0')
            mp = "models/Face Recognition/yolov5n-face_320_cut.q.onnx";

        int w = config_get_int(config, "face.input_size.0", 320);
        int h = config_get_int(config, "face.input_size.1", 320);
        float conf = config_get_float(config, "face.confidence_threshold", 0.30f);
        float iou  = config_get_float(config, "face.iou_threshold", 0.4f);

        BMModelResult r = benchmark_yolov5_face(mp, w, h, conf, iou, warmup_runs, timed_runs);

        printf(" %s (%.1f ms avg, %d runs)\n",
               r.valid ? "OK" : "FAIL", r.valid ? r.mean_ms : 0.0, r.timed_runs);
        report.models[report.num_models++] = r;
    }

    /* ══════════════════════════════════════════════════════════════
     * 4. ArcFace (INT8)
     * ══════════════════════════════════════════════════════════════ */
    if (run_all || strstr(model_filter, "arcface") || strstr(model_filter, "face")) {
        printf("[4/5] Benchmarking ArcFace (INT8+EP)...");
        fflush(stdout);

        const char* mp = config_get_string(config, "face.recognition_model_path", NULL);
        if (!mp || mp[0] == '\0')
            mp = "models/Face Recognition/arcface_mobilefacenet_cut.q.onnx";

        BMModelResult r = benchmark_arcface(mp, warmup_runs, timed_runs);

        printf(" %s (%.1f ms avg, %d runs)\n",
               r.valid ? "OK" : "FAIL", r.valid ? r.mean_ms : 0.0, r.timed_runs);
        report.models[report.num_models++] = r;
    }

    /* ══════════════════════════════════════════════════════════════
     * 5. ST-GCN (FP32 CPU EP)
     * ══════════════════════════════════════════════════════════════ */
    if (run_all || strstr(model_filter, "stgcn") || strstr(model_filter, "action")) {
        printf("[5/5] Benchmarking ST-GCN (FP32 CPU EP)...");
        fflush(stdout);

        const char* mp = config_get_string(config, "action.model_path", NULL);
        if (!mp || mp[0] == '\0')
            mp = "models/Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx";

        int nf = config_get_int(config, "action.num_frames", 30);
        int nk = config_get_int(config, "action.num_keypoints", 14);
        int np = config_get_int(config, "action.num_persons", 1);
        int nc = config_get_int(config, "action.num_classes", 7);
        float conf = config_get_float(config, "action.confidence_threshold", 0.5f);

        BMModelResult r = benchmark_stgcn(mp, nf, nk, np, nc, conf, warmup_runs, timed_runs);

        printf(" %s (%.1f ms avg, %d runs)\n",
               r.valid ? "OK" : "FAIL", r.valid ? r.mean_ms : 0.0, r.timed_runs);
        report.models[report.num_models++] = r;
    }

    printf("\n");

    /* ══════════════════════════════════════════════════════════════
     * Pipeline E2E profiling (only if --benchmark-video provided)
     * ══════════════════════════════════════════════════════════════ */
    if (video_path && video_path[0] != '\0') {
        printf("Profiling end-to-end pipeline with video: %s\n", video_path);
        fflush(stdout);
        report.pipeline = benchmark_pipeline_profile(config_path, video_path, 100);
        if (report.pipeline.valid) {
            report.pipeline_valid = true;
            report.total_frames_processed = report.pipeline.num_samples;
            printf("  OK (%d frames profiled)\n\n", report.pipeline.num_samples);
        } else {
            printf("  FAIL (could not open video or run pipeline)\n\n");
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * Print formatted report
     * ══════════════════════════════════════════════════════════════ */
    bm_print_report(&report);

    /* ── Also print machine-readable JSON summary ── */
    printf("\n");
    bm_title("Machine-Readable JSON Summary");
    printf("{\n");
    printf("  \"platform\": \"%s\",\n", report.platform);
    printf("  \"cpu_cores\": %d,\n", report.cpu_cores);
    printf("  \"rvv_version\": \"%s\",\n", report.rvv_version);
    printf("  \"spacemit_ep\": %s,\n", report.spacemit_ep_available ? "true" : "false");
    printf("  \"tcm_size_kb\": %d,\n", report.tcm_size_kb);
    printf("  \"benchmark_config\": {\n");
    printf("    \"warmup_runs\": %d,\n", warmup_runs);
    printf("    \"timed_runs\": %d\n", timed_runs);
    printf("  },\n");
    printf("  \"models\": [\n");
    for (int i = 0; i < report.num_models; i++) {
        const BMModelResult* r = &report.models[i];
        printf("    {\n");
        printf("      \"name\": \"%s\",\n", r->model_name);
        printf("      \"precision\": \"%s\",\n", r->precision);
        printf("      \"input_shape\": \"%s\",\n", r->input_shape);
        printf("      \"valid\": %s,\n", r->valid ? "true" : "false");
        if (r->valid) {
            printf("      \"runs\": %d,\n", r->timed_runs);
            printf("      \"min_ms\": %.2f,\n", r->min_ms);
            printf("      \"max_ms\": %.2f,\n", r->max_ms);
            printf("      \"mean_ms\": %.2f,\n", r->mean_ms);
            printf("      \"median_ms\": %.2f,\n", r->median_ms);
            printf("      \"stddev_ms\": %.2f,\n", r->stddev_ms);
            printf("      \"p95_ms\": %.2f\n", r->p95_ms);
        } else {
            printf("      \"error\": \"%s\"\n", r->error_msg);
        }
        printf("    }%s\n", (i < report.num_models - 1) ? "," : "");
    }
    printf("  ]\n");
    printf("}\n\n");

    config_manager_destroy(config);
    return 0;
}
