/**
 * benchmark.h — LingQi TanTong Performance Benchmark Module
 *
 * Provides comprehensive model inference timing, pipeline profiling,
 * and tracking statistics suitable for paper screenshots and reports.
 *
 * Usage:
 *   ./lingqi_tantong --benchmark [--benchmark-model <name>] [--benchmark-runs N]
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "core_types.h"
#include "config_manager.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Benchmark configuration ── */
#define BM_DEFAULT_WARMUP_RUNS   5
#define BM_DEFAULT_TIMED_RUNS    30
#define BM_MAX_RUNS              200
#define BM_MAX_NAME_LEN          64

typedef enum {
    BM_MODEL_YOLOV8_POSE = 0,
    BM_MODEL_YOLO11N_POSE,
    BM_MODEL_YOLOV5_FACE,
    BM_MODEL_ARCFACE,
    BM_MODEL_STGCN,
    BM_MODEL_COUNT
} BMModel;

/* ── Single model benchmark result ── */
typedef struct {
    char   model_name[BM_MAX_NAME_LEN];
    char   precision[16];          /* "FP32" or "INT8+EP" */
    char   input_shape[64];        /* e.g. "640×640×3" */
    int    warmup_runs;
    int    timed_runs;
    double latencies_ms[BM_MAX_RUNS];
    double min_ms;
    double max_ms;
    double mean_ms;
    double median_ms;
    double stddev_ms;
    double p95_ms;                 /* 95th percentile */
    bool   valid;                  /* false if model failed to load */
    char   error_msg[256];
} BMModelResult;

/* ── Pipeline stage breakdown ── */
typedef struct {
    bool   valid;               /* true if profiling completed successfully */
    double capture_ms;
    double preprocess_ms;
    double inference_pose_ms;
    double inference_face_ms;
    double inference_arcface_ms;
    double inference_stgcn_ms;
    double postprocess_ms;
    double tracking_ms;
    double spatial_ms;
    double render_ms;
    double total_ms;
    int    num_samples;
} BMPipelineProfile;

/* ── Overall benchmark report ── */
typedef struct {
    BMModelResult models[BM_MODEL_COUNT];
    int           num_models;
    BMPipelineProfile pipeline;
    bool          pipeline_valid;
    int           total_frames_processed;

    /* System info */
    char   platform[64];
    int    cpu_cores;
    char   rvv_version[16];
    bool   spacemit_ep_available;
    int    tcm_size_kb;
} BMBenchmarkReport;

/* ── Main entry point ── */

/**
 * Run the full benchmark suite and print formatted results.
 *
 * @param config_path   Path to YAML config file (model paths, thresholds)
 * @param model_filter  Model name to benchmark, or NULL for all models
 * @param timed_runs    Number of timed iterations per model (0 = default 30)
 * @param video_path    Optional video file for pipeline profile (NULL = skip)
 * @return 0 on success, -1 on error
 */
int benchmark_run(const char* config_path, const char* model_filter, int timed_runs,
                  const char* video_path);

/* ── Individual benchmarks (callable externally) ── */

/**
 * Benchmark a single YOLO pose model (YOLOv8-Pose or YOLO11n-Pose).
 * Creates the estimator, runs warmup + timed inference on synthetic frame.
 */
BMModelResult benchmark_yolo_pose(const char* model_path, int input_w, int input_h,
                                   float conf_thresh, float iou_thresh,
                                   int warmup_runs, int timed_runs);

/** Benchmark YOLOv5-Face detector. */
BMModelResult benchmark_yolov5_face(const char* model_path, int input_w, int input_h,
                                     float conf_thresh, float nms_thresh,
                                     int warmup_runs, int timed_runs);

/** Benchmark ArcFace feature extractor. */
BMModelResult benchmark_arcface(const char* model_path, int warmup_runs, int timed_runs);

/** Benchmark ST-GCN action recognizer. */
BMModelResult benchmark_stgcn(const char* model_path, int num_frames, int num_kpts,
                               int num_persons, int num_classes, float conf_thresh,
                               int warmup_runs, int timed_runs);

/** Profile full inference pipeline over N frames (video file or camera). */
BMPipelineProfile benchmark_pipeline_profile(const char* config_path,
                                              const char* video_path,
                                              int max_frames);

/* ── Formatting utilities ── */

/** Print a horizontal table separator line. */
void bm_print_separator(int cols, const int* widths);

/** Print a model result as a formatted table row. */
void bm_print_result_row(const BMModelResult* r);

/** Print the full benchmark report with all tables. */
void bm_print_report(const BMBenchmarkReport* report);

/** Compute statistics (min/max/mean/median/stddev/p95) from latency array. */
void bm_compute_stats(double* latencies, int n, double* out_min, double* out_max,
                      double* out_mean, double* out_median, double* out_stddev,
                      double* out_p95);

/** Sort double array in-place (ascending). */
void bm_sort_double(double* arr, int n);

#ifdef __cplusplus
}
#endif

#endif /* BENCHMARK_H */
