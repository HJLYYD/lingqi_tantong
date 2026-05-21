#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "core_types.h"
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BENCHMARK_MAX_METRICS     32
#define BENCHMARK_MAX_SAMPLES     10000
#define BENCHMARK_TARGET_FPS      20.0
#define BENCHMARK_TARGET_AVG_MS   50.0
#define BENCHMARK_TARGET_P95_MS   100.0

typedef struct {
    char  name[64];
    float value;
    char  unit[16];
    float target;
    bool  passed;
} BenchmarkMetric;

typedef struct {
    const char* video_path;
    int         max_frames;
    int         frames_processed;

    double      frame_times[BENCHMARK_MAX_SAMPLES];
    int         num_frame_times;
    int         detection_counts[BENCHMARK_MAX_SAMPLES];
    int         num_detection_samples;

    double      total_frame_time_ms;
    int         total_objects;
    int         total_tracks;
    int         active_tracks;

    clock_t     start_wall_time;
    double      start_cpu_time;

    BenchmarkMetric metrics[BENCHMARK_MAX_METRICS];
    int             num_metrics;
} BenchmarkResult;

BenchmarkResult* benchmark_create(const char* video_path, int max_frames);
void benchmark_destroy(BenchmarkResult* bench);

void benchmark_record_frame(BenchmarkResult* bench, double frame_time_ms, int num_objects);
void benchmark_finalize(BenchmarkResult* bench);

void benchmark_generate_report(const BenchmarkResult* bench, char* out_buffer, int buffer_size);
int  benchmark_save_report(const BenchmarkResult* bench, const char* output_path);

int  benchmark_run(BenchmarkResult* bench);

#ifdef __cplusplus
}
#endif

#endif