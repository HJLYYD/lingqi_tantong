#include "benchmark.h"
#include "logger.h"
#include "utils.h"
#include "system_controller.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

BenchmarkResult* benchmark_create(const char* video_path, int max_frames) {
    BenchmarkResult* bench = (BenchmarkResult*)calloc(1, sizeof(BenchmarkResult));
    if (!bench) return NULL;

    bench->video_path = video_path;
    bench->max_frames = max_frames > 0 && max_frames <= BENCHMARK_MAX_SAMPLES
                        ? max_frames : BENCHMARK_MAX_SAMPLES;
    bench->frames_processed = 0;
    bench->num_frame_times = 0;
    bench->num_detection_samples = 0;
    bench->total_frame_time_ms = 0.0;
    bench->total_objects = 0;
    bench->total_tracks = 0;
    bench->active_tracks = 0;
    bench->start_wall_time = (double)utils_get_time_ms();
    bench->start_cpu_time = (double)utils_get_time_ms() / 1000.0;
    bench->num_metrics = 0;

    return bench;
}

void benchmark_destroy(BenchmarkResult* bench) {
    if (!bench) return;
    free(bench);
}

void benchmark_record_frame(BenchmarkResult* bench, double frame_time_ms, int num_objects) {
    if (!bench) return;

    if (bench->num_frame_times < BENCHMARK_MAX_SAMPLES) {
        bench->frame_times[bench->num_frame_times++] = frame_time_ms;
    }
    if (bench->num_detection_samples < BENCHMARK_MAX_SAMPLES) {
        bench->detection_counts[bench->num_detection_samples++] = num_objects;
    }

    bench->total_frame_time_ms += frame_time_ms;
    bench->total_objects += num_objects;
    bench->frames_processed++;
}

static int cmp_double(const void* a, const void* b) {
    double diff = *(const double*)a - *(const double*)b;
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}

static void add_metric_impl(BenchmarkResult* bench, const char* name,
                             float val, const char* unit,
                             float target, bool pass) {
    if (bench->num_metrics >= BENCHMARK_MAX_METRICS) return;
    BenchmarkMetric* m = &bench->metrics[bench->num_metrics++];
    snprintf(m->name, sizeof(m->name), "%s", name);
    m->value = val;
    snprintf(m->unit, sizeof(m->unit), "%s", unit);
    m->target = target;
    m->passed = pass;
}

static double compute_percentile(double* arr, int n, double p) {
    if (n <= 0) return 0.0;
    qsort(arr, n, sizeof(double), cmp_double);
    int k = (int)ceil(p / 100.0 * n) - 1;
    if (k < 0) k = 0;
    if (k >= n) k = n - 1;
    return arr[k];
}

void benchmark_finalize(BenchmarkResult* bench) {
    if (!bench || bench->num_frame_times <= 0) return;

    double times_copy[BENCHMARK_MAX_SAMPLES];
    memcpy(times_copy, bench->frame_times, bench->num_frame_times * sizeof(double));

    double avg_ms = bench->total_frame_time_ms / bench->num_frame_times;
    double max_ms = times_copy[0];
    double min_ms = times_copy[0];

    for (int i = 1; i < bench->num_frame_times; i++) {
        if (times_copy[i] > max_ms) max_ms = times_copy[i];
        if (times_copy[i] < min_ms) min_ms = times_copy[i];
    }

    double p95 = compute_percentile(times_copy, bench->num_frame_times, 95.0);
    double p99 = compute_percentile(times_copy, bench->num_frame_times, 99.0);
    double avg_objects = bench->num_detection_samples > 0
        ? (double)bench->total_objects / bench->num_detection_samples : 0.0;
    double fps = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;

    bench->num_metrics = 0;

#define ADD_METRIC(_name, _val, _unit, _target, _pass) \
    add_metric_impl(bench, _name, (float)(_val), _unit, (float)(_target), (_pass))

    ADD_METRIC("avg_frame_time_ms", avg_ms, "ms", BENCHMARK_TARGET_AVG_MS,
               avg_ms <= BENCHMARK_TARGET_AVG_MS);
    ADD_METRIC("p95_frame_time_ms", p95, "ms", BENCHMARK_TARGET_P95_MS,
               p95 <= BENCHMARK_TARGET_P95_MS);
    ADD_METRIC("max_frame_time_ms", max_ms, "ms", 150.0, max_ms <= 150.0);
    ADD_METRIC("min_frame_time_ms", min_ms, "ms", 50.0, min_ms <= 50.0);
    ADD_METRIC("avg_fps", fps, "fps", BENCHMARK_TARGET_FPS,
               fps >= BENCHMARK_TARGET_FPS);
    ADD_METRIC("avg_objects_per_frame", avg_objects, "obj/frame", 5.0,
               avg_objects <= 5.0);
    ADD_METRIC("frames_processed", (double)bench->frames_processed, "frames",
               (double)bench->max_frames, true);
    ADD_METRIC("total_objects", (double)bench->total_objects, "objects",
               0.0, true);

#undef ADD_METRIC
}

void benchmark_generate_report(const BenchmarkResult* bench, char* out_buffer, int buffer_size) {
    if (!bench || !out_buffer || buffer_size <= 0) return;

    int pos = 0;

    pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
        "======================================================================\n"
        "LINGQI TANTONG C - PERFORMANCE BENCHMARK REPORT\n"
        "======================================================================\n\n");

    pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
        "Video: %s\n"
        "Frames processed: %d / %d\n"
        "Total objects: %d\n"
        "Total frame time: %.1f ms\n\n",
        bench->video_path ? bench->video_path : "N/A",
        bench->frames_processed, bench->max_frames,
        bench->total_objects,
        bench->total_frame_time_ms);

    pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
        "----------------------------------------------------------------------\n"
        "%-40s %10s %10s %10s\n"
        "----------------------------------------------------------------------\n",
        "Metric", "Value", "Target", "Status");

    for (int i = 0; i < bench->num_metrics; i++) {
        const BenchmarkMetric* m = &bench->metrics[i];
        pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
            "%-40s %9.2f %s %9.1f %9s\n",
            m->name, m->value, m->unit, m->target,
            m->passed ? "PASS" : "FAIL");
    }

    pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
        "----------------------------------------------------------------------\n");

    int pass_count = 0;
    for (int i = 0; i < bench->num_metrics; i++) {
        if (bench->metrics[i].passed) pass_count++;
    }

    pos += snprintf(out_buffer + pos, UTILS_MAX(0, buffer_size - pos),
        "\n======================================================================\n"
        "SUMMARY: %d / %d benchmarks PASSED\n"
        "======================================================================\n",
        pass_count, bench->num_metrics);
}

int benchmark_save_report(const BenchmarkResult* bench, const char* output_path) {
    if (!bench || !output_path) return -1;

    char report[8192];
    memset(report, 0, sizeof(report));
    benchmark_generate_report(bench, report, sizeof(report));

    FILE* f = fopen(output_path, "w");
    if (!f) {
        log_error("Cannot save benchmark report: %s", output_path);
        return -1;
    }

    fprintf(f, "%s", report);
    fclose(f);
    log_info("Benchmark report saved: %s", output_path);
    return 0;
}

int benchmark_run(BenchmarkResult* bench) {
    if (!bench) return -1;

    log_info("==========================================");
    log_info("PERFORMANCE BENCHMARK - Starting");
    log_info("==========================================");
    log_info("Video: %s", bench->video_path ? bench->video_path : "N/A");
    log_info("Max frames: %d", bench->max_frames);

    SystemController* sys = system_controller_create(NULL);
    if (!sys) {
        log_error("Failed to create system controller");
        return -1;
    }

    double bench_start_ms = utils_get_time_ms();

    SystemStatus status = system_controller_process_video(
        sys, bench->video_path, NULL, bench->max_frames, false, 0);

    double bench_end_ms = utils_get_time_ms();
    double total_wall_time = bench_end_ms - bench_start_ms;

    for (int i = 0; i < sys->proc_times_count && i < BENCHMARK_MAX_SAMPLES; i++) {
        double frame_ms = sys->processing_times[i] * 1000.0;
        benchmark_record_frame(bench, frame_ms, 0);
    }

    system_controller_destroy(sys);

    benchmark_finalize(bench);

    log_info("Benchmark completed: %d frames in %.1fs (status: %s)",
             status.frame_count, total_wall_time / 1000.0, status.message);

    char report[8192];
    benchmark_generate_report(bench, report, sizeof(report));
    log_info("\n%s", report);

    return 0;
}