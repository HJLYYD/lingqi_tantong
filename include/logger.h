/*
 * logger.h — Lock-free structured logging for K1 edge AI pipeline
 *
 * Architecture (NanoLog + spdlog + OpenTelemetry):
 *   Per-thread SPSC ring buffers → background writer thread → stdout + rotating file
 *
 * Key properties:
 *   - ~5μs per log call (vs ~50μs for mutex-based)
 *   - JSON structured output for machine parsing
 *   - Built-in trace spans + latency histograms
 *   - Zero heap allocation in hot path
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Types
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    LOG_LV_TRACE = -1,
    LOG_LV_DEBUG = 0,
    LOG_LV_INFO,
    LOG_LV_WARN,
    LOG_LV_ERROR,
    LOG_LV_FATAL,
} LogLevel;

/* Format for file output (stdout is always human-readable color text) */
typedef enum {
    LOG_FMT_TEXT = 0,   /* "HH:MM:SS.MMM LVL [thread] msg" */
    LOG_FMT_JSON = 1,   /* {"ts":N,"lvl":"INFO","th":"inf","msg":"..."} */
} LogFormat;

/* Latency histogram bucket boundaries (microseconds) */
enum { LOG_HIST_BUCKETS = 8 };
extern const int64_t LOG_HIST_BOUNDARIES_US[LOG_HIST_BUCKETS];
/* Buckets: <1ms, <5ms, <10ms, <20ms, <50ms, <100ms, <200ms, <500ms */

/* ── Trace span ── */
typedef struct LogSpan_ {
    const char*         name;
    int64_t             start_ns;
    int64_t             end_ns;
    uint64_t            span_id;
    uint64_t            trace_id;
    uint64_t            parent_id;
    int                 status;         /* 0=ok, non-zero=error */
    char                attrs[256];
    struct LogSpan_*    next;           /* intrusive stack */
} LogSpan;

/* ═══════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ═══════════════════════════════════════════════════════════════════════ */

/** Initialize logger. log_path may be NULL (stdout only). */
void log_init(const char* log_path, LogLevel min_level);

/** Shutdown: drain all buffers, close files, join writer thread. */
void log_shutdown(void);

/** Synchronous flush: drain all rings + fsync. Call before critical exit. */
void log_flush(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Basic logging (hot path — lock-free, allocation-free)
 * ═══════════════════════════════════════════════════════════════════════ */

void log_write(LogLevel lv, const char* fmt, ...) __attribute__((format(printf,2,3)));

#define log_trace(...)  log_write(LOG_LV_TRACE, __VA_ARGS__)
#define log_debug(...)  log_write(LOG_LV_DEBUG, __VA_ARGS__)
#define log_info(...)   log_write(LOG_LV_INFO,  __VA_ARGS__)
#define log_warn(...)   log_write(LOG_LV_WARN,  __VA_ARGS__)
#define log_warning     log_warn
#define log_error(...)  log_write(LOG_LV_ERROR, __VA_ARGS__)
#define log_fatal(...)    log_write(LOG_LV_FATAL, __VA_ARGS__)
#define log_critical(...) log_write(LOG_LV_FATAL, __VA_ARGS__)

/* ── Backward-compat aliases for old caller code ── */
#define LOG_LEVEL_TRACE    LOG_LV_TRACE
#define LOG_LEVEL_DEBUG    LOG_LV_DEBUG
#define LOG_LEVEL_INFO     LOG_LV_INFO
#define LOG_LEVEL_WARNING  LOG_LV_WARN
#define LOG_LEVEL_ERROR    LOG_LV_ERROR
#define LOG_LEVEL_CRITICAL LOG_LV_FATAL
#define log_event          log_ev
#define log_latency_record log_hist_record
#define log_latency_dump   log_hist_dump
#define logger_init        log_init
#define logger_close       log_shutdown
#define log_set_thread_name log_thread_name
#define log_event_info(e, ...)  log_ev(LOG_LV_INFO,  e, __VA_ARGS__)
#define log_event_warn(e, ...)  log_ev(LOG_LV_WARN,  e, __VA_ARGS__)
#define log_event_error(e, ...) log_ev(LOG_LV_ERROR, e, __VA_ARGS__)

/** Set minimum level at runtime. Thread-safe. */
void log_set_level(LogLevel lv);
LogLevel log_get_level(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Structured event logging (OpenTelemetry-aligned)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Log a structured event with key=value attributes.
 * Terminate pairs with NULL key.
 *
 * Example:
 *   log_ev(LOG_LV_INFO, "inference.frame",
 *          "frame",    "%d",   42,
 *          "latency",  "%lld", 12345LL,
 *          "dets",     "%d",   3,
 *          NULL);
 */
void log_ev(LogLevel lv, const char* event, ...);

/* ═══════════════════════════════════════════════════════════════════════
 * Trace spans (OpenTelemetry-compatible)
 * ═══════════════════════════════════════════════════════════════════════ */

/** Start a span. Returns handle; NULL if depth exceeds max (8). */
LogSpan* log_span_begin(const char* name);

/** End span and emit to log. attrs may be NULL. */
void log_span_end(LogSpan* s, int status, const char* attrs);

/** Add timestamped event to an active span. */
void log_span_ev(LogSpan* s, const char* name, const char* attrs);

/** End span with status=0, no attrs. */
static inline void log_span_done(LogSpan* s) { log_span_end(s, 0, NULL); }

/* ═══════════════════════════════════════════════════════════════════════
 * Thread context (for cross-thread log correlation)
 * ═══════════════════════════════════════════════════════════════════════ */

/** Set display name for calling thread (max 15 chars). */
void log_thread_name(const char* name);

/** Set current frame sequence ID for calling thread. */
void log_frame_id(uint64_t id);

/** Set current trace ID (all spans inherit this). */
void log_trace_id(uint64_t id);

/* ═══════════════════════════════════════════════════════════════════════
 * Latency histograms
 * ═══════════════════════════════════════════════════════════════════════ */

/** Record a latency sample into named histogram. ~10ns amortized. */
void log_hist_record(const char* stage, int64_t latency_us);

/** Dump all histogram summaries at INFO level and reset counters. */
void log_hist_dump(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════ */

/** Set output format for file sink. */
void log_set_format(LogFormat fmt);

/** Enable auto-rotation: max bytes per file, max rotated files to keep. */
void log_set_rotation(size_t max_bytes, int max_files);

/** Force immediate rotation. */
void log_rotate(void);

/** Install SIGUSR1 (cycle level) and SIGHUP (rotate) handlers. */
void log_install_signal_handlers(void);

#ifdef __cplusplus
}
#endif

#endif /* LOGGER_H */
