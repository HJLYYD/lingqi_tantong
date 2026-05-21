#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static CRITICAL_SECTION g_log_mutex;
static bool g_log_mutex_initialized = false;
#define LOG_MUTEX_INIT() do { \
    if (!g_log_mutex_initialized) { \
        InitializeCriticalSection(&g_log_mutex); \
        g_log_mutex_initialized = true; \
    } \
} while(0)
#define LOG_MUTEX_LOCK()   EnterCriticalSection(&g_log_mutex)
#define LOG_MUTEX_UNLOCK() LeaveCriticalSection(&g_log_mutex)
#define LOG_MUTEX_DESTROY() do { \
    if (g_log_mutex_initialized) { \
        DeleteCriticalSection(&g_log_mutex); \
        g_log_mutex_initialized = false; \
    } \
} while(0)
#else
#include <pthread.h>
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;
#define LOG_MUTEX_INIT()   ((void)0)
#define LOG_MUTEX_LOCK()   pthread_mutex_lock(&g_log_mutex)
#define LOG_MUTEX_UNLOCK() pthread_mutex_unlock(&g_log_mutex)
#define LOG_MUTEX_DESTROY() pthread_mutex_destroy(&g_log_mutex)
#endif

static FILE* g_log_file = NULL;
static LogLevel g_log_level = LOG_LEVEL_INFO;
static const char* g_level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};

void logger_init(const char* log_file, LogLevel level) {
    LOG_MUTEX_INIT();
    LOG_MUTEX_LOCK();
    g_log_level = level;
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    if (log_file) {
        g_log_file = fopen(log_file, "a");
        if (!g_log_file) {
            fprintf(stderr, "Failed to open log file: %s\n", log_file);
        }
    }
    LOG_MUTEX_UNLOCK();
}

void logger_close(void) {
    LOG_MUTEX_LOCK();
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
    LOG_MUTEX_UNLOCK();
    LOG_MUTEX_DESTROY();
}

static void log_write(LogLevel level, const char* fmt, va_list args) {
    LOG_MUTEX_LOCK();

    if (level < g_log_level) {
        LOG_MUTEX_UNLOCK();
        return;
    }

    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    char msg[MAX_LOG_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, args);

    const char* level_name = g_level_names[level];

    printf("%s | %-8s | %s\n", time_str, level_name, msg);
    fflush(stdout);

    if (g_log_file) {
        fprintf(g_log_file, "%s | %-8s | %s\n", time_str, level_name, msg);
        fflush(g_log_file);
    }

    LOG_MUTEX_UNLOCK();
}

void log_debug(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(LOG_LEVEL_DEBUG, fmt, args);
    va_end(args);
}

void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(LOG_LEVEL_INFO, fmt, args);
    va_end(args);
}

void log_warning(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(LOG_LEVEL_WARNING, fmt, args);
    va_end(args);
}

void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(LOG_LEVEL_ERROR, fmt, args);
    va_end(args);
}

void log_critical(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(LOG_LEVEL_CRITICAL, fmt, args);
    va_end(args);
}

void log_set_level(LogLevel level) {
    g_log_level = level;
}

LogLevel log_get_level(void) {
    return g_log_level;
}
