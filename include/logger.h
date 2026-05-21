#ifndef LOGGER_H
#define LOGGER_H

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL
} LogLevel;

void logger_init(const char* log_file, LogLevel level);
void logger_close(void);

void log_debug(const char* fmt, ...);
void log_info(const char* fmt, ...);
void log_warning(const char* fmt, ...);
void log_error(const char* fmt, ...);
void log_critical(const char* fmt, ...);

void log_set_level(LogLevel level);
LogLevel log_get_level(void);

#ifdef __cplusplus
}
#endif

#endif
