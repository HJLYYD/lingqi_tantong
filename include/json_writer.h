/*
 * json_writer.h — Minimal streaming JSON writer for K1 embedded
 *
 * ~300 lines of C. No heap allocation. Proper escaping. Streaming FILE* output.
 * Patterned after cJSON's API but write-only, not a DOM.
 */

#ifndef JSON_WRITER_H
#define JSON_WRITER_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    FILE* f;
    int   depth;
    int   indent;      /* spaces per level; 0 = compact */
    bool  comma;
} JsonW;

/** Initialize writer attached to a FILE. indent=2 for pretty, 0 for compact. */
void jw_init(JsonW* w, FILE* f, int indent);

/* ── Container open/close ── */
void jw_obj_begin(JsonW* w);
void jw_obj_end(JsonW* w);
void jw_arr_begin(JsonW* w);
void jw_arr_end(JsonW* w);

/* ── Key (only valid inside an object, before a value) ── */
void jw_key(JsonW* w, const char* key);

/* ── Value writers (with proper JSON escaping for strings) ── */
void jw_str(JsonW* w, const char* val);
void jw_int(JsonW* w, int64_t val);
void jw_float(JsonW* w, double val, int decimals);
void jw_bool(JsonW* w, bool val);
void jw_null(JsonW* w);

/* ── Convenience: key + value in one call ── */
static inline void jw_kstr(JsonW* w, const char* k, const char* v)
    { jw_key(w,k); jw_str(w,v); }
static inline void jw_kint(JsonW* w, const char* k, int64_t v)
    { jw_key(w,k); jw_int(w,v); }
static inline void jw_kflt(JsonW* w, const char* k, double v, int d)
    { jw_key(w,k); jw_float(w,v,d); }
static inline void jw_kbool(JsonW* w, const char* k, bool v)
    { jw_key(w,k); jw_bool(w,v); }

/** key: [a, b, c] */
void jw_kvec3(JsonW* w, const char* k, double x, double y, double z, int dec);

/** Flush underlying file */
void jw_flush(JsonW* w);

/* ── Backward-compat long names ── */
#define JsonWriter              JsonW
#define json_writer_init(w,f,i) jw_init(w,f,i)
#define json_writer_object_begin jw_obj_begin
#define json_writer_object_end   jw_obj_end
#define json_writer_array_begin  jw_arr_begin
#define json_writer_array_end    jw_arr_end
#define json_writer_key          jw_key
#define json_writer_string       jw_str
#define json_writer_int          jw_int
#define json_writer_float        jw_float
#define json_writer_bool         jw_bool
#define json_writer_null         jw_null
#define json_writer_key_string   jw_kstr
#define json_writer_key_int      jw_kint
#define json_writer_key_float    jw_kflt
#define json_writer_key_bool     jw_kbool
#define json_writer_key_float3   jw_kvec3
#define json_writer_flush        jw_flush

#ifdef __cplusplus
}
#endif
#endif
