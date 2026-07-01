/*
 * json_writer.c — Implementation
 *
 * Proper JSON escaping: " → \", \ → \\, \n → \n, \r → \r, \t → \t,
 * control chars (0x00-0x1F) → \u00XX.
 */

#include "json_writer.h"
#include <string.h>
#include <stdio.h>

/* ── Internal ── */

static void indent(JsonW* w) {
    if (w->indent <= 0) return;
    fputc('\n', w->f);
    for (int i = 0; i < w->depth * w->indent; i++) fputc(' ', w->f);
}

static void comma(JsonW* w) {
    if (w->comma) fputc(',', w->f);
    w->comma = true;
}

static void write_escaped(FILE* f, const char* s) {
    if (!s) { fputs("null", f); return; }
    fputc('"', f);
    for (const char* p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\b': fputs("\\b",  f); break;
        case '\f': fputs("\\f",  f); break;
        case '\n': fputs("\\n",  f); break;
        case '\r': fputs("\\r",  f); break;
        case '\t': fputs("\\t",  f); break;
        default:
            if (c < 0x20) fprintf(f, "\\u%04x", (unsigned)c);
            else fputc(c, f);
            break;
        }
    }
    fputc('"', f);
}

/* ── Public ── */

void jw_init(JsonW* w, FILE* f, int indent) {
    memset(w, 0, sizeof(*w));
    w->f = f; w->indent = indent;
}

void jw_obj_begin(JsonW* w) {
    comma(w); indent(w); fputc('{', w->f);
    w->comma = false; w->depth++;
}

void jw_obj_end(JsonW* w) {
    w->depth--; indent(w); fputc('}', w->f);
    w->comma = true;
}

void jw_arr_begin(JsonW* w) {
    comma(w); indent(w); fputc('[', w->f);
    w->comma = false; w->depth++;
}

void jw_arr_end(JsonW* w) {
    w->depth--; indent(w); fputc(']', w->f);
    w->comma = true;
}

void jw_key(JsonW* w, const char* key) {
    comma(w); indent(w);
    write_escaped(w->f, key);
    fputs(": ", w->f);
    w->comma = false;
}

void jw_str(JsonW* w, const char* val) {
    comma(w); indent(w);
    write_escaped(w->f, val);
}

void jw_int(JsonW* w, int64_t val) {
    comma(w); indent(w);
    fprintf(w->f, "%lld", (long long)val);
}

void jw_float(JsonW* w, double val, int dec) {
    comma(w); indent(w);
    fprintf(w->f, "%.*f", dec, val);
}

void jw_bool(JsonW* w, bool val) {
    comma(w); indent(w);
    fputs(val ? "true" : "false", w->f);
}

void jw_null(JsonW* w) {
    comma(w); indent(w);
    fputs("null", w->f);
}

void jw_kvec3(JsonW* w, const char* k, double x, double y, double z, int dec) {
    jw_key(w, k);
    fprintf(w->f, "[%.*f, %.*f, %.*f]", dec, x, dec, y, dec, z);
    w->comma = true;
}

void jw_flush(JsonW* w) { if (w->f) fflush(w->f); }
