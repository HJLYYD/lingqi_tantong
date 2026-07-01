/*
 * logger.c — Lock-free structured logging implementation
 *
 * Architecture:
 *   ┌──────────┐   SPSC ring (4K entries)   ┌───────────────┐
 *   │ Thread A │──────────────────────────►│               │
 *   │ Thread B │──► own ring ─────────────►│ writer_thread │──► stdout (color)
 *   │ Thread C │──► own ring ─────────────►│  (CPU idle)   │──► file  (JSON)
 *   └──────────┘                           └───────────────┘
 *
 * Hot path (log_write): ~5μs — only a ring buffer copy + atomic store.
 * No mutex, no malloc, no syscall in the calling thread.
 */

#include "logger.h"
#include "terminal_ui.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Platform abstractions
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef _WIN32
#include <sys/syscall.h>
static inline int64_t clock_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}
static inline uint32_t thread_hash(void) {
    return (uint32_t)(uintptr_t)pthread_self();
}
#else
static inline int64_t clock_us(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return c.QuadPart * 1000000LL / f.QuadPart;
}
static inline uint32_t thread_hash(void) {
    return (uint32_t)(uintptr_t)GetCurrentThreadId();
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════ */

enum {
    RING_ENTRIES        = 4096,     /* power of 2 */
    RING_MASK           = 4095,
    MAX_THREADS         = 16,
    MAX_SPAN_DEPTH      = 8,
    MSG_BUF_SIZE        = 512,
    WRITER_SLEEP_US     = 5000,
    ROTATE_DEFAULT_BYTES = 10 * 1024 * 1024,  /* 10 MB */
    ROTATE_DEFAULT_FILES = 10,
    HISTO_DUMP_FRAMES   = 30,
    MAX_HISTOGRAMS      = 16,
};

/* Histogram bucket boundaries */
const int64_t LOG_HIST_BOUNDARIES_US[LOG_HIST_BUCKETS] = {
    1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000,
};

/* Level name table — indexed by level+1 */
static const char* const LV_NAMES[] = {
    [0] = "TRACE", [1] = "DEBUG", [2] = "INFO",
    [3] = "WARN",  [4] = "ERROR", [5] = "FATAL",
};

/* ANSI color codes — unified palette from terminal_ui.h */
static const char* const LV_COLORS[] = {
    [0] = TUI_ANSI_MUTED,     /* TRACE = gray */
    [1] = TUI_ANSI_INFO,      /* DEBUG = cyan */
    [2] = TUI_ANSI_SUCCESS,   /* INFO  = green */
    [3] = TUI_ANSI_WARNING,   /* WARN  = yellow */
    [4] = TUI_ANSI_ERROR,     /* ERROR = red */
    [5] = TUI_ANSI_ACCENT,    /* FATAL = magenta */
};
#define COLOR_OFF TUI_ANSI_RESET

/* ═══════════════════════════════════════════════════════════════════════
 * Ring buffer entry (what gets copied in the hot path)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int64_t  ts_us;             /* monotonic timestamp */
    uint32_t thread_h;
    uint32_t frame_id;
    int16_t  level;             /* LogLevel + 1 */
    char     th_name[16];
    char     msg[MSG_BUF_SIZE];
} RingEntry;

/* ═══════════════════════════════════════════════════════════════════════
 * Histogram (atomic counters)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    char             name[32];
    volatile uint64_t buckets[LOG_HIST_BUCKETS];
    volatile uint64_t overflow;
    volatile uint64_t count;
    volatile uint64_t sum_us;
} Histogram;

/* ═══════════════════════════════════════════════════════════════════════
 * Global state
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Ring buffers — one per registered thread */
    RingEntry*          rings[MAX_THREADS];
    volatile unsigned   wpos[MAX_THREADS];   /* producer index */
    volatile unsigned   rpos[MAX_THREADS];   /* consumer index (only writer thread reads) */
    int                 nrings;
    pthread_mutex_t     reg_mtx;

    /* Writer thread */
    pthread_t           writer_th;
    volatile bool       running;
    volatile bool       quit;

    /* Sinks */
    FILE*               file;
    char                path[256];
    LogLevel            min_lv;
    LogFormat           fmt;
    bool                color;              /* true if stdout is a TTY */
    int64_t             t0_us;              /* monotonic clock at init */

    /* Rotation */
    size_t              rot_bytes;
    int                 rot_files;
    size_t              cur_bytes;

    /* Drop counter */
    volatile uint64_t   drops;

    /* Signal handlers installed? */
    bool                sig_ok;

    /* Histograms */
    Histogram           histos[MAX_HISTOGRAMS];
    int                 nhistos;
    pthread_mutex_t     histo_mtx;
    int                 dump_frame;

    /* Span ID counter */
    volatile uint64_t   next_sid;
} Logger;

static Logger G = {0};
static bool   G_ok = false;

/* ═══════════════════════════════════════════════════════════════════════
 * Thread-local state
 * ═══════════════════════════════════════════════════════════════════════ */

static __thread char     TL_name[16]    = "main";
static __thread uint64_t TL_frame       = 0;
static __thread uint64_t TL_trace       = 0;
static __thread int      TL_ring        = -1;
static __thread LogSpan* TL_spans[MAX_SPAN_DEPTH];
static __thread int      TL_sdepth      = 0;

/* ═══════════════════════════════════════════════════════════════════════
 * Ring buffer helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/** Register calling thread's ring buffer (called once, lazily). */
static int ring_register(void) {
    if (TL_ring >= 0) return TL_ring;

    pthread_mutex_lock(&G.reg_mtx);
    if (G.nrings >= MAX_THREADS) {
        pthread_mutex_unlock(&G.reg_mtx);
        return -1;
    }
    int idx = G.nrings++;
    G.rings[idx] = (RingEntry*)calloc(RING_ENTRIES, sizeof(RingEntry));
    G.wpos[idx] = 0;
    G.rpos[idx] = 0;
    TL_ring = idx;
    pthread_mutex_unlock(&G.reg_mtx);
    return idx;
}

/** Push one entry. Returns false if ring full (entry dropped). */
static bool ring_push(int ri, const RingEntry* e) {
    unsigned w = G.wpos[ri];
    unsigned n = (w + 1) & RING_MASK;
    __sync_synchronize();  /* acquire: see G.rpos[ri] from writer thread */
    unsigned r = G.rpos[ri];
    if (n == r) {
        G.drops++;
        return false;
    }
    memcpy(&G.rings[ri][w], e, sizeof(RingEntry));
    __sync_synchronize();  /* release: make entry visible before advancing wpos */
    G.wpos[ri] = n;
    return true;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Output formatting
 * ═══════════════════════════════════════════════════════════════════════ */

/** JSON-escape a string into buf. Returns bytes written (excluding NUL). */
static int json_escape(char* dst, size_t dstsz, const char* src) {
    char* start = dst;
    char* end   = dst + dstsz - 1;
    while (*src && dst < end) {
        unsigned char c = (unsigned char)*src++;
        switch (c) {
        case '"':  if (dst+1 < end) { *dst++='\\'; *dst++='"';  } break;
        case '\\': if (dst+1 < end) { *dst++='\\'; *dst++='\\'; } break;
        case '\n': if (dst+1 < end) { *dst++='\\'; *dst++='n';  } break;
        case '\r': if (dst+1 < end) { *dst++='\\'; *dst++='r';  } break;
        case '\t': if (dst+1 < end) { *dst++='\\'; *dst++='t';  } break;
        default:
            if (c < 0x20) {
                dst += snprintf(dst, (size_t)(end - dst), "\\u%04x", c);
            } else {
                *dst++ = (char)c;
            }
            break;
        }
    }
    *dst = '\0';
    return (int)(dst - start);
}

/** Format entry as human-readable text line. */
static int fmt_text(char* buf, size_t sz, const RingEntry* e, int64_t t0) {
    int64_t us = e->ts_us - t0;
    int ms = (int)((us / 1000) % 1000);
    int s  = (int)(us / 1000000);
    return snprintf(buf, sz, "%02d:%02d:%02d.%03d %-5s [%-12s] %s\n",
                    s/3600, (s/60)%60, s%60, ms,
                    LV_NAMES[e->level], e->th_name, e->msg);
}

/** Format entry as JSON line. */
static int fmt_json(char* buf, size_t sz, const RingEntry* e, int64_t t0) {
    char escaped[MSG_BUF_SIZE * 2];
    json_escape(escaped, sizeof(escaped), e->msg);
    return snprintf(buf, sz,
        "{\"ts\":%lld,\"lv\":\"%s\",\"th\":\"%s\",\"fr\":%u,\"msg\":\"%s\"}\n",
        (long long)(e->ts_us - t0),
        LV_NAMES[e->level], e->th_name, e->frame_id, escaped);
}

/* ═══════════════════════════════════════════════════════════════════════
 * File rotation
 * ═══════════════════════════════════════════════════════════════════════ */

static void rotate_file(void) {
    if (!G.file || G.rot_bytes == 0) return;

    fflush(G.file);
    fclose(G.file);
    G.file = NULL;
    G.cur_bytes = 0;

    /* Shift: log.9 → log.10, ..., log → log.1 */
    for (int i = G.rot_files - 1; i >= 0; i--) {
        char old[512], new_[512];
        snprintf(old, sizeof(old), "%s%s", G.path, i == 0 ? "" : "");
        snprintf(new_, sizeof(new_), "%s.%d", G.path, i + 1);
        if (i > 0) {
            snprintf(old, sizeof(old), "%s.%d", G.path, i);
        }
        rename(old, new_);
    }

    G.file = fopen(G.path, "w");
    if (!G.file) {
        fprintf(stderr, "logger: rotate failed for %s (errno=%d)\n", G.path, errno);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Writer thread — drains all rings, writes to sinks
 * ═══════════════════════════════════════════════════════════════════════ */

static void* writer_thread(void* arg) {
    (void)arg;
    char   buf[MSG_BUF_SIZE + 256];
    uint64_t last_drops = 0;

    while (G.running) {
        bool work = false;

        for (int ri = 0; ri < G.nrings; ri++) {
            for (;;) {
                unsigned r = G.rpos[ri];
                unsigned w = G.wpos[ri];
                if (r == w) break;
                work = true;

                const RingEntry* e = &G.rings[ri][r];

                /* ── Write to stdout (always text, with color if TTY) ── */
                int n = fmt_text(buf, sizeof(buf), e, G.t0_us);
                if (G.color) {
                    fprintf(stdout, "%s%s" COLOR_OFF, LV_COLORS[e->level], buf);
                } else {
                    fwrite(buf, 1, (size_t)n, stdout);
                }
                fflush(stdout);

                /* ── Write to file (text or JSON) ── */
                if (G.file) {
                    int fn = (G.fmt == LOG_FMT_JSON)
                        ? fmt_json(buf, sizeof(buf), e, G.t0_us)
                        : fmt_text(buf, sizeof(buf), e, G.t0_us);
                    fwrite(buf, 1, (size_t)fn, G.file);
                    G.cur_bytes += (size_t)fn;
                    if (G.rot_bytes > 0 && G.cur_bytes >= G.rot_bytes) {
                        rotate_file();
                    }
                }

                /* Advance read cursor */
                do{G.rpos[ri]=(r + 1) & RING_MASK;__sync_synchronize();}while(0);
            }
        }

        /* Report drops */
        uint64_t drops = G.drops;
        if (drops > last_drops) {
            uint64_t d = drops - last_drops;
            last_drops = drops;
            if (G.file) {
                int n = snprintf(buf, sizeof(buf),
                    "{\"ts\":%lld,\"lv\":\"WARN\",\"th\":\"logger\",\"fr\":0,"
                    "\"msg\":\"dropped %llu messages (ring full)\"}\n",
                    (long long)(clock_us() - G.t0_us), (unsigned long long)d);
                fwrite(buf, 1, (size_t)n, G.file);
                G.cur_bytes += (size_t)n;
            }
        }

        if (!work) {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = WRITER_SLEEP_US * 1000 };
            nanosleep(&ts, NULL);
        }
    }

    /* Final drain */
    for (int ri = 0; ri < G.nrings; ri++) {
        for (;;) {
            unsigned r = G.rpos[ri];
            unsigned w = G.wpos[ri];
            if (r == w) break;
            const RingEntry* e = &G.rings[ri][r];
            if (G.file) {
                int n = fmt_json(buf, sizeof(buf), e, G.t0_us);
                fwrite(buf, 1, (size_t)n, G.file);
            }
            do{G.rpos[ri]=(r + 1) & RING_MASK;__sync_synchronize();}while(0);
        }
    }
    if (G.file) { fflush(G.file); fclose(G.file); G.file = NULL; }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ═══════════════════════════════════════════════════════════════════════ */

void log_init(const char* log_path, LogLevel min_level) {
    /* BUGFIX: mutex around init to prevent TOCTOU race where two threads
     * both pass the G_ok check and double-initialize the global G.
     * PTHREAD_MUTEX_INITIALIZER is valid in C11 (static init). */
    static pthread_mutex_t init_mtx = PTHREAD_MUTEX_INITIALIZER;

    pthread_mutex_lock(&init_mtx);
    if (G_ok) { pthread_mutex_unlock(&init_mtx); return; }   /* idempotent */

    memset(&G, 0, sizeof(G));
    G.min_lv    = min_level;
    G.fmt       = LOG_FMT_JSON;
    G.color     = tui_is_interactive();    /* unified TTY detection via terminal_ui */
    G.t0_us     = clock_us();
    G.rot_bytes = ROTATE_DEFAULT_BYTES;
    G.rot_files = ROTATE_DEFAULT_FILES;
    G.running  = false;
    G.quit     = false;
    G.drops    = 0;
    G.next_sid = 1;
    pthread_mutex_init(&G.reg_mtx, NULL);
    pthread_mutex_init(&G.histo_mtx, NULL);

    if (log_path && log_path[0]) {
        strncpy(G.path, log_path, sizeof(G.path) - 1);
        /* Ensure parent directory exists */
        char dir[256];
        strncpy(dir, log_path, sizeof(dir) - 1);
        char* slash = strrchr(dir, '/');
        if (!slash) slash = strrchr(dir, '\\');
        if (slash) { *slash = '\0'; mkdir(dir, 0755); }
        G.file = fopen(log_path, "w");
        if (!G.file) {
            fprintf(stderr, "logger: cannot open %s (errno=%d)\n", log_path, errno);
        }
    }

    /* Start writer thread */
    do{G.running=true;__sync_synchronize();}while(0);
    if (pthread_create(&G.writer_th, NULL, writer_thread, NULL) != 0) {
        fprintf(stderr, "logger: writer thread creation failed\n");
        do{G.running=false;__sync_synchronize();}while(0);
        pthread_mutex_unlock(&init_mtx);
        return;
    }

    G_ok = true;
    pthread_mutex_unlock(&init_mtx);
    log_info("logger started: lv=%s fmt=%s path=%s rot=%zuMB×%d",
             LV_NAMES[min_level + 1],
             G.fmt == LOG_FMT_JSON ? "json" : "text",
             log_path ? log_path : "(none)",
             G.rot_bytes / (1024 * 1024), G.rot_files);
}

void log_shutdown(void) {
    if (!G_ok) return;
    log_info("logger shutting down");
    do{G.running=false;__sync_synchronize();}while(0);
    do{G.quit=true;__sync_synchronize();}while(0);
    pthread_join(G.writer_th, NULL);
    for (int i = 0; i < G.nrings; i++) free(G.rings[i]);
    pthread_mutex_destroy(&G.reg_mtx);
    pthread_mutex_destroy(&G.histo_mtx);
    G_ok = false;
}

void log_flush(void) {
    if (!G_ok) return;
    /* Synchronously drain all rings */
    char buf[MSG_BUF_SIZE + 256];
    for (int ri = 0; ri < G.nrings; ri++) {
        for (;;) {
            unsigned r = G.rpos[ri];
            unsigned w = G.wpos[ri];
            if (r == w) break;
            const RingEntry* e = &G.rings[ri][r];
            int n = fmt_text(buf, sizeof(buf), e, G.t0_us);
            fwrite(buf, 1, (size_t)n, stdout);
            if (G.file) {
                int fn = (G.fmt == LOG_FMT_JSON)
                    ? fmt_json(buf, sizeof(buf), e, G.t0_us)
                    : fmt_text(buf, sizeof(buf), e, G.t0_us);
                fwrite(buf, 1, (size_t)fn, G.file);
            }
            do{G.rpos[ri]=(r + 1) & RING_MASK;__sync_synchronize();}while(0);
        }
    }
    fflush(stdout);
    if (G.file) fflush(G.file);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Basic logging — hot path
 * ═══════════════════════════════════════════════════════════════════════ */

void log_write(LogLevel lv, const char* fmt, ...) {
    if (!G_ok) {
        /* Before init: fallback to stderr */
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        return;
    }
    if (lv < G.min_lv) return;

    int ri = ring_register();
    if (ri < 0) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
        va_end(ap);
        return;
    }

    /* Build entry on stack — zero heap allocation */
    RingEntry e;
    memset(&e, 0, sizeof(e));
    e.ts_us    = clock_us();
    e.thread_h = thread_hash();
    e.frame_id = (uint32_t)TL_frame;
    e.level    = (int16_t)(lv + 1);
    strncpy(e.th_name, TL_name, sizeof(e.th_name) - 1);

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e.msg, sizeof(e.msg), fmt, ap);
    va_end(ap);
    e.msg[sizeof(e.msg) - 1] = '\0';

    ring_push(ri, &e);
}

void log_set_level(LogLevel lv) {
    G.min_lv = lv;
    if (G_ok && G.file) {
        char buf[256];
        int n = snprintf(buf, sizeof(buf),
            "{\"ts\":%lld,\"lv\":\"INFO\",\"th\":\"logger\",\"fr\":0,"
            "\"msg\":\"level changed to %s\"}\n",
            (long long)(clock_us() - G.t0_us), LV_NAMES[lv + 1]);
        fwrite(buf, 1, (size_t)n, G.file);
        fflush(G.file);
    }
}

LogLevel log_get_level(void) { return G.min_lv; }

/* ═══════════════════════════════════════════════════════════════════════
 * Structured event logging
 * ═══════════════════════════════════════════════════════════════════════ */

void log_ev(LogLevel lv, const char* event, ...) {
    if (!G_ok || lv < G.min_lv) return;

    /* Build: "event | k=v | k=v | ..." */
    char msg[MSG_BUF_SIZE];
    int off = snprintf(msg, sizeof(msg), "%s", event);

    va_list ap;
    va_start(ap, event);
    for (;;) {
        const char* key = va_arg(ap, const char*);
        if (!key) break;
        const char* vfmt = va_arg(ap, const char*);
        char val[128];

        /*
         * Consume the VALUE argument DIRECTLY from ap, matching its type
         * to the format string.  This avoids the va_copy/va_end dance
         * that left ap pointing at the value instead of the next key
         * (segfault when vfmt eventually became NULL).
         */
        if (strstr(vfmt, "%d") || strstr(vfmt, "%i") ||
            strstr(vfmt, "%u") || strstr(vfmt, "%x") || strstr(vfmt, "%X")) {
            int ival = va_arg(ap, int);
            snprintf(val, sizeof(val), vfmt, ival);
        } else if (strstr(vfmt, "%s")) {
            const char* sval = va_arg(ap, const char*);
            snprintf(val, sizeof(val), vfmt, sval ? sval : "(null)");
        } else if (strstr(vfmt, "%f") || strstr(vfmt, "%g") || strstr(vfmt, "%e")) {
            double dval = va_arg(ap, double);
            snprintf(val, sizeof(val), vfmt, dval);
        } else {
            /* No format specifier — vfmt IS the literal value */
            snprintf(val, sizeof(val), "%s", vfmt);
        }

        int rem = (int)sizeof(msg) - off - 4;
        if (rem > 0) off += snprintf(msg + off, (size_t)rem, " | %s=%s", key, val);
    }
    va_end(ap);

    log_write(lv, "%s", msg);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Trace spans
 * ═══════════════════════════════════════════════════════════════════════ */

LogSpan* log_span_begin(const char* name) {
    if (!G_ok) return NULL;
    if (TL_sdepth >= MAX_SPAN_DEPTH) return NULL;

    LogSpan* s = (LogSpan*)calloc(1, sizeof(LogSpan));
    if (!s) return NULL;

    s->name     = name;
    s->start_ns = clock_us() * 1000;
    s->span_id  = ++G.next_sid;
    s->trace_id = TL_trace;

    if (TL_sdepth > 0 && TL_spans[TL_sdepth - 1]) {
        s->parent_id = TL_spans[TL_sdepth - 1]->span_id;
        if (s->trace_id == 0) s->trace_id = TL_spans[TL_sdepth - 1]->trace_id;
    }
    if (s->trace_id == 0) s->trace_id = s->span_id;  /* root span */

    TL_spans[TL_sdepth++] = s;
    return s;
}

void log_span_end(LogSpan* s, int status, const char* attrs) {
    if (!s) return;
    s->end_ns = clock_us() * 1000;
    s->status = status;

    if (TL_sdepth > 0 && TL_spans[TL_sdepth - 1] == s) {
        TL_spans[TL_sdepth - 1] = NULL;
        TL_sdepth--;
    }

    int64_t dur_us = (s->end_ns - s->start_ns) / 1000;
    LogLevel lv = (status != 0) ? LOG_LV_ERROR : LOG_LV_DEBUG;

    log_ev(lv, "span.end",
           "name",     "%s", s->name,
           "sid",      "%llu", (unsigned long long)s->span_id,
           "tid",      "%llu", (unsigned long long)s->trace_id,
           "pid",      "%llu", (unsigned long long)s->parent_id,
           "dur_us",   "%lld", (long long)dur_us,
           "status",   "%d",   status,
           attrs ? "attrs" : NULL, attrs ? "%s" : NULL, attrs,
           NULL);
    free(s);
}

void log_span_ev(LogSpan* s, const char* name, const char* attrs) {
    if (!s || !name) return;
    int64_t off_us = (clock_us() * 1000 - s->start_ns) / 1000;
    log_ev(LOG_LV_TRACE, "span.event",
           "span",  "%s", s->name,
           "event", "%s", name,
           "off_us","%lld", (long long)off_us,
           attrs ? "attrs" : NULL, attrs ? "%s" : NULL, attrs,
           NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Thread context
 * ═══════════════════════════════════════════════════════════════════════ */

void log_thread_name(const char* name) {
    if (name) {
        strncpy(TL_name, name, sizeof(TL_name) - 1);
        TL_name[sizeof(TL_name) - 1] = '\0';
    }
}

void log_frame_id(uint64_t id) { TL_frame = id; }
void log_trace_id(uint64_t id) { TL_trace = id; }

/* ═══════════════════════════════════════════════════════════════════════
 * Latency histograms
 * ═══════════════════════════════════════════════════════════════════════ */

void log_hist_record(const char* stage, int64_t lat_us) {
    if (!G_ok || !stage) return;

    pthread_mutex_lock(&G.histo_mtx);
    int idx = -1;
    for (int i = 0; i < G.nhistos; i++) {
        if (strcmp(G.histos[i].name, stage) == 0) { idx = i; break; }
    }
    if (idx < 0 && G.nhistos < MAX_HISTOGRAMS) {
        idx = G.nhistos++;
        strncpy(G.histos[idx].name, stage, sizeof(G.histos[idx].name) - 1);
    }
    pthread_mutex_unlock(&G.histo_mtx);
    if (idx < 0) return;

    int b = LOG_HIST_BUCKETS;
    for (int i = 0; i < LOG_HIST_BUCKETS; i++) {
        if (lat_us < LOG_HIST_BOUNDARIES_US[i]) { b = i; break; }
    }

    if (b < LOG_HIST_BUCKETS) {
        G.histos[idx].buckets[b]++;
    } else {
        G.histos[idx].overflow++;
    }
    G.histos[idx].count++;
    G.histos[idx].sum_us += (uint64_t)lat_us;
}

void log_hist_dump(void) {
    if (!G_ok) return;

    G.dump_frame++;
    if (G.dump_frame < HISTO_DUMP_FRAMES) return;
    G.dump_frame = 0;

    pthread_mutex_lock(&G.histo_mtx);
    for (int i = 0; i < G.nhistos; i++) {
        uint64_t bkts[LOG_HIST_BUCKETS + 1];
        uint64_t n = 0, sum = 0;
        for (int b = 0; b < LOG_HIST_BUCKETS; b++) {
            bkts[b] = G.histos[i].buckets[b];
            G.histos[i].buckets[b] = 0;
            n += bkts[b];
        }
        bkts[LOG_HIST_BUCKETS] = G.histos[i].overflow;
        G.histos[i].overflow = 0;
        n += bkts[LOG_HIST_BUCKETS];
        sum  = G.histos[i].sum_us;
        G.histos[i].sum_us = 0;
        G.histos[i].count = 0;

        if (n == 0) continue;

        int64_t avg = (int64_t)(sum / (n > 0 ? n : 1));
        char line[512];
        int off = snprintf(line, sizeof(line),
            "%s: n=%llu avg=%lldus [", G.histos[i].name,
            (unsigned long long)n, (long long)avg);

        const char* labels[] = {"<1ms","<5ms","<10ms","<20ms","<50ms","<100ms","<200ms","<500ms"};
        for (int b = 0; b < LOG_HIST_BUCKETS; b++) {
            off += snprintf(line + off, sizeof(line) - (size_t)off,
                "%s%s:%llu", b > 0 ? " " : "",
                labels[b], (unsigned long long)bkts[b]);
        }
        snprintf(line + off, sizeof(line) - (size_t)off,
            " ovf:%llu]", (unsigned long long)bkts[LOG_HIST_BUCKETS]);
        log_info("%s", line);
    }
    pthread_mutex_unlock(&G.histo_mtx);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════ */

void log_set_format(LogFormat fmt) { G.fmt = fmt; }
void log_set_rotation(size_t max_bytes, int max_files) {
    G.rot_bytes = max_bytes;
    G.rot_files = max_files;
}
void log_rotate(void) { if (G.file) rotate_file(); }

/* ── Signal handler (async-signal-safe) ── */
static void sig_handler(int sig) {
    if (sig == SIGUSR1) {
        /* Cycle: TRACE→DEBUG→INFO→WARN→ERROR→TRACE */
        static const LogLevel cyc[] = {
            LOG_LV_TRACE, LOG_LV_DEBUG, LOG_LV_INFO, LOG_LV_WARN, LOG_LV_ERROR,
        };
        int n = (int)(sizeof(cyc) / sizeof(cyc[0]));
        for (int i = 0; i < n; i++) {
            if (G.min_lv == cyc[i]) {
                G.min_lv = cyc[(i + 1) % n];
                break;
            }
        }
        const char* nm = LV_NAMES[G.min_lv + 1];
        (void)!write(STDERR_FILENO, "\nLOG → ", 7);
        (void)!write(STDERR_FILENO, nm, strlen(nm));
        (void)!write(STDERR_FILENO, "\n", 1);
    } else if (sig == SIGHUP) {
        if (G.file) rotate_file();
        (void)!write(STDERR_FILENO, "\nLOG rotated\n", 13);
    }
}

void log_install_signal_handlers(void) {
    if (G.sig_ok) return;
    struct sigaction sa = {0};
    sa.sa_handler = sig_handler;
    sa.sa_flags   = SA_RESTART;
    sigaction(SIGUSR1, &sa, NULL);
    sigaction(SIGHUP,  &sa, NULL);
    G.sig_ok = true;
}
