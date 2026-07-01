/*
 * terminal_ui.c — Semantic Terminal UI Engine (v2 — Theme System)
 *
 * Architecture inspired by Claude Code CIL, ratatui, Textual:
 *   Theme Slots → Semantic Primitives → Layout → Dynamic Elements
 *
 * Three-tier color degradation:
 *   Tier 3 — TrueColor (COLORTERM=truecolor)       → full RGB
 *   Tier 2 — 256-color (TERM contains "256color")   → xterm palette
 *   Tier 1 — 16 ANSI  (foundation — usable everywhere)
 *   Tier 0 — Monochrome (NO_COLOR, CI, pipe)
 *
 * Spacing rhythm: 1-2-3-5 (xs-sm-md-lg)
 *
 * Output routing: stderr = UI, stdout = data/logs
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "terminal_ui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Spinner Braille Frames (10-frame smooth animation)
 * ═══════════════════════════════════════════════════════════════════════ */

static const char* SPINNER_FRAMES[TUI_SPINNER_FRAMES] = {
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
};

/* ═══════════════════════════════════════════════════════════════════════
 * Symbol Table (single-cell Unicode → ASCII fallback)
 * ═══════════════════════════════════════════════════════════════════════ */

static const char* SYMBOLS_UNICODE[TUI_SYM_COUNT] = {
    "✔", "✖", "◉", "○", "→", "·", "…", "·"
};
static const char* SYMBOLS_ASCII[TUI_SYM_COUNT] = {
    "OK", "FAIL", "(*)", "( )", "->", ".", "...", "."
};

/* ═══════════════════════════════════════════════════════════════════════
 * Box-Drawing Character Tables (by style)
 *      tl — h — tr
 *      v         v
 *      bl — h — br
 * ═══════════════════════════════════════════════════════════════════════ */

static const char* BOX_CHARS[5][6] = {
    /* SINGLE */  {"┌","┐","└","┘","─","│"},
    /* DOUBLE */  {"╔","╗","╚","╝","═","║"},
    /* ROUNDED */ {"╭","╮","╰","╯","─","│"},
    /* HEAVY */   {"┏","┓","┗","┛","━","┃"},
    /* ASCII */   {"+","+","+","+","-","|"},
};

/* ═══════════════════════════════════════════════════════════════════════
 * ANSI 3/4-bit Color Palette (Tier 1 — Foundation)
 *
 * We use the 16-color palette with semantic mapping.  These are the
 * "relative" colors — the terminal theme determines the actual rendered
 * color.  Bold/bright variants provide 8 additional foregrounds.
 *
 *   FG: 30-37 (normal), 90-97 (bright)
 *   BG: 40-47 (normal), 100-107 (bright)
 *
 * Semantic mapping:
 *   fg_default    → 37 (white) / bright white
 *   fg_muted      → 90 (bright black = gray)
 *   fg_emphasis   → 1;37 (bold white)
 *   fg_accent     → 36 (cyan) / 35 (magenta)
 *   bg_base       → (none — terminal default)
 *   bg_surface    → 40 (black) or 100
 *   status_ok     → 32 (green)
 *   status_warn   → 33 (yellow)
 *   status_error  → 31 (red)
 *   status_info   → 36 (cyan)
 *   border        → 90 (gray)
 *   bar_bg        → 100 (bright black bg)
 *   bar_fg        → 42 (green bg)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Tier 1: 16-color ANSI slot→escape mapping ── */
static const char* COLOR_16_FG[TUI_CLR_COUNT] = {
    "\033[37m",     /* FG_DEFAULT    — white */
    "\033[90m",     /* FG_MUTED      — bright black (gray) */
    "\033[1;37m",   /* FG_EMPHASIS   — bold white */
    "\033[36m",     /* FG_ACCENT     — cyan */
    "",              /* BG_BASE       — terminal default */
    "\033[100m",    /* BG_SURFACE    — bright black bg */
    "\033[47m",     /* BG_OVERLAY    — white bg */
    "\033[32m",     /* STATUS_OK     — green */
    "\033[33m",     /* STATUS_WARN   — yellow */
    "\033[31m",     /* STATUS_ERROR  — red */
    "\033[36m",     /* STATUS_INFO   — cyan */
    "\033[100m",    /* BAR_BG        — bright black bg */
    "\033[42m",     /* BAR_FG        — green bg */
    "\033[90m",     /* BORDER        — gray */
};

__attribute__((unused))
static const char* COLOR_16_BG[TUI_CLR_COUNT] = {
    "",              /* FG_DEFAULT */
    "",              /* FG_MUTED */
    "",              /* FG_EMPHASIS */
    "",              /* FG_ACCENT */
    "",              /* BG_BASE */
    "\033[100m",    /* BG_SURFACE */
    "\033[47m",     /* BG_OVERLAY */
    "",              /* STATUS_OK */
    "",              /* STATUS_WARN */
    "",              /* STATUS_ERROR */
    "",              /* STATUS_INFO */
    "\033[100m",    /* BAR_BG */
    "\033[42m",     /* BAR_FG */
    "",              /* BORDER */
};

/* ── Tier 3: TrueColor (24-bit) slot→escape mapping ──
 * Dark-theme palette optimized for readability on modern terminals.
 * Light-theme users can swap these values via tui_force_color_tier(). */
static const char* TC_FG[TUI_CLR_COUNT] = {
    "\033[38;2;220;223;238m",  /* FG_DEFAULT    — #dcdfee (off-white) */
    "\033[38;2;86;95;137m",    /* FG_MUTED      — #565f89 (gray-blue) */
    "\033[38;2;255;255;255m",  /* FG_EMPHASIS   — pure white */
    "\033[38;2;122;162;247m",  /* FG_ACCENT     — #7aa2f7 (soft blue) */
    "",                         /* BG_BASE       — terminal default */
    "\033[48;2;36;40;59m",     /* BG_SURFACE    — #24283b */
    "\033[48;2;65;72;104m",    /* BG_OVERLAY    — #414868 */
    "\033[38;2;158;206;106m",  /* STATUS_OK     — #9ece6a (green) */
    "\033[38;2;224;175;104m",  /* STATUS_WARN   — #e0af68 (amber) */
    "\033[38;2;247;118;142m",  /* STATUS_ERROR  — #f7768e (red) */
    "\033[38;2;125;207;255m",  /* STATUS_INFO   — #7dcfff (cyan) */
    "\033[48;2;36;40;59m",     /* BAR_BG        — #24283b */
    "\033[48;2;122;162;247m",  /* BAR_FG        — #7aa2f7 */
    "\033[38;2;86;95;137m",    /* BORDER        — #565f89 */
};

__attribute__((unused))
static const char* TC_BG[TUI_CLR_COUNT] = {
    "", "", "", "", "",
    "\033[48;2;36;40;59m",
    "\033[48;2;65;72;104m",
    "", "", "", "",
    "\033[48;2;36;40;59m",
    "\033[48;2;122;162;247m",
    "",
};

#define TUI_ANSI_RESET  "\033[0m"
#define TUI_ANSI_BOLD   "\033[1m"
#define TUI_ANSI_DIM    "\033[2m"

/* ═══════════════════════════════════════════════════════════════════════
 * Global State
 * ═══════════════════════════════════════════════════════════════════════ */

static TUIMode      G_mode        = TUI_AUTO;
static TUIColorTier G_color_tier  = TUI_COLOR_16;
static bool         G_initialized = false;
static bool         G_is_tty      = false;
static bool         G_json_mode   = false;
static int          G_term_width  = 80;
__attribute__((unused))
static TUIBoxStyle  G_box_style   = TUI_BOX_SINGLE;
static pthread_mutex_t G_mutex    = PTHREAD_MUTEX_INITIALIZER;

/* ── Status bar segment accumulator ── */
typedef struct {
    char label[32];
    char value[32];
    TUIColorSlot color;   /* TUI_CLR_COUNT = use default (muted label, bright value) */
} SBarSegment;
static SBarSegment G_sbar_segs[TUI_SBAR_MAX_SEGMENTS];
static int G_sbar_count = 0;

/* ═══════════════════════════════════════════════════════════════════════
 * Environment Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static bool env_flag(const char* name) {
    const char* v = getenv(name);
    return v && (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 ||
                 strcmp(v, "yes") == 0 || strcmp(v, "TRUE") == 0);
}

static bool env_nonempty(const char* name) {
    const char* v = getenv(name);
    return v && v[0] != '\0';
}

static int detect_term_width(void) {
    const char* cols = getenv("COLUMNS");
    if (cols) { int w = atoi(cols); if (w >= 40) return w; }
    return 80;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Color Tier Detection
 * ═══════════════════════════════════════════════════════════════════════ */

static TUIColorTier detect_color_tier(void) {
    if (env_nonempty("COLORTERM")) {
        const char* ct = getenv("COLORTERM");
        if (strstr(ct, "truecolor") || strstr(ct, "24bit"))
            return TUI_COLOR_TRUECOLOR;
    }
    if (env_nonempty("TERM")) {
        const char* t = getenv("TERM");
        if (strstr(t, "256color") || strstr(t, "256"))
            return TUI_COLOR_256;
        if (strstr(t, "color") || strstr(t, "xterm") || strstr(t, "screen"))
            return TUI_COLOR_16;
    }
    return TUI_COLOR_MONO;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_init(TUIMode mode) {
    if (G_initialized) return;

    if (mode == TUI_AUTO) {
        if (env_flag("NO_COLOR")) {
            G_mode = TUI_PLAIN;
        } else if (env_flag("CI") || env_nonempty("GITHUB_ACTIONS") ||
                   env_nonempty("JENKINS_HOME") || env_nonempty("TRAVIS")) {
            G_mode = TUI_PLAIN;
        } else if (!isatty(STDERR_FILENO)) {
            G_mode = TUI_PLAIN;
        } else {
            G_mode = TUI_HUMAN;
        }
    } else {
        G_mode = mode;
    }

    G_is_tty    = (isatty(STDERR_FILENO) == 1);
    G_json_mode = (G_mode == TUI_MACHINE);

    if (G_mode == TUI_HUMAN && G_is_tty) {
        G_color_tier = detect_color_tier();
    } else {
        G_color_tier = TUI_COLOR_MONO;
    }

    G_term_width  = detect_term_width();
    G_initialized = true;
}

void tui_shutdown(void) {
    if (!G_initialized) return;
    if (G_is_tty && G_mode == TUI_HUMAN) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    G_initialized = false;
}

TUIMode tui_get_mode(void)           { return G_mode; }
TUIColorTier tui_get_color_tier(void) { return G_color_tier; }
bool tui_is_interactive(void)         { return G_mode == TUI_HUMAN && G_is_tty; }

void tui_force_color_tier(TUIColorTier tier) {
    G_color_tier = tier;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Theme Accessors — resolve semantic slot → ANSI escape
 * ═══════════════════════════════════════════════════════════════════════ */

const char* tui_theme_color(TUIColorSlot slot) {
    if (G_color_tier == TUI_COLOR_MONO || G_mode == TUI_PLAIN ||
        G_mode == TUI_MACHINE)
        return "";

    if (G_color_tier == TUI_COLOR_TRUECOLOR && TC_FG[slot][0] != '\0')
        return TC_FG[slot];

    if (G_color_tier >= TUI_COLOR_16 && COLOR_16_FG[slot][0] != '\0')
        return COLOR_16_FG[slot];

    return "";
}

const char* tui_theme_reset(void) {
    if (G_color_tier == TUI_COLOR_MONO || G_mode == TUI_PLAIN ||
        G_mode == TUI_MACHINE)
        return "";
    return TUI_ANSI_RESET;
}

const char* tui_theme_bold(void) {
    if (G_color_tier == TUI_COLOR_MONO || G_mode == TUI_PLAIN ||
        G_mode == TUI_MACHINE)
        return "";
    return TUI_ANSI_BOLD;
}

const char* tui_theme_dim(void) {
    if (G_color_tier == TUI_COLOR_MONO || G_mode == TUI_PLAIN ||
        G_mode == TUI_MACHINE)
        return "";
    return TUI_ANSI_DIM;
}

const char* tui_theme_symbol(TUISymbol sym) {
    if (G_mode == TUI_PLAIN || G_mode == TUI_MACHINE || !G_is_tty)
        return SYMBOLS_ASCII[sym];
    return SYMBOLS_UNICODE[sym];
}

/* ── Box drawing accessors ── */
static int g_box = TUI_BOX_SINGLE;
#define BOX_C(i) BOX_CHARS[g_box][i]
const char* tui_theme_box_tl(void) { return BOX_C(0); }
const char* tui_theme_box_tr(void) { return BOX_C(1); }
const char* tui_theme_box_bl(void) { return BOX_C(2); }
const char* tui_theme_box_br(void) { return BOX_C(3); }
const char* tui_theme_box_h(void)  { return BOX_C(4); }
const char* tui_theme_box_v(void)  { return BOX_C(5); }

/* ═══════════════════════════════════════════════════════════════════════
 * Core Output Helper
 * ═══════════════════════════════════════════════════════════════════════ */

static void fmt_out(const char* badge, TUIColorSlot badge_color,
                    const char* fmt, va_list ap) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char msg[512];
        vsnprintf(msg, sizeof(msg), fmt, ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"%s\",\"msg\":\"", badge);
        /* BUGFIX: proper JSON escaping — handle all control chars, not just " and \ */
        for (const char* p = msg; *p && p < msg + sizeof(msg) - 1; p++) {
            unsigned char c = (unsigned char)*p;
            switch (c) {
            case '"':  fputs("\\\"", stderr); break;
            case '\\': fputs("\\\\", stderr); break;
            case '\b': fputs("\\b", stderr);  break;
            case '\f': fputs("\\f", stderr);  break;
            case '\n': fputs("\\n", stderr);  break;
            case '\r': fputs("\\r", stderr);  break;
            case '\t': fputs("\\t", stderr);  break;
            default:
                if (c < 0x20) fprintf(stderr, "\\u%04x", (unsigned)c);
                else fputc(c, stderr);
                break;
            }
        }
        fprintf(stderr, "\"}\n");
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        const char* clr = tui_theme_color(badge_color);
        const char* rst = tui_theme_reset();
        fprintf(stderr, "  %s%-4s%s  ", clr, badge, rst);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "  %-4s  ", badge);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Semantic Output Primitives
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_ok(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fmt_out(tui_theme_symbol(TUI_SYM_CHECKBOX_OK), TUI_CLR_STATUS_SUCCESS, fmt, ap);
    va_end(ap);
}

void tui_warn(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fmt_out("WARN", TUI_CLR_STATUS_WARNING, fmt, ap);
    va_end(ap);
}

void tui_fail(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fmt_out(tui_theme_symbol(TUI_SYM_CHECKBOX_FAIL), TUI_CLR_STATUS_ERROR, fmt, ap);
    va_end(ap);
}

void tui_info(const char* fmt, ...) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char msg[512];
        va_list ap; va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"INFO\",\"msg\":\"");
        for (const char* p = msg; *p; p++) {
            if (*p == '"') fputc('\\', stderr);
            fputc(*p, stderr);
        }
        fprintf(stderr, "\"}\n");
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "%s", tui_theme_bold());
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "%s\n", tui_theme_reset());
    } else {
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_muted(const char* fmt, ...) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char msg[512];
        va_list ap; va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"DIAG\",\"msg\":\"%s\"}\n", msg);
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "  %s", tui_theme_color(TUI_CLR_FG_MUTED));
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "%s\n", tui_theme_reset());
    } else {
        fprintf(stderr, "  ");
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_metric(const char* label, const char* valfmt, ...) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char val[128];
        va_list ap; va_start(ap, valfmt);
        vsnprintf(val, sizeof(val), valfmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"METRIC\","
                "\"label\":\"%s\",\"value\":\"%s\"}\n", label, val);
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "  %s%-20s%s %s",
                tui_theme_color(TUI_CLR_FG_MUTED), label, tui_theme_reset(),
                tui_theme_bold());
        va_list ap; va_start(ap, valfmt);
        vfprintf(stderr, valfmt, ap);
        va_end(ap);
        fprintf(stderr, "%s\n", tui_theme_reset());
    } else {
        fprintf(stderr, "  %-20s ", label);
        va_list ap; va_start(ap, valfmt);
        vfprintf(stderr, valfmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_keyval(const char* key, const char* valfmt, ...) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char val[256];
        va_list ap; va_start(ap, valfmt);
        vsnprintf(val, sizeof(val), valfmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"KV\",\"key\":\"%s\",\"val\":\"%s\"}\n", key, val);
    } else {
        int key_len = (int)strlen(key);
        int pad = (key_len < 24) ? (24 - key_len) : 1;
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "  %s%s%s%*s",
                    tui_theme_color(TUI_CLR_FG_MUTED), key, tui_theme_reset(),
                    pad, "");
        } else {
            fprintf(stderr, "  %s%*s", key, pad, "");
        }
        va_list ap; va_start(ap, valfmt);
        vfprintf(stderr, valfmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Layout Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_banner(const char* title, const char* subtitle) {
    int w = G_term_width;
    if (w > 100) w = 72;  /* cap for readability */

    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"BANNER\","
                "\"title\":\"%s\",\"subtitle\":\"%s\"}\n", title, subtitle);
        pthread_mutex_unlock(&G_mutex);
        return;
    }

    fprintf(stderr, "\n");

    const char* clr_accent = tui_theme_color(TUI_CLR_FG_ACCENT);
    const char* clr_border = tui_theme_color(TUI_CLR_BORDER);
    const char* clr_bold   = tui_theme_bold();
    const char* rst        = tui_theme_reset();

    /* Top border */
    fprintf(stderr, "  %s%s", clr_border, tui_theme_box_tl());
    for (int i = 0; i < w - 6; i++) fputs(tui_theme_box_h(), stderr);
    fprintf(stderr, "%s%s\n", tui_theme_box_tr(), rst);

    /* Title line (centered) */
    int tlen = (int)strlen(title);
    int tpad = (w - 6 - tlen) / 2;
    if (tpad < 1) tpad = 1;
    fprintf(stderr, "  %s%s", clr_border, tui_theme_box_v());
    fprintf(stderr, "%s%*s%s%s%s%s%*s",
            rst, tpad, "",                           /* reset + left-pad */
            clr_accent, clr_bold, title,              /* accent+bold + title */
            rst,                                      /* reset styles */
            w - 6 - tpad - tlen, "");                 /* right-pad */
    fprintf(stderr, "%s%s%s\n", clr_border, tui_theme_box_v(), rst);

    /* Subtitle line (centered) */
    int slen = (int)strlen(subtitle);
    int spad = (w - 6 - slen) / 2;
    if (spad < 1) spad = 1;
    fprintf(stderr, "  %s%s", clr_border, tui_theme_box_v());
    fprintf(stderr, "%s%*s%s%s%*s",
            rst, spad, "",                           /* reset + left-pad */
            tui_theme_color(TUI_CLR_FG_MUTED), subtitle,  /* muted subtitle */
            w - 6 - spad - slen, "");                /* right-pad */
    fprintf(stderr, "%s%s%s\n", clr_border, tui_theme_box_v(), rst);

    /* Bottom border */
    fprintf(stderr, "  %s%s", clr_border, tui_theme_box_bl());
    for (int i = 0; i < w - 6; i++) fputs(tui_theme_box_h(), stderr);
    fprintf(stderr, "%s%s\n", tui_theme_box_br(), rst);

    fprintf(stderr, "\n");
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_section(const char* name) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"SECTION\",\"name\":\"%s\"}\n", name);
    } else {
        int w = (G_term_width < 72) ? G_term_width : 72;
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "\n  %s%s%s%s\n",
                    tui_theme_bold(), tui_theme_color(TUI_CLR_FG_EMPHASIS),
                    name, tui_theme_reset());
            fprintf(stderr, "%s", tui_theme_color(TUI_CLR_BORDER));
        } else {
            fprintf(stderr, "\n  %s\n", name);
        }
        const char* hline = tui_theme_box_h();
        for (int i = 0; i < w - 2; i++) fputs(hline, stderr);
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_reset());
        }
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_separator(void) {
    int w = (G_term_width < 72) ? G_term_width : 72;
    pthread_mutex_lock(&G_mutex);
    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"SEP\"}\n");
    } else {
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_color(TUI_CLR_BORDER));
        }
        fprintf(stderr, "  ");
        const char* hline = tui_theme_box_h();
        for (int i = 0; i < w - 4; i++) fputs(hline, stderr);
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_reset());
        }
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_separator_thick(void) {
    int w = (G_term_width < 72) ? G_term_width : 72;
    pthread_mutex_lock(&G_mutex);
    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"SEP_THICK\"}\n");
    } else {
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_color(TUI_CLR_FG_ACCENT));
        }
        fprintf(stderr, "  ");
        /* Thick separator uses HEAVY horizontal, falling back to ASCII '-' */
        const char* hheavy = (G_mode == TUI_HUMAN && G_is_tty) ? "━" : "-";
        for (int i = 0; i < w - 4; i++) fputs(hheavy, stderr);
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_reset());
        }
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_blank(void) {
    pthread_mutex_lock(&G_mutex);
    if (!G_json_mode) fprintf(stderr, "\n");
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_box_header(const char* title, int content_width) {
    (void)content_width;
    pthread_mutex_lock(&G_mutex);
    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"BOX_OPEN\",\"title\":\"%s\"}\n", title);
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "  %s%s %s %s%s\n",
                tui_theme_color(TUI_CLR_BORDER), tui_theme_box_tl(),
                title, tui_theme_box_h(), tui_theme_box_tr());
    } else {
        fprintf(stderr, "  +-- %s ----+\n", title);
    }
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_box_footer(void) {
    pthread_mutex_lock(&G_mutex);
    if (!G_json_mode) {
        fprintf(stderr, "  %s%s%s%s%s\n",
                G_is_tty ? tui_theme_color(TUI_CLR_BORDER) : "",
                G_is_tty ? tui_theme_box_bl() : "+",
                G_is_tty ? tui_theme_box_h() : "-",
                G_is_tty ? tui_theme_box_h() : "-",
                G_is_tty ? tui_theme_box_br() : "+");
    }
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Spinner (checklist-style with action verbs)
 * ═══════════════════════════════════════════════════════════════════════ */

#define SPINNER_MAX_ITEMS 32

struct TuiSpinner {
    char    main_msg[256];
    char    verb_msg[256];         /* current action verb */
    char    items[SPINNER_MAX_ITEMS][256];
    bool    item_ok[SPINNER_MAX_ITEMS];
    int     num_items;
    bool    done;
    bool    aborted;
    volatile int  frame;
    volatile bool animating;
    pthread_t     anim_thread;
    pthread_mutex_t lock;
};

static void spinner_render_line(TuiSpinner* s) {
    if (G_json_mode) return;
    int frame_idx = s->frame % TUI_SPINNER_FRAMES;
    const char* verb = s->verb_msg[0] ? s->verb_msg : s->main_msg;

    if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "\r\033[K  %s%s%s %s%s%s  %s",
                tui_theme_color(TUI_CLR_STATUS_INFO),
                SPINNER_FRAMES[frame_idx],
                tui_theme_reset(),
                tui_theme_bold(), verb, tui_theme_reset(),
                tui_theme_color(TUI_CLR_FG_MUTED));
    } else {
        fprintf(stderr, "\r  ... %s", verb);
    }
    fflush(stderr);
}

static void spinner_render_all(TuiSpinner* s) {
    if (G_json_mode) {
        for (int i = 0; i < s->num_items; i++) {
            fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"%s\",\"msg\":\"%s\"}\n",
                    s->item_ok[i] ? "OK" : "FAIL", s->items[i]);
        }
        return;
    }

    spinner_render_line(s);
    fprintf(stderr, "\n");

    for (int i = 0; i < s->num_items; i++) {
        const char* sym  = tui_theme_symbol(s->item_ok[i] ? TUI_SYM_CHECKBOX_OK : TUI_SYM_CHECKBOX_FAIL);
        TUIColorSlot clr = s->item_ok[i] ? TUI_CLR_STATUS_SUCCESS : TUI_CLR_STATUS_ERROR;
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "    %s%s%s %s\n",
                    tui_theme_color(clr), sym, tui_theme_reset(), s->items[i]);
        } else {
            fprintf(stderr, "    [%s] %s\n",
                    s->item_ok[i] ? "OK" : "FAIL", s->items[i]);
        }
    }
}

static void* spinner_anim_thread(void* arg) {
    TuiSpinner* s = (TuiSpinner*)arg;
    while (s->animating) {
        s->frame++;
        pthread_mutex_lock(&s->lock);
        if (!s->done && !s->aborted && s->animating) {
            spinner_render_line(s);
        }
        pthread_mutex_unlock(&s->lock);
        struct timespec ts = {0, TUI_SPINNER_INTERVAL_MS * 1000000L};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

TuiSpinner* tui_spinner_start(const char* msg) {
    TuiSpinner* s = (TuiSpinner*)calloc(1, sizeof(TuiSpinner));
    if (!s) return NULL;

    strncpy(s->main_msg, msg, sizeof(s->main_msg) - 1);
    s->num_items = 0;
    s->done = false;
    s->aborted = false;
    s->frame = 0;
    s->verb_msg[0] = '\0';
    pthread_mutex_init(&s->lock, NULL);

    if (tui_is_interactive()) {
        s->animating = true;
        if (pthread_create(&s->anim_thread, NULL, spinner_anim_thread, s) == 0) {
            struct timespec ts = {0, 20 * 1000000L};
            nanosleep(&ts, NULL);
        } else {
            s->animating = false;
        }
    } else {
        s->animating = false;
        pthread_mutex_lock(&G_mutex);
        if (!G_json_mode) {
            fprintf(stderr, "  ... %s\n", msg);
            fflush(stderr);
        }
        pthread_mutex_unlock(&G_mutex);
    }

    return s;
}

void tui_spinner_verb(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    va_list ap; va_start(ap, fmt);
    vsnprintf(s->verb_msg, sizeof(s->verb_msg), fmt, ap);
    va_end(ap);
    if (s->animating) spinner_render_line(s);
    pthread_mutex_unlock(&s->lock);
}

void tui_spinner_ok(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    if (s->num_items < SPINNER_MAX_ITEMS) {
        va_list ap; va_start(ap, fmt);
        vsnprintf(s->items[s->num_items], sizeof(s->items[0]), fmt, ap);
        va_end(ap);
        s->item_ok[s->num_items] = true;
        s->num_items++;
    }
    if (s->animating) {
        spinner_render_all(s);
    } else if (!G_json_mode) {
        fprintf(stderr, "    %s ", tui_theme_symbol(TUI_SYM_CHECKBOX_OK));
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }
    pthread_mutex_unlock(&s->lock);
}

void tui_spinner_fail(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    if (s->num_items < SPINNER_MAX_ITEMS) {
        va_list ap; va_start(ap, fmt);
        vsnprintf(s->items[s->num_items], sizeof(s->items[0]), fmt, ap);
        va_end(ap);
        s->item_ok[s->num_items] = false;
        s->num_items++;
    }
    if (s->animating) {
        spinner_render_all(s);
    } else if (!G_json_mode) {
        fprintf(stderr, "    %s ", tui_theme_symbol(TUI_SYM_CHECKBOX_FAIL));
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }
    pthread_mutex_unlock(&s->lock);
}

void tui_spinner_update(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    va_list ap; va_start(ap, fmt);
    vsnprintf(s->main_msg, sizeof(s->main_msg), fmt, ap);
    va_end(ap);
    if (s->animating) spinner_render_line(s);
    pthread_mutex_unlock(&s->lock);
}

void tui_spinner_done(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    bool had_thread = s->animating;
    s->animating = false;
    s->done = true;
    pthread_mutex_unlock(&s->lock);
    if (had_thread) pthread_join(s->anim_thread, NULL);
    pthread_mutex_lock(&s->lock);

    if (G_json_mode) {
        char msg[256];
        va_list ap; va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"DONE\",\"msg\":\"%s\"}\n", msg);
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "\r\033[K  %s%s%s ",
                tui_theme_color(TUI_CLR_STATUS_SUCCESS),
                tui_theme_symbol(TUI_SYM_CHECKBOX_OK),
                tui_theme_reset());
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "  OK  ");
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    pthread_mutex_unlock(&s->lock);
    pthread_mutex_destroy(&s->lock);
    free(s);
}

void tui_spinner_abort(TuiSpinner* s, const char* fmt, ...) {
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    bool had_thread = s->animating;
    s->animating = false;
    s->aborted = true;
    pthread_mutex_unlock(&s->lock);
    if (had_thread) pthread_join(s->anim_thread, NULL);
    pthread_mutex_lock(&s->lock);

    if (!G_json_mode) {
        const char* sym = tui_theme_symbol(TUI_SYM_CHECKBOX_FAIL);
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "\r\033[K  %s%s%s ",
                    tui_theme_color(TUI_CLR_STATUS_ERROR), sym, tui_theme_reset());
        } else {
            fprintf(stderr, "  FAIL ");
        }
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    pthread_mutex_unlock(&s->lock);
    pthread_mutex_destroy(&s->lock);
    free(s);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Progress Bar (determinate, with percentage)
 * ═══════════════════════════════════════════════════════════════════════ */

struct TuiProgress {
    char    label[128];
    int     total;
    int     current;
    bool    done;
};

TuiProgress* tui_progress_start(const char* label, int total) {
    TuiProgress* p = (TuiProgress*)calloc(1, sizeof(TuiProgress));
    if (!p) return NULL;
    strncpy(p->label, label, sizeof(p->label) - 1);
    p->total = (total > 0) ? total : 1;
    p->current = 0;
    p->done = false;

    if (!G_json_mode) {
        fprintf(stderr, "\r\033[K  %s [", p->label);
        for (int i = 0; i < TUI_PROGRESS_WIDTH; i++) fputc(' ', stderr);
        fprintf(stderr, "]   0%%");
        fflush(stderr);
    }
    return p;
}

void tui_progress_tick(TuiProgress* p, int n) {
    if (!p || p->done) return;
    p->current += n;
    if (p->current > p->total) p->current = p->total;
    if (G_json_mode) return;

    int pct  = p->current * 100 / p->total;
    int fill = p->current * TUI_PROGRESS_WIDTH / p->total;

    fprintf(stderr, "\r\033[K  %s [", p->label);
    if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "%s", tui_theme_color(TUI_CLR_DIM_BAR_FG));
        for (int i = 0; i < fill; i++) fputc(' ', stderr);
        fprintf(stderr, "%s", tui_theme_reset());
        fprintf(stderr, "%s", tui_theme_color(TUI_CLR_DIM_BAR_BG));
        for (int i = fill; i < TUI_PROGRESS_WIDTH; i++) fputc(' ', stderr);
        fprintf(stderr, "%s", tui_theme_reset());
    } else {
        for (int i = 0; i < TUI_PROGRESS_WIDTH; i++) {
            if (i < fill)      fputc('=', stderr);
            else if (i == fill) fputc('>', stderr);
            else                fputc(' ', stderr);
        }
    }
    fprintf(stderr, "] %3d%% | %d/%d", pct, p->current, p->total);
    fflush(stderr);
}

void tui_progress_done(TuiProgress* p) {
    if (!p || p->done) return;
    p->done = true;

    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"DONE\","
                "\"label\":\"%s\",\"total\":%d}\n", p->label, p->total);
    } else {
        fprintf(stderr, "\r\033[K  %s [", p->label);
        if (G_mode == TUI_HUMAN && G_is_tty) {
            fprintf(stderr, "%s", tui_theme_color(TUI_CLR_STATUS_SUCCESS));
            for (int i = 0; i < TUI_PROGRESS_WIDTH; i++) fputc(' ', stderr);
            fprintf(stderr, "%s", tui_theme_reset());
        } else {
            for (int i = 0; i < TUI_PROGRESS_WIDTH; i++) fputc('=', stderr);
        }
        fprintf(stderr, "] 100%% | %d/%d\n", p->total, p->total);
        fflush(stderr);
    }
    free(p);
}

void tui_progress_update_msg(TuiProgress* p, const char* fmt, ...) {
    if (!p) return;
    va_list ap; va_start(ap, fmt);
    vsnprintf(p->label, sizeof(p->label), fmt, ap);
    va_end(ap);
    if (!G_json_mode && !p->done) {
        int pct  = p->current * 100 / p->total;
        int fill = p->current * TUI_PROGRESS_WIDTH / p->total;
        fprintf(stderr, "\r\033[K  %s [", p->label);
        for (int i = 0; i < TUI_PROGRESS_WIDTH; i++) {
            if (i < fill - 1)      fputc('=', stderr);
            else if (i == fill - 1) fputc('>', stderr);
            else                    fputc(' ', stderr);
        }
        fprintf(stderr, "] %3d%%", pct);
        fflush(stderr);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Status Bar — Multi-Segment Bottom Bar (Claude Code style)
 *
 * Format: " LABEL value │ LABEL value │ LABEL value "
 *
 * Responsive: segments drop from right when width insufficient.
 * Rendered with \r\033[K (carriage return + clear to end of line).
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_status_bar_begin(void) {
    G_sbar_count = 0;
    memset(G_sbar_segs, 0, sizeof(G_sbar_segs));
}

void tui_status_bar_segment(const char* label, const char* valfmt, ...) {
    if (G_sbar_count >= TUI_SBAR_MAX_SEGMENTS) return;

    SBarSegment* seg = &G_sbar_segs[G_sbar_count++];
    strncpy(seg->label, label, sizeof(seg->label) - 1);
    va_list ap; va_start(ap, valfmt);
    vsnprintf(seg->value, sizeof(seg->value), valfmt, ap);
    va_end(ap);
    seg->color = TUI_CLR_COUNT;  /* default: muted label, bright value */
}

void tui_status_bar_segment_colored(TUIColorSlot color,
                                     const char* label, const char* valfmt, ...) {
    if (G_sbar_count >= TUI_SBAR_MAX_SEGMENTS) return;

    SBarSegment* seg = &G_sbar_segs[G_sbar_count++];
    strncpy(seg->label, label, sizeof(seg->label) - 1);
    va_list ap; va_start(ap, valfmt);
    vsnprintf(seg->value, sizeof(seg->value), valfmt, ap);
    va_end(ap);
    seg->color = color;
}

void tui_status_bar_end(void) {
    if (G_json_mode) return;

    pthread_mutex_lock(&G_mutex);

    /* ── Responsive layout: calculate total width, drop rightmost
     * segments until it fits within terminal width ── */
    int n = G_sbar_count;
    int widths[TUI_SBAR_MAX_SEGMENTS];
    int total_w = 0;
    for (int i = 0; i < n; i++) {
        /* " LABEL value " + separator "│" */
        widths[i] = (int)strlen(G_sbar_segs[i].label)
                  + (int)strlen(G_sbar_segs[i].value) + 3;
        total_w += widths[i];
    }
    total_w += (n - 1);  /* separators */

    int max_w = G_term_width - 2;  /* margin */
    while (n > 1 && total_w > max_w) {
        total_w -= widths[n - 1] + 1;  /* +1 for separator */
        n--;
    }

    if (n <= 0) { pthread_mutex_unlock(&G_mutex); return; }

    /* ── Render ── */
    fprintf(stderr, "\r\033[K ");  /* CR + clear line + left margin */

    const char* muted  = tui_theme_color(TUI_CLR_FG_MUTED);
    const char* bright = tui_theme_bold();
    const char* rst    = tui_theme_reset();

    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fprintf(stderr, " %s│%s ", muted, rst);
        }

        /* Label (muted) */
        fprintf(stderr, "%s%s%s ", muted, G_sbar_segs[i].label, rst);

        /* Value (bright, optionally color-coded) */
        if (G_sbar_segs[i].color != TUI_CLR_COUNT) {
            fprintf(stderr, "%s%s%s%s",
                    tui_theme_color(G_sbar_segs[i].color), bright,
                    G_sbar_segs[i].value, rst);
        } else {
            fprintf(stderr, "%s%s%s", bright, G_sbar_segs[i].value, rst);
        }
    }

    /* Pad to end of line with muted background to create a bar effect */
    fprintf(stderr, " %s", muted);
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_status_bar_clear(void) {
    if (G_json_mode) return;
    pthread_mutex_lock(&G_mutex);
    fprintf(stderr, "\r\033[K");
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

/* ── Legacy single-segment status line ── */

void tui_status_line(const char* fmt, ...) {
    if (G_json_mode) return;
    pthread_mutex_lock(&G_mutex);

    char buf[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "\r\033[K%s%s  %s%s",
                tui_theme_color(TUI_CLR_FG_MUTED), tui_theme_dim(),
                buf, tui_theme_reset());
    } else {
        fprintf(stderr, "\r  %s", buf);
    }
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

void tui_status_line_clear(void) {
    if (G_json_mode) return;
    pthread_mutex_lock(&G_mutex);
    fprintf(stderr, "\r\033[K");
    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Session Framing
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_intro(const char* session_id) {
    (void)session_id;
    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"INTRO\","
                "\"session\":\"%s\"}\n", session_id);
    } else {
        tui_info("Pipeline Running");
        tui_separator();
        fprintf(stderr, "  %sCtrl+C%s to stop\n\n",
                G_is_tty ? tui_theme_color(TUI_CLR_FG_MUTED) : "",
                G_is_tty ? tui_theme_reset() : "");
    }
}

void tui_outro(int frame_count, float avg_fps, float avg_ms,
               int total_tracks, int errors, const char* mode,
               double elapsed_s) {
    if (G_json_mode) {
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"OUTRO\","
                "\"frames\":%d,\"avg_fps\":%.1f,\"avg_ms\":%.1f,"
                "\"tracks\":%d,\"errors\":%d,\"mode\":\"%s\","
                "\"elapsed_s\":%.1f}\n",
                frame_count, (double)avg_fps, (double)avg_ms,
                total_tracks, errors, mode, elapsed_s);
        return;
    }

    tui_blank();
    tui_separator_thick();
    tui_section("Session Summary");
    tui_blank();
    tui_metric("Mode",        "%s", mode);
    tui_metric("Frames",      "%d", frame_count);
    tui_metric("Avg FPS",     "%.1f", (double)avg_fps);
    tui_metric("Avg latency", "%.1f ms", (double)avg_ms);
    tui_metric("Total tracks","%d", total_tracks);
    if (errors > 0) tui_metric("Errors", "%d", errors);
    tui_metric("Elapsed",     "%.1f s", elapsed_s);
    tui_blank();
    tui_separator();

    if (errors == 0 && frame_count > 0) {
        tui_ok("Session completed successfully");
    } else if (frame_count > 0) {
        tui_warn("Session completed with %d warning(s)", errors);
    } else {
        tui_fail("Session failed — no frames processed");
    }
    tui_blank();
}

/* ═══════════════════════════════════════════════════════════════════════
 * Diagnostic Output
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_diag(const char* component, const char* fmt, ...) {
    pthread_mutex_lock(&G_mutex);

    if (G_json_mode) {
        char msg[256];
        va_list ap; va_start(ap, fmt);
        vsnprintf(msg, sizeof(msg), fmt, ap);
        va_end(ap);
        fprintf(stderr, "{\"event\":\"tui\",\"badge\":\"DIAG\","
                "\"component\":\"%s\",\"msg\":\"%s\"}\n", component, msg);
    } else if (G_mode == TUI_HUMAN && G_is_tty) {
        fprintf(stderr, "%s  [%s]%s ",
                tui_theme_color(TUI_CLR_FG_MUTED), component, tui_theme_reset());
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "  [%s] ", component);
        va_list ap; va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        fprintf(stderr, "\n");
    }

    fflush(stderr);
    pthread_mutex_unlock(&G_mutex);
}
