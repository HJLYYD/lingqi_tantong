/*
 * terminal_ui.h — Semantic Terminal UI Primitives (v2 — Theme System)
 *
 * Architecture inspired by Claude Code's CIL, ratatui, and Textual:
 *   Theme Slots → Semantic Primitives → Layout → Dynamic Elements → Session Framing
 *
 * Three-tier color degradation:
 *   Tier 1 — TrueColor (24-bit, COLORTERM=truecolor)
 *   Tier 2 — 256-color (TERM=xterm-256color)
 *   Tier 3 — 16 ANSI  (foundation — must be usable here)
 *   Tier 0 — Monochrome (NO_COLOR=1, CI, pipe)
 *
 * All output → stderr.  stdout reserved for structured data / logs.
 *
 * Design principles (2025-2026 CLI best practices):
 *   - Semantic color — encode meaning, never hardcode hex
 *   - Single-cell Unicode — never double-width emoji
 *   - Progressive disclosure — 3 modes: HUMAN / PLAIN / MACHINE
 *   - Flicker-free — batched \r updates, no full redraws
 *   - Responsive — adapt to terminal width (80→120→200 cols)
 */

#ifndef TERMINAL_UI_H
#define TERMINAL_UI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Output Mode
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    TUI_AUTO    = 0,   /* auto-detect: TTY→HUMAN, CI→PLAIN, pipe→PLAIN */
    TUI_HUMAN   = 1,   /* color + Unicode, TTY assumed */
    TUI_PLAIN   = 2,   /* no color, ASCII-only, safe for CI/pipes */
    TUI_MACHINE = 3,   /* JSON Lines on stderr, for automation */
} TUIMode;

/* ═══════════════════════════════════════════════════════════════════════
 * Color Tier (auto-detected at tui_init)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    TUI_COLOR_MONO      = 0,  /* no color support */
    TUI_COLOR_16        = 1,  /* 4-bit ANSI (foundation) */
    TUI_COLOR_256       = 2,  /* 8-bit extended */
    TUI_COLOR_TRUECOLOR = 3,  /* 24-bit RGB */
} TUIColorTier;

/* ═══════════════════════════════════════════════════════════════════════
 * ANSI SGR Escape Codes (compile-time constants)
 *
 * These are the foundational 4-bit ANSI SGR sequences.  They are available
 * as #define string literals so they can be used in:
 *   - Static array initializers (e.g. LV_COLORS[])
 *   - Compile-time string concatenation in fprintf calls
 *
 * For dynamic/theme-aware output, use tui_theme_color(slot) instead.
 * ═══════════════════════════════════════════════════════════════════════ */

#define TUI_ANSI_RESET      "\033[0m"
#define TUI_ANSI_BOLD       "\033[1m"
#define TUI_ANSI_DIM        "\033[2m"
#define TUI_ANSI_RED        "\033[31m"
#define TUI_ANSI_GREEN      "\033[32m"
#define TUI_ANSI_YELLOW     "\033[33m"
#define TUI_ANSI_BLUE       "\033[34m"
#define TUI_ANSI_MAGENTA    "\033[35m"
#define TUI_ANSI_CYAN       "\033[36m"
#define TUI_ANSI_GRAY       "\033[90m"

/* Semantic aliases (backward-compatible names) */
#define TUI_ANSI_MUTED      TUI_ANSI_GRAY
#define TUI_ANSI_INFO       TUI_ANSI_CYAN
#define TUI_ANSI_SUCCESS    TUI_ANSI_GREEN
#define TUI_ANSI_WARNING    TUI_ANSI_YELLOW
#define TUI_ANSI_ERROR      TUI_ANSI_RED
#define TUI_ANSI_ACCENT     TUI_ANSI_MAGENTA
#define TUI_ANSI_HIGHLIGHT  TUI_ANSI_BOLD

/* ═══════════════════════════════════════════════════════════════════════
 * Semantic Color Slots
 *
 * Every visual element references a semantic SLOT, never a raw ANSI code.
 * The theme engine resolves slots → concrete escape sequences based on
 * the detected color tier.  This makes dark/light theme support trivial:
 * swap the slot→color mapping, all widgets update automatically.
 *
 * Background layering (creates depth without borders):
 *   bg_base (darkest) → bg_surface → bg_overlay (lightest)
 * Each step ~5-8% lighter in dark themes.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    TUI_CLR_FG_DEFAULT = 0,   /* body text */
    TUI_CLR_FG_MUTED,         /* secondary, metadata, timestamps */
    TUI_CLR_FG_EMPHASIS,      /* headers, labels, active items */
    TUI_CLR_FG_ACCENT,        /* brand color, interactive elements */
    TUI_CLR_BG_BASE,          /* primary background */
    TUI_CLR_BG_SURFACE,       /* panel/widget backgrounds */
    TUI_CLR_BG_OVERLAY,       /* popup/dialog backgrounds */
    TUI_CLR_STATUS_SUCCESS,   /* ✔ OK, healthy */
    TUI_CLR_STATUS_WARNING,   /* ⚠ WARN, degraded */
    TUI_CLR_STATUS_ERROR,     /* ✖ FAIL, critical */
    TUI_CLR_STATUS_INFO,      /* ℹ INFO, neutral */
    TUI_CLR_DIM_BAR_BG,       /* progress bar background */
    TUI_CLR_DIM_BAR_FG,       /* progress bar filled */
    TUI_CLR_BORDER,           /* box-drawing characters */
    TUI_CLR_COUNT
} TUIColorSlot;

/* ═══════════════════════════════════════════════════════════════════════
 * Semantic Symbols (single-cell Unicode, never double-width emoji)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    TUI_SYM_CHECKBOX_OK = 0,  /* ✔ */
    TUI_SYM_CHECKBOX_FAIL,    /* ✖ */
    TUI_SYM_RADIO_ON,         /* ◉ */
    TUI_SYM_RADIO_OFF,        /* ○ */
    TUI_SYM_ARROW_RIGHT,      /* → */
    TUI_SYM_BULLET,           /* · */
    TUI_SYM_ELLIPSIS,         /* … */
    TUI_SYM_MIDDLE_DOT,       /* · */
    TUI_SYM_COUNT
} TUISymbol;

/* ═══════════════════════════════════════════════════════════════════════
 * Box-Drawing Style
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    TUI_BOX_SINGLE = 0,       /* ┌─────┐  clean, modern (default) */
    TUI_BOX_DOUBLE,           /* ╔═════╗  bold, formal */
    TUI_BOX_ROUNDED,          /* ╭─────╮  soft, friendly */
    TUI_BOX_HEAVY,            /* ┏━━━━━┓  strong, industrial */
    TUI_BOX_ASCII,            /* +-----+  universal compatibility */
} TUIBoxStyle;

/* ═══════════════════════════════════════════════════════════════════════
 * Spacing Tokens (design system — never magic numbers in widget code)
 * ═══════════════════════════════════════════════════════════════════════ */

#define TUI_SPACE_XS  1    /* between tightly-related elements */
#define TUI_SPACE_SM  2    /* between menu items, inline elements */
#define TUI_SPACE_MD  3    /* between content sections */
#define TUI_SPACE_LG  5    /* above/below major sections */

#define TUI_SPINNER_FRAMES      10
#define TUI_SPINNER_INTERVAL_MS 80

/* ── Status bar segments ── */
#define TUI_SBAR_MAX_SEGMENTS   8
#define TUI_SBAR_SEGMENT_WIDTH  18

/* ── Progress bar ── */
#define TUI_PROGRESS_WIDTH      30

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_init(TUIMode mode);
void tui_shutdown(void);
TUIMode tui_get_mode(void);
TUIColorTier tui_get_color_tier(void);
bool tui_is_interactive(void);

/* ── Manual color tier override (for headless/testing) ── */
void tui_force_color_tier(TUIColorTier tier);

/* ═══════════════════════════════════════════════════════════════════════
 * Theme accessors — widgets call these, never hardcode escape codes
 * ═══════════════════════════════════════════════════════════════════════ */

/** Get ANSI escape for opening a semantic color slot. */
const char* tui_theme_color(TUIColorSlot slot);

/** Get ANSI reset sequence. */
const char* tui_theme_reset(void);

/** Get bold/bright ANSI sequence. */
const char* tui_theme_bold(void);

/** Get dim/muted ANSI sequence. */
const char* tui_theme_dim(void);

/** Get single-cell Unicode symbol. Falls back to ASCII in PLAIN mode. */
const char* tui_theme_symbol(TUISymbol sym);

/** Get box-drawing corner/edge characters for the current style. */
const char* tui_theme_box_tl(void);  /* top-left */
const char* tui_theme_box_tr(void);  /* top-right */
const char* tui_theme_box_bl(void);  /* bottom-left */
const char* tui_theme_box_br(void);  /* bottom-right */
const char* tui_theme_box_h(void);   /* horizontal */
const char* tui_theme_box_v(void);   /* vertical */

/* ═══════════════════════════════════════════════════════════════════════
 * Semantic Output Primitives (→ stderr, auto-adapt to mode)
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_ok(const char* fmt, ...)    __attribute__((format(printf,1,2)));
void tui_warn(const char* fmt, ...)  __attribute__((format(printf,1,2)));
void tui_fail(const char* fmt, ...)  __attribute__((format(printf,1,2)));
void tui_info(const char* fmt, ...)  __attribute__((format(printf,1,2)));
void tui_muted(const char* fmt, ...) __attribute__((format(printf,1,2)));

/** Compact metric: "  Label │ value unit" with muted label, bright value. */
void tui_metric(const char* label, const char* valfmt, ...) __attribute__((format(printf,2,3)));

/** Key-value display with aligned columns. */
void tui_keyval(const char* key, const char* valfmt, ...) __attribute__((format(printf,2,3)));

/* ═══════════════════════════════════════════════════════════════════════
 * Layout Helpers (→ stderr)
 * ═══════════════════════════════════════════════════════════════════════ */

/** Full-width banner with box-drawing border, centered title + subtitle. */
void tui_banner(const char* title, const char* subtitle);

/** Section header with accent color and separator line. */
void tui_section(const char* name);

/** Thin separator line (muted). */
void tui_separator(void);

/** Thick separator line (for major section boundaries). */
void tui_separator_thick(void);

/** Blank line for vertical rhythm. */
void tui_blank(void);

/** Draw a box around content using box-drawing characters.
 *  Call tui_box_header() to start, then print content lines,
 *  then tui_box_footer() to close. */
void tui_box_header(const char* title, int content_width);
void tui_box_footer(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Spinner (checklist-style with action verbs)
 *
 * Modern CLI convention: spinner frame + verb phrase describes what's
 * happening RIGHT NOW, not generic "Loading…"
 *
 * Usage:
 *   TuiSpinner* sp = tui_spinner_start("Loading models");
 *   tui_spinner_verb(sp, "Compiling YOLOv8n-pose graph");
 *   tui_spinner_ok(sp,   "YOLOv8n-pose  → SpacemiT EP, 320×320");
 *   tui_spinner_verb(sp, "Compiling YOLOv5n-face graph");
 *   tui_spinner_ok(sp,   "YOLOv5n-face  → CPU EP, 320×320");
 *   tui_spinner_done(sp, "4/4 models loaded (4.6s)");
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct TuiSpinner TuiSpinner;

TuiSpinner* tui_spinner_start(const char* msg);
void tui_spinner_verb(TuiSpinner* s, const char* fmt, ...) __attribute__((format(printf,2,3)));
void tui_spinner_ok(TuiSpinner* s, const char* fmt, ...)   __attribute__((format(printf,2,3)));
void tui_spinner_fail(TuiSpinner* s, const char* fmt, ...) __attribute__((format(printf,2,3)));
void tui_spinner_update(TuiSpinner* s, const char* fmt, ...) __attribute__((format(printf,2,3)));
void tui_spinner_done(TuiSpinner* s, const char* fmt, ...) __attribute__((format(printf,2,3)));
void tui_spinner_abort(TuiSpinner* s, const char* fmt, ...) __attribute__((format(printf,2,3)));

/* ═══════════════════════════════════════════════════════════════════════
 * Progress Bar (determinate, with ETA)
 *
 * Layout: "Label [====>      ] 42%  ⏱ 2m15s"
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct TuiProgress TuiProgress;

TuiProgress* tui_progress_start(const char* label, int total);
void tui_progress_tick(TuiProgress* p, int n);
void tui_progress_done(TuiProgress* p);
void tui_progress_update_msg(TuiProgress* p, const char* fmt, ...) __attribute__((format(printf,2,3)));

/* ═══════════════════════════════════════════════════════════════════════
 * Status Bar — Multi-Segment Bottom Bar (Claude Code style)
 *
 * A single-line \r-updated status bar composed of independent segments.
 * Each segment has a label (muted) and a value (bright).  Segments are
 * joined with muted separators.  Layout is responsive: segments drop
 * from the right when terminal width is insufficient.
 *
 * Format: " LABEL value │ LABEL value │ LABEL value "
 *
 * Usage:
 *   tui_status_bar_begin();
 *   tui_status_bar_segment("Frames",  "%d",  fc);
 *   tui_status_bar_segment("Tracks",  "%d",  n_tracks);
 *   tui_status_bar_segment("FPS",     "%.1f", fps);
 *   tui_status_bar_segment("Mode",    "%s",   "TRCK");
 *   tui_status_bar_segment("Uptime",  "%ds",  uptime);
 *   tui_status_bar_end();
 * ═══════════════════════════════════════════════════════════════════════ */

/** Begin building a status bar. Resets segment counter. */
void tui_status_bar_begin(void);

/** Add a segment to the status bar. Silently drops if bar is full. */
void tui_status_bar_segment(const char* label, const char* valfmt, ...) __attribute__((format(printf,2,3)));

/** Add a color-coded segment (uses semantic status color for value). */
void tui_status_bar_segment_colored(TUIColorSlot color,
                                     const char* label, const char* valfmt, ...) __attribute__((format(printf,3,4)));

/** Render the status bar and flush.  Uses \r\033[K to overwrite current line. */
void tui_status_bar_end(void);

/** Clear the status bar line (call before normal output). */
void tui_status_bar_clear(void);

/* ── Legacy single-segment status line (kept for backward compat) ── */
void tui_status_line(const char* fmt, ...) __attribute__((format(printf,1,2)));
void tui_status_line_clear(void);

/* ═══════════════════════════════════════════════════════════════════════
 * Session Framing (→ stderr)
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_intro(const char* session_id);
void tui_outro(int frame_count, float avg_fps, float avg_ms,
               int total_tracks, int errors, const char* mode,
               double elapsed_s);

/* ═══════════════════════════════════════════════════════════════════════
 * Diagnostic Output (for C++ bridge, JPU, etc.)
 * ═══════════════════════════════════════════════════════════════════════ */

void tui_diag(const char* component, const char* fmt, ...) __attribute__((format(printf,2,3)));

#ifdef __cplusplus
}
#endif

#endif /* TERMINAL_UI_H */
