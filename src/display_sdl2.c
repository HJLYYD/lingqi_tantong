/*
 * SDL2 desktop window display backend.
 *
 * Replaces raw /dev/fb0 framebuffer writes with a proper SDL2 window
 * that appears on top of the desktop (X11/Wayland).  Falls back to
 * fbdev at runtime if SDL2 init fails (e.g. no desktop running).
 *
 * Compiled only when HAS_SDL2 is defined (CMake detects SDL2).
 */

#include "display_sdl2.h"
#include "logger.h"
#include <SDL2/SDL.h>
#include <stdlib.h>

static SDL_Window*   sdl_window   = NULL;
static SDL_Renderer* sdl_renderer = NULL;
static SDL_Texture*  sdl_texture  = NULL;
static volatile bool sdl_running = false;

/* ── Initialise SDL2 video + create window & texture ─────────────── */

bool sdl2_display_init(int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        log_warning("[SDL2] SDL_Init failed: %s (no desktop? fallback to fbdev)",
                    SDL_GetError());
        return false;
    }

    sdl_window = SDL_CreateWindow("LingQi TanTong — Real-time Detection",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        width, height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALWAYS_ON_TOP);
    if (!sdl_window) {
        log_warning("[SDL2] CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return false;
    }

    /* K1 GPU (PowerVR BXE-2-32) Vulkan driver is incomplete —
     * SDL_RENDERER_ACCELERATED picks Vulkan and crashes on first RenderCopy.
     * Use software renderer; 640x480 RGB24 is only ~0.9 MB/frame, fine on CPU. */
    sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_SOFTWARE);
    if (!sdl_renderer) {
        /* last-resort: try accelerated anyway */
        sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
    }

    sdl_texture = SDL_CreateTexture(sdl_renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!sdl_texture) {
        log_warning("[SDL2] CreateTexture failed: %s", SDL_GetError());
        SDL_DestroyRenderer(sdl_renderer);
        SDL_DestroyWindow(sdl_window);
        SDL_Quit();
        return false;
    }

    sdl_running = true;

    log_info("[SDL2] Window created: %dx%d — drag/resize as needed", width, height);
    return true;
}

/* ── Push one RGB24 frame to the window ──────────────────────────── */

void sdl2_display_frame(const uint8_t* rgb24, int width, int height) {
    if (!sdl_running || !sdl_texture || !rgb24) return;

    /* Update texture (SDL_TEXTUREACCESS_STREAMING is efficient) */
    SDL_UpdateTexture(sdl_texture, NULL, rgb24, width * 3);

    /* Get current window size for letterbox / stretch */
    int win_w, win_h;
    SDL_GetWindowSize(sdl_window, &win_w, &win_h);

    SDL_Rect dst;
    /* Letterbox: maintain aspect ratio, centre in window */
    float scale_w = (float)win_w / (float)width;
    float scale_h = (float)win_h / (float)height;
    float scale   = scale_w < scale_h ? scale_w : scale_h;
    dst.w = (int)(width  * scale);
    dst.h = (int)(height * scale);
    dst.x = (win_w - dst.w) / 2;
    dst.y = (win_h - dst.h) / 2;

    SDL_RenderClear(sdl_renderer);
    SDL_RenderCopy(sdl_renderer, sdl_texture, NULL, &dst);
    SDL_RenderPresent(sdl_renderer);

    /* Drain SDL events so the window stays responsive (non-blocking) */
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        if (ev.type == SDL_QUIT) {
            sdl_running = false;
        }
    }
}

/* ── Query running state (window closed → stop thread) ───────────── */

bool sdl2_display_is_running(void) {
    return sdl_running;
}

/* ── Tear-down ───────────────────────────────────────────────────── */

void sdl2_display_close(void) {
    sdl_running = false;
    if (sdl_texture)  { SDL_DestroyTexture(sdl_texture);  sdl_texture  = NULL; }
    if (sdl_renderer) { SDL_DestroyRenderer(sdl_renderer); sdl_renderer = NULL; }
    if (sdl_window)   { SDL_DestroyWindow(sdl_window);     sdl_window   = NULL; }
    SDL_Quit();
    log_info("[SDL2] Window closed");
}
