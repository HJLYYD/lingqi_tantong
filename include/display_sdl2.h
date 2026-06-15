/*
 * SDL2 desktop window display backend — public API.
 *
 * Used by display_output.c as an alternative to raw framebuffer (/dev/fb0).
 * When SDL2 is available and a desktop is running, the window floats on top
 * of the compositor and is visible without switching to a tty.
 */
#ifndef DISPLAY_SDL2_H
#define DISPLAY_SDL2_H

#include <SDL2/SDL.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool sdl2_display_init(int width, int height);
void sdl2_display_frame(const uint8_t* rgb24, int width, int height);
bool sdl2_display_is_running(void);
void sdl2_display_close(void);

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_SDL2_H */
