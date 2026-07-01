/*
 * frame_diff.h — Fast Subsampled Frame Differencing for Adaptive Frame Skip
 *
 * Detects whether two consecutive frames are "similar enough" to skip
 * expensive YOLO inference.  Uses grid-based subsampled Mean Absolute
 * Difference (MAD) — designed for embedded deployment on K1 (RISC-V).
 *
 * Algorithm:
 *   1. Divide frame into GRID_H × GRID_W cells (default 8×8 = 64 cells)
 *   2. In each cell, sample every SUBSAMPLE-th pixel (default stride=4)
 *   3. Compute MAD per cell over RGB channels
 *   4. A cell is "changed" if MAD > cell_threshold
 *   5. Frame is "different" if changed_cell_ratio > change_threshold
 *
 * This is O(width*height / stride²) ≈ 1/16 of full-frame comparison,
 * and about 100-1000× cheaper than YOLO inference (which is ~150-400ms).
 *
 * For 640×480 with stride=4: 640*480*3/16 = 57,600 comparisons ≈ 0.2ms on K1.
 *
 * References:
 *   - Cucchiara et al., "Detecting Moving Objects, Ghosts, and Shadows
 *     in Video Streams" (2003) — grid-based motion detection
 *   - Kim et al., "Real-time Motion Detection Based on Discrete Wavelet
 *     Transform" (2012) — subsampled frame differencing
 */

#ifndef FRAME_DIFF_H
#define FRAME_DIFF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Default grid dimensions ── */
#define FD_DEFAULT_GRID_W       8
#define FD_DEFAULT_GRID_H       8
#define FD_DEFAULT_SUBSAMPLE    4       /* stride for pixel sampling within cells */
#define FD_DEFAULT_CELL_THRESH  8.0f    /* MAD per-channel threshold per cell */
#define FD_DEFAULT_CHANGE_RATIO 0.15f   /* fraction of changed cells to trigger full inference */
#define FD_MIN_CELL_SIZE        8       /* minimum pixels per cell dimension */

/* ── Scene activity levels (for adaptive skipping) ── */
typedef enum {
    FD_SCENE_STATIC    = 0,   /* < 5% cells changed — skip up to max_static_skip frames */
    FD_SCENE_LOW_MOTION = 1,  /* 5-15% cells changed — skip up to max_low_motion_skip */
    FD_SCENE_ACTIVE     = 2,  /* > 15% cells changed — process every frame */
} FrameDiffActivity;

typedef struct {
    /* ── Configuration ── */
    int     grid_w;              /* horizontal cells (default 8) */
    int     grid_h;              /* vertical cells (default 8) */
    int     subsample;           /* pixel stride within each cell (default 4) */
    float   cell_threshold;      /* MAD per channel to mark cell changed (default 8.0) */
    float   change_threshold;    /* fraction of changed cells to trigger full inference (default 0.15) */

    /* ── Adaptive skip settings ── */
    bool    adaptive_enabled;    /* enable adaptive frame skip */
    int     max_static_skip;     /* max consecutive skips in static scene (default 20) */
    int     max_low_motion_skip; /* max consecutive skips in low-motion (default 5) */
    int     force_process_every; /* force full inference every N frames (default 30) */

    /* ── State ── */
    uint8_t* reference_frame;    /* last fully-processed frame (width*height*3 bytes) */
    int     ref_width;
    int     ref_height;
    bool    has_reference;

    int     consecutive_skips;   /* frames skipped since last full inference */
    int     total_skipped;       /* cumulative skip counter */
    int     total_processed;     /* cumulative process counter */
    float   last_change_ratio;   /* change ratio from last comparison (0.0-1.0) */
    FrameDiffActivity last_activity;

    /* ── Pre-allocated work buffers (grid cell MAD values) ── */
    float*  cell_mads;           /* grid_w * grid_h floats */
    int     cell_mads_size;
} FrameDiff;

/*
 * Create a frame differencing instance.
 *
 * @param grid_w          number of horizontal grid cells
 * @param grid_h          number of vertical grid cells
 * @param subsample       pixel sampling stride (1 = every pixel, 4 = every 4th)
 * @param cell_threshold  MAD per-channel threshold for cell "changed" status
 * @param change_threshold fraction of changed cells to trigger full inference
 * @return new FrameDiff instance (call frame_diff_destroy to free)
 */
FrameDiff* frame_diff_create(int grid_w, int grid_h, int subsample,
                              float cell_threshold, float change_threshold);

/*
 * Destroy a frame differencing instance. Frees reference frame and work buffers.
 */
void frame_diff_destroy(FrameDiff* fd);

/*
 * Reset the reference frame and internal counters.
 * Call when source changes (camera switch, new video file) or after scene cuts.
 */
void frame_diff_reset(FrameDiff* fd);

/*
 * Set the current frame as the reference frame for future comparisons.
 * Call after a full inference has been completed on this frame.
 * Copies frame data into the internal reference buffer.
 *
 * @param fd     frame differencing instance
 * @param rgb    RGB24 frame data (width*height*3 bytes)
 * @param width  frame width in pixels
 * @param height frame height in pixels
 * @return 0 on success, -1 on allocation failure
 */
int frame_diff_set_reference(FrameDiff* fd, const uint8_t* rgb, int width, int height);

/*
 * Compute the frame difference between the reference frame and a new frame.
 * Returns the change ratio (0.0 = identical, 1.0 = completely different).
 * Also updates `last_change_ratio` and `last_activity` in the fd struct.
 *
 * If no reference frame exists, returns 1.0 (process frame).
 *
 * @param fd     frame differencing instance
 * @param rgb    new RGB24 frame data (width*height*3 bytes)
 * @param width  frame width in pixels
 * @param height frame height in pixels
 * @return change ratio [0.0, 1.0]
 */
float frame_diff_compute(FrameDiff* fd, const uint8_t* rgb, int width, int height);

/*
 * Decision function: should this frame be fully processed by the inference pipeline?
 *
 * Uses the change ratio + adaptive skip logic:
 *   - Always process if no reference frame exists
 *   - Always process if force_process_every is hit
 *   - Always process if change_ratio > change_threshold
 *   - Skip if adaptive_enabled and within skip limits for the activity level
 *   - Otherwise process
 *
 * @param fd frame differencing instance
 * @return true if frame should be processed, false if it can be skipped
 */
bool frame_diff_should_process(FrameDiff* fd);

/*
 * Quick inline: notify the differencer that a frame was processed (to update counters).
 * Call after full inference completes on this frame.
 */
void frame_diff_mark_processed(FrameDiff* fd);

/*
 * Quick inline: notify the differencer that a frame was skipped.
 * Call when inference is skipped for this frame.
 */
void frame_diff_mark_skipped(FrameDiff* fd);

/*
 * Get a human-readable description of the current activity level.
 */
const char* frame_diff_activity_name(FrameDiffActivity level);

/*
 * Get statistics: total processed, total skipped, skip ratio.
 */
void frame_diff_get_stats(const FrameDiff* fd, int* processed, int* skipped, float* skip_ratio);

#ifdef __cplusplus
}
#endif

#endif /* FRAME_DIFF_H */
