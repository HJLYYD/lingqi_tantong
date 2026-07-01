/*
 * frame_diff.c — Fast Subsampled Frame Differencing Implementation
 *
 * See frame_diff.h for algorithm description and references.
 */

#include "frame_diff.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

FrameDiff* frame_diff_create(int grid_w, int grid_h, int subsample,
                              float cell_threshold, float change_threshold) {
    FrameDiff* fd = (FrameDiff*)calloc(1, sizeof(FrameDiff));
    if (!fd) return NULL;

    fd->grid_w = (grid_w > 0) ? grid_w : FD_DEFAULT_GRID_W;
    fd->grid_h = (grid_h > 0) ? grid_h : FD_DEFAULT_GRID_H;
    fd->subsample = (subsample > 0) ? subsample : FD_DEFAULT_SUBSAMPLE;
    fd->cell_threshold = (cell_threshold > 0.0f) ? cell_threshold : FD_DEFAULT_CELL_THRESH;
    fd->change_threshold = (change_threshold > 0.0f) ? change_threshold : FD_DEFAULT_CHANGE_RATIO;

    /* Adaptive defaults */
    fd->adaptive_enabled = true;
    fd->max_static_skip = 20;
    fd->max_low_motion_skip = 5;
    fd->force_process_every = 30;

    fd->reference_frame = NULL;
    fd->has_reference = false;
    fd->consecutive_skips = 0;
    fd->total_skipped = 0;
    fd->total_processed = 0;
    fd->last_change_ratio = 1.0f;
    fd->last_activity = FD_SCENE_ACTIVE;

    /* Allocate cell MAD buffer */
    fd->cell_mads_size = fd->grid_w * fd->grid_h;
    fd->cell_mads = (float*)calloc((size_t)fd->cell_mads_size, sizeof(float));
    if (!fd->cell_mads) {
        free(fd);
        return NULL;
    }

    log_info("FrameDiff: grid=%dx%d subsample=%d cell_thresh=%.0f change_thresh=%.2f "
             "adaptive=%d max_static=%d max_low=%d force=%d",
             fd->grid_w, fd->grid_h, fd->subsample,
             (double)fd->cell_threshold, (double)fd->change_threshold,
             fd->adaptive_enabled, fd->max_static_skip,
             fd->max_low_motion_skip, fd->force_process_every);

    return fd;
}

void frame_diff_destroy(FrameDiff* fd) {
    if (!fd) return;
    free(fd->reference_frame);
    free(fd->cell_mads);
    int processed, skipped;
    float ratio;
    frame_diff_get_stats(fd, &processed, &skipped, &ratio);
    log_info("FrameDiff: destroyed (processed=%d skipped=%d ratio=%.1f%%)",
             processed, skipped, (double)(ratio * 100.0f));
    free(fd);
}

void frame_diff_reset(FrameDiff* fd) {
    if (!fd) return;
    free(fd->reference_frame);
    fd->reference_frame = NULL;
    fd->has_reference = false;
    fd->consecutive_skips = 0;
    fd->last_change_ratio = 1.0f;
    fd->last_activity = FD_SCENE_ACTIVE;
    fd->total_skipped = 0;
    fd->total_processed = 0;
    log_debug("FrameDiff: reset");
}

int frame_diff_set_reference(FrameDiff* fd, const uint8_t* rgb, int width, int height) {
    if (!fd || !rgb || width <= 0 || height <= 0) return -1;

    size_t needed = (size_t)width * height * 3;
    if (!fd->reference_frame || fd->ref_width != width || fd->ref_height != height) {
        uint8_t* new_buf = (uint8_t*)realloc(fd->reference_frame, needed);
        if (!new_buf) {
            log_error("FrameDiff: failed to allocate reference buffer (%zux%zu)",
                      (size_t)width, (size_t)height);
            return -1;
        }
        fd->reference_frame = new_buf;
    }

    memcpy(fd->reference_frame, rgb, needed);
    fd->ref_width = width;
    fd->ref_height = height;
    fd->has_reference = true;

    return 0;
}

/*
 * Internal: compute MAD for a single grid cell.
 *
 * The cell covers pixel region [cell_x_start, cell_x_end) × [cell_y_start, cell_y_end).
 * Within this region, sample every `subsample`-th pixel.
 * MAD = mean(|R_new - R_ref| + |G_new - G_ref| + |B_new - B_ref|) / 3
 *
 * Returns the per-channel MAD for this cell.
 */
static float compute_cell_mad(const uint8_t* ref, const uint8_t* cur,
                               int width, int height,
                               int cell_x_start, int cell_x_end,
                               int cell_y_start, int cell_y_end,
                               int subsample) {
    int64_t sum_diff = 0;
    int count = 0;

    for (int y = cell_y_start; y < cell_y_end; y += subsample) {
        if (y >= height) break;
        for (int x = cell_x_start; x < cell_x_end; x += subsample) {
            if (x >= width) break;
            size_t off = ((size_t)y * (size_t)width + (size_t)x) * 3;
            int dr = (int)cur[off + 0] - (int)ref[off + 0];
            int dg = (int)cur[off + 1] - (int)ref[off + 1];
            int db = (int)cur[off + 2] - (int)ref[off + 2];
            sum_diff += (dr < 0 ? -dr : dr) + (dg < 0 ? -dg : dg) + (db < 0 ? -db : db);
            count++;
        }
    }

    if (count == 0) return 0.0f;
    /* Per-channel MAD: divide total RGB diff sum by (3 * count) */
    return (float)((double)sum_diff / (3.0 * (double)count));
}

float frame_diff_compute(FrameDiff* fd, const uint8_t* rgb, int width, int height) {
    if (!fd || !rgb || width <= 0 || height <= 0) return 1.0f;

    /* No reference → must process */
    if (!fd->has_reference || !fd->reference_frame) {
        fd->last_change_ratio = 1.0f;
        fd->last_activity = FD_SCENE_ACTIVE;
        return 1.0f;
    }

    /* Dimension mismatch → treat as completely different */
    if (fd->ref_width != width || fd->ref_height != height) {
        log_debug("FrameDiff: dimension mismatch %dx%d vs %dx%d — full process",
                  fd->ref_width, fd->ref_height, width, height);
        fd->last_change_ratio = 1.0f;
        fd->last_activity = FD_SCENE_ACTIVE;
        return 1.0f;
    }

    /* Compute grid cell dimensions */
    int cell_w = width / fd->grid_w;
    int cell_h = height / fd->grid_h;

    if (cell_w < FD_MIN_CELL_SIZE || cell_h < FD_MIN_CELL_SIZE) {
        /* Frame too small for grid — fall back to single-cell comparison */
        float mad = compute_cell_mad(fd->reference_frame, rgb, width, height,
                                      0, width, 0, height, fd->subsample);
        fd->last_change_ratio = (mad > fd->cell_threshold) ? 1.0f : 0.0f;
        fd->last_activity = (fd->last_change_ratio > fd->change_threshold)
            ? FD_SCENE_ACTIVE : FD_SCENE_STATIC;
        return fd->last_change_ratio;
    }

    /* Compute MAD for each grid cell */
    int changed_cells = 0;
    int total_cells = 0;

    for (int gy = 0; gy < fd->grid_h; gy++) {
        for (int gx = 0; gx < fd->grid_w; gx++) {
            int cell_x_start = gx * cell_w;
            int cell_x_end   = (gx == fd->grid_w - 1) ? width  : (gx + 1) * cell_w;
            int cell_y_start = gy * cell_h;
            int cell_y_end   = (gy == fd->grid_h - 1) ? height : (gy + 1) * cell_h;

            float mad = compute_cell_mad(fd->reference_frame, rgb, width, height,
                                          cell_x_start, cell_x_end,
                                          cell_y_start, cell_y_end,
                                          fd->subsample);
            int idx = gy * fd->grid_w + gx;
            if (idx < fd->cell_mads_size) {
                fd->cell_mads[idx] = mad;
            }

            if (mad > fd->cell_threshold) {
                changed_cells++;
            }
            total_cells++;
        }
    }

    float change_ratio = (total_cells > 0)
        ? (float)changed_cells / (float)total_cells
        : 1.0f;

    fd->last_change_ratio = change_ratio;

    /* Classify activity level */
    if (change_ratio < 0.05f) {
        fd->last_activity = FD_SCENE_STATIC;
    } else if (change_ratio < fd->change_threshold) {
        fd->last_activity = FD_SCENE_LOW_MOTION;
    } else {
        fd->last_activity = FD_SCENE_ACTIVE;
    }

    return change_ratio;
}

bool frame_diff_should_process(FrameDiff* fd) {
    if (!fd) return true;

    /* No reference → must process */
    if (!fd->has_reference) return true;

    /* Force process every N frames to prevent drift */
    int total_frames = fd->total_processed + fd->total_skipped;
    if (fd->force_process_every > 0 &&
        total_frames > 0 &&
        total_frames % fd->force_process_every == 0) {
        return true;
    }

    /* Active scene → always process */
    if (fd->last_activity == FD_SCENE_ACTIVE) {
        return true;
    }

    /* Change above threshold → process */
    if (fd->last_change_ratio >= fd->change_threshold) {
        return true;
    }

    /* Adaptive skip: check limits */
    if (fd->adaptive_enabled) {
        int max_skip;
        switch (fd->last_activity) {
        case FD_SCENE_STATIC:
            max_skip = fd->max_static_skip;
            break;
        case FD_SCENE_LOW_MOTION:
            max_skip = fd->max_low_motion_skip;
            break;
        default:
            max_skip = 0;
            break;
        }

        if (fd->consecutive_skips < max_skip) {
            /* Within skip budget — skip this frame */
            return false;
        }
        /* Exceeded skip budget — force process */
    }

    return true;
}

void frame_diff_mark_processed(FrameDiff* fd) {
    if (!fd) return;
    fd->consecutive_skips = 0;
    fd->total_processed++;
}

void frame_diff_mark_skipped(FrameDiff* fd) {
    if (!fd) return;
    fd->consecutive_skips++;
    fd->total_skipped++;
}

const char* frame_diff_activity_name(FrameDiffActivity level) {
    switch (level) {
    case FD_SCENE_STATIC:     return "STATIC";
    case FD_SCENE_LOW_MOTION: return "LOW";
    case FD_SCENE_ACTIVE:     return "ACTIVE";
    default:                  return "UNKNOWN";
    }
}

void frame_diff_get_stats(const FrameDiff* fd, int* processed, int* skipped, float* skip_ratio) {
    if (!fd) {
        if (processed) *processed = 0;
        if (skipped)   *skipped = 0;
        if (skip_ratio) *skip_ratio = 0.0f;
        return;
    }
    if (processed) *processed = fd->total_processed;
    if (skipped)   *skipped = fd->total_skipped;
    if (skip_ratio) {
        int total = fd->total_processed + fd->total_skipped;
        *skip_ratio = (total > 0) ? (float)fd->total_skipped / (float)total : 0.0f;
    }
}
