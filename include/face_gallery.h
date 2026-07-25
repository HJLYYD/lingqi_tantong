/*
 * face_gallery.h — Face Gallery module for LingQi TanTong
 *
 * Maintains a time-windowed collection of detected face thumbnails
 * with recognition results, keyed by track_id for deduplication.
 *
 * Architecture:
 *   - Updated from push_frame_to_web() in the viz thread (face crop extraction)
 *   - Read by gallery_broadcast_timer_fn() in the mongoose server thread
 *   - Thread-safe via internal mutex
 *
 * Deduplication strategy:
 *   1. track_id key: same track → update in place (latest thumbnail wins)
 *   2. identity merge: same identity across track_ids → keep highest similarity
 *   3. time-based expiry: entries older than ttl_sec are pruned
 */

#ifndef FACE_GALLERY_H
#define FACE_GALLERY_H

#include "core_types.h"
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FACE_GALLERY_MAX         256
#define FACE_GALLERY_THUMB_W     112
#define FACE_GALLERY_THUMB_H     112
#define FACE_GALLERY_THUMB_MAX   8192    /* max JPEG thumbnail bytes */

typedef struct {
    int      track_id;
    char     identity[STR_LEN];
    float    similarity;
    float    feature_vector[FEAT_DIM];   /* 128-dim L2-normalized embedding */
    bool     has_feature;
    BBox     bbox;
    uint8_t  thumbnail_jpeg[FACE_GALLERY_THUMB_MAX];
    int      thumbnail_len;
    int64_t  last_seen_ms;
    bool     active;
} FaceGalleryEntry;

typedef struct {
    FaceGalleryEntry entries[FACE_GALLERY_MAX];
    int         num_entries;
    int         ttl_sec;            /* expiry timeout in seconds (default 60) */
    pthread_mutex_t mutex;
} FaceGallery;

/* ── Lifecycle ── */
void face_gallery_init(FaceGallery* g, int ttl_sec);
void face_gallery_destroy(FaceGallery* g);

/* ── Core operations ── */

/**
 * Update or insert a gallery entry.
 * Called from the viz thread each time a face is detected.
 * track_id acts as the primary dedup key.
 */
void face_gallery_update(FaceGallery* g, int track_id,
                          const char* identity, float similarity,
                          const float* feature_vector, bool has_feature,
                          const BBox* bbox,
                          const uint8_t* thumb_jpeg, int thumb_len);

/** Remove entries older than ttl_sec. Call once per frame push. */
void face_gallery_prune(FaceGallery* g);

/** Build the complete "type":"g" JSON string for WS broadcast.
 *  Returns a heap-allocated string (caller must free). */
char* face_gallery_build_json(FaceGallery* g, int* out_len);

/** Find an entry by track_id. Returns NULL if not found. */
FaceGalleryEntry* face_gallery_find(FaceGallery* g, int track_id);

/** Clear all gallery entries. */
void face_gallery_clear(FaceGallery* g);

#ifdef __cplusplus
}
#endif

#endif /* FACE_GALLERY_H */
