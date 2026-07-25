/*
 * face_gallery.c — Face Gallery implementation
 *
 * Thread-safe face thumbnail collection for web UI display.
 * See face_gallery.h for the full design rationale.
 */

#include "face_gallery.h"
#include "utils.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

void face_gallery_init(FaceGallery* g, int ttl_sec) {
    if (!g) return;
    memset(g, 0, sizeof(*g));
    g->ttl_sec = ttl_sec > 0 ? ttl_sec : 60;
    pthread_mutex_init(&g->mutex, NULL);
}

void face_gallery_destroy(FaceGallery* g) {
    if (!g) return;
    pthread_mutex_destroy(&g->mutex);
    memset(g, 0, sizeof(*g));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Core operations
 * ═══════════════════════════════════════════════════════════════════════ */

void face_gallery_update(FaceGallery* g, int track_id,
                          const char* identity, float similarity,
                          const float* feature_vector, bool has_feature,
                          const BBox* bbox,
                          const uint8_t* thumb_jpeg, int thumb_len) {
    if (!g || track_id < 0) return;

    pthread_mutex_lock(&g->mutex);

    /* 1. Find existing entry by track_id */
    int slot = -1;
    for (int i = 0; i < g->num_entries; i++) {
        if (g->entries[i].active && g->entries[i].track_id == track_id) {
            slot = i;
            break;
        }
    }

    /* 2. If not found, find an empty or expired slot */
    if (slot < 0) {
        int64_t now = utils_get_time_ms();
        int64_t ttl_ms = (int64_t)g->ttl_sec * 1000;

        /* Look for an inactive slot first */
        for (int i = 0; i < FACE_GALLERY_MAX; i++) {
            if (!g->entries[i].active ||
                (now - g->entries[i].last_seen_ms > ttl_ms)) {
                slot = i;
                break;
            }
        }

        /* If still no slot, evict the oldest entry */
        if (slot < 0) {
            int64_t oldest = INT64_MAX;
            for (int i = 0; i < FACE_GALLERY_MAX; i++) {
                if (g->entries[i].last_seen_ms < oldest) {
                    oldest = g->entries[i].last_seen_ms;
                    slot = i;
                }
            }
        }

        if (slot < 0) { pthread_mutex_unlock(&g->mutex); return; }

        /* Update num_entries if needed */
        if (slot >= g->num_entries) {
            g->num_entries = slot + 1;
            if (g->num_entries > FACE_GALLERY_MAX) g->num_entries = FACE_GALLERY_MAX;
        }
    }

    /* 3. Fill in the entry */
    FaceGalleryEntry* e = &g->entries[slot];
    e->track_id = track_id;
    if (identity) {
        strncpy(e->identity, identity, STR_LEN - 1);
        e->identity[STR_LEN - 1] = '\0';
    } else {
        e->identity[0] = '\0';
    }
    e->similarity = similarity;
    e->has_feature = has_feature;
    if (has_feature && feature_vector) {
        memcpy(e->feature_vector, feature_vector, FEAT_DIM * sizeof(float));
    }
    if (bbox) {
        e->bbox = *bbox;
    }
    if (thumb_jpeg && thumb_len > 0 && thumb_len <= FACE_GALLERY_THUMB_MAX) {
        memcpy(e->thumbnail_jpeg, thumb_jpeg, (size_t)thumb_len);
        e->thumbnail_len = thumb_len;
    } else {
        e->thumbnail_len = 0;
    }
    e->last_seen_ms = utils_get_time_ms();
    e->active = true;

    /* 4. Identity-level dedup: if same identity appears on another track_id,
     *    keep the entry with higher similarity, deactivate the other */
    if (identity && identity[0] != '\0' && strcmp(identity, "unknown") != 0) {
        for (int i = 0; i < g->num_entries; i++) {
            if (i == slot) continue;
            if (!g->entries[i].active) continue;
            if (strcmp(g->entries[i].identity, identity) == 0) {
                /* Same identity on different track — keep higher similarity */
                if (g->entries[i].similarity >= e->similarity) {
                    /* Old entry is better, deactivate new one */
                    e->active = false;
                } else {
                    /* New entry is better, deactivate old one */
                    g->entries[i].active = false;
                }
                break;
            }
        }
    }

    pthread_mutex_unlock(&g->mutex);
}

void face_gallery_prune(FaceGallery* g) {
    if (!g) return;

    pthread_mutex_lock(&g->mutex);

    int64_t now = utils_get_time_ms();
    int64_t ttl_ms = (int64_t)g->ttl_sec * 1000;

    for (int i = 0; i < g->num_entries; i++) {
        if (g->entries[i].active && (now - g->entries[i].last_seen_ms > ttl_ms)) {
            g->entries[i].active = false;
        }
    }

    pthread_mutex_unlock(&g->mutex);
}

char* face_gallery_build_json(FaceGallery* g, int* out_len) {
    if (!g) { if (out_len) *out_len = 0; return NULL; }

    pthread_mutex_lock(&g->mutex);

    /* Estimate buffer size: ~2KB per entry (with base64 thumbnail) */
    int cap = 256 + g->num_entries * 8192;
    char* buf = (char*)malloc((size_t)cap);
    if (!buf) { pthread_mutex_unlock(&g->mutex); return NULL; }

    int w = 0;
    #define GA_APPEND(fmt, ...) do { \
        int n = snprintf(buf + w, (size_t)(cap - w > 0 ? cap - w : 0), fmt, ##__VA_ARGS__); \
        if (n > 0) w += n; if (w >= cap) w = cap; \
    } while(0)

    GA_APPEND("{\"type\":\"g\",\"entries\":[");

    int first = 1;
    int64_t now = utils_get_time_ms();

    for (int i = 0; i < g->num_entries; i++) {
        if (!g->entries[i].active) continue;
        FaceGalleryEntry* e = &g->entries[i];

        if (!first) GA_APPEND(",");
        first = 0;

        /* Escape identity string */
        char id_esc[STR_LEN * 2];
        {
            const char* src = e->identity[0] ? e->identity : "unknown";
            char* dst = id_esc;
            while (*src && (size_t)(dst - id_esc) < sizeof(id_esc) - 2) {
                if (*src == '\\') { *dst++ = '\\'; *dst++ = '\\'; }
                else if (*src == '"') { *dst++ = '\\'; *dst++ = '"'; }
                else { *dst++ = *src; }
                src++;
            }
            *dst = '\0';
        }

        GA_APPEND("{\"t\":%d,\"id\":\"%s\",\"s\":%.2f,",
                  e->track_id, id_esc, (double)e->similarity);

        /* Face bbox */
        GA_APPEND("\"b\":[%.0f,%.0f,%.0f,%.0f],",
                  (double)e->bbox.x_min, (double)e->bbox.y_min,
                  (double)e->bbox.x_max, (double)e->bbox.y_max);

        /* Age in seconds */
        double age = (double)(now - e->last_seen_ms) / 1000.0;
        GA_APPEND("\"age\":%.1f", age);

        /* Base64-encoded thumbnail */
        if (e->thumbnail_len > 0) {
            int b64_len = (e->thumbnail_len + 2) / 3 * 4 + 1;
            char* b64 = (char*)malloc((size_t)b64_len);
            if (b64) {
                int enc_len = utils_base64_encode(e->thumbnail_jpeg, e->thumbnail_len,
                                                   b64, b64_len);
                GA_APPEND(",\"thumb\":\"%s\"", b64);
                free(b64);
            }
        }

        GA_APPEND("}");
    }

    GA_APPEND("]}");
    #undef GA_APPEND

    if (out_len) *out_len = w;
    pthread_mutex_unlock(&g->mutex);
    return buf;
}

FaceGalleryEntry* face_gallery_find(FaceGallery* g, int track_id) {
    if (!g || track_id < 0) return NULL;

    pthread_mutex_lock(&g->mutex);
    for (int i = 0; i < g->num_entries; i++) {
        if (g->entries[i].active && g->entries[i].track_id == track_id) {
            pthread_mutex_unlock(&g->mutex);
            return &g->entries[i];
        }
    }
    pthread_mutex_unlock(&g->mutex);
    return NULL;
}

void face_gallery_clear(FaceGallery* g) {
    if (!g) return;
    pthread_mutex_lock(&g->mutex);
    g->num_entries = 0;
    pthread_mutex_unlock(&g->mutex);
}
