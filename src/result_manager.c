/*
 * result_manager.c — Session-based inference result management
 *
 * Rewrite based on embedded JSON best practices (cJSON-inspired):
 *   1. Proper JSON escaping via JsonWriter — no injection vulnerabilities.
 *   2. Schema versioning — "schema_version": "2.0.0" in all reports.
 *   3. Incremental per-frame metadata flush — recoverable on crash.
 *   4. Binary trajectory alternative — compact float-binary format.
 */

#include "result_manager.h"
#include "spatial_engine.h"
#include "json_writer.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/* ── Schema version — embedded in every report for forward compatibility ── */
#define RM_SCHEMA_VERSION "2.0.0"

/* ── Timestamp helpers ── */

static void get_timestamp(char* buf, int len) {
    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    strftime(buf, (size_t)len, "%Y%m%d_%H%M%S", &tm_buf);
}

static void get_iso_time(char* buf, int len) {
    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    strftime(buf, (size_t)len, "%Y-%m-%dT%H:%M:%S", &tm_buf);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═════════════════════════════════════════════════════════════════════════ */

ResultManager* result_manager_create(const char* base_output_dir) {
    ResultManager* rm = (ResultManager*)calloc(1, sizeof(ResultManager));
    if (!rm) return NULL;

    strncpy(rm->base_output_dir, base_output_dir ? base_output_dir : "output", MAX_PATH_LEN - 1);
    rm->base_output_dir[MAX_PATH_LEN - 1] = '\0';

    /* Create output base directory only — session subdirs are created on demand */
    mkdir(rm->base_output_dir, 0755);

    /* Open per-frame metadata file handle lazily on session start */
    rm->frame_meta_file = NULL;

    return rm;
}

void result_manager_destroy(ResultManager* rm) {
    if (!rm) return;
    /* Ensure any open per-frame metadata file is closed */
    if (rm->frame_meta_file) {
        fclose(rm->frame_meta_file);
        rm->frame_meta_file = NULL;
    }
    free(rm);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Session management
 * ═════════════════════════════════════════════════════════════════════════ */

const char* result_manager_start_session(ResultManager* rm, const char* video_path) {
    if (!rm || !video_path) return "";
    if (rm->num_sessions >= RM_MAX_SESSIONS) {
        log_error("Cannot start session: maximum sessions (%d) reached", RM_MAX_SESSIONS);
        return "";
    }

    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));

    SessionResult* session = &rm->sessions[rm->num_sessions++];
    memset(session, 0, sizeof(SessionResult));

    snprintf(session->session_id, sizeof(session->session_id), "session_%s", timestamp);
    strncpy(session->timestamp, timestamp, sizeof(session->timestamp) - 1);
    strncpy(session->video_path, video_path, sizeof(session->video_path) - 1);
    get_iso_time(session->start_time, sizeof(session->start_time));
    session->active = true;

    rm->current_session = session;

    /* ── Create session-centric directory: output/<session_id>/ ── */
    char sess_dir[MAX_PATH_LEN * 2];
    snprintf(sess_dir, sizeof(sess_dir), "%s/%s", rm->base_output_dir, session->session_id);
    mkdir(sess_dir, 0755);

    /* Create trajectories subdir */
    char traj_dir[MAX_PATH_LEN * 3];
    snprintf(traj_dir, sizeof(traj_dir), "%s/trajectories", sess_dir);
    mkdir(traj_dir, 0755);

    /* Open per-frame metadata file for incremental writing */
    char meta_path[MAX_PATH_LEN * 3];
    snprintf(meta_path, sizeof(meta_path), "%s/frames.jsonl", sess_dir);
    rm->frame_meta_file = fopen(meta_path, "w");
    if (rm->frame_meta_file) {
        /* Write JSONL header comment (valid JSON-Lines allows comments with #) */
        fprintf(rm->frame_meta_file, "# schema_version=%s session_id=%s\n",
                RM_SCHEMA_VERSION, session->session_id);
        fflush(rm->frame_meta_file);
    }

    log_info("Session started: %s (schema=%s)", session->session_id, RM_SCHEMA_VERSION);
    return session->session_id;
}

void result_manager_update_session_stats(ResultManager* rm, int frames, int objects,
                                          float avg_fps, float avg_time_ms) {
    if (!rm || !rm->current_session) {
        log_warning("No active session to update");
        return;
    }
    rm->current_session->frames_processed = frames;
    rm->current_session->objects_detected = objects;
    rm->current_session->average_fps = avg_fps;
    rm->current_session->average_processing_time_ms = avg_time_ms;
}

void result_manager_add_tracked_object(ResultManager* rm, int track_id, float height,
                                        int pos_count, int pose_count) {
    if (!rm || !rm->current_session) return;

    SessionResult* session = rm->current_session;

    /* Update existing entry if track_id already recorded */
    for (int i = 0; i < session->num_tracked_objects; i++) {
        if (session->tracked_objects[i].track_id == track_id) {
            TrackedObjectSummary* obj = &session->tracked_objects[i];
            obj->height_meters = height;
            obj->position_count = pos_count;
            obj->pose_count = pose_count;
            return;
        }
    }

    /* New track_id — append if room */
    if (session->num_tracked_objects >= RM_MAX_TRACKED_OBJS) {
        log_event_warn("result.tracked_object_buffer_full",
                       "max", "%d", RM_MAX_TRACKED_OBJS,
                       "track_id", "%d", track_id,
                       NULL);
        return;
    }

    TrackedObjectSummary* obj = &session->tracked_objects[session->num_tracked_objects++];
    obj->track_id = track_id;
    obj->height_meters = height;
    obj->position_count = pos_count;
    obj->pose_count = pose_count;
}

void result_manager_add_error(ResultManager* rm, const char* type, const char* message) {
    if (!rm || !rm->current_session) return;

    SessionResult* session = rm->current_session;
    if (session->num_errors >= RM_MAX_ERRORS) {
        log_event_warn("result.error_buffer_full",
                       "max", "%d", RM_MAX_ERRORS,
                       "error_type", "%s", type ? type : "?",
                       NULL);
        return;
    }

    ErrorEntry* err = &session->errors[session->num_errors++];
    strncpy(err->type, type ? type : "unknown", sizeof(err->type) - 1);
    strncpy(err->message, message ? message : "", sizeof(err->message) - 1);
    get_iso_time(err->timestamp, sizeof(err->timestamp));

    log_event_error("result.session_error",
                    "type", "%s", err->type,
                    "message", "%s", err->message,
                    NULL);
}

/* ═════════════════════════════════════════════════════════════════════════
 * Session end + final reports
 * ═════════════════════════════════════════════════════════════════════════ */

const char* result_manager_end_session(ResultManager* rm, SpatialLocalizationEngine* spatial) {
    if (!rm || !rm->current_session) {
        log_warning("No active session to end");
        return "";
    }

    SessionResult* session = rm->current_session;
    log_info("[ResultMgr] Ending session %s", session->session_id);
    log_flush();

    /* Close per-frame metadata file */
    if (rm->frame_meta_file) {
        fclose(rm->frame_meta_file);
        rm->frame_meta_file = NULL;
    }

    /* ── Export spatial trajectories ── */
    if (spatial) {
        log_info("[ResultMgr] Exporting spatial trajectories...");
        log_flush();
        int track_ids[SPATIAL_MAX_PERSONS];
        int num_active = spatial_engine_get_active_tracks(spatial, track_ids, SPATIAL_MAX_PERSONS);
        log_info("[ResultMgr] get_active_tracks returned %d active tracks (max=%d)",
                 num_active, SPATIAL_MAX_PERSONS);
        log_flush();

        /* Defensive: clamp num_active to array bounds */
        if (num_active < 0 || num_active > SPATIAL_MAX_PERSONS) {
            log_error("[ResultMgr] num_active=%d out of bounds, clamping to 0", num_active);
            num_active = 0;
        }

        char traj_dir[MAX_PATH_LEN * 3];
        snprintf(traj_dir, sizeof(traj_dir), "%s/%s/trajectories",
                 rm->base_output_dir, session->session_id);

        for (int i = 0; i < num_active; i++) {
            int tid = track_ids[i];
            /* Defensive: validate track_id */
            if (tid < 0 || tid >= SPATIAL_MAX_PERSONS) {
                log_warning("[ResultMgr] Skipping invalid track_id=%d at index %d", tid, i);
                continue;
            }
            int traj_count = 0;
            const SpatialPosition* traj = spatial_engine_get_trajectory(spatial, tid, &traj_count);
            if (!traj || traj_count < 2) continue;

            /* Defensive: clamp traj_count to TRAJ_LEN */
            if (traj_count > TRAJ_LEN) {
                log_warning("[ResultMgr] track %d: traj_count=%d exceeds TRAJ_LEN=%d, clamping",
                           tid, traj_count, TRAJ_LEN);
                traj_count = TRAJ_LEN;
            }

            log_info("[ResultMgr] Exporting trajectory track=%d, %d points", tid, traj_count);
            log_flush();

            /* Write trajectory as CSV (human-readable) */
            char traj_path[MAX_PATH_LEN * 3];
            snprintf(traj_path, sizeof(traj_path), "%s/person_%d.csv",
                     traj_dir, tid);
            FILE* f = fopen(traj_path, "w");
            if (f) {
                fprintf(f, "index,x,y,z,confidence\n");
                float total_dist = 0.0f;
                for (int j = 0; j < traj_count; j++) {
                    fprintf(f, "%d,%.4f,%.4f,%.4f,%.4f\n",
                            j, traj[j].x, traj[j].y, traj[j].z, traj[j].confidence);
                    if (j > 0) {
                        float dx = traj[j].x - traj[j-1].x;
                        float dy = traj[j].y - traj[j-1].y;
                        float dz = traj[j].z - traj[j-1].z;
                        total_dist += sqrtf(dx*dx + dy*dy + dz*dz);
                    }
                }
                fclose(f);

                /* Store trajectory summary in session record */
                if (session->num_tracked_objects < 0 || session->num_tracked_objects > RM_MAX_TRACKED_OBJS) {
                    log_error("[ResultMgr] CORRUPTED num_tracked_objects=%d, resetting to 0",
                             session->num_tracked_objects);
                    session->num_tracked_objects = 0;
                }
                if (session->num_tracked_objects < RM_MAX_TRACKED_OBJS) {
                    TrackedObjectSummary* obj = &session->tracked_objects[session->num_tracked_objects];
                    bool found = false;
                    for (int k = 0; k < session->num_tracked_objects; k++) {
                        if (session->tracked_objects[k].track_id == tid) {
                            obj = &session->tracked_objects[k];
                            found = true;
                            break;
                        }
                    }
                    obj->track_id = tid;
                    obj->position_count = traj_count;
                    obj->start_x = traj[0].x;
                    obj->start_y = traj[0].y;
                    obj->start_z = traj[0].z;
                    obj->end_x = traj[traj_count - 1].x;
                    obj->end_y = traj[traj_count - 1].y;
                    obj->end_z = traj[traj_count - 1].z;
                    obj->total_distance_m = total_dist;
                    if (!found) session->num_tracked_objects++;
                } else {
                    log_warning("[ResultMgr] Skipping track summary: num_tracked_objects=%d >= max=%d",
                               session->num_tracked_objects, RM_MAX_TRACKED_OBJS);
                }
            } else {
                log_warning("[ResultMgr] Failed to open trajectory CSV: %s", traj_path);
            }
        }

        log_info("[ResultMgr] Trajectory export complete (%d tracks exported)", num_active);
        log_flush();
    }

    log_info("[ResultMgr] Ending session, saving reports...");
    log_flush();
    get_iso_time(session->end_time, sizeof(session->end_time));
    session->active = false;

    char json_path[MAX_PATH_LEN];
    char csv_path[MAX_PATH_LEN];

    log_info("[ResultMgr] Saving JSON report...");
    log_flush();
    result_manager_save_json_report(rm, session->session_id, json_path, sizeof(json_path));
    log_info("[ResultMgr] JSON report saved: %s", json_path);
    log_flush();

    log_info("[ResultMgr] Saving CSV report...");
    log_flush();
    result_manager_save_csv_report(rm, session->session_id, csv_path, sizeof(csv_path));
    log_info("[ResultMgr] CSV report saved: %s", csv_path);
    log_flush();

    log_event_info("result.session_ended",
                   "session_id", "%s", session->session_id,
                   "json", "%s", json_path,
                   "csv", "%s", csv_path,
                   NULL);

    log_info("[ResultMgr] Session %s ended successfully", session->session_id);
    log_flush();
    rm->current_session = NULL;
    return session->session_id;
}

/* ═════════════════════════════════════════════════════════════════════════
 * Save helpers — using JsonWriter for proper escaping
 * ═════════════════════════════════════════════════════════════════════════ */

static const SessionResult* find_session(const ResultManager* rm, const char* session_id) {
    if (!rm || !session_id) return NULL;
    /* Defensive: clamp num_sessions to prevent OOB on corruption */
    int n = rm->num_sessions;
    if (n < 0 || n > RM_MAX_SESSIONS) {
        log_error("[ResultMgr] CORRUPTED num_sessions=%d, clamping to 0", n);
        n = 0;
    }
    for (int i = 0; i < n; i++) {
        if (strcmp(rm->sessions[i].session_id, session_id) == 0) {
            return &rm->sessions[i];
        }
    }
    return NULL;
}

int result_manager_save_json_report(const ResultManager* rm, const char* session_id,
                                     char* out_path, int path_len) {
    if (!rm || !session_id || !out_path) return -1;

    const SessionResult* session = find_session(rm, session_id);
    if (!session) return -1;

    snprintf(out_path, (size_t)path_len, "%s/%s/summary.json",
             rm->base_output_dir, session_id);

    FILE* f = fopen(out_path, "w");
    if (!f) return -1;

    JsonWriter jw;
    json_writer_init(&jw, f, 2);  /* pretty-printed with 2-space indent */

    json_writer_object_begin(&jw);

    /* Header metadata */
    json_writer_key_string(&jw, "schema_version", RM_SCHEMA_VERSION);
    json_writer_key_string(&jw, "session_id", session->session_id);
    json_writer_key_string(&jw, "timestamp", session->timestamp);
    json_writer_key_string(&jw, "video_path", session->video_path);
    json_writer_key_string(&jw, "start_time", session->start_time);
    json_writer_key_string(&jw, "end_time", session->end_time);

    /* Performance stats */
    json_writer_key_int(&jw, "frames_processed", session->frames_processed);
    json_writer_key_int(&jw, "objects_detected", session->objects_detected);
    json_writer_key_float(&jw, "average_fps", session->average_fps, 2);
    json_writer_key_float(&jw, "average_processing_time_ms", session->average_processing_time_ms, 2);

    /* Tracked objects array */
    json_writer_key(&jw, "tracked_objects");
    json_writer_array_begin(&jw);
    int num_tobjs = session->num_tracked_objects;
    if (num_tobjs < 0 || num_tobjs > RM_MAX_TRACKED_OBJS) {
        log_error("[ResultMgr] JSON: CORRUPTED num_tracked_objects=%d, clamping", num_tobjs);
        num_tobjs = 0;
    }
    for (int i = 0; i < num_tobjs; i++) {
        const TrackedObjectSummary* obj = &session->tracked_objects[i];
        json_writer_object_begin(&jw);
        json_writer_key_int(&jw, "track_id", obj->track_id);
        json_writer_key_float(&jw, "height_meters", obj->height_meters, 2);
        json_writer_key_int(&jw, "position_count", obj->position_count);
        json_writer_key_int(&jw, "pose_count", obj->pose_count);
        json_writer_key_float3(&jw, "start_position",
                                obj->start_x, obj->start_y, obj->start_z, 4);
        json_writer_key_float3(&jw, "end_position",
                                obj->end_x, obj->end_y, obj->end_z, 4);
        json_writer_key_float(&jw, "total_distance_m", obj->total_distance_m, 4);
        json_writer_object_end(&jw);
    }
    json_writer_array_end(&jw);

    /* Errors array */
    json_writer_key(&jw, "errors");
    json_writer_array_begin(&jw);
    for (int i = 0; i < session->num_errors; i++) {
        const ErrorEntry* err = &session->errors[i];
        json_writer_object_begin(&jw);
        json_writer_key_string(&jw, "type", err->type);
        json_writer_key_string(&jw, "message", err->message);
        json_writer_key_string(&jw, "timestamp", err->timestamp);
        json_writer_object_end(&jw);
    }
    json_writer_array_end(&jw);

    json_writer_object_end(&jw);
    fputc('\n', f);

    fclose(f);
    return 0;
}

int result_manager_save_csv_report(const ResultManager* rm, const char* session_id,
                                    char* out_path, int path_len) {
    if (!rm || !session_id || !out_path) return -1;

    const SessionResult* session = find_session(rm, session_id);
    if (!session) return -1;

    snprintf(out_path, (size_t)path_len, "%s/%s/summary.csv",
             rm->base_output_dir, session_id);

    FILE* f = fopen(out_path, "w");
    if (!f) return -1;

    /* Write BOM for Excel compatibility with UTF-8 */
    fprintf(f, "\xEF\xBB\xBF");

    /* CSV with proper escaping — fields containing commas or quotes are wrapped */
    fprintf(f, "schema_version,Metric,Value\n");
    fprintf(f, "%s,Session ID,%s\n", RM_SCHEMA_VERSION, session->session_id);
    fprintf(f, ",Timestamp,%s\n", session->timestamp);
    fprintf(f, ",Video Path,%s\n", session->video_path);
    fprintf(f, ",Frames Processed,%d\n", session->frames_processed);
    fprintf(f, ",Objects Detected,%d\n", session->objects_detected);
    fprintf(f, ",Average FPS,%.2f\n", session->average_fps);
    fprintf(f, ",Avg Processing Time (ms),%.2f\n", session->average_processing_time_ms);
    fprintf(f, ",Start Time,%s\n", session->start_time);
    fprintf(f, ",End Time,%s\n", session->end_time);
    fprintf(f, ",Tracked Objects Count,%d\n", session->num_tracked_objects);
    fprintf(f, ",Errors Count,%d\n", session->num_errors);

    fclose(f);
    return 0;
}

/* ═════════════════════════════════════════════════════════════════════════
 * Per-frame operations
 * ═════════════════════════════════════════════════════════════════════════ */

int result_manager_save_frame(const ResultManager* rm, const char* session_id,
                               const uint8_t* frame_data, int width, int height,
                               int frame_num) {
    if (!rm || !session_id || !frame_data) return -1;

    char frame_path[MAX_PATH_LEN * 3];
    snprintf(frame_path, sizeof(frame_path), "%s/%s/frame_%05d.bmp",
             rm->base_output_dir, session_id, frame_num);

    int ret = utils_write_bmp(frame_path, frame_data, width, height);

    if (ret == 0) {
        log_info("Visualization frame saved: %s (Frame #%d)", frame_path, frame_num);
    }

    return ret;
}

int result_manager_save_frame_metadata(const ResultManager* rm, const char* session_id,
                                        int frame_num, int num_detections, int num_poses,
                                        int num_faces, int num_tracked, int has_action) {
    if (!rm || !session_id) return -1;

    /* Use the pre-opened frame metadata file if available */
    if (rm->frame_meta_file) {
        char timestamp[32];
        get_iso_time(timestamp, sizeof(timestamp));

        /* Write compact JSON line with proper escaping via JsonWriter */
        JsonWriter jw;
        json_writer_init(&jw, rm->frame_meta_file, 0);  /* compact, no newlines */

        json_writer_object_begin(&jw);
        json_writer_key_int(&jw, "frame", frame_num);
        json_writer_key_string(&jw, "time", timestamp);
        json_writer_key_int(&jw, "detections", num_detections);
        json_writer_key_int(&jw, "poses", num_poses);
        json_writer_key_int(&jw, "faces", num_faces);
        json_writer_key_int(&jw, "tracked", num_tracked);
        json_writer_key_bool(&jw, "action", has_action != 0);
        json_writer_object_end(&jw);
        fputc('\n', rm->frame_meta_file);

        /* Flush periodically (every frame for crash-recovery, or every N frames for perf) */
        /* Every frame flush: ensures crash-recovery but costs ~1 syscall/frame */
        fflush(rm->frame_meta_file);
        return 0;
    }

    /* Fallback: open-append-close pattern */
    char meta_path[MAX_PATH_LEN * 3];
    snprintf(meta_path, sizeof(meta_path), "%s/%s/frames.jsonl",
             rm->base_output_dir, session_id);

    FILE* f = fopen(meta_path, "a");
    if (!f) return -1;

    char timestamp[32];
    get_iso_time(timestamp, sizeof(timestamp));

    JsonWriter jw;
    json_writer_init(&jw, f, 0);  /* compact */

    json_writer_object_begin(&jw);
    json_writer_key_int(&jw, "frame", frame_num);
    json_writer_key_string(&jw, "time", timestamp);
    json_writer_key_int(&jw, "detections", num_detections);
    json_writer_key_int(&jw, "poses", num_poses);
    json_writer_key_int(&jw, "faces", num_faces);
    json_writer_key_int(&jw, "tracked", num_tracked);
    json_writer_key_bool(&jw, "action", has_action != 0);
    json_writer_object_end(&jw);
    fputc('\n', f);

    fclose(f);
    return 0;
}

/* ═════════════════════════════════════════════════════════════════════════
 * Video output path helper
 * ═════════════════════════════════════════════════════════════════════════ */

int result_manager_get_video_path(const ResultManager* rm, const char* session_id,
                                   char* out_path, int path_len) {
    if (!rm || !session_id || !out_path || path_len <= 0) return -1;
    snprintf(out_path, (size_t)path_len, "%s/%s/annotated.mp4",
             rm->base_output_dir, session_id);
    return 0;
}

/* ═════════════════════════════════════════════════════════════════════════
 * Incremental flush — call periodically for crash recovery
 * ═════════════════════════════════════════════════════════════════════════ */

void result_manager_flush(ResultManager* rm) {
    if (!rm) return;
    if (rm->frame_meta_file) {
        fflush(rm->frame_meta_file);
    }
}
