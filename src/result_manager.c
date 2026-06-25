#include "result_manager.h"
#include "spatial_engine.h"
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

static void get_timestamp(char* buf, int len) {
    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    strftime(buf, len, "%Y%m%d_%H%M%S", &tm_buf);
}

static void get_iso_time(char* buf, int len) {
    time_t now = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    strftime(buf, len, "%Y-%m-%dT%H:%M:%S", &tm_buf);
}

ResultManager* result_manager_create(const char* base_output_dir) {
    ResultManager* rm = (ResultManager*)calloc(1, sizeof(ResultManager));
    if (!rm) return NULL;

    strncpy(rm->base_output_dir, base_output_dir ? base_output_dir : "output", MAX_PATH_LEN - 1);
    rm->base_output_dir[MAX_PATH_LEN - 1] = '\0';

    mkdir(rm->base_output_dir, 0755);

    char path[MAX_PATH_LEN * 2];
    snprintf(path, sizeof(path), "%s/results", rm->base_output_dir);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/models", rm->base_output_dir);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/reports", rm->base_output_dir);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/frames", rm->base_output_dir);
    mkdir(path, 0755);

    return rm;
}

void result_manager_destroy(ResultManager* rm) {
    free(rm);
}

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

    log_info("Session started: %s", session->session_id);
    return session->session_id;
}

void result_manager_update_session_stats(ResultManager* rm, int frames, int objects, float avg_fps, float avg_time_ms) {
    if (!rm || !rm->current_session) {
        log_warning("No active session to update");
        return;
    }

    rm->current_session->frames_processed = frames;
    rm->current_session->objects_detected = objects;
    rm->current_session->average_fps = avg_fps;
    rm->current_session->average_processing_time_ms = avg_time_ms;
}

void result_manager_add_tracked_object(ResultManager* rm, int track_id, float height, int pos_count, int pose_count) {
    if (!rm || !rm->current_session) return;

    SessionResult* session = rm->current_session;
    if (session->num_tracked_objects >= RM_MAX_TRACKED_OBJS) {
        log_warning("Tracked objects buffer full (max %d), skipping track_id=%d", RM_MAX_TRACKED_OBJS, track_id);
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
        log_warning("Error buffer full (max %d), skipping: %s", RM_MAX_ERRORS, message ? message : "");
        return;
    }

    ErrorEntry* err = &session->errors[session->num_errors++];
    strncpy(err->type, type, sizeof(err->type) - 1);
    strncpy(err->message, message, sizeof(err->message) - 1);
    get_iso_time(err->timestamp, sizeof(err->timestamp));
}

const char* result_manager_end_session(ResultManager* rm, SpatialLocalizationEngine* spatial) {
    if (!rm || !rm->current_session) {
        log_warning("No active session to end");
        return "";
    }

    SessionResult* session = rm->current_session;

    /* ── Export spatial trajectories before ending the session ── */
    if (spatial) {
        int track_ids[SPATIAL_MAX_PERSONS];
        int num_active = spatial_engine_get_active_tracks(spatial, track_ids, SPATIAL_MAX_PERSONS);

        /* Per-track CSV: output/trajectories/{session_id}_track_{id}.csv */
        char traj_dir[MAX_PATH_LEN * 2];
        snprintf(traj_dir, sizeof(traj_dir), "%s/trajectories", rm->base_output_dir);
        mkdir(traj_dir, 0755);

        for (int i = 0; i < num_active; i++) {
            int tid = track_ids[i];
            int traj_count = 0;
            const SpatialPosition* traj = spatial_engine_get_trajectory(spatial, tid, &traj_count);
            if (!traj || traj_count < 2) continue;

            char traj_path[MAX_PATH_LEN * 3];
            snprintf(traj_path, sizeof(traj_path), "%s/%s_track_%d.csv",
                     traj_dir, session->session_id, tid);
            FILE* f = fopen(traj_path, "w");
            if (!f) continue;
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
            if (session->num_tracked_objects < RM_MAX_TRACKED_OBJS) {
                TrackedObjectSummary* obj = &session->tracked_objects[session->num_tracked_objects];
                /* Update existing entry or create new one */
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
            }
        }
    }

    get_iso_time(session->end_time, sizeof(session->end_time));
    session->active = false;

    char json_path[MAX_PATH_LEN];
    char csv_path[MAX_PATH_LEN];

    result_manager_save_json_report(rm, session->session_id, json_path, sizeof(json_path));
    result_manager_save_csv_report(rm, session->session_id, csv_path, sizeof(csv_path));

    log_info("Session %s ended. Results saved to:", session->session_id);
    log_info("  JSON: %s", json_path);
    log_info("  CSV: %s", csv_path);

    rm->current_session = NULL;
    return session->session_id;
}

int result_manager_save_json_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len) {
    if (!rm || !session_id || !out_path) return -1;

    const SessionResult* session = NULL;
    for (int i = 0; i < rm->num_sessions; i++) {
        if (strcmp(rm->sessions[i].session_id, session_id) == 0) {
            session = &rm->sessions[i];
            break;
        }
    }
    if (!session) return -1;

    snprintf(out_path, path_len, "%s/reports/%s_report.json", rm->base_output_dir, session_id);

    FILE* f = fopen(out_path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"session_id\": \"%s\",\n", session->session_id);
    fprintf(f, "  \"timestamp\": \"%s\",\n", session->timestamp);
    fprintf(f, "  \"video_path\": \"%s\",\n", session->video_path);
    fprintf(f, "  \"start_time\": \"%s\",\n", session->start_time);
    fprintf(f, "  \"end_time\": \"%s\",\n", session->end_time);
    fprintf(f, "  \"frames_processed\": %d,\n", session->frames_processed);
    fprintf(f, "  \"objects_detected\": %d,\n", session->objects_detected);
    fprintf(f, "  \"average_fps\": %.2f,\n", session->average_fps);
    fprintf(f, "  \"average_processing_time_ms\": %.2f,\n", session->average_processing_time_ms);
    fprintf(f, "  \"tracked_objects\": [\n");

    for (int i = 0; i < session->num_tracked_objects; i++) {
        const TrackedObjectSummary* obj = &session->tracked_objects[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"track_id\": %d,\n", obj->track_id);
        fprintf(f, "      \"height_meters\": %.2f,\n", obj->height_meters);
        fprintf(f, "      \"position_count\": %d,\n", obj->position_count);
        fprintf(f, "      \"pose_count\": %d,\n", obj->pose_count);
        fprintf(f, "      \"start_position\": [%.4f, %.4f, %.4f],\n",
                obj->start_x, obj->start_y, obj->start_z);
        fprintf(f, "      \"end_position\": [%.4f, %.4f, %.4f],\n",
                obj->end_x, obj->end_y, obj->end_z);
        fprintf(f, "      \"total_distance_m\": %.4f\n", obj->total_distance_m);
        fprintf(f, "    }%s\n", (i < session->num_tracked_objects - 1) ? "," : "");
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"errors\": [\n");

    for (int i = 0; i < session->num_errors; i++) {
        const ErrorEntry* err = &session->errors[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"type\": \"%s\",\n", err->type);
        fprintf(f, "      \"message\": \"%s\",\n", err->message);
        fprintf(f, "      \"timestamp\": \"%s\"\n", err->timestamp);
        fprintf(f, "    }%s\n", (i < session->num_errors - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

int result_manager_save_csv_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len) {
    if (!rm || !session_id || !out_path) return -1;

    const SessionResult* session = NULL;
    for (int i = 0; i < rm->num_sessions; i++) {
        if (strcmp(rm->sessions[i].session_id, session_id) == 0) {
            session = &rm->sessions[i];
            break;
        }
    }
    if (!session) return -1;

    snprintf(out_path, path_len, "%s/reports/%s_summary.csv", rm->base_output_dir, session_id);

    FILE* f = fopen(out_path, "w");
    if (!f) return -1;

    fprintf(f, "Metric,Value\n");
    fprintf(f, "Session ID,%s\n", session->session_id);
    fprintf(f, "Timestamp,%s\n", session->timestamp);
    fprintf(f, "Video Path,%s\n", session->video_path);
    fprintf(f, "Frames Processed,%d\n", session->frames_processed);
    fprintf(f, "Objects Detected,%d\n", session->objects_detected);
    fprintf(f, "Average FPS,%.2f\n", session->average_fps);
    fprintf(f, "Avg Processing Time (ms),%.2f\n", session->average_processing_time_ms);
    fprintf(f, "Start Time,%s\n", session->start_time);
    fprintf(f, "End Time,%s\n", session->end_time);
    fprintf(f, "Tracked Objects Count,%d\n", session->num_tracked_objects);
    fprintf(f, "Errors Count,%d\n", session->num_errors);

    fclose(f);
    return 0;
}

int result_manager_save_frame(const ResultManager* rm, const char* session_id, const uint8_t* frame_data, int width, int height, int frame_num) {
    if (!rm || !session_id || !frame_data) return -1;

    char frame_path[MAX_PATH_LEN * 3];
    snprintf(frame_path, sizeof(frame_path), "%s/frames/%s_frame_%05d.bmp",
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

    char meta_path[MAX_PATH_LEN * 3];
    snprintf(meta_path, sizeof(meta_path), "%s/reports/%s_frames.jsonl",
             rm->base_output_dir, session_id);

    FILE* f = fopen(meta_path, "a");
    if (!f) return -1;

    char timestamp[32];
    get_iso_time(timestamp, sizeof(timestamp));

    fprintf(f, "{\"frame\":%d,\"time\":\"%s\",\"detections\":%d,\"poses\":%d,"
               "\"faces\":%d,\"tracked\":%d,\"action\":%d}\n",
            frame_num, timestamp, num_detections, num_poses,
            num_faces, num_tracked, has_action);

    fclose(f);
    return 0;
}
