#ifndef RESULT_MANAGER_H
#define RESULT_MANAGER_H

#include "core_types.h"
#include "spatial_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RM_MAX_SESSIONS         128
#define RM_MAX_ERRORS           64
#define RM_MAX_TRACKED_OBJS     256

typedef struct {
    char type[MAX_STRING_LEN];
    char message[MAX_LOG_MSG_LEN];
    char timestamp[32];
} ErrorEntry;

typedef struct {
    int track_id;
    float height_meters;
    int position_count;
    int pose_count;
    float start_x, start_y, start_z;  /* 轨迹起点 (世界坐标) */
    float end_x, end_y, end_z;        /* 轨迹终点 (世界坐标) */
    float total_distance_m;           /* 累积路径长度 */
} TrackedObjectSummary;

typedef struct {
    char session_id[MAX_STRING_LEN];
    char timestamp[32];
    char video_path[MAX_PATH_LEN];
    char start_time[32];
    char end_time[32];
    int frames_processed;
    int objects_detected;
    float average_fps;
    float average_processing_time_ms;
    TrackedObjectSummary tracked_objects[RM_MAX_TRACKED_OBJS];
    int num_tracked_objects;
    ErrorEntry errors[RM_MAX_ERRORS];
    int num_errors;
    bool active;
} SessionResult;

typedef struct {
    char base_output_dir[MAX_PATH_LEN];
    SessionResult sessions[RM_MAX_SESSIONS];
    int num_sessions;
    SessionResult* current_session;
} ResultManager;

ResultManager* result_manager_create(const char* base_output_dir);
void result_manager_destroy(ResultManager* rm);

const char* result_manager_start_session(ResultManager* rm, const char* video_path);
void result_manager_update_session_stats(ResultManager* rm, int frames, int objects, float avg_fps, float avg_time_ms);
void result_manager_add_tracked_object(ResultManager* rm, int track_id, float height, int pos_count, int pose_count);
void result_manager_add_error(ResultManager* rm, const char* type, const char* message);
const char* result_manager_end_session(ResultManager* rm, SpatialLocalizationEngine* spatial);

int result_manager_save_json_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len);
int result_manager_save_csv_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len);
int result_manager_save_frame(const ResultManager* rm, const char* session_id, const uint8_t* frame_data, int width, int height, int frame_num);

/**
 * Save per-frame inference metadata as JSON lines (.jsonl).
 * Each line is a JSON object with frame number, detections, poses, faces, actions.
 */
int result_manager_save_frame_metadata(const ResultManager* rm, const char* session_id,
                                        int frame_num, int num_detections, int num_poses,
                                        int num_faces, int num_tracked, int has_action);

#ifdef __cplusplus
}
#endif

#endif
