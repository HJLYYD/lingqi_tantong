#ifndef RESULT_MANAGER_H
#define RESULT_MANAGER_H

#include "core_types.h"

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
const char* result_manager_end_session(ResultManager* rm);

int result_manager_save_json_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len);
int result_manager_save_csv_report(const ResultManager* rm, const char* session_id, char* out_path, int path_len);
int result_manager_save_frame(const ResultManager* rm, const char* session_id, const uint8_t* frame_data, int width, int height, int frame_num);

#ifdef __cplusplus
}
#endif

#endif
