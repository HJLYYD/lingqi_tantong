#ifndef STGCN_ACTION_RECOGNIZER_H
#define STGCN_ACTION_RECOGNIZER_H

#include "core_types.h"
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STGCN_MAX_FRAMES        300
#define STGCN_NUM_KEYPOINTS     25   /* NTU-RGB+D standard (25 joints); COCO-17 is remapped on push */
#define STGCN_MAX_PERSONS       2
#define STGCN_NUM_CHANNELS      3
#define STGCN_MAX_CLASSES       60   /* max known class labels; actual model output auto-detected at load time */
#define STGCN_MAX_ACTIONS       MAX_ACTIONS_PER_FRAME

typedef struct OrtSession OrtSession;
typedef struct OrtInferenceContext OrtInferenceContext;

typedef struct {
    OrtSession* session;
    OrtInferenceContext* ctx;
    char input_name_pts[MAX_STRING_LEN];   /* primary input: skeleton keypoints [1, 3, T, V, M] */
    char input_name_mot[MAX_STRING_LEN];   /* secondary input: motion vectors [1, 2, T, V, M] */
    int num_frames;
    int num_keypoints;
    int num_persons;
    int num_classes;                        /* auto-detected from model output shape at load time */
    float confidence_threshold;

    float skeleton_buffer[STGCN_NUM_CHANNELS * STGCN_MAX_FRAMES * STGCN_NUM_KEYPOINTS * STGCN_MAX_PERSONS];
    int buffer_frames;
    int buffer_person_id;

    bool has_mot_input;                     /* true if model requires motion vector input (2-input ST-GCN) */
    bool model_loaded;

    /* ── Pre-allocated inference tensors (eliminates per-frame malloc/free) ── */
    float* prealloc_pts;                    /* [C * T * V * M] input keypoints tensor */
    float* prealloc_mot;                    /* [2 * T * V * M] input motion tensor */
    float* prealloc_padded;                 /* [C * T * V * M] padded pts for mot computation */
    size_t prealloc_pts_size;               /* element count of pts tensor */
    size_t prealloc_mot_size;               /* element count of mot tensor */
    bool prealloc_valid;                    /* true if all pre-allocations succeeded */

    /* ── Thread safety for async recognition ──
     * push_pose (inference thread, CPU1) and recognize (ST-GCN thread, CPU2)
     * both access skeleton_buffer.  Lock protects the buffer + buffer_frames. */
    pthread_mutex_t mutex;

    /* ── Latest async recognition result ──
     * Written by ST-GCN thread, read by postprocess thread.
     * Protected by a separate read-write convention:
     *   writer sets has_new_result=true after writing
     *   reader sets has_new_result=false after reading */
    ActionResult latest_action;
    bool has_new_action;
    pthread_mutex_t result_mutex;
} STGCNActionRecognizer;

STGCNActionRecognizer* stgcn_action_recognizer_create(const char* model_path,
                                                        int num_frames,
                                                        int num_keypoints,
                                                        int num_persons,
                                                        int num_classes,
                                                        float conf_thresh);
void stgcn_action_recognizer_destroy(STGCNActionRecognizer* recognizer);

bool stgcn_action_recognizer_load_model(STGCNActionRecognizer* recognizer, const char* model_path);

void stgcn_action_recognizer_push_pose(STGCNActionRecognizer* recognizer,
                                        const PoseEstimation* pose,
                                        int img_width, int img_height);

ActionResult stgcn_action_recognizer_recognize(STGCNActionRecognizer* recognizer);

void stgcn_action_recognizer_reset(STGCNActionRecognizer* recognizer);

/* ── Async recognition API (multi-threaded K1 pipeline) ── */

/** Lock, run ST-GCN inference, store result, unlock.
 *  Called from ST-GCN async thread (CPU 2). */
void stgcn_action_recognizer_run_async(STGCNActionRecognizer* recognizer);

/** Non-blocking read of latest async result.
 *  Returns true if a new result is available, copies into *out. */
bool stgcn_action_recognizer_get_latest(const STGCNActionRecognizer* recognizer,
                                        ActionResult* out);

const char* stgcn_get_action_name(int action_id);

#ifdef __cplusplus
}
#endif

#endif
