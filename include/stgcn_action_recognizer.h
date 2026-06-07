#ifndef STGCN_ACTION_RECOGNIZER_H
#define STGCN_ACTION_RECOGNIZER_H

#include "core_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STGCN_MAX_FRAMES        300
#define STGCN_NUM_KEYPOINTS     17
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

const char* stgcn_get_action_name(int action_id);

#ifdef __cplusplus
}
#endif

#endif
