#ifndef TCN_ACTION_PREDICTOR_H
#define TCN_ACTION_PREDICTOR_H

#include "core_types.h"
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 1D-TCN Skeleton-based Action Prediction
 *
 * The model (1D-TCN_Skeleton_INT8.onnx) uses 1D temporal convolutions
 * (not graph convolutions like ST-GCN), so it can leverage SpacemiT EP
 * hardware acceleration (RVV 1.0 + IME) for INT8 inference.
 *
 * Input:  skeleton keypoints  [1, C, T, V]  (4D, 18 OpenPose keypoints, 300 frames)
 * Output: action class scores  [1, N]  (400 classes)
 *
 * COCO-17 keypoints (from YOLOv8-pose) are remapped to OpenPose-18
 * format at push time via COCO_TO_OPENPOSE18 mapping table.
 */

#define TCN_MAX_FRAMES        300
#define TCN_NUM_KEYPOINTS     18   /* OpenPose-18 joints; COCO-17 remapped on push */
#define TCN_MAX_PERSONS       2
#define TCN_NUM_CHANNELS      3
#define TCN_MAX_CLASSES       400
#define TCN_MAX_ACTIONS       MAX_ACTIONS_PER_FRAME

typedef struct OrtSession OrtSession;
typedef struct OrtInferenceContext OrtInferenceContext;

typedef struct {
    OrtSession* session;
    OrtInferenceContext* ctx;
    char input_name[MAX_STRING_LEN];
    int num_frames;
    int num_keypoints;
    int num_persons;
    int num_channels;
    int num_classes;                        /* auto-detected from model output shape at load time */
    int input_ndim;                         /* actual input rank detected from model (4 or 5) */
    int min_frames_to_predict;              /* minimum frames before running inference (default: 30) */
    float confidence_threshold;

    float skeleton_buffer[TCN_NUM_CHANNELS * TCN_MAX_FRAMES * TCN_NUM_KEYPOINTS * TCN_MAX_PERSONS];
    int buffer_frames;
    int buffer_person_id;

    bool model_loaded;

    /* ── Pre-allocated inference tensor (eliminates per-frame malloc/free) ── */
    float* prealloc_input;                  /* [C * T * V * M] input tensor */
    size_t prealloc_input_size;             /* element count */
    bool prealloc_valid;

    /* ── Thread safety for async prediction ── */
    pthread_mutex_t mutex;

    /* ── Latest async prediction result ── */
    ActionResult latest_action;
    bool has_new_action;
    pthread_mutex_t result_mutex;
} TcnActionPredictor;

TcnActionPredictor* tcn_action_predictor_create(const char* model_path,
                                                  int num_frames,
                                                  int num_keypoints,
                                                  int num_persons,
                                                  int num_classes,
                                                  float conf_thresh);
void tcn_action_predictor_destroy(TcnActionPredictor* predictor);

bool tcn_action_predictor_load_model(TcnActionPredictor* predictor, const char* model_path);

void tcn_action_predictor_push_pose(TcnActionPredictor* predictor,
                                     const PoseEstimation* pose,
                                     int img_width, int img_height);

ActionResult tcn_action_predictor_predict(TcnActionPredictor* predictor);

void tcn_action_predictor_reset(TcnActionPredictor* predictor);

/* ── Async prediction API (multi-threaded K1 pipeline) ── */

/** Lock, run TCN inference, store result, unlock.
 *  Called from TCN async thread. */
void tcn_action_predictor_run_async(TcnActionPredictor* predictor);

/** Non-blocking read of latest async result.
 *  Returns true if a new result is available, copies into *out. */
bool tcn_action_predictor_get_latest(const TcnActionPredictor* predictor,
                                     ActionResult* out);

/** Query current skeleton buffer fill count (thread-safe).
 *  Returns 0..num_frames.  Callers should skip run_async() until
 *  buffer is full to avoid zero-padded inference producing empty results. */
int tcn_action_predictor_get_buffer_fill(const TcnActionPredictor* predictor);

/** Set minimum frames required before running inference.
 *  When buffer_frames < num_frames, available frames are replicated to fill
 *  the temporal window (instead of zero-padding), enabling fast-start predictions.
 *  Default: 30 frames (~2s at 16 FPS).  Set to num_frames to require full buffer. */
void tcn_action_predictor_set_min_frames(TcnActionPredictor* predictor, int min_frames);

const char* tcn_get_action_name(int action_id);

#ifdef __cplusplus
}
#endif

#endif
