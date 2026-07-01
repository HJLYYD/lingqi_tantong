#ifndef INFERENCE_PIPELINE_H
#define INFERENCE_PIPELINE_H

#include "core_types.h"
#include <stdatomic.h>
#include "config_manager.h"
#include "yolov8_pose_estimator.h"
#include "yolov5_face_detector.h"
#include "arcface_recognizer.h"
#include "stgcn_action_recognizer.h"
#include "keypoint_validator.h"
#include "frame_diff.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIPELINE_ENABLE_DETECTION   0x01
#define PIPELINE_ENABLE_POSE        0x02
#define PIPELINE_ENABLE_FACE        0x04
#define PIPELINE_ENABLE_ACTION      0x08
#define PIPELINE_ENABLE_ALL         0x0F

/*
 * ── Adaptive Cascade States ──
 *
 * The pipeline dynamically switches between full-resolution search and
 * reduced-resolution tracking to save inference time.  When tracks are
 * confirmed (≥3 consecutive frames with tracked persons), it enters
 * TRACKING mode: run YOLOv8-Pose only at 320×320, skip YOLO11n.
 * Every cascade_validation_interval frames, it briefly returns to full
 * 640×640 to re-validate and catch new entries.
 */
typedef enum {
    PIPELINE_CASCADE_SEARCHING  = 0,  /* no confirmed tracks → both models at 640×640 */
    PIPELINE_CASCADE_TRACKING   = 1,  /* ≥1 confirmed track → pose-only at 320×320 */
    PIPELINE_CASCADE_VALIDATING = 2,  /* periodic full-res check (1 frame) */
} PipelineCascadeState;

typedef struct {
    YOLOv8PoseEstimator* pose_estimator;
    YOLOv5FaceDetector* face_detector;
    ArcFaceRecognizer* face_recognizer;
    STGCNActionRecognizer* action_recognizer;

    uint32_t enabled_stages;
    bool models_loaded[5];

    int frame_counter;

    /* ── Cascade state machine ── */
    PipelineCascadeState cascade_state;
    int cascade_frames_in_state;       /* consecutive frames in current state */
    int cascade_validation_interval;   /* frames between full-res validation checks */
    int cascade_secondary_interval;    /* frames between secondary detector runs in TRACKING mode */
    bool cascade_enabled;              /* master switch from config */
    int cascade_tracking_w;            /* reduced resolution width when tracking */
    int cascade_tracking_h;            /* reduced resolution height when tracking */
    int cascade_lost_counter;          /* consecutive frames with 0 confirmed tracks */
    /* BUGFIX: _Atomic for RISC-V weak memory (was volatile).
     * Written by PostProcess thread, read by Inference thread.
     * RVWMO requires acquire/release ordering — plain ld/sd (volatile)
     * does NOT provide cross-hart visibility.  C11 _Atomic with
     * memory_order_relaxed is sufficient here since stale-by-1-frame
     * is acceptable for cascade state machine. */
    _Atomic int confirmed_track_count;
    _Atomic int total_track_count;

    /* ── Keypoint validator ── */
    KeypointValidator* keypoint_validator;
    bool keypoint_filter_enabled;

    /* ── Enhanced filter config ── */
    float fallback_conf_threshold;     /* stricter conf when no pose data */
    float fallback_area_ratio_min;     /* stricter area min when no pose data */

    /* ── Action recognition config ── */
    int action_inference_interval;     /* frames between ST-GCN inference runs */

    /* ── NEW: Model warm-up ── */
    bool warmed_up;                    /* true after pipeline_warmup() completes */

    /* ── NEW: Dynamic frame skip ── */
    bool dynamic_skip_enabled;         /* enable frame skipping when saturated */
    int  skip_counter;                 /* consecutive skipped face/action frames */
    int  max_consecutive_skip;         /* max skips before forcing full frame */

    /* ── NEW: Frame differencing for adaptive frame skip ── */
    FrameDiff* frame_diff;             /* fast subsampled MAD frame comparison */
    bool      frame_diff_enabled;      /* master switch from config */
    InferResult last_full_result;      /* cached result from last processed frame */
    bool      has_last_result;         /* true after first processed frame */
} AIInferencePipeline;

AIInferencePipeline* inference_pipeline_create(void);
void inference_pipeline_destroy(AIInferencePipeline* pipeline);

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config);
/*
 * Process a single frame through the inference pipeline.
 * Writes results into the caller-provided out_result buffer (no large struct copy).
 * Returns 0 on success, -1 on error.
 */
int inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                     const uint8_t* frame_data, int width, int height,
                                     InferenceResult* out_result);

void inference_pipeline_configure(AIInferencePipeline* pipeline, uint32_t stages);
void inference_pipeline_reset(AIInferencePipeline* pipeline);

/**
 * Run model warm-up: 3 dummy inferences to populate TCM/NPU caches and
 * initialize ONNX Runtime session memory pools.
 * Must be called after load_models() and before first process_frame().
 * Skipped if already warmed up (idempotent).
 *
 * @return 0 = success, -1 = warmup failed
 */
int inference_pipeline_warmup(AIInferencePipeline* pipeline, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
