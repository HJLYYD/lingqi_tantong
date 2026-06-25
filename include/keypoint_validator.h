#ifndef KEYPOINT_VALIDATOR_H
#define KEYPOINT_VALIDATOR_H

#include "core_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ── Keypoint Anatomical Validator ──
 *
 * Validates whether a PoseEstimation's keypoint distribution is anatomically
 * plausible for a human.  The INT8-quantized YOLO models lack functional
 * classification heads, so DFL peakiness alone cannot distinguish humans
 * from chair-shaped objects.  This module checks the SPATIAL ARRANGEMENT
 * of keypoints — real humans have consistent limb proportions, symmetric
 * left/right pairs, and head-above-shoulders ordering.  Non-human objects
 * produce random keypoint scatter that fails these checks.
 *
 * Usage:
 *   KeypointValidator* kv = keypoint_validator_create(config);
 *   KeypointValidityResult r = keypoint_validator_validate(kv, &pose, &bbox);
 *   if (r.is_valid_human) { ... accept detection ... }
 *   keypoint_validator_destroy(kv);
 */

/* ── Tunable thresholds ── */
#define KV_DEFAULT_MIN_KEYPOINTS       3     /* lowered from 6: 320x320 model weak keypoints */
#define KV_DEFAULT_KPT_CONF_THRESHOLD  0.10f /* lowered from 0.30: INT8-quantized kpt confidence */
#define KV_DEFAULT_VALIDITY_THRESHOLD  0.25f /* lowered from 0.50: accept partial-body detections */
#define KV_DEFAULT_IN_BBOX_RATIO       0.70f
#define KV_DEFAULT_SYMMETRY_TOLERANCE  0.15f

/* Anatomical ratio bounds (torso / limb proportions) */
#define KV_TORSO_RATIO_MIN       0.40f
#define KV_TORSO_RATIO_MAX       2.80f
#define KV_LIMB_RATIO_MIN        0.35f
#define KV_LIMB_RATIO_MAX        2.80f
#define KV_SHOULDER_HIP_MIN      0.40f
#define KV_SHOULDER_HIP_MAX      2.50f

/* ── NEW: Body aspect ratio bounds ──
 * Human bounding boxes in typical surveillance/indoor views have
 * characteristic height/width ratios.  Standing adults: 2.5-4.0.
 * Sitting/crouching: 1.2-2.5.  Objects (chairs, boxes) often fall
 * outside this range or are nearly square. */
#define KV_BBOX_ASPECT_MIN       1.2f   /* sitting/crouching human */
#define KV_BBOX_ASPECT_MAX       4.5f   /* standing adult + hat/bag */
#define KV_BBOX_AREA_MIN_RATIO   0.003f /* ~0.3% of frame = distant person */
#define KV_BBOX_AREA_MAX_RATIO   0.45f  /* ~45% of frame = very close person */

/* ── NEW: Mandatory anatomical gates ──
 * At least one of these paired-limb groups must pass validation for
 * a detection to be considered human.  Prevents objects with scattered
 * keypoints from accumulating enough partial scores. */
#define KV_MANDATORY_SHOULDER_Y_MAX  0.30f  /* max shoulder dy / bbox_h for valid pair */
#define KV_MANDATORY_HEAD_AT_TOP     0.40f  /* nose must be in top 40% of bbox */
#define KV_MIN_SYMMETRIC_PAIRS       1      /* at least 1 symmetric pair must be visible */

/* COCO keypoint indices (17-keypoint model) */
#define KPT_NOSE          0
#define KPT_LEFT_EYE      1
#define KPT_RIGHT_EYE     2
#define KPT_LEFT_EAR      3
#define KPT_RIGHT_EAR     4
#define KPT_LEFT_SHOULDER 5
#define KPT_RIGHT_SHOULDER 6
#define KPT_LEFT_ELBOW    7
#define KPT_RIGHT_ELBOW   8
#define KPT_LEFT_WRIST    9
#define KPT_RIGHT_WRIST   10
#define KPT_LEFT_HIP      11
#define KPT_RIGHT_HIP     12
#define KPT_LEFT_KNEE     13
#define KPT_RIGHT_KNEE    14
#define KPT_LEFT_ANKLE    15
#define KPT_RIGHT_ANKLE   16

typedef struct {
    bool is_valid_human;         /* overall verdict */
    float anatomical_score;      /* 0.0 – 1.0 weighted composite */
    int valid_keypoint_count;    /* keypoints above confidence threshold */
    int total_keypoints;         /* total keypoints in pose */
    float failure_reasons[8];    /* per-check scores for diagnostics */
    int num_failures;            /* count of checks below threshold */
} KeypointValidityResult;

typedef struct {
    int min_keypoints;           /* KV_DEFAULT_MIN_KEYPOINTS */
    float kpt_conf_threshold;    /* KV_DEFAULT_KPT_CONF_THRESHOLD */
    float validity_threshold;    /* KV_DEFAULT_VALIDITY_THRESHOLD */
    float in_bbox_ratio;         /* KV_DEFAULT_IN_BBOX_RATIO */
    float symmetry_tolerance;    /* KV_DEFAULT_SYMMETRY_TOLERANCE */
    int debug_frame_interval;    /* 0 = off, N = log every N frames */
} KeypointValidatorConfig;

typedef struct KeypointValidator KeypointValidator;

/* ── Lifecycle ── */
KeypointValidator* keypoint_validator_create(const KeypointValidatorConfig* config);
void keypoint_validator_destroy(KeypointValidator* kv);

/* ── Core validation ──
 * Returns a detailed result struct.  The `is_valid_human` field is the
 * go/no-go verdict (anatomical_score >= validity_threshold). */
KeypointValidityResult keypoint_validator_validate(
    KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox);

/* ── Quick check (no allocation, no logging) ──
 * Returns true if the pose passes all anatomical checks.  Faster than
 * the full validate() because it skips the result struct and scoring. */
bool keypoint_validator_quick_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox);

/* ── Partial-body quick check (upper body only) ──
 * For occluded persons where lower body is not visible.  Checks the
 * upper-body keypoint subset (indices 0-10) for anatomical plausibility:
 * head-above-shoulders, shoulder symmetry, and minimum keypoint count.
 * Returns true if the upper-body keypoints form a plausible human pose. */
bool keypoint_validator_upper_body_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox);

/* ── Side-body quick check (one side visible) ──
 * For lateral-view occluded persons.  Checks that at least one side
 * (left or right) has a plausible keypoint chain: eye→shoulder→elbow→
 * wrist→hip→knee→ankle, with reasonable limb proportions.
 * Returns true if at least one side is anatomically plausible. */
bool keypoint_validator_side_body_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox);

/* ── Utility: count keypoints above confidence threshold ── */
int keypoint_validator_count_valid(const PoseEstimation* pose, float min_conf);

/* ── Utility: compute COCO-standard OKS (Object Keypoint Similarity)
 * between two poses.  Returns 0.0 – 1.0.  Used by pose NMS. ── */
float keypoint_validator_compute_oks(const PoseEstimation* a, const PoseEstimation* b);

/* ── NEW: Body aspect ratio check ──
 * Returns true if bbox has a human-like aspect ratio (height/width)
 * and area ratio relative to the full frame.  frame_w/frame_h are
 * the full image dimensions. */
bool keypoint_validator_check_body_aspect(const BoundingBox* bbox,
                                           int frame_w, int frame_h);

/* ── NEW: Detection consensus check ──
 * Two detections (from different models, e.g. YOLOv8-pose + YOLO11n)
 * are considered "consensus" if their IoU ≥ min_iou.  Returns the
 * combined confidence (max of the two) if consensus, 0.0 otherwise. */
float keypoint_validator_detection_consensus(const Detection* a, const Detection* b,
                                              float min_iou);

#ifdef __cplusplus
}
#endif

#endif
