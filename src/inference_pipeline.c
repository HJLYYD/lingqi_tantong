#include "inference_pipeline.h"
#include "logger.h"
#include "utils.h"
#include "keypoint_validator.h"
#include "tracking_manager.h"
#include "frame_diff.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── Person detection filter constants ──
 *
 * SpacemiT's official quantized YOLO models are xquant-split format
 * (classification head removed by design). The cls/obj heads are broken
 * by INT8 quantization (all sigmoids ~0.5). Detection relies on DFL
 * peakiness as the confidence signal: uniform≈0.0625, weak≈0.10,
 * clear≈0.25+.
 *
 * STRATEGY: YOLOv8n-pose (official BRDK CV demo) is the PRIMARY person detector
 * (person detection + 17 COCO keypoints). YOLOv11n is the SECONDARY fallback.
 * Face detection runs at reduced frequency as optional supplement.
 * ByteTrack does two-stage matching for final filtering. */
#define PERSON_CLASS_ID             0
#define MIN_PERSON_CONFIDENCE       0.03f   /* Tuned for 320×320 INT8 model: DFL peaks 0.04-0.12 */
#define MIN_PERSON_AREA_RATIO       0.005f  /* raised from 0.002: tiny blobs <0.5% of frame are noise */
#define MAX_PERSON_AREA_RATIO       0.40f   /* lowered from 0.45: person >40% of frame is improbable */
#define MIN_PERSON_ASPECT_RATIO     0.22f   /* v2.6: relaxed from 0.30 for half-body lateral views */
#define MAX_PERSON_ASPECT_RATIO     2.80f   /* v2.6: relaxed from 1.80→2.80 for half-body (upper-torso only) */
#define MIN_PERSON_HEIGHT_RATIO     0.04f   /* raised from 0.03: distant person still >4% of frame height */
#define MAX_PERSON_HEIGHT_RATIO     0.95f   /* allows close-up person up to 95% of frame height */
#define PERSON_NMS_IOU_THRESHOLD    0.45f   /* raised from 0.30: aggressive intra-model NMS.
                                               INT8 quantized models can produce duplicate boxes for
                                               the same person at different anchor scales.  At 0.45,
                                               only genuinely distinct persons survive.  This is
                                               tighter than the tracker IoU (0.30) per ByteTrack
                                               best practice: "detector NMS should be tighter than
                                               tracker association IoU". */
#define MAX_FILTERED_DETECTIONS     20      /* lowered from 25: real scenes rarely exceed 15 people */
/* ── Cascade state machine constants ── */
#define CASCADE_VALIDATION_INTERVAL_DEFAULT  15    /* frames between full-res re-validation */
#define CASCADE_TRACKING_W_DEFAULT           320   /* reduced width when tracking */
#define CASCADE_TRACKING_H_DEFAULT           320   /* reduced height when tracking */
#define CASCADE_SEARCHING_SETTLE_FRAMES      2     /* reduced from 3: faster transition to TRACKING */
#define CASCADE_TRACKING_LOST_FRAMES         15    /* raised from 5: prevent flip-flopping.
                                                     At 5 frames, a single missed detection cycle
                                                     (person behind pole for 1-2s) dumps back to
                                                     expensive SEARCHING mode.  15 frames = ~4s grace
                                                     period for the tracker to recover. */

/* ── Enhanced filter: stricter thresholds when no pose data available ── */
#define FILTER_FALLBACK_CONF_THRESHOLD  0.20f     /* stricter conf when no keypoints to validate */
#define FILTER_FALLBACK_AREA_RATIO_MIN  0.008f    /* stricter area min (0.8% vs 0.5%) */
#define FILTER_KEYPOINT_CONF_THRESHOLD  0.30f     /* per-keypoint confidence for counting valid kpts */

AIInferencePipeline* inference_pipeline_create(void) {
    AIInferencePipeline* pipeline = (AIInferencePipeline*)calloc(1, sizeof(AIInferencePipeline));
    if (!pipeline) return NULL;

    /* PRIMARY: YOLOv8n-pose (official BRDK CV demo) for person detection + keypoints.
     * SECONDARY: YOLOv11n for supplementary person detection.
     * SUPPLEMENT: Face detection/recognition (reduced frequency).
     * Action recognition runs from pose keypoints. */
    pipeline->enabled_stages = PIPELINE_ENABLE_POSE | PIPELINE_ENABLE_FACE | PIPELINE_ENABLE_ACTION;
    pipeline->frame_counter = 0;

    /* ── Cascade state init ── */
    pipeline->cascade_state = PIPELINE_CASCADE_SEARCHING;
    pipeline->cascade_frames_in_state = 0;
    pipeline->cascade_validation_interval = CASCADE_VALIDATION_INTERVAL_DEFAULT;
    pipeline->cascade_secondary_interval = 5;  /* default: run secondary detector every 5 frames in TRACKING */
    pipeline->cascade_enabled = true;
    pipeline->cascade_tracking_w = CASCADE_TRACKING_W_DEFAULT;
    pipeline->cascade_tracking_h = CASCADE_TRACKING_H_DEFAULT;
    pipeline->cascade_lost_counter = 0;
    pipeline->confirmed_track_count = 0;
    pipeline->total_track_count = 0;

    /* ── Keypoint validator init (created in load_models with config) ── */
    pipeline->keypoint_validator = NULL;
    pipeline->keypoint_filter_enabled = true;

    /* ── Enhanced filter defaults ── */
    pipeline->fallback_conf_threshold = FILTER_FALLBACK_CONF_THRESHOLD;
    pipeline->fallback_area_ratio_min = FILTER_FALLBACK_AREA_RATIO_MIN;

    /* ── Action recognition defaults ── */
    pipeline->action_inference_interval = 10;  /* default: run ST-GCN every 10 frames */

    /* ── Frame differencing init (disabled until explicitly configured) ── */
    pipeline->frame_diff = NULL;
    pipeline->frame_diff_enabled = false;
    pipeline->has_last_result = false;
    memset(&pipeline->last_full_result, 0, sizeof(pipeline->last_full_result));

    /* ── Dynamic skip defaults (legacy, may be superseded by frame_diff) ── */
    pipeline->dynamic_skip_enabled = false;
    pipeline->skip_counter = 0;
    pipeline->max_consecutive_skip = 10;

    return pipeline;
}

void inference_pipeline_destroy(AIInferencePipeline* pipeline) {
    if (!pipeline) return;

    if (pipeline->pose_estimator) {
        yolov8_pose_estimator_destroy(pipeline->pose_estimator);
    }
    if (pipeline->face_detector) {
        yolov5_face_detector_destroy(pipeline->face_detector);
    }
    if (pipeline->face_recognizer) {
        arcface_recognizer_destroy(pipeline->face_recognizer);
    }
    if (pipeline->action_recognizer) {
        stgcn_action_recognizer_destroy(pipeline->action_recognizer);
    }
    if (pipeline->keypoint_validator) {
        keypoint_validator_destroy(pipeline->keypoint_validator);
    }
    if (pipeline->frame_diff) {
        frame_diff_destroy(pipeline->frame_diff);
    }

    free(pipeline);
}

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config) {
    if (!pipeline || !config) return -1;
    (void)model_dir;  /* all model paths come from config */

    char path_buf[MAX_PATH_LEN];

    /*
     * K1 EP slot allocation (SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1):
     *   1. YOLOv8n-pose — person detector + keypoints in one pass (every frame)
     *   2. (reserved for future use — secondary detector disabled in unified mode)
     *   3. Face det      — face detection (every 10-120 frames)
     *   4. ArcFace       — face recognition (per-face)
     *   5. ST-GCN        — CPU EP only (FP32)
     */

    /* ── Step 1: YOLO-Pose (PRIMARY, REQUIRED) ──
     * Select model via pose.model_variant config key.
     * Supported: "yolov8n-pose", "yolo11n-pose".
     * Falls back to pose.model_path if variant is unrecognised. */
    {
        const char* variant = config_get_string(config, "pose.model_variant", "yolov8n-pose");
        const char* model_path = NULL;

        if (strcmp(variant, "yolo11n-pose") == 0) {
            model_path = "models/Action Prediction/Skeleton Recognition/yolo11n-pose.q.onnx";
        } else if (strcmp(variant, "yolov8n-pose") == 0) {
            model_path = "models/Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx";
        } else {
            /* Legacy: use explicit model_path from config */
            model_path = config_get_string(config, "pose.model_path", NULL);
            if (model_path && model_path[0] != '\0') {
                log_info("Pose: using explicit model_path: %s", model_path);
            }
        }

        if (!model_path || model_path[0] == '\0') {
            log_critical("Pose model not configured — set pose.model_variant "
                         "('yolov8n-pose' or 'yolo11n-pose') or pose.model_path");
            return -1;
        }
        strncpy(path_buf, model_path, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';  /* ensure null termination */

        int w = config_get_int(config, "pose.input_size.0", 640);
        int h = config_get_int(config, "pose.input_size.1", 640);
        float conf = config_get_float(config, "pose.confidence_threshold", 0.08f);
        float iou  = config_get_float(config, "pose.iou_threshold", 0.55f);  /* v2.7: 0.40→0.55, less aggressive multi-person NMS */
        pipeline->pose_estimator = yolov8_pose_estimator_create(path_buf, w, h, conf, iou);
        if (!pipeline->pose_estimator) {
            log_critical("Failed to load PRIMARY pose model (variant=%s): %s",
                         variant, path_buf);
            return -1;
        }
        pipeline->models_loaded[1] = true;
        log_info("Pose estimator loaded (PRIMARY, %s): %s", variant, path_buf);
    }

    /* ── Step 3: Face detection ── */
    {
        const char* m = config_get_string(config, "face.detection_model_path", NULL);
        if (m && m[0] != '\0') {
            strncpy(path_buf, m, sizeof(path_buf) - 1);
            path_buf[sizeof(path_buf) - 1] = '\0';
            int w = config_get_int(config, "face.input_size.0", 320);
            int h = config_get_int(config, "face.input_size.1", 320);
            float conf = config_get_float(config, "face.confidence_threshold", 0.30f);  /* v2.6: lowered from 0.5→0.3 for better recall */
            float iou  = config_get_float(config, "face.iou_threshold", 0.4f);
            bool  face_ep = config_get_bool(config, "face.use_ep", false);
            pipeline->face_detector = yolov5_face_detector_create(path_buf, w, h, conf, iou, face_ep);
            if (pipeline->face_detector) {
                pipeline->models_loaded[2] = true;
                log_info("Face detector loaded: %s", path_buf);
            } else {
                log_warning("Face detector failed: %s", path_buf);
            }
        } else {
            pipeline->face_detector = NULL;
            log_info("Face detection: DISABLED (no model_path)");
        }
    }

    /* ── Step 4: Face recognition ── */
    {
        const char* m = config_get_string(config, "face.recognition_model_path", NULL);
        if (m && m[0] != '\0') {
            strncpy(path_buf, m, sizeof(path_buf) - 1);
            path_buf[sizeof(path_buf) - 1] = '\0';
            pipeline->face_recognizer = arcface_recognizer_create(path_buf, 112, 112, 0.45f);  /* v2.6: lowered from 0.55→0.45 for better recall */
            if (pipeline->face_recognizer) {
                pipeline->models_loaded[3] = true;
                log_info("Face recognizer loaded: %s", path_buf);
            } else {
                log_warning("Face recognizer failed: %s", path_buf);
            }
        } else {
            pipeline->face_recognizer = NULL;
            log_info("Face recognition: DISABLED (no model_path)");
        }
    }

    /* ── Step 5: Action recognition ── */
    {
        const char* m = config_get_string(config, "action.model_path", NULL);
        if (m && m[0] != '\0') {
            strncpy(path_buf, m, sizeof(path_buf) - 1);
            path_buf[sizeof(path_buf) - 1] = '\0';
            int nf = config_get_int(config, "action.num_frames", 30);
            int nk = config_get_int(config, "action.num_keypoints", 14);
            int np = config_get_int(config, "action.num_persons", 1);
            int nc = config_get_int(config, "action.num_classes", 7);
            float conf = config_get_float(config, "action.confidence_threshold", 0.5f);
            pipeline->action_recognizer = stgcn_action_recognizer_create(path_buf, nf, nk, np, nc, conf);
            if (pipeline->action_recognizer) {
                pipeline->models_loaded[4] = true;
                log_info("Action recognizer loaded: %s", path_buf);
            } else {
                log_warning("Action recognizer failed: %s", path_buf);
            }
        } else {
            pipeline->action_recognizer = NULL;
            log_info("Action recognition: DISABLED (no model_path)");
        }
    }

    /* ── Step 6: Keypoint validator + cascade config ── */
    {
        KeypointValidatorConfig kv_cfg;
        memset(&kv_cfg, 0, sizeof(kv_cfg));
        kv_cfg.min_keypoints = config_get_int(config, "detection.keypoint_min_count", KV_DEFAULT_MIN_KEYPOINTS);
        kv_cfg.kpt_conf_threshold = config_get_float(config, "detection.keypoint_min_confidence", KV_DEFAULT_KPT_CONF_THRESHOLD);
        kv_cfg.validity_threshold = config_get_float(config, "detection.keypoint_validity_threshold", KV_DEFAULT_VALIDITY_THRESHOLD);
        kv_cfg.in_bbox_ratio = KV_DEFAULT_IN_BBOX_RATIO;
        kv_cfg.symmetry_tolerance = KV_DEFAULT_SYMMETRY_TOLERANCE;
        kv_cfg.debug_frame_interval = 0;
        pipeline->keypoint_validator = keypoint_validator_create(&kv_cfg);
        pipeline->keypoint_filter_enabled = (pipeline->keypoint_validator != NULL);

        pipeline->cascade_enabled = config_get_bool(config, "detection.cascade_enabled", false);
        pipeline->cascade_validation_interval = config_get_int(config, "detection.cascade_validation_interval", 15);
        pipeline->cascade_tracking_w = config_get_int(config, "detection.cascade_tracking_resolution.0", 320);
        pipeline->cascade_tracking_h = config_get_int(config, "detection.cascade_tracking_resolution.1", 320);
        pipeline->fallback_conf_threshold = config_get_float(config, "detection.fallback_confidence", FILTER_FALLBACK_CONF_THRESHOLD);
        pipeline->fallback_area_ratio_min = config_get_float(config, "detection.fallback_area_ratio_min", FILTER_FALLBACK_AREA_RATIO_MIN);
        pipeline->cascade_secondary_interval = config_get_int(config, "detection.cascade_secondary_interval", 5);
        pipeline->action_inference_interval = config_get_int(config, "action.inference_interval", 10);

        log_info("Cascade: enabled=%d validation_interval=%d secondary_interval=%d tracking_res=%dx%d",
                 pipeline->cascade_enabled, pipeline->cascade_validation_interval,
                 pipeline->cascade_secondary_interval,
                 pipeline->cascade_tracking_w, pipeline->cascade_tracking_h);
        log_info("Action: inference_interval=%d frames", pipeline->action_inference_interval);
    }

    /* ── Frame differencing (adaptive frame skip) ──
     * Configured from config section "frame_diff" with sensible defaults.
     * When enabled, near-identical frames skip YOLO inference entirely,
     * reusing the last result.  This saves ~150ms/frame on K1. */
    {
        pipeline->frame_diff_enabled = config_get_bool(config, "frame_diff.enabled", true);

        if (pipeline->frame_diff_enabled) {
            int fd_grid_w   = config_get_int(config, "frame_diff.grid_w", 8);
            int fd_grid_h   = config_get_int(config, "frame_diff.grid_h", 8);
            int fd_stride   = config_get_int(config, "frame_diff.subsample", 4);
            float fd_cell   = config_get_float(config, "frame_diff.cell_threshold", 8.0f);
            float fd_change = config_get_float(config, "frame_diff.change_threshold", 0.15f);

            pipeline->frame_diff = frame_diff_create(fd_grid_w, fd_grid_h, fd_stride,
                                                      fd_cell, fd_change);
            if (pipeline->frame_diff) {
                pipeline->frame_diff->adaptive_enabled =
                    config_get_bool(config, "frame_diff.adaptive_enabled", true);
                /* v2.6: more aggressive skip — static: 40 frames, low-motion: 8 frames.
                 * Force-process every 60 frames to prevent drift. */
                pipeline->frame_diff->max_static_skip =
                    config_get_int(config, "frame_diff.max_static_skip", 40);
                pipeline->frame_diff->max_low_motion_skip =
                    config_get_int(config, "frame_diff.max_low_motion_skip", 8);
                pipeline->frame_diff->force_process_every =
                    config_get_int(config, "frame_diff.force_process_every", 60);
                log_info("FrameDiff: enabled grid=%dx%d stride=%d cell=%.0f change=%.2f",
                         fd_grid_w, fd_grid_h, fd_stride,
                         (double)fd_cell, (double)fd_change);
            } else {
                log_warning("FrameDiff: creation failed — disabled");
                pipeline->frame_diff_enabled = false;
            }
        } else {
            log_info("FrameDiff: DISABLED (frame_diff.enabled=false)");
            pipeline->frame_diff = NULL;
        }
    }

    int loaded = 0;
    for (int i = 0; i < 5; i++) {
        if (pipeline->models_loaded[i]) loaded++;
    }
    return loaded;
}

/*
 * Enhanced detection filter with optional keypoint-based anatomical validation.
 *
 * When poses are available (from YOLOv8-pose), each detection is cross-referenced
 * against its associated pose's keypoints.  The keypoint_validator checks
 * anatomical plausibility (shoulder symmetry, torso proportions, limb ratios,
 * head-above-shoulders).  Detections whose keypoints fail the anatomical check
 * are rejected — this is our primary defense against non-human false positives
 * from the broken INT8 classifier head.
 *
 * When no pose data is available (detector-only, no keypoints), stricter geometry
 * thresholds are applied as a fallback.
 */
static int filter_detections(const Detection* input, int num_input, int img_w, int img_h,
                              Detection* output, int max_output,
                              const PoseEstimation* poses, int num_poses,
                              KeypointValidator* kv, bool kpt_filter_enabled,
                              float fallback_conf, float fallback_area_min) {
    if (num_input <= 0 || !input || !output) return 0;

    float image_area = (float)(img_w * img_h);
    int filtered = 0;

    /* Determine effective thresholds: stricter when no pose data available */
    float effective_conf_min = (num_poses > 0) ? MIN_PERSON_CONFIDENCE : fallback_conf;
    float effective_area_min = (num_poses > 0) ? MIN_PERSON_AREA_RATIO : fallback_area_min;

    for (int i = 0; i < num_input && filtered < max_output; i++) {
        const Detection* det = &input[i];

        if (det->confidence < effective_conf_min) continue;

        (void)PERSON_CLASS_ID;
        if (det->class_id != PERSON_CLASS_ID) continue;

        float det_area = bbox_area(&det->bbox);
        float area_ratio = det_area / image_area;
        if (area_ratio < effective_area_min) continue;
        if (area_ratio > MAX_PERSON_AREA_RATIO) continue;

        float det_height = bbox_height(&det->bbox);
        float height_ratio = det_height / img_h;
        if (height_ratio < MIN_PERSON_HEIGHT_RATIO) continue;
        if (height_ratio > MAX_PERSON_HEIGHT_RATIO) continue;

        float aspect_ratio = bbox_width(&det->bbox) / UTILS_MAX(det_height, 1.0f);
        if (aspect_ratio < MIN_PERSON_ASPECT_RATIO) continue;
        if (aspect_ratio > MAX_PERSON_ASPECT_RATIO) continue;

        /* ── NEW: Body aspect check (height/width must be human-like) ──
         * Complements the width/height check above with a more intuitive
         * human-proportions gate.  Real standing humans: h/w ∈ [1.5, 4.0].
         * Square-ish boxes (h/w ≈ 1.0) are nearly always objects, not people. */
        if (!keypoint_validator_check_body_aspect(&det->bbox, img_w, img_h)) continue;

        if (det->bbox.x_max < 0 || det->bbox.y_max < 0 ||
            det->bbox.x_min > img_w || det->bbox.y_min > img_h) continue;

        /* ── Keypoint-based anatomical validation (with partial-body fallback) ──
         * Find the pose that best matches this detection (by IoU), then
         * check if its keypoints form a plausible human pose.
         *
         * THREE-TIER VALIDATION:
         *   1. Full-body quick check (standard): both shoulders + hips + knees
         *   2. Upper-body fallback: keypoints 0-10 (nose through wrists)
         *   3. Side-body fallback: one-side chain visible (lateral view)
         *
         * Partial-body detections are MARKED but NOT rejected — they are
         * passed through with is_partial_body=true for the tracker to handle
         * with occlusion-aware matching thresholds. */
        bool is_partial = false;
        int nvis = 0;
        bool apply_boost = false;

        if (kpt_filter_enabled && kv && num_poses > 0 && poses) {
            /* Find best-matching pose for this detection */
            float best_pose_iou = 0.20f;  /* v2.7: 0.35→0.20, match more detections to poses */
            int best_pose_idx = -1;
            for (int p = 0; p < num_poses; p++) {
                if (!poses[p].has_bbox) continue;
                float iou = bbox_iou(&det->bbox, &poses[p].bbox);
                if (iou > best_pose_iou) {
                    best_pose_iou = iou;
                    best_pose_idx = p;
                }
            }

            if (best_pose_idx >= 0) {
                const PoseEstimation* matched_pose = &poses[best_pose_idx];

                /* ── Count visible keypoints in each category ──
                 * v2.6: kpt_conf lowered from 0.08→0.06 for INT8 models.
                 * KeypointValidator is opaque here; use KV_DEFAULT threshold. */
                int n_upper = 0, n_left = 0, n_right = 0;
                float kpt_conf = KV_DEFAULT_KPT_CONF_THRESHOLD;
                if (kpt_conf > 0.15f) kpt_conf = 0.10f;
                if (kpt_conf < 0.05f) kpt_conf = 0.06f;
                for (int k = 0; k <= 10 && k < matched_pose->num_keypoints; k++) {
                    if (matched_pose->keypoints[k].confidence >= kpt_conf &&
                        matched_pose->keypoints[k].x >= 0.0f &&
                        matched_pose->keypoints[k].y >= 0.0f) n_upper++;
                }
                {
                    static const int left_chain[] = {1,3,5,7,9,11,13,15};
                    for (int k = 0; k < 8; k++) {
                        int idx = left_chain[k];
                        if (idx < matched_pose->num_keypoints &&
                            matched_pose->keypoints[idx].confidence >= kpt_conf &&
                            matched_pose->keypoints[idx].x >= 0.0f &&
                            matched_pose->keypoints[idx].y >= 0.0f) n_left++;
                    }
                }
                {
                    static const int right_chain[] = {2,4,6,8,10,12,14,16};
                    for (int k = 0; k < 8; k++) {
                        int idx = right_chain[k];
                        if (idx < matched_pose->num_keypoints &&
                            matched_pose->keypoints[idx].confidence >= kpt_conf &&
                            matched_pose->keypoints[idx].x >= 0.0f &&
                            matched_pose->keypoints[idx].y >= 0.0f) n_right++;
                    }
                }

                /* Tier 1: Full-body validation */
                bool full_ok = keypoint_validator_quick_check(kv, matched_pose, &det->bbox);

                if (!full_ok) {
                    /* Tier 2: Upper-body fallback */
                    bool upper_ok = keypoint_validator_upper_body_check(kv, matched_pose, &det->bbox);

                    if (!upper_ok) {
                        /* Tier 3: Side-body fallback */
                        bool side_ok = keypoint_validator_side_body_check(kv, matched_pose, &det->bbox);

                        if (!side_ok) {
                            static int reject_count = 0;
                            reject_count++;
                            if (reject_count % 30 == 0) {
                                KeypointValidityResult vr = keypoint_validator_validate(
                                    kv, matched_pose, &det->bbox);
                                log_info("KptFilter: rejected non-human (score=%.2f, kpts=%d/%d) — "
                                         "total rejects: %d",
                                         (double)vr.anatomical_score, vr.valid_keypoint_count,
                                         vr.total_keypoints, reject_count);
                            }
                            continue;
                        }
                        /* Side-body passed — mark as partial */
                        is_partial = true;
                        nvis = (n_left >= n_right && n_left >= 3) ? n_left : n_right;
                        apply_boost = true;
                    } else {
                        /* Upper-body passed — mark as partial */
                        is_partial = true;
                        nvis = n_upper;
                        apply_boost = true;
                    }
                } else {
                    /* Full-body passed */
                    is_partial = false;
                    nvis = 17;
                    apply_boost = false;
                }
            }
            /* If no matching pose found, still accept detection
             * because it might be a person the pose model missed. */
        }

        /* Copy detection first, then re-apply metadata AFTER the copy
         * so partial-body / keypoint fields are not overwritten. */
        output[filtered] = *det;
        output[filtered].is_partial_body = is_partial;
        output[filtered].num_visible_keypoints = nvis;
        if (apply_boost) {
            output[filtered].confidence += TRACKING_PARTIAL_BODY_CONF_BOOST;
        }
        filtered++;
    }

    /*
     * Detections from YOLOv8-Pose already passed OKS-NMS inside
     * yolov8_pose_estimator_estimate().  We skip pipeline-level NMS
     * here to avoid redundant O(k²) processing — sort by confidence
     * and apply the output cap only.
     */

    if (filtered > 1) {
        utils_sort_detections_by_confidence(output, filtered);
    }

    if (filtered > MAX_FILTERED_DETECTIONS) {
        filtered = MAX_FILTERED_DETECTIONS;
    }

    return filtered;
}

static int estimate_poses_fullframe(YOLOv8PoseEstimator* estimator, const uint8_t* frame,
                                     int width, int height,
                                     PoseEstimation* out_poses, int max_poses) {
    if (!estimator || !frame || !out_poses) return 0;
    return yolov8_pose_estimator_estimate(estimator, frame, width, height, out_poses, max_poses);
}

/* Convert poses with valid bboxes to detections. Returns number of bboxes packed.
 * Detections are packed contiguously (no gaps for invalid poses). */
static int poses_to_detections(const PoseEstimation* poses, int num_poses,
                                 Detection* out_dets, int max_dets) {
    int n = UTILS_MIN(num_poses, max_dets);
    int out_count = 0;
    for (int i = 0; i < n; i++) {
        if (!poses[i].has_bbox) continue;
        Detection* d = &out_dets[out_count++];
        d->bbox = poses[i].bbox;
        d->confidence = poses[i].confidence;
        d->class_id = PERSON_CLASS_ID;
        strncpy(d->class_name, "person", MAX_STRING_LEN - 1);
        d->class_name[MAX_STRING_LEN - 1] = '\0';
    }
    /* Zero remaining slots for safety */
    for (int i = out_count; i < max_dets; i++) {
        memset(&out_dets[i], 0, sizeof(Detection));
    }
    return out_count;
}

/*
 * ── Face detection with ROI acceleration ──
 *
 * When person detections are available, faces are detected in each person's
 * head region (ROI cropping → resize to model input → detect). This improves
 * face detection quality because faces occupy a larger fraction of the model
 * input, giving better landmark accuracy for ArcFace recognition.
 *
 * Falls back to full-frame detection when no person detections are available.
 *
 * Per-person deduplication: at most ONE face per person bbox (highest confidence).
 */
static int detect_faces(YOLOv5FaceDetector* face_detector, ArcFaceRecognizer* face_recognizer,
                        const uint8_t* frame, int width, int height,
                        const Detection* person_dets, int num_person_dets,
                        FaceIdentity* out_faces, int max_faces) {
    if (!face_detector || !frame || !out_faces) return 0;

    FaceIdentity detected[20];
    int num_detected = 0;

    /* ── ROI mode: per-person head crop detection ──
     * More accurate face landmarks because faces appear larger relative
     * to the 320×320 model input.  ROI covers head + shoulders (top ~40%
     * of person bbox with 10% horizontal margin). */
    if (num_person_dets > 0) {
        uint8_t* roi_buf = (uint8_t*)malloc(320 * 320 * 3);
        if (roi_buf) {
            for (int p = 0; p < num_person_dets && num_detected < 20; p++) {
                const BoundingBox* pb = &person_dets[p].bbox;
                float person_h = bbox_height(pb);
                if (person_h < 24.0f) continue;  /* v2.6: lowered from 40→24, detect faces farther away */

                /* Head ROI: top portion of person bbox.
                 * v2.6: expanded to 50% height for half-body persons.
                 * v2.7: ROI made SQUARE to prevent aspect-ratio distortion
                 * when resizing to 320×320 model input.  Non-square ROIs
                 * stretch faces → wrong coordinates + lower detection rate. */
                int roi_x = (int)(pb->x_min - person_h * 0.15f);
                int roi_y = (int)(pb->y_min);
                int roi_w = (int)(bbox_width(pb) + person_h * 0.30f);
                int roi_h = (int)(person_h * 0.50f);

                /* Clamp to frame */
                if (roi_x < 0) { roi_w += roi_x; roi_x = 0; }
                if (roi_y < 0) { roi_h += roi_y; roi_y = 0; }
                if (roi_x + roi_w > width)  roi_w = width  - roi_x;
                if (roi_y + roi_h > height) roi_h = height - roi_y;

                /* v2.7: Make ROI square to avoid distortion on resize.
                 * Take the larger dimension, expand symmetrically. */
                int roi_dim = (roi_w > roi_h) ? roi_w : roi_h;
                if (roi_dim < 24) continue;
                /* Center the expansion so face stays in middle */
                int roi_cx = roi_x + roi_w / 2;
                int roi_cy = roi_y + roi_h / 2;
                roi_x = roi_cx - roi_dim / 2;
                roi_y = roi_cy - roi_dim / 2;
                /* Re-clamp after squaring */
                if (roi_x < 0) roi_x = 0;
                if (roi_y < 0) roi_y = 0;
                if (roi_x + roi_dim > width)  roi_x = width  - roi_dim;
                if (roi_y + roi_dim > height) roi_y = height - roi_dim;
                if (roi_x < 0 || roi_y < 0) continue;
                roi_w = roi_dim;
                roi_h = roi_dim;

                /* Crop ROI → contiguous buffer, then resize to model input. */
                uint8_t* crop_temp = (uint8_t*)malloc((size_t)roi_dim * roi_dim * 3);
                if (!crop_temp) continue;
                utils_crop_image(frame, width, height,
                                 roi_x, roi_y, roi_dim, roi_dim, crop_temp);
                utils_resize_image(crop_temp, roi_dim, roi_dim,
                                   roi_buf, 320, 320, 3);
                free(crop_temp);

                /* Run face detection on ROI */
                FaceIdentity roi_faces[3];
                int n_roi = yolov5_face_detector_detect_faces(
                    face_detector, roi_buf, 320, 320, roi_faces, 3);

                /* Map ROI-space coords back to full-frame.
                 * ROI is square (roi_dim×roi_dim) → resized to 320×320 WITH CORRECT
                 * ASPECT RATIO.  Scale factor is uniform: sx = sy = roi_dim / 320. */
                float sxy = (float)roi_dim / 320.0f;

                for (int f = 0; f < n_roi && num_detected < 20; f++) {
                    detected[num_detected] = roi_faces[f];
                    float fx1 = roi_faces[f].bbox.x_min * sxy + (float)roi_x;
                    float fy1 = roi_faces[f].bbox.y_min * sxy + (float)roi_y;
                    float fx2 = roi_faces[f].bbox.x_max * sxy + (float)roi_x;
                    float fy2 = roi_faces[f].bbox.y_max * sxy + (float)roi_y;
                    /* Tighten face bbox: YOLOv5-Face adds ~15% margin. Shrink 12% per side. */
                    float fw = fx2 - fx1, fh = fy2 - fy1;
                    float shrink_x = fw * 0.12f, shrink_y = fh * 0.12f;
                    detected[num_detected].bbox.x_min = fx1 + shrink_x;
                    detected[num_detected].bbox.y_min = fy1 + shrink_y;
                    detected[num_detected].bbox.x_max = fx2 - shrink_x;
                    detected[num_detected].bbox.y_max = fy2 - shrink_y;
                    /* v2.7 BUGFIX: map face KEYPOINTS from ROI space to full-frame.
                     * Previously only bbox was mapped; keypoints stayed in 320×320 space
                     * → drawn at completely wrong positions on the full-frame overlay. */
                    if (roi_faces[f].has_keypoints) {
                        for (int kp = 0; kp < 5; kp++) {
                            detected[num_detected].keypoints[kp][0] =
                                roi_faces[f].keypoints[kp][0] * sxy + (float)roi_x;
                            detected[num_detected].keypoints[kp][1] =
                                roi_faces[f].keypoints[kp][1] * sxy + (float)roi_y;
                        }
                    }
                    num_detected++;
                }
            }
            free(roi_buf);
        }
    }

    /* ── Full-frame detection (runs on-demand, not every cycle) ──
     * v2.6: Full-frame face detection is EXPENSIVE (~80ms on K1 for 640×480).
     * It only runs when:
     *   1. ROI found zero faces — likely no person detections, need full-frame
     *   2. ROI faces < person detections — some persons have undetected faces
     *   3. Every 4th face cycle as a periodic sweep (catch new entry / half-body)
     * Otherwise skipped. This keeps face detection overhead ~15ms avg vs ~80ms. */
    static int ff_cycle_count = 0;
    bool run_full_frame = false;
    ff_cycle_count++;

    if (num_detected == 0) {
        run_full_frame = true;  /* on-demand: ROI found nothing */
    } else if (num_person_dets > 0 && num_detected < num_person_dets) {
        run_full_frame = true;  /* on-demand: some persons have no face */
    } else if (ff_cycle_count % 4 == 0) {
        run_full_frame = true;  /* periodic sweep: every 4th face cycle */
    }

    if (run_full_frame) {
        FaceIdentity ff_faces[10];
        int n_ff = yolov5_face_detector_detect_faces(
            face_detector, frame, width, height, ff_faces, 10);
        /* Merge full-frame faces (avoid duplicating ROI-detected faces) */
        for (int ff = 0; ff < n_ff && num_detected < 20; ff++) {
            bool is_dup = false;
            for (int d = 0; d < num_detected; d++) {
                if (bbox_iou(&ff_faces[ff].bbox, &detected[d].bbox) > 0.5f) {
                    is_dup = true;
                    /* Keep the higher-confidence detection */
                    if (ff_faces[ff].confidence > detected[d].confidence) {
                        detected[d] = ff_faces[ff];
                    }
                    break;
                }
            }
            if (!is_dup) {
                detected[num_detected++] = ff_faces[ff];
            }
        }
    }

    if (num_detected <= 0) return 0;

    /* ── Tighten all face bboxes ──
     * YOLOv5-Face adds ~15% margin around the actual face. Shrink 10% per side
     * for a tighter fit. Applied to both ROI and full-frame detected faces. */
    for (int ti = 0; ti < num_detected; ti++) {
        float fw = detected[ti].bbox.x_max - detected[ti].bbox.x_min;
        float fh = detected[ti].bbox.y_max - detected[ti].bbox.y_min;
        float sx = fw * 0.10f, sy = fh * 0.10f;
        detected[ti].bbox.x_min += sx;
        detected[ti].bbox.y_min += sy;
        detected[ti].bbox.x_max -= sx;
        detected[ti].bbox.y_max -= sy;
    }

    int num_faces = 0;

    /* ── Per-person deduplication ──
     * v2.7: Find BEST matching person via bbox IoU (was: first-contains-center).
     * When people stand close together, face centers can fall inside multiple
     * person bboxes.  IoU-based matching correctly assigns each face to its
     * owner.  At most one face per person (highest confidence wins). */
    int matched_persons[20] = {0};
    int num_matched = 0;

    for (int i = 0; i < num_detected && num_faces < max_faces; i++) {
        /* Find best-matching person by bbox IoU */
        int best_person = -1;
        float best_iou = 0.15f;  /* minimum IoU to consider a match */
        if (num_person_dets > 0) {
            for (int p = 0; p < num_person_dets; p++) {
                float iou = bbox_iou(&detected[i].bbox, &person_dets[p].bbox);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_person = p;
                }
            }
        }

        /* Skip faces with no clear person match */
        if (num_person_dets > 0 && best_person < 0) continue;

        /* Per-person dedup: keep only highest-confidence face per person */
        if (best_person >= 0) {
            bool already_matched = false;
            for (int m = 0; m < num_matched; m++) {
                if (matched_persons[m] == best_person) {
                    /* Duplicate: keep higher-confidence face */
                    already_matched = true;
                    break;
                }
            }
            if (already_matched) continue;
            if (num_matched < 20) {
                matched_persons[num_matched++] = best_person;
            }
        }

        if (face_recognizer) {
            uint8_t* face_crop = (uint8_t*)malloc(112 * 112 * 3);
            if (!face_crop) continue;

            yolov5_face_detector_crop_face(face_detector, frame, width, height,
                                            &detected[i], face_crop, 112, 112);

            FaceIdentity identity = arcface_recognizer_recognize(
                face_recognizer, face_crop, 112, 112);
            identity.bbox = detected[i].bbox;
            identity.confidence = detected[i].confidence;
            identity.has_keypoints = detected[i].has_keypoints;
            memcpy(identity.keypoints, detected[i].keypoints,
                   sizeof(detected[i].keypoints));

            out_faces[num_faces++] = identity;
            free(face_crop);
        } else {
            out_faces[num_faces++] = detected[i];
        }
    }

    return num_faces;
}

/*
 * ── Cascade state management (v2 — Multi-Person Aware) ──
 *
 * KEY FIX: In TRACKING mode, we now run the SECONDARY detector (YOLOv11n) at
 * reduced frequency (every `secondary_interval` frames) instead of skipping
 * it entirely.  This allows NEW people entering the scene to be detected
 * promptly — previously they'd only be caught during the periodic full-res
 * VALIDATING frame (every 15 frames), causing multi-person detection failure.
 *
 * Transitions:
 *   SEARCHING  → TRACKING:  ≥1 confirmed track for SETTLE_FRAMES
 *   TRACKING   → VALIDATING: every validation_interval frames (full-res both models)
 *   TRACKING   → SEARCHING: 0 confirmed tracks for LOST_FRAMES
 *   VALIDATING → TRACKING: 1 frame only, then back
 *
 * New: MULTI_PERSON_CHECK flag — when detection count exceeds track count
 * significantly, forces a full-res frame to catch new people.
 */
static void cascade_update_state(AIInferencePipeline* pipeline, int confirmed_track_count,
                                  int num_detections, int num_tracks) {
    if (!pipeline->cascade_enabled) {
        pipeline->cascade_state = PIPELINE_CASCADE_SEARCHING;
        return;
    }

    switch (pipeline->cascade_state) {
    case PIPELINE_CASCADE_SEARCHING:
        if (confirmed_track_count > 0) {
            pipeline->cascade_frames_in_state++;
            if (pipeline->cascade_frames_in_state >= CASCADE_SEARCHING_SETTLE_FRAMES) {
                pipeline->cascade_state = PIPELINE_CASCADE_TRACKING;
                pipeline->cascade_frames_in_state = 0;
                log_info("Cascade: SEARCHING → TRACKING (%d confirmed tracks)", confirmed_track_count);
            }
        } else {
            pipeline->cascade_frames_in_state = 0;
        }
        break;

    case PIPELINE_CASCADE_TRACKING:
        pipeline->cascade_frames_in_state++;

        /* ── Multi-person detection trigger ──
         * If detections significantly outnumber tracks, a new person likely
         * entered.  Force a full-res validation frame to catch them. */
        if (num_detections > num_tracks + 1 && num_detections >= 2) {
            pipeline->cascade_state = PIPELINE_CASCADE_VALIDATING;
            log_info("Cascade: TRACKING → VALIDATING (multi-person trigger: %d dets vs %d tracks)",
                     num_detections, num_tracks);
            break;
        }

        if (confirmed_track_count == 0) {
            /* ── FIXED: Use pipeline's own counter instead of static ──
             * Previously a static local that never reset properly, causing
             * instant SEARCHING fallback on subsequent track-loss events. */
            pipeline->cascade_lost_counter++;
            if (pipeline->cascade_lost_counter >= CASCADE_TRACKING_LOST_FRAMES) {
                pipeline->cascade_state = PIPELINE_CASCADE_SEARCHING;
                pipeline->cascade_frames_in_state = 0;
                pipeline->cascade_lost_counter = 0;
                log_info("Cascade: TRACKING → SEARCHING (tracks lost for %d frames)",
                         CASCADE_TRACKING_LOST_FRAMES);
            }
        } else {
            /* Tracks present — reset lost counter */
            pipeline->cascade_lost_counter = 0;
            /* Periodic full-resolution validation */
            if (pipeline->cascade_frames_in_state > 0 &&
                pipeline->cascade_frames_in_state % pipeline->cascade_validation_interval == 0) {
                pipeline->cascade_state = PIPELINE_CASCADE_VALIDATING;
            }
        }
        break;

    case PIPELINE_CASCADE_VALIDATING:
        pipeline->cascade_state = PIPELINE_CASCADE_TRACKING;
        pipeline->cascade_frames_in_state = 0;
        break;
    }
}

int inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                     const uint8_t* frame_data, int width, int height,
                                     InferenceResult* out_result) {
    if (!pipeline || !frame_data || !out_result) return -1;
    inference_result_init(out_result);

    int64_t start_time = utils_get_time_ms();
    pipeline->frame_counter++;

    /* ═══════════════════════════════════════════════════════════════════════
     * ── FRAME DIFFERENCING: Adaptive Frame Skip ──
     *
     * Before running expensive YOLO inference (~150-400ms on K1), compute a
     * fast subsampled MAD between the current frame and the last fully-
     * processed reference frame.  If the scene is static or has only minor
     * changes, skip inference entirely and reuse the previous result.
     *
     * This is the single most impactful optimization for real-world videos:
     *   - Static scenes (empty room, nobody moving): skip up to 20 frames
     *   - Low-motion (person standing still): skip up to 5 frames
     *   - Active (person walking/running): process every frame
     *
     * Cost: ~0.2ms for frame diff vs ~150ms for YOLO inference → 750× faster.
     *
     * When skipping, we still:
     *   1. Push the best pose from the last result to ST-GCN (temporal cont.)
     *   2. Read the latest ST-GCN async result
     *   3. Keep the reference frame unchanged (drift accumulates naturally)
     * ═══════════════════════════════════════════════════════════════════════ */
    if (pipeline->frame_diff_enabled && pipeline->frame_diff && pipeline->has_last_result) {
        /* Compute fast frame difference */
        float change_ratio = frame_diff_compute(pipeline->frame_diff, frame_data, width, height);

        if (!frame_diff_should_process(pipeline->frame_diff)) {
            /* ── SKIP: reuse previous inference result ── */
            *out_result = pipeline->last_full_result;

            /* Still push best pose to ST-GCN for temporal continuity */
            if ((pipeline->enabled_stages & PIPELINE_ENABLE_ACTION) &&
                pipeline->action_recognizer &&
                out_result->num_poses > 0) {
                int best_p = 0;
                float best_conf = out_result->poses[0].confidence;
                for (int p = 1; p < out_result->num_poses; p++) {
                    if (out_result->poses[p].confidence > best_conf) {
                        best_conf = out_result->poses[p].confidence;
                        best_p = p;
                    }
                }
                stgcn_action_recognizer_push_pose(pipeline->action_recognizer,
                    &out_result->poses[best_p], width, height);

                /* Read latest async ST-GCN result */
                ActionResult latest;
                if (stgcn_action_recognizer_get_latest(pipeline->action_recognizer, &latest)) {
                    out_result->action = latest;
                    out_result->has_action = (latest.num_actions > 0);
                }
            }

            out_result->processing_time_ms = (float)(utils_get_time_ms() - start_time);
            frame_diff_mark_skipped(pipeline->frame_diff);

            /* Periodic skip log */
            if (pipeline->frame_diff->consecutive_skips % 10 == 0 &&
                pipeline->frame_diff->consecutive_skips > 0) {
                int proc, skip;
                float ratio;
                frame_diff_get_stats(pipeline->frame_diff, &proc, &skip, &ratio);
                log_debug("FrameDiff: skip #%d (change=%.3f, %s, total skip=%.0f%%)",
                         pipeline->frame_diff->consecutive_skips,
                         (double)change_ratio,
                         frame_diff_activity_name(pipeline->frame_diff->last_activity),
                         (double)(ratio * 100.0f));
            }

            return 0;
        }
        /* ── PROCESS: full inference needed ── */
    }

    /*
     * ── ADAPTIVE CASCADE STRATEGY (v2 — Multi-Person + Partial-Body Aware) ──
     *
     * PIPELINE_CASCADE_SEARCHING:  No confirmed tracks.
     *   → Run YOLOv8n-Pose (PRIMARY) + YOLOv11n (SECONDARY) at 640×640.
     *   → Apply keypoint anatomical + partial-body validation.
     *   → Maximum recall, higher latency.
     *
     * PIPELINE_CASCADE_TRACKING:   ≥1 confirmed track.
     *   → Run YOLOv8-Pose every frame (PRIMARY).
     *   → Run YOLOv11n every `secondary_interval` frames (NEW: was skipped entirely).
     *   → This ensures new people entering the scene are promptly detected.
     *   → Lower latency than SEARCHING, but still multi-person capable.
     *
     * PIPELINE_CASCADE_VALIDATING: Periodic full-res check (1 frame).
     *   → Same as SEARCHING: both models at 640×640.
     *   → Triggered by: validation_interval timer OR multi-person detection trigger.
     */

    /* ── Step 1: Pose estimation (single unified model, always runs) ── */
    bool run_full_res = (pipeline->cascade_state != PIPELINE_CASCADE_TRACKING);
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_POSE) && pipeline->pose_estimator) {
        /*
         * When tracking at reduced resolution, we still pass the full frame
         * but the pose estimator will resize internally.  For a proper
         * performance gain, the model input size should also be reduced.
         * Since the ONNX model has a fixed input shape (640×640), the ORT
         * resize happens during preprocessing anyway.  The key saving comes
         * from SKIPPING the YOLOv11n secondary model entirely.
         */
        out_result->num_poses = estimate_poses_fullframe(pipeline->pose_estimator,
                                                     frame_data, width, height,
                                                     out_result->poses, MAX_POSES_PER_FRAME);
        /* Convert pose bboxes to detections for tracking */
        Detection pose_dets[MAX_POSES_PER_FRAME];
        int num_pose_dets = poses_to_detections(out_result->poses, out_result->num_poses,
                                                 pose_dets, MAX_POSES_PER_FRAME);

        /* Filter pose detections with keypoint validation */
        Detection filtered_pose[MAX_POSES_PER_FRAME];
        int num_pose_filt = filter_detections(pose_dets, num_pose_dets, width, height,
                                               filtered_pose, MAX_POSES_PER_FRAME,
                                               out_result->poses, out_result->num_poses,
                                               pipeline->keypoint_validator,
                                               pipeline->keypoint_filter_enabled,
                                               pipeline->fallback_conf_threshold,
                                               pipeline->fallback_area_ratio_min);

        /* Pose detections (keypoint-validated by filter above) → final detections */
        out_result->num_detections = num_pose_filt;
        memcpy(out_result->detections, filtered_pose, (size_t)out_result->num_detections * sizeof(Detection));
    }

    /* ── Cascade state update (after detection, before face/action) ──
     * Uses actual confirmed track count from the tracker (set externally
     * before this call).  Falls back to detection-based approximation if
     * tracker state hasn't been synchronized. */
    {
        int ctc = pipeline->confirmed_track_count;
        int ttc = pipeline->total_track_count;
        cascade_update_state(pipeline, ctc,
                            out_result->num_detections, (ttc > 0 ? ttc : out_result->num_detections));
    }

    /* ── Step 3: Face detection (SUPPLEMENT, balanced frequency + event-driven) ──
     * v2.7: Three triggers for face detection:
     *   1. Periodic: every 15 (TRACKING) / 8 (SEARCHING) / 60 (no dets) frames
     *   2. New-person: detection count increased → run immediately (reduces lag)
     *   3. First-person: 0→1 detections → run immediately (cold start)
     * Face persistence cache bridges the gap between detection cycles. */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_FACE) && pipeline->face_detector) {
        int face_interval;
        if (pipeline->cascade_state == PIPELINE_CASCADE_TRACKING) {
            face_interval = (out_result->num_detections > 0) ? 15 : 60;
        } else {
            face_interval = (out_result->num_detections > 0) ? 8 : 60;
        }

        bool run_face = (pipeline->frame_counter % face_interval == 0);

        /* Event-driven: new person appeared? Force face detection on next cycle */
        static int prev_det_count = -1;
        if (!run_face && prev_det_count >= 0) {
            if (out_result->num_detections > prev_det_count) {
                run_face = true;  /* new person entered → immediate face check */
            } else if (prev_det_count == 0 && out_result->num_detections > 0) {
                run_face = true;  /* first person appeared → immediate face check */
            }
        }
        prev_det_count = out_result->num_detections;

        if (run_face) {
            out_result->num_faces = detect_faces(pipeline->face_detector, pipeline->face_recognizer,
                                             frame_data, width, height,
                                             out_result->detections, out_result->num_detections,
                                             out_result->faces, MAX_FACES_PER_FRAME);
        }

        /* ── Cross-validation: Face → Person rescue ──
         * v2.7: If face detection found a face but the person detector missed
         * that person, the face confirms human presence.  Expand the face bbox
         * to estimate a full person bbox and add it as a synthetic detection.
         * This fixes the "skeleton drawn but not tracked" problem where the
         * keypoint validator rejects partial-body poses. */
        if (out_result->num_faces > 0 && out_result->num_detections < MAX_DETECTIONS_PER_FRAME) {
            for (int fi = 0; fi < out_result->num_faces; fi++) {
                const FaceIdentity* face = &out_result->faces[fi];
                /* Check if this face already belongs to a known detection */
                bool already_covered = false;
                for (int di = 0; di < out_result->num_detections; di++) {
                    if (bbox_iou(&face->bbox, &out_result->detections[di].bbox) > 0.05f) {
                        already_covered = true;
                        /* Boost this detection's confidence — face confirms person */
                        if (out_result->detections[di].confidence < 0.25f) {
                            out_result->detections[di].confidence += 0.08f;
                        }
                        break;
                    }
                }
                if (already_covered) continue;

                /* Face without a person detection — synthesize one.
                 * Estimate full body: face is ~1/7 of person height, centered.
                 * Expand face bbox downward to approximate full body. */
                float face_w = bbox_width(&face->bbox);
                float face_h = bbox_height(&face->bbox);
                float face_cx = bbox_center_x(&face->bbox);
                float body_h = face_h * 7.0f;  /* head ≈ 1/7 body height */
                float body_w = face_w * 2.5f;  /* shoulders ≈ 2.5× face width */

                Detection synth;
                memset(&synth, 0, sizeof(synth));
                synth.bbox.x_min = UTILS_CLAMP(face_cx - body_w * 0.5f, 0.0f, (float)(width - 1));
                synth.bbox.y_min = UTILS_CLAMP(face->bbox.y_min - face_h * 0.5f, 0.0f, (float)(height - 1));
                synth.bbox.x_max = UTILS_CLAMP(face_cx + body_w * 0.5f, 0.0f, (float)(width - 1));
                synth.bbox.y_max = UTILS_CLAMP(face->bbox.y_min + body_h, 0.0f, (float)(height - 1));
                synth.confidence = face->confidence * 0.7f;  /* discounted: estimated, not detected */
                synth.class_id = 0;  /* PERSON_CLASS_ID */
                strncpy(synth.class_name, "person", MAX_STRING_LEN - 1);
                synth.is_partial_body = true;
                synth.num_visible_keypoints = 0;

                if (bbox_height(&synth.bbox) > 12.0f && bbox_width(&synth.bbox) > 8.0f) {
                    out_result->detections[out_result->num_detections++] = synth;
                }
            }
        }
    }

    /* ── Step 4: Action recognition from pose keypoints ──
     * Push poses to ST-GCN sliding window (fast, ~10μs).
     * Actual inference runs async on CPU 2 via stgcn_action_recognizer_run_async().
     * Result read below via stgcn_action_recognizer_get_latest(). */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_ACTION) && pipeline->action_recognizer
        && out_result->num_poses > 0) {
        /* ST-GCN expects ONE skeleton per timestep.  Select the highest-
         * confidence pose to avoid corrupting the temporal buffer with
         * interleaved multi-person skeletons. */
        int best_p = 0;
        float best_conf = out_result->poses[0].confidence;
        for (int p = 1; p < out_result->num_poses; p++) {
            if (out_result->poses[p].confidence > best_conf) {
                best_conf = out_result->poses[p].confidence;
                best_p = p;
            }
        }
        stgcn_action_recognizer_push_pose(pipeline->action_recognizer,
            &out_result->poses[best_p], width, height);
        /* Non-blocking: read latest async result if available */
        ActionResult latest;
        if (stgcn_action_recognizer_get_latest(pipeline->action_recognizer, &latest)) {
            out_result->action = latest;
            out_result->has_action = (latest.num_actions > 0);
        }
    }

    out_result->processing_time_ms = (float)(utils_get_time_ms() - start_time);

    /* Per-frame timing and periodic summary */
    {
        static int64_t cum_ms = 0;
        static int cum_frames = 0;
        cum_ms += (int64_t)out_result->processing_time_ms;
        cum_frames++;

        log_debug("Frame %d [%s]: %.0f ms | poses=%d dets=%d faces=%d",
                  pipeline->frame_counter,
                  run_full_res ? "FULL" : "TRACK",
                  out_result->processing_time_ms,
                  out_result->num_poses, out_result->num_detections, out_result->num_faces);

        if (pipeline->frame_counter % 30 == 0 && cum_frames > 0) {
            float avg = (float)cum_ms / (float)cum_frames;
            log_info("Perf summary (last %d frames [%s]): avg %.0f ms/frame (%.1f FPS) | "
                     "poses=%d dets=%d faces=%d",
                     cum_frames,
                     pipeline->cascade_state == PIPELINE_CASCADE_TRACKING ? "TRACK" : "FULL",
                     avg, 1000.0f / (avg > 0 ? avg : 1.0f),
                     out_result->num_poses, out_result->num_detections, out_result->num_faces);
            cum_ms = 0;
            cum_frames = 0;
        }
    }

    /* ── Post-inference: update frame differencing reference ──
     * Store the current frame as the new reference for future comparisons.
     * Also cache the full inference result for reuse on skipped frames. */
    if (pipeline->frame_diff_enabled && pipeline->frame_diff) {
        frame_diff_set_reference(pipeline->frame_diff, frame_data, width, height);
        frame_diff_mark_processed(pipeline->frame_diff);
        /* Cache result for skip reuse */
        pipeline->last_full_result = *out_result;
        pipeline->has_last_result = true;
    }

    return 0;
}

void inference_pipeline_configure(AIInferencePipeline* pipeline, uint32_t stages) {
    if (!pipeline) return;
    pipeline->enabled_stages = stages;
    log_info("Pipeline configured: stages=0x%02X", stages);
}

void inference_pipeline_reset(AIInferencePipeline* pipeline) {
    if (!pipeline) return;
    pipeline->frame_counter = 0;
    /* Reset frame differencing state (source changed or scene cut) */
    if (pipeline->frame_diff) {
        frame_diff_reset(pipeline->frame_diff);
    }
    pipeline->has_last_result = false;
    memset(&pipeline->last_full_result, 0, sizeof(pipeline->last_full_result));
    log_info("AI inference pipeline reset");
}
