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
#define MIN_PERSON_ASPECT_RATIO     0.30f   /* tightened from 0.20: standing/walking person is 0.35-0.65 */
#define MAX_PERSON_ASPECT_RATIO     1.80f   /* tightened from 2.0: arms-out pose at most ~1.5-1.8 */
#define MIN_PERSON_HEIGHT_RATIO     0.04f   /* raised from 0.03: distant person still >4% of frame height */
#define MAX_PERSON_HEIGHT_RATIO     0.95f   /* allows close-up person up to 95% of frame height */
#define PERSON_NMS_IOU_THRESHOLD    0.30f   /* lowered from 0.40: more aggressive intra-model NMS.
                                               ByteTrack best practice: detector NMS should be tighter
                                               than tracker IOU to avoid passing duplicate boxes. */
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
        float iou  = config_get_float(config, "pose.iou_threshold", 0.40f);
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
            float conf = config_get_float(config, "face.confidence_threshold", 0.5f);
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
            pipeline->face_recognizer = arcface_recognizer_create(path_buf, 112, 112, 0.55f);
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
                pipeline->frame_diff->max_static_skip =
                    config_get_int(config, "frame_diff.max_static_skip", 20);
                pipeline->frame_diff->max_low_motion_skip =
                    config_get_int(config, "frame_diff.max_low_motion_skip", 5);
                pipeline->frame_diff->force_process_every =
                    config_get_int(config, "frame_diff.force_process_every", 30);
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
            float best_pose_iou = 0.35f;
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

                /* ── Count visible keypoints in each category ── */
                int n_upper = 0, n_left = 0, n_right = 0;
                float kpt_conf = 0.08f;  /* INT8 at 320×320 rarely >0.15 */
                {
                    for (int k = 0; k <= 10 && k < matched_pose->num_keypoints; k++) {
                        if (matched_pose->keypoints[k].confidence >= kpt_conf &&
                            matched_pose->keypoints[k].x >= 0.0f &&
                            matched_pose->keypoints[k].y >= 0.0f) n_upper++;
                    }
                    static const int left_chain[] = {1,3,5,7,9,11,13,15};
                    for (int k = 0; k < 8; k++) {
                        int idx = left_chain[k];
                        if (idx < matched_pose->num_keypoints &&
                            matched_pose->keypoints[idx].confidence >= kpt_conf &&
                            matched_pose->keypoints[idx].x >= 0.0f &&
                            matched_pose->keypoints[idx].y >= 0.0f) n_left++;
                    }
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
                if (person_h < 40.0f) continue;  /* too distant to detect face */

                /* Head ROI: top portion of person bbox */
                int roi_x = (int)(pb->x_min - person_h * 0.10f);
                int roi_y = (int)(pb->y_min);
                int roi_w = (int)(bbox_width(pb) + person_h * 0.20f);
                int roi_h = (int)(person_h * 0.40f);

                /* Clamp to frame */
                if (roi_x < 0) { roi_w += roi_x; roi_x = 0; }
                if (roi_y < 0) { roi_h += roi_y; roi_y = 0; }
                if (roi_x + roi_w > width)  roi_w = width  - roi_x;
                if (roi_y + roi_h > height) roi_h = height - roi_y;
                if (roi_w < 24 || roi_h < 24) continue;

                /* Crop ROI → contiguous buffer, then resize to model input.
                 * Cannot pass frame+offset directly to utils_resize_image because
                 * the source stride is full-frame width, not roi_w. */
                uint8_t* crop_temp = (uint8_t*)malloc((size_t)roi_w * roi_h * 3);
                if (!crop_temp) continue;
                utils_crop_image(frame, width, height,
                                 roi_x, roi_y, roi_w, roi_h, crop_temp);
                utils_resize_image(crop_temp, roi_w, roi_h,
                                   roi_buf, 320, 320, 3);
                free(crop_temp);

                /* Run face detection on ROI */
                FaceIdentity roi_faces[3];
                int n_roi = yolov5_face_detector_detect_faces(
                    face_detector, roi_buf, 320, 320, roi_faces, 3);

                /* Map ROI-space coords back to full-frame.
                 * The face detector's internal yolo_preprocess maps back via
                 * scale/pad_x/pad_y (letterbox). For ROI mode, input is already
                 * 320×320 → scale≈1.0, pad≈0.  The detector returns coords in
                 * the 320×320 space; we scale back to ROI dimensions then offset. */
                float sx = (float)roi_w / 320.0f;
                float sy = (float)roi_h / 320.0f;

                for (int f = 0; f < n_roi && num_detected < 20; f++) {
                    detected[num_detected] = roi_faces[f];
                    detected[num_detected].bbox.x_min =
                        roi_faces[f].bbox.x_min * sx + (float)roi_x;
                    detected[num_detected].bbox.y_min =
                        roi_faces[f].bbox.y_min * sy + (float)roi_y;
                    detected[num_detected].bbox.x_max =
                        roi_faces[f].bbox.x_max * sx + (float)roi_x;
                    detected[num_detected].bbox.y_max =
                        roi_faces[f].bbox.y_max * sy + (float)roi_y;
                    num_detected++;
                }
            }
            free(roi_buf);

            if (num_detected > 0) goto face_recognition;
        }
    }

    /* ── Fallback: full-frame detection ── */
    num_detected = yolov5_face_detector_detect_faces(
        face_detector, frame, width, height, detected, 20);

face_recognition:
    if (num_detected <= 0) return 0;

    int num_faces = 0;

    /* ── Per-person deduplication ──
     * At most one face per person: keep the highest-confidence face whose
     * center falls inside the person bbox. ROI-detected faces are already
     * per-person by construction, but full-frame fallback may have multiples. */
    int matched_persons[20] = {0};  /* track which person index got a face */
    int num_matched = 0;

    for (int i = 0; i < num_detected && num_faces < max_faces; i++) {
        /* Find which person this face belongs to */
        int best_person = -1;
        if (num_person_dets > 0) {
            float face_cx = bbox_center_x(&detected[i].bbox);
            float face_cy = bbox_center_y(&detected[i].bbox);
            for (int p = 0; p < num_person_dets; p++) {
                if (face_cx >= person_dets[p].bbox.x_min &&
                    face_cx <= person_dets[p].bbox.x_max &&
                    face_cy >= person_dets[p].bbox.y_min &&
                    face_cy <= person_dets[p].bbox.y_max) {
                    best_person = p;
                    break;
                }
            }
        }

        /* Skip faces not inside any person (when persons are known) */
        if (num_person_dets > 0 && best_person < 0) continue;

        /* Per-person dedup: keep only highest-confidence face per person */
        if (best_person >= 0) {
            bool already_matched = false;
            for (int m = 0; m < num_matched; m++) {
                if (matched_persons[m] == best_person) {
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

    /* ── Step 3: Face detection (SUPPLEMENT, reduced frequency) ──
     * In TRACKING mode, further reduce face detection frequency since
     * we already have confirmed person tracks. */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_FACE) && pipeline->face_detector) {
        int face_interval;
        if (pipeline->cascade_state == PIPELINE_CASCADE_TRACKING) {
            face_interval = (out_result->num_detections > 0) ? 30 : 120;
        } else {
            face_interval = (out_result->num_detections > 0) ? 10 : 120;
        }
        if (pipeline->frame_counter % face_interval == 0) {
            out_result->num_faces = detect_faces(pipeline->face_detector, pipeline->face_recognizer,
                                             frame_data, width, height,
                                             out_result->detections, out_result->num_detections,
                                             out_result->faces, MAX_FACES_PER_FRAME);
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
