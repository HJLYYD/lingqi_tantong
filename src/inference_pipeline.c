#include "inference_pipeline.h"
#include "logger.h"
#include "utils.h"
#include "keypoint_validator.h"
#include "tracking_manager.h"
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
 * STRATEGY: YOLOv8-pose is the PRIMARY person detector (trained specifically
 * for person detection + keypoints). YOLO11n is the SECONDARY fallback.
 * Face detection runs at reduced frequency as optional supplement.
 * ByteTrack does two-stage matching for final filtering. */
#define PERSON_CLASS_ID             0
#define MIN_PERSON_CONFIDENCE       0.15f   /* DFL geomean: uniform=0.0625, weak=0.10, clear=0.25+.
                                               Raised from 0.10 to 0.15 (2.4× uniform baseline) to
                                               suppress false positives from the broken cls head.
                                               Only detections with moderately peaked DFL distributions
                                               pass — effectively filters ~60% of noise. */
#define MIN_PERSON_AREA_RATIO       0.005f  /* raised from 0.002: tiny blobs <0.5% of frame are noise */
#define MAX_PERSON_AREA_RATIO       0.40f   /* lowered from 0.45: person >40% of frame is improbable */
#define MIN_PERSON_ASPECT_RATIO     0.30f   /* tightened from 0.20: standing/walking person is 0.35-0.65 */
#define MAX_PERSON_ASPECT_RATIO     1.80f   /* tightened from 2.0: arms-out pose at most ~1.5-1.8 */
#define MIN_PERSON_HEIGHT_RATIO     0.04f   /* raised from 0.03: distant person still >4% of frame height */
#define MAX_PERSON_HEIGHT_RATIO     0.75f   /* lowered from 0.80: very close person <75% of frame */
#define PERSON_NMS_IOU_THRESHOLD    0.30f   /* lowered from 0.40: more aggressive intra-model NMS.
                                               ByteTrack best practice: detector NMS should be tighter
                                               than tracker IOU to avoid passing duplicate boxes. */
#define MAX_FILTERED_DETECTIONS     20      /* lowered from 25: real scenes rarely exceed 15 people */
#define MERGE_IOU_DUPLICATE         0.35f   /* lowered from 0.50: aggressively merge overlapping detections
                                               from pose model + YOLO model.  Two models detecting the same
                                               person at slightly different scales should be merged, not
                                               both kept. 0.35 catches 85%+ of duplicate pairs. */
#define FINAL_NMS_IOU_THRESHOLD     0.35f   /* lowered from 0.45: final cleanup after merge.
                                               At 0.35, any two boxes with >35% overlap are considered
                                               the same person. This is the standard COCO NMS default. */

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

    /* PRIMARY: YOLOv8-pose for person detection + keypoints.
     * SECONDARY: YOLO11n for supplementary person detection.
     * SUPPLEMENT: Face detection/recognition (reduced frequency).
     * Action recognition runs from pose keypoints. */
    pipeline->enabled_stages = PIPELINE_ENABLE_DETECTION | PIPELINE_ENABLE_POSE
                             | PIPELINE_ENABLE_FACE | PIPELINE_ENABLE_ACTION;
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

    return pipeline;
}

void inference_pipeline_destroy(AIInferencePipeline* pipeline) {
    if (!pipeline) return;

    if (pipeline->detector) {
        yolov8_detector_destroy(pipeline->detector);
    }
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

    free(pipeline);
}

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config) {
    if (!pipeline || !model_dir) return -1;

    char path_buf[MAX_PATH_LEN];

    /*
     * EP slot allocation (SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1 → shared pool):
     *   1. YOLOv8-Pose — PRIMARY person detector + keypoints (runs every frame) → EP slot 1
     *   2. YOLO11      — SECONDARY person detector (runs every frame)           → EP slot 2
     *   3. Face det    — face detection (runs every 10 frames)                  → EP slot 3
     *   4. ArcFace     — face recognition (runs per-face, small model)          → EP slot 4
     *   5. ST-GCN      — CPU only (FP32 model, can't use EP)                    → CPU
     *
     * All 4 quantized models share one global EP intra-op thread pool.
     * Models run sequentially → no TCM contention.
     */

    /* ── Step 1: YOLOv8-Pose (PRIMARY person detector + keypoints, REQUIRED) ── */
    int pose_w = config ? config_get_int(config, "pose.input_size.0", 640) : 640;
    int pose_h = config ? config_get_int(config, "pose.input_size.1", 640) : 640;
    float pose_conf = config ? config_get_float(config, "pose.confidence_threshold", 0.3f) : 0.3f;
    float pose_iou = config ? config_get_float(config, "pose.iou_threshold", 0.45f) : 0.45f;

    const char* pose_model = config ? config_get_string(config, "pose.model_path", NULL) : NULL;
    if (pose_model) {
        strncpy(path_buf, pose_model, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
    } else {
        snprintf(path_buf, sizeof(path_buf), "%s/Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx", model_dir);
    }
    pipeline->pose_estimator = yolov8_pose_estimator_create(path_buf, pose_w, pose_h, pose_conf, pose_iou);
    if (pipeline->pose_estimator) {
        pipeline->models_loaded[1] = true;
        log_info("Pose estimator loaded (PRIMARY, EP): %s", path_buf);
    } else {
        log_error("PRIMARY pose model not found: %s", path_buf);
        return -1;
    }

    /* ── Step 2: YOLO11 (SECONDARY person detector) ── */
    int det_w = config ? config_get_int(config, "detection.input_size.0", 320) : 320;
    int det_h = config ? config_get_int(config, "detection.input_size.1", 320) : 320;
    float det_conf = config ? config_get_float(config, "detection.confidence_threshold", 0.25f) : 0.25f;
    float det_iou = config ? config_get_float(config, "detection.iou_threshold", 0.45f) : 0.45f;

    const char* det_model = config ? config_get_string(config, "detection.model_path", NULL) : NULL;
    if (det_model) {
        strncpy(path_buf, det_model, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
    } else {
        snprintf(path_buf, sizeof(path_buf), "%s/Human Recognition/yolo11n.q.onnx", model_dir);
    }
    pipeline->detector = yolov8_detector_create(path_buf, det_w, det_h, det_conf, det_iou);
    if (pipeline->detector) {
        pipeline->models_loaded[0] = true;
        log_info("Detector loaded (SECONDARY, EP): %s", path_buf);
    } else {
        log_warning("Secondary detector model not found: %s (pose-only mode)", path_buf);
    }

    int face_w = config ? config_get_int(config, "face.input_size.0", 320) : 320;
    int face_h = config ? config_get_int(config, "face.input_size.1", 320) : 320;
    float face_conf = config ? config_get_float(config, "face.confidence_threshold", 0.5f) : 0.5f;
    float face_iou = config ? config_get_float(config, "face.iou_threshold", 0.4f) : 0.4f;

    const char* face_det_model = config ? config_get_string(config, "face.detection_model_path", NULL) : NULL;
    if (face_det_model) {
        strncpy(path_buf, face_det_model, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
    } else {
        snprintf(path_buf, sizeof(path_buf), "%s/Face Recognition/yolov5n-face_cut.q.onnx", model_dir);
    }
    pipeline->face_detector = yolov5_face_detector_create(path_buf, face_w, face_h, face_conf, face_iou);
    if (pipeline->face_detector) {
        pipeline->models_loaded[2] = true;
        log_info("Face detector loaded (EP slot 3): %s", path_buf);
    } else {
        log_warning("Face detector model not found: %s", path_buf);
    }

    const char* face_rec_model = config ? config_get_string(config, "face.recognition_model_path", NULL) : NULL;
    if (face_rec_model) {
        strncpy(path_buf, face_rec_model, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
    } else {
        snprintf(path_buf, sizeof(path_buf), "%s/Face Recognition/arcface_mobilefacenet_cut.q.onnx", model_dir);
    }
    pipeline->face_recognizer = arcface_recognizer_create(path_buf, 112, 112, 0.55f);
    if (pipeline->face_recognizer) {
        pipeline->models_loaded[3] = true;
        log_info("Face recognizer loaded (EP slot 4): %s", path_buf);
    } else {
        log_warning("Face recognizer model not found: %s", path_buf);
    }

    int action_num_frames = config ? config_get_int(config, "action.num_frames", 300) : 300;
    int action_num_keypoints = config ? config_get_int(config, "action.num_keypoints", 17) : 17;
    int action_num_persons = config ? config_get_int(config, "action.num_persons", 2) : 2;
    int action_num_classes = config ? config_get_int(config, "action.num_classes", 60) : 60;
    float action_conf = config ? config_get_float(config, "action.confidence_threshold", 0.5f) : 0.5f;

    const char* action_model = config ? config_get_string(config, "action.model_path", NULL) : NULL;
    if (action_model) {
        strncpy(path_buf, action_model, sizeof(path_buf) - 1);
        path_buf[sizeof(path_buf) - 1] = '\0';
    } else {
        snprintf(path_buf, sizeof(path_buf), "%s/Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx", model_dir);
    }
    pipeline->action_recognizer = stgcn_action_recognizer_create(path_buf, action_num_frames,
                                                                   action_num_keypoints, action_num_persons,
                                                                   action_num_classes, action_conf);
    if (pipeline->action_recognizer) {
        pipeline->models_loaded[4] = true;
        log_info("Action recognizer loaded: %s", path_buf);
    } else {
        log_warning("Action recognizer model not found: %s", path_buf);
    }

    /* ── Step 6: Keypoint validator + cascade config ── */
    {
        KeypointValidatorConfig kv_cfg;
        memset(&kv_cfg, 0, sizeof(kv_cfg));
        kv_cfg.min_keypoints = config ? config_get_int(config, "detection.keypoint_min_count", KV_DEFAULT_MIN_KEYPOINTS) : KV_DEFAULT_MIN_KEYPOINTS;
        kv_cfg.kpt_conf_threshold = config ? config_get_float(config, "detection.keypoint_min_confidence", KV_DEFAULT_KPT_CONF_THRESHOLD) : KV_DEFAULT_KPT_CONF_THRESHOLD;
        kv_cfg.validity_threshold = config ? config_get_float(config, "detection.keypoint_validity_threshold", KV_DEFAULT_VALIDITY_THRESHOLD) : KV_DEFAULT_VALIDITY_THRESHOLD;
        kv_cfg.in_bbox_ratio = KV_DEFAULT_IN_BBOX_RATIO;
        kv_cfg.symmetry_tolerance = KV_DEFAULT_SYMMETRY_TOLERANCE;
        kv_cfg.debug_frame_interval = 0;  /* set to 30 for debug logging */
        pipeline->keypoint_validator = keypoint_validator_create(&kv_cfg);
        pipeline->keypoint_filter_enabled = (pipeline->keypoint_validator != NULL);

        /* Cascade config */
        pipeline->cascade_enabled = config ? config_get_bool(config, "detection.cascade_enabled", true) : true;
        pipeline->cascade_validation_interval = config ? config_get_int(config, "detection.cascade_validation_interval", CASCADE_VALIDATION_INTERVAL_DEFAULT) : CASCADE_VALIDATION_INTERVAL_DEFAULT;
        int tracking_w = config ? config_get_int(config, "detection.cascade_tracking_resolution.0", CASCADE_TRACKING_W_DEFAULT) : CASCADE_TRACKING_W_DEFAULT;
        int tracking_h = config ? config_get_int(config, "detection.cascade_tracking_resolution.1", CASCADE_TRACKING_H_DEFAULT) : CASCADE_TRACKING_H_DEFAULT;
        pipeline->cascade_tracking_w = tracking_w > 0 ? tracking_w : CASCADE_TRACKING_W_DEFAULT;
        pipeline->cascade_tracking_h = tracking_h > 0 ? tracking_h : CASCADE_TRACKING_H_DEFAULT;

        /* Enhanced fallback filter thresholds */
        pipeline->fallback_conf_threshold = config ? config_get_float(config, "detection.fallback_confidence", FILTER_FALLBACK_CONF_THRESHOLD) : FILTER_FALLBACK_CONF_THRESHOLD;
        pipeline->fallback_area_ratio_min = config ? config_get_float(config, "detection.fallback_area_ratio_min", FILTER_FALLBACK_AREA_RATIO_MIN) : FILTER_FALLBACK_AREA_RATIO_MIN;

        /* Cascade secondary detector interval (configurable, default 5 in TRACKING mode) */
        pipeline->cascade_secondary_interval = config ? config_get_int(config, "detection.cascade_secondary_interval", 5) : 5;

        /* Action recognition inference interval */
        pipeline->action_inference_interval = config ? config_get_int(config, "action.inference_interval", 10) : 10;

        log_info("Cascade: enabled=%d validation_interval=%d secondary_interval=%d tracking_res=%dx%d",
                 pipeline->cascade_enabled, pipeline->cascade_validation_interval,
                 pipeline->cascade_secondary_interval,
                 pipeline->cascade_tracking_w, pipeline->cascade_tracking_h);
        log_info("Action: inference_interval=%d frames", pipeline->action_inference_interval);
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
 * When no pose data is available (YOLO11-only detections), stricter geometry
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
                float kpt_conf = 0.30f;
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
                        output[filtered].is_partial_body = true;
                        output[filtered].num_visible_keypoints =
                            (n_left >= n_right && n_left >= 3) ? n_left : n_right;
                        output[filtered].confidence += TRACKING_PARTIAL_BODY_CONF_BOOST;
                    } else {
                        /* Upper-body passed — mark as partial */
                        output[filtered].is_partial_body = true;
                        output[filtered].num_visible_keypoints = n_upper;
                        output[filtered].confidence += TRACKING_PARTIAL_BODY_CONF_BOOST;
                    }
                } else {
                    /* Full-body passed */
                    output[filtered].is_partial_body = false;
                    output[filtered].num_visible_keypoints = 17;
                }
            }
            /* If no matching pose found, still accept detection
             * because it might be a person the pose model missed. */
        }

        output[filtered++] = *det;
    }

    if (filtered > 0) {
        utils_sort_detections_by_confidence(output, filtered);

        bool* suppressed = (bool*)calloc(filtered, sizeof(bool));
        if (suppressed) {
            int keep = 0;
            for (int i = 0; i < filtered; i++) {
                if (suppressed[i]) continue;
                output[keep++] = output[i];
                for (int j = i + 1; j < filtered; j++) {
                    if (suppressed[j]) continue;
                    if (bbox_iou(&output[i].bbox, &output[j].bbox) > PERSON_NMS_IOU_THRESHOLD) {
                        suppressed[j] = true;
                    }
                }
            }
            filtered = keep;
            free(suppressed);
        }
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

static int merge_detection_sets(Detection* primary, int num_primary,
                                 Detection* secondary, int num_secondary,
                                 Detection* out, int max_out,
                                 float iou_thresh) {
    int count = 0;
    bool* sec_used = (bool*)calloc((size_t)num_secondary, sizeof(bool));
    if (!sec_used) {
        int n = UTILS_MIN(num_primary + num_secondary, max_out);
        if (num_primary > 0) memcpy(out, primary, (size_t)num_primary * sizeof(Detection));
        if (num_secondary > 0 && num_primary < n) {
            memcpy(out + num_primary, secondary,
                   (size_t)UTILS_MIN(num_secondary, n - num_primary) * sizeof(Detection));
        }
        return n;
    }

    for (int i = 0; i < num_primary && count < max_out; i++) {
        out[count++] = primary[i];
    }

    for (int j = 0; j < num_secondary && count < max_out; j++) {
        bool duplicate = false;
        for (int i = 0; i < num_primary; i++) {
            if (bbox_iou(&secondary[j].bbox, &primary[i].bbox) > iou_thresh) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            out[count++] = secondary[j];
        }
    }

    free(sec_used);
    return count;
}

static int detect_faces(YOLOv5FaceDetector* face_detector, ArcFaceRecognizer* face_recognizer,
                        const uint8_t* frame, int width, int height,
                        const Detection* person_dets, int num_person_dets,
                        FaceIdentity* out_faces, int max_faces) {
    if (!face_detector || !frame || !out_faces) return 0;

    FaceIdentity detected[20];
    int num_detected = yolov5_face_detector_detect_faces(face_detector, frame, width, height, detected, 20);

    int num_faces = 0;
    for (int i = 0; i < num_detected && num_faces < max_faces; i++) {
        /* ── Face-to-person association ──
         * Only accept faces that fall within a detected person bounding box.
         * This prevents face recognition on false positives (e.g. faces on
         * posters, reflections, or non-person objects).  The face center must
         * be inside at least one person bbox. */
        bool inside_person = (num_person_dets == 0);  /* if no person dets, accept all */
        if (num_person_dets > 0) {
            float face_cx = bbox_center_x(&detected[i].bbox);
            float face_cy = bbox_center_y(&detected[i].bbox);
            for (int p = 0; p < num_person_dets; p++) {
                if (face_cx >= person_dets[p].bbox.x_min && face_cx <= person_dets[p].bbox.x_max &&
                    face_cy >= person_dets[p].bbox.y_min && face_cy <= person_dets[p].bbox.y_max) {
                    inside_person = true;
                    break;
                }
            }
        }
        if (!inside_person) continue;

        if (face_recognizer) {
            uint8_t* face_crop = (uint8_t*)malloc(112 * 112 * 3);
            if (!face_crop) continue;

            yolov5_face_detector_crop_face(face_detector, frame, width, height, &detected[i], face_crop, 112, 112);

            FaceIdentity identity = arcface_recognizer_recognize(face_recognizer, face_crop, 112, 112);
            identity.bbox = detected[i].bbox;
            identity.confidence = detected[i].confidence;
            identity.has_keypoints = detected[i].has_keypoints;
            memcpy(identity.keypoints, detected[i].keypoints, sizeof(detected[i].keypoints));

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
 * KEY FIX: In TRACKING mode, we now run the SECONDARY detector (YOLO11n) at
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

InferenceResult inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                                  const uint8_t* frame_data, int width, int height) {
    InferenceResult result;
    inference_result_init(&result);

    if (!pipeline || !frame_data) return result;

    int64_t start_time = utils_get_time_ms();
    pipeline->frame_counter++;

    /*
     * ── ADAPTIVE CASCADE STRATEGY (v2 — Multi-Person + Partial-Body Aware) ──
     *
     * PIPELINE_CASCADE_SEARCHING:  No confirmed tracks.
     *   → Run YOLOv8-Pose (PRIMARY) + YOLO11n (SECONDARY) at 640×640.
     *   → Apply keypoint anatomical + partial-body validation.
     *   → Maximum recall, higher latency.
     *
     * PIPELINE_CASCADE_TRACKING:   ≥1 confirmed track.
     *   → Run YOLOv8-Pose every frame (PRIMARY).
     *   → Run YOLO11n every `secondary_interval` frames (NEW: was skipped entirely).
     *   → This ensures new people entering the scene are promptly detected.
     *   → Lower latency than SEARCHING, but still multi-person capable.
     *
     * PIPELINE_CASCADE_VALIDATING: Periodic full-res check (1 frame).
     *   → Same as SEARCHING: both models at 640×640.
     *   → Triggered by: validation_interval timer OR multi-person detection trigger.
     */

    /* ── Determine cascade state for THIS frame ── */
    bool run_full_res = (pipeline->cascade_state == PIPELINE_CASCADE_SEARCHING ||
                         pipeline->cascade_state == PIPELINE_CASCADE_VALIDATING);

    /* ── NEW: In TRACKING mode, run secondary detector every N frames ──
     * This is the key fix for multi-person detection.  Previously, YOLO11n
     * was skipped entirely in TRACKING mode, meaning new people entering the
     * scene wouldn't be detected until the next VALIDATING frame (every 15
     * frames).  Now we run it at reduced frequency (every 5 frames) to catch
     * new entries while still saving most of the inference time. */
    bool run_secondary = run_full_res ||
        (pipeline->cascade_state == PIPELINE_CASCADE_TRACKING &&
         pipeline->frame_counter % pipeline->cascade_secondary_interval == 0);

    /* infer_w/infer_h retained for future reduced-resolution inference path */
    (void)run_full_res;  /* used for diagnostics */

    /* ── Step 1: Pose estimation (PRIMARY, always runs) ── */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_POSE) && pipeline->pose_estimator) {
        /*
         * When tracking at reduced resolution, we still pass the full frame
         * but the pose estimator will resize internally.  For a proper
         * performance gain, the model input size should also be reduced.
         * Since the ONNX model has a fixed input shape (640×640), the ORT
         * resize happens during preprocessing anyway.  The key saving comes
         * from SKIPPING the YOLO11n secondary model entirely.
         */
        result.num_poses = estimate_poses_fullframe(pipeline->pose_estimator,
                                                     frame_data, width, height,
                                                     result.poses, MAX_POSES_PER_FRAME);
        /* Convert pose bboxes to detections for tracking */
        Detection pose_dets[MAX_POSES_PER_FRAME];
        int num_pose_dets = poses_to_detections(result.poses, result.num_poses,
                                                 pose_dets, MAX_POSES_PER_FRAME);

        /* Filter pose detections with keypoint validation */
        Detection filtered_pose[MAX_POSES_PER_FRAME];
        int num_pose_filt = filter_detections(pose_dets, num_pose_dets, width, height,
                                               filtered_pose, MAX_POSES_PER_FRAME,
                                               result.poses, result.num_poses,
                                               pipeline->keypoint_validator,
                                               pipeline->keypoint_filter_enabled,
                                               pipeline->fallback_conf_threshold,
                                               pipeline->fallback_area_ratio_min);

        /* ── Step 2: YOLO11 detection (SECONDARY — only in SEARCHING/VALIDATING) ── */
        if (run_secondary &&
            (pipeline->enabled_stages & PIPELINE_ENABLE_DETECTION) && pipeline->detector) {
            Detection yolo_dets[MAX_DETECTIONS_PER_FRAME];
            int num_yolo = yolov8_detector_detect_persons(pipeline->detector, frame_data, width, height,
                                                           yolo_dets, MAX_DETECTIONS_PER_FRAME);
            /* YOLO-only detections have no pose data → stricter geometry thresholds */
            Detection filtered_yolo[MAX_DETECTIONS_PER_FRAME];
            int num_yolo_filt = filter_detections(yolo_dets, num_yolo, width, height,
                                                   filtered_yolo, MAX_DETECTIONS_PER_FRAME,
                                                   NULL, 0,  /* no pose data for YOLO dets */
                                                   pipeline->keypoint_validator,
                                                   pipeline->keypoint_filter_enabled,
                                                   pipeline->fallback_conf_threshold,
                                                   pipeline->fallback_area_ratio_min);

            /* ── NEW: Dual-consensus filter ──
             * When both models ran, each detection must either:
             *   (a) pass keypoint anatomical validation (already handled above), OR
             *   (b) be confirmed by BOTH models (IoU ≥ 0.35 between pose_det and yolo_det).
             * Solo detections without keypoint validation are rejected.
             * This eliminates objects that fool only one DFL-based detector. */
            Detection consensus_pose[MAX_POSES_PER_FRAME];
            int num_consensus_pose = 0;
            Detection consensus_yolo[MAX_DETECTIONS_PER_FRAME];
            int num_consensus_yolo = 0;
            bool yolo_has_consensus[MAX_DETECTIONS_PER_FRAME];
            memset(yolo_has_consensus, 0, sizeof(yolo_has_consensus));

            /* Find pose detections that have YOLO consensus.
             * Pose detections ALREADY passed keypoint validation — these are
             * trusted.  We just check for YOLO consensus to boost confidence
             * and mark which YOLO dets are confirmed. */
            for (int pi = 0; pi < num_pose_filt; pi++) {
                if (num_consensus_pose >= MAX_POSES_PER_FRAME) break;

                bool found = false;
                for (int yi = 0; yi < num_yolo_filt; yi++) {
                    float iou = bbox_iou(&filtered_pose[pi].bbox, &filtered_yolo[yi].bbox);
                    if (iou >= 0.35f) {
                        found = true;
                        yolo_has_consensus[yi] = true;
                        break;
                    }
                }
                /* Always keep pose detections (keypoint-validated) */
                consensus_pose[num_consensus_pose] = filtered_pose[pi];
                /* Boost confidence when BOTH models agree */
                if (found) {
                    consensus_pose[num_consensus_pose].confidence =
                        UTILS_MIN(consensus_pose[num_consensus_pose].confidence * 1.15f, 1.0f);
                }
                num_consensus_pose++;
            }
            /* Keep YOLO-only detections that have consensus with a pose detection */
            for (int yi = 0; yi < num_yolo_filt; yi++) {
                if (yolo_has_consensus[yi]) {
                    if (num_consensus_yolo < MAX_DETECTIONS_PER_FRAME) {
                        consensus_yolo[num_consensus_yolo++] = filtered_yolo[yi];
                    }
                }
                /* ── YOLO-only detections WITHOUT consensus are DISCARDED ──
                 * Rationale: YOLO11n's broken cls head cannot distinguish
                 * person from chair. Without pose keypoint confirmation,
                 * these are overwhelmingly false positives. */
            }

            /* Merge confirmed detections */
            Detection merged[MAX_DETECTIONS_PER_FRAME];
            result.num_detections = merge_detection_sets(consensus_pose, num_consensus_pose,
                                                          consensus_yolo, num_consensus_yolo,
                                                          merged, MAX_DETECTIONS_PER_FRAME,
                                                          MERGE_IOU_DUPLICATE);

            /* ── Final NMS pass after merge ── */
            {
                bool* final_suppressed = (bool*)calloc((size_t)result.num_detections, sizeof(bool));
                if (final_suppressed) {
                    int final_keep = 0;
                    for (int i = 0; i < result.num_detections; i++) {
                        if (final_suppressed[i]) continue;
                        merged[final_keep++] = merged[i];
                        for (int j = i + 1; j < result.num_detections; j++) {
                            if (final_suppressed[j]) continue;
                            if (bbox_iou(&merged[i].bbox, &merged[j].bbox) > FINAL_NMS_IOU_THRESHOLD) {
                                final_suppressed[j] = true;
                            }
                        }
                    }
                    result.num_detections = final_keep;
                    free(final_suppressed);
                }
            }
            memcpy(result.detections, merged, (size_t)result.num_detections * sizeof(Detection));

            log_debug("Cascade[%s]: pose=%d(filt=%d) yolo=%d(raw=%d) merged=%d",
                      run_full_res ? "FULL" : "TRACK",
                      num_pose_dets, num_pose_filt,
                      num_yolo_filt, num_yolo, result.num_detections);
        } else {
            /* TRACKING mode or no YOLO detector — use filtered pose detections directly */
            result.num_detections = num_pose_filt;
            memcpy(result.detections, filtered_pose, (size_t)result.num_detections * sizeof(Detection));

            log_debug("Cascade[%s]: pose=%d(filt=%d) — YOLO skipped",
                      run_full_res ? "FULL" : "TRACK",
                      num_pose_dets, num_pose_filt);
        }
    } else if ((pipeline->enabled_stages & PIPELINE_ENABLE_DETECTION) && pipeline->detector) {
        /* Fallback: YOLO11 only (no pose model available) */
        Detection raw_dets[MAX_DETECTIONS_PER_FRAME];
        int num_raw = yolov8_detector_detect_persons(pipeline->detector, frame_data, width, height,
                                                      raw_dets, MAX_DETECTIONS_PER_FRAME);
        result.num_detections = filter_detections(raw_dets, num_raw, width, height,
                                                   result.detections, MAX_DETECTIONS_PER_FRAME,
                                                   NULL, 0,
                                                   pipeline->keypoint_validator,
                                                   pipeline->keypoint_filter_enabled,
                                                   pipeline->fallback_conf_threshold,
                                                   pipeline->fallback_area_ratio_min);
        log_debug("YOLO only (no pose): raw=%d, filtered=%d", num_raw, result.num_detections);
    }

    /* ── Cascade state update (after detection, before face/action) ──
     * Uses actual confirmed track count from the tracker (set externally
     * before this call).  Falls back to detection-based approximation if
     * tracker state hasn't been synchronized. */
    {
        int ctc = pipeline->confirmed_track_count;
        int ttc = pipeline->total_track_count;
        cascade_update_state(pipeline, ctc,
                            result.num_detections, (ttc > 0 ? ttc : result.num_detections));
    }

    /* ── Step 3: Face detection (SUPPLEMENT, reduced frequency) ──
     * In TRACKING mode, further reduce face detection frequency since
     * we already have confirmed person tracks. */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_FACE) && pipeline->face_detector) {
        int face_interval;
        if (pipeline->cascade_state == PIPELINE_CASCADE_TRACKING) {
            face_interval = (result.num_detections > 0) ? 30 : 120;
        } else {
            face_interval = (result.num_detections > 0) ? 10 : 120;
        }
        if (pipeline->frame_counter % face_interval == 0) {
            result.num_faces = detect_faces(pipeline->face_detector, pipeline->face_recognizer,
                                             frame_data, width, height,
                                             result.detections, result.num_detections,
                                             result.faces, MAX_FACES_PER_FRAME);
        }
    }

    /* ── Step 4: Action recognition from pose keypoints ── */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_ACTION) && pipeline->action_recognizer
        && result.num_poses > 0) {
        for (int p = 0; p < result.num_poses; p++) {
            stgcn_action_recognizer_push_pose(pipeline->action_recognizer, &result.poses[p], width, height);
        }
        if (pipeline->frame_counter % pipeline->action_inference_interval == 0) {
            result.action = stgcn_action_recognizer_recognize(pipeline->action_recognizer);
            result.has_action = (result.action.num_actions > 0);
        }
    }

    result.processing_time_ms = (float)(utils_get_time_ms() - start_time);

    /* Per-frame timing and periodic summary */
    {
        static int64_t cum_ms = 0;
        static int cum_frames = 0;
        cum_ms += (int64_t)result.processing_time_ms;
        cum_frames++;

        log_debug("Frame %d [%s]: %.0f ms | poses=%d dets=%d faces=%d",
                  pipeline->frame_counter,
                  run_full_res ? "FULL" : "TRACK",
                  result.processing_time_ms,
                  result.num_poses, result.num_detections, result.num_faces);

        if (pipeline->frame_counter % 30 == 0 && cum_frames > 0) {
            float avg = (float)cum_ms / (float)cum_frames;
            log_info("Perf summary (last %d frames [%s]): avg %.0f ms/frame (%.1f FPS) | "
                     "poses=%d dets=%d faces=%d",
                     cum_frames,
                     pipeline->cascade_state == PIPELINE_CASCADE_TRACKING ? "TRACK" : "FULL",
                     avg, 1000.0f / (avg > 0 ? avg : 1.0f),
                     result.num_poses, result.num_detections, result.num_faces);
            cum_ms = 0;
            cum_frames = 0;
        }
    }

    return result;
}

void inference_pipeline_configure(AIInferencePipeline* pipeline, uint32_t stages) {
    if (!pipeline) return;
    pipeline->enabled_stages = stages;
    log_info("Pipeline configured: stages=0x%02X", stages);
}

void inference_pipeline_reset(AIInferencePipeline* pipeline) {
    if (!pipeline) return;
    pipeline->frame_counter = 0;
    log_info("AI inference pipeline reset");
}
