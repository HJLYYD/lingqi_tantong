#include "inference_pipeline.h"
#include "logger.h"
#include "utils.h"
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
#define MIN_PERSON_CONFIDENCE       0.08f   /* DFL geomean: uniform=0.0625, weak=0.10, clear=0.25+ */
#define MIN_PERSON_AREA_RATIO       0.002f
#define MAX_PERSON_AREA_RATIO       0.45f
#define MIN_PERSON_ASPECT_RATIO     0.20f
#define MAX_PERSON_ASPECT_RATIO     2.0f
#define MIN_PERSON_HEIGHT_RATIO     0.03f
#define MAX_PERSON_HEIGHT_RATIO     0.80f
#define PERSON_NMS_IOU_THRESHOLD    0.45f  /* tighter NMS to suppress overlapping boxes on same person */
#define MAX_FILTERED_DETECTIONS     25

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

    free(pipeline);
}

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir, const ConfigManager* config) {
    if (!pipeline || !model_dir) return -1;

    char path_buf[MAX_PATH_LEN];

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
        log_info("Detector loaded: %s", path_buf);
    } else {
        log_error("Required detector model not found: %s", path_buf);
        return -1;
    }

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
        log_info("Pose estimator loaded: %s", path_buf);
    } else {
        log_warning("Pose model not found: %s", path_buf);
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
        log_info("Face detector loaded: %s", path_buf);
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
        log_info("Face recognizer loaded: %s", path_buf);
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

    int loaded = 0;
    for (int i = 0; i < 5; i++) {
        if (pipeline->models_loaded[i]) loaded++;
    }
    return loaded;
}

static int filter_detections(const Detection* input, int num_input, int img_w, int img_h,
                              Detection* output, int max_output) {
    if (num_input <= 0 || !input || !output) return 0;

    float image_area = (float)(img_w * img_h);
    int filtered = 0;

    for (int i = 0; i < num_input && filtered < max_output; i++) {
        const Detection* det = &input[i];

        if (det->confidence < MIN_PERSON_CONFIDENCE) continue;

        /* All detections are forced to class_id=0 (person) by the detector.
         * Class labels from the quantized model are random noise — we filter
         * using DFL peakiness + person geometry instead. */
        (void)PERSON_CLASS_ID;  /* all detections are class_id=0 */
        if (det->class_id != PERSON_CLASS_ID) continue;  /* safety: skip non-person */

        float det_area = bbox_area(&det->bbox);
        float area_ratio = det_area / image_area;
        if (area_ratio < MIN_PERSON_AREA_RATIO) continue;
        if (area_ratio > MAX_PERSON_AREA_RATIO) continue;

        float det_height = bbox_height(&det->bbox);
        float height_ratio = det_height / img_h;
        if (height_ratio < MIN_PERSON_HEIGHT_RATIO) continue;
        if (height_ratio > MAX_PERSON_HEIGHT_RATIO) continue;

        float aspect_ratio = bbox_width(&det->bbox) / UTILS_MAX(det_height, 1.0f);
        if (aspect_ratio < MIN_PERSON_ASPECT_RATIO) continue;
        if (aspect_ratio > MAX_PERSON_ASPECT_RATIO) continue;

        if (det->bbox.x_max < 0 || det->bbox.y_max < 0 || det->bbox.x_min > img_w || det->bbox.y_min > img_h) continue;

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

    /* Hard cap: INT8-quantized models can produce excessive false positives.
     * A real scene rarely has >25 people visible at once. */
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

static void poses_to_detections(const PoseEstimation* poses, int num_poses,
                                 Detection* out_dets, int max_dets) {
    int n = UTILS_MIN(num_poses, max_dets);
    for (int i = 0; i < n; i++) {
        if (!poses[i].has_bbox) continue;
        Detection* d = &out_dets[i];
        d->bbox = poses[i].bbox;
        d->confidence = poses[i].confidence;
        d->class_id = PERSON_CLASS_ID;
        strncpy(d->class_name, "person", MAX_STRING_LEN - 1);
        d->class_name[MAX_STRING_LEN - 1] = '\0';
    }
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
                        FaceIdentity* out_faces, int max_faces) {
    if (!face_detector || !frame || !out_faces) return 0;

    FaceIdentity detected[20];
    int num_detected = yolov5_face_detector_detect_faces(face_detector, frame, width, height, detected, 20);

    int num_faces = 0;
    for (int i = 0; i < num_detected && num_faces < max_faces; i++) {
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

InferenceResult inference_pipeline_process_frame(AIInferencePipeline* pipeline,
                                                  const uint8_t* frame_data, int width, int height) {
    InferenceResult result;
    inference_result_init(&result);

    if (!pipeline || !frame_data) return result;

    int64_t start_time = utils_get_time_ms();
    pipeline->frame_counter++;

    /*
     * PIPELINE STRATEGY:
     * 1. PRIMARY: YOLOv8-pose on full frame → person bboxes + keypoints
     * 2. SECONDARY: YOLO11n detector (supplementary, merged with pose bboxes)
     * 3. SUPPLEMENT: Face detection/recognition at reduced frequency
     * 4. Action recognition from pose keypoints
     *
     * The pose model is trained specifically for person detection + keypoints.
     * Even with broken cls head (INT8 quantization), the DFL regression gives
     * usable bounding boxes. Keypoints may also be partially broken but bboxes
     * are the primary output for tracking.
     */

    /* ── Step 1: Pose estimation (PRIMARY person detector) ── */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_POSE) && pipeline->pose_estimator) {
        result.num_poses = estimate_poses_fullframe(pipeline->pose_estimator,
                                                     frame_data, width, height,
                                                     result.poses, MAX_POSES_PER_FRAME);
        /* Convert pose bboxes to detections for tracking */
        int num_pose_dets = 0;
        Detection pose_dets[MAX_POSES_PER_FRAME];
        poses_to_detections(result.poses, result.num_poses, pose_dets, MAX_POSES_PER_FRAME);
        for (int i = 0; i < result.num_poses && i < MAX_POSES_PER_FRAME; i++) {
            if (result.poses[i].has_bbox) num_pose_dets++;
        }
        num_pose_dets = UTILS_MIN(num_pose_dets, MAX_POSES_PER_FRAME);

        /* ── Step 2: YOLO11 detection (SECONDARY) ── */
        if ((pipeline->enabled_stages & PIPELINE_ENABLE_DETECTION) && pipeline->detector) {
            Detection yolo_dets[MAX_DETECTIONS_PER_FRAME];
            int num_yolo = yolov8_detector_detect_persons(pipeline->detector, frame_data, width, height,
                                                           yolo_dets, MAX_DETECTIONS_PER_FRAME);
            Detection filtered_yolo[MAX_DETECTIONS_PER_FRAME];
            int num_yolo_filt = filter_detections(yolo_dets, num_yolo, width, height,
                                                   filtered_yolo, MAX_DETECTIONS_PER_FRAME);

            /* Merge pose detections (primary) + YOLO detections (secondary) */
            Detection merged[MAX_DETECTIONS_PER_FRAME];
            result.num_detections = merge_detection_sets(pose_dets, num_pose_dets,
                                                          filtered_yolo, num_yolo_filt,
                                                          merged, MAX_DETECTIONS_PER_FRAME,
                                                          0.35f);  /* aggressive merge — same person from two models */
            memcpy(result.detections, merged, (size_t)result.num_detections * sizeof(Detection));

            log_debug("Pose dets: %d, YOLO dets: %d (raw:%d), Merged: %d",
                      num_pose_dets, num_yolo_filt, num_yolo, result.num_detections);
        } else {
            /* No YOLO detector — use pose detections directly */
            Detection filtered_pose[MAX_DETECTIONS_PER_FRAME];
            result.num_detections = filter_detections(pose_dets, num_pose_dets, width, height,
                                                       filtered_pose, MAX_DETECTIONS_PER_FRAME);
            memcpy(result.detections, filtered_pose, (size_t)result.num_detections * sizeof(Detection));
        }
    } else if ((pipeline->enabled_stages & PIPELINE_ENABLE_DETECTION) && pipeline->detector) {
        /* Fallback: YOLO11 only (no pose model available) */
        Detection raw_dets[MAX_DETECTIONS_PER_FRAME];
        int num_raw = yolov8_detector_detect_persons(pipeline->detector, frame_data, width, height,
                                                      raw_dets, MAX_DETECTIONS_PER_FRAME);
        result.num_detections = filter_detections(raw_dets, num_raw, width, height,
                                                   result.detections, MAX_DETECTIONS_PER_FRAME);
        log_debug("YOLO only: raw=%d, filtered=%d", num_raw, result.num_detections);
    }

    /* ── Step 3: Face detection (SUPPLEMENT, reduced frequency) ── */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_FACE) && pipeline->face_detector) {
        /* When people detected: every 5 frames. Without people: every 60 frames.
         * Face detection is expensive (640x640 model) and not critical for
         * the primary task of person detection + pose estimation. */
        int face_interval = (result.num_detections > 0) ? 5 : 60;
        if (pipeline->frame_counter % face_interval == 0) {
            result.num_faces = detect_faces(pipeline->face_detector, pipeline->face_recognizer,
                                             frame_data, width, height,
                                             result.faces, MAX_FACES_PER_FRAME);
        }
    }

    /* ── Step 4: Action recognition from pose keypoints ── */
    if ((pipeline->enabled_stages & PIPELINE_ENABLE_ACTION) && pipeline->action_recognizer
        && result.num_poses > 0) {
        for (int p = 0; p < result.num_poses; p++) {
            stgcn_action_recognizer_push_pose(pipeline->action_recognizer, &result.poses[p], width, height);
        }
        if (pipeline->frame_counter % 10 == 0) {
            result.action = stgcn_action_recognizer_recognize(pipeline->action_recognizer);
            result.has_action = (result.action.num_actions > 0);
        }
    }

    result.processing_time_ms = (float)(utils_get_time_ms() - start_time);
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
