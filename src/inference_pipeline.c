#include "inference_pipeline.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MIN_PERSON_CONFIDENCE       0.45f
#define MIN_PERSON_AREA_RATIO       0.003f
#define MAX_PERSON_AREA_RATIO       0.7f
#define MIN_PERSON_ASPECT_RATIO     0.25f
#define MAX_PERSON_ASPECT_RATIO     3.5f
#define MIN_PERSON_HEIGHT_RATIO     0.05f
#define MAX_PERSON_HEIGHT_RATIO     0.95f
#define PERSON_NMS_IOU_THRESHOLD    0.5f

AIInferencePipeline* inference_pipeline_create(bool use_onnx) {
    AIInferencePipeline* pipeline = (AIInferencePipeline*)calloc(1, sizeof(AIInferencePipeline));
    if (!pipeline) return NULL;

    pipeline->use_onnx = use_onnx;
    pipeline->enabled_stages = PIPELINE_ENABLE_ALL;
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
        scrfd_detector_destroy(pipeline->face_detector);
    }
    if (pipeline->face_recognizer) {
        arcface_recognizer_destroy(pipeline->face_recognizer);
    }

    free(pipeline);
}

int inference_pipeline_load_models(AIInferencePipeline* pipeline, const char* model_dir) {
    if (!pipeline || !model_dir) return -1;

    char path_buf[MAX_PATH_LEN];

    snprintf(path_buf, sizeof(path_buf), "%s/Human Recognition/yolov8n.onnx", model_dir);
    pipeline->detector = yolov8_detector_create(path_buf, 640, 640, 0.40f, 0.45f, pipeline->use_onnx);
    if (pipeline->detector) {
        pipeline->models_loaded[0] = true;
        log_info("Detector loaded: %s", path_buf);
    } else {
        log_error("Required detector model not found: %s", path_buf);
        return -1;
    }

    snprintf(path_buf, sizeof(path_buf), "%s/Action Prediction/Skeleton Recognition/yolov8n-pose.onnx", model_dir);
    pipeline->pose_estimator = yolov8_pose_estimator_create(path_buf, 640, 640, 0.3f, 0.45f, pipeline->use_onnx);
    if (pipeline->pose_estimator) {
        pipeline->models_loaded[1] = true;
        log_info("Pose estimator loaded: %s", path_buf);
    } else {
        log_warning("Pose model not found: %s", path_buf);
    }

    snprintf(path_buf, sizeof(path_buf), "%s/Face Recognition/scrfd_10g_bnkps.onnx", model_dir);
    pipeline->face_detector = scrfd_detector_create(path_buf, 640, 640, 0.5f, 0.4f, pipeline->use_onnx);
    if (pipeline->face_detector) {
        pipeline->models_loaded[2] = true;
        log_info("Face detector loaded: %s", path_buf);
    } else {
        log_warning("Face detector model not found: %s", path_buf);
    }

    snprintf(path_buf, sizeof(path_buf), "%s/Face Recognition/glintr100.onnx", model_dir);
    pipeline->face_recognizer = arcface_recognizer_create(path_buf, 112, 112, 0.55f, pipeline->use_onnx);
    if (pipeline->face_recognizer) {
        pipeline->models_loaded[3] = true;
        log_info("Face recognizer loaded: %s", path_buf);
    } else {
        log_warning("Face recognizer model not found: %s", path_buf);
    }

    int loaded = 0;
    for (int i = 0; i < 4; i++) {
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

    return filtered;
}

static int estimate_poses(YOLOv8PoseEstimator* estimator, const uint8_t* frame, int width, int height,
                          const Detection* detections, int num_detections,
                          PoseEstimation* out_poses, int max_poses) {
    if (!estimator || !frame || !detections || !out_poses) return 0;

    int num_poses = 0;

    for (int i = 0; i < num_detections && num_poses < max_poses; i++) {
        const Detection* det = &detections[i];
        int x1 = (int)det->bbox.x_min;
        int y1 = (int)det->bbox.y_min;
        int x2 = (int)det->bbox.x_max;
        int y2 = (int)det->bbox.y_max;

        int margin_x = (int)((x2 - x1) * 0.2f);
        int margin_y = (int)((y2 - y1) * 0.2f);

        int crop_x1 = UTILS_MAX(0, x1 - margin_x);
        int crop_y1 = UTILS_MAX(0, y1 - margin_y);
        int crop_x2 = UTILS_MIN(width, x2 + margin_x);
        int crop_y2 = UTILS_MIN(height, y2 + margin_y);

        int crop_w = crop_x2 - crop_x1;
        int crop_h = crop_y2 - crop_y1;

        if (crop_w <= 0 || crop_h <= 0) continue;

        uint8_t* person_crop = (uint8_t*)malloc(crop_w * crop_h * 3);
        if (!person_crop) continue;

        for (int y = 0; y < crop_h; y++) {
            for (int x = 0; x < crop_w; x++) {
                int src_idx = ((crop_y1 + y) * width + (crop_x1 + x)) * 3;
                int dst_idx = (y * crop_w + x) * 3;
                person_crop[dst_idx + 0] = frame[src_idx + 0];
                person_crop[dst_idx + 1] = frame[src_idx + 1];
                person_crop[dst_idx + 2] = frame[src_idx + 2];
            }
        }

        PoseEstimation person_poses[5];
        int n = yolov8_pose_estimator_estimate(estimator, person_crop, crop_w, crop_h, person_poses, 5);

        for (int p = 0; p < n && num_poses < max_poses; p++) {
            for (int k = 0; k < person_poses[p].num_keypoints; k++) {
                person_poses[p].keypoints[k].x += crop_x1;
                person_poses[p].keypoints[k].y += crop_y1;
            }
            if (person_poses[p].has_bbox) {
                person_poses[p].bbox.x_min += crop_x1;
                person_poses[p].bbox.y_min += crop_y1;
                person_poses[p].bbox.x_max += crop_x1;
                person_poses[p].bbox.y_max += crop_y1;
            }
            out_poses[num_poses++] = person_poses[p];
        }

        free(person_crop);
    }

    return num_poses;
}

static int detect_faces(SCRFDDetector* face_detector, ArcFaceRecognizer* face_recognizer,
                        const uint8_t* frame, int width, int height,
                        FaceIdentity* out_faces, int max_faces) {
    if (!face_detector || !frame || !out_faces) return 0;

    FaceIdentity detected[20];
    int num_detected = scrfd_detector_detect_faces(face_detector, frame, width, height, detected, 20);

    int num_faces = 0;
    for (int i = 0; i < num_detected && num_faces < max_faces; i++) {
        if (face_recognizer) {
            uint8_t* face_crop = (uint8_t*)malloc(112 * 112 * 3);
            if (!face_crop) continue;

            scrfd_detector_crop_face(face_detector, frame, width, height, &detected[i], face_crop, 112, 112);

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

    if ((pipeline->enabled_stages & PIPELINE_ENABLE_DETECTION) && pipeline->detector) {
        Detection raw_dets[MAX_DETECTIONS_PER_FRAME];
        int num_raw = yolov8_detector_detect_persons(pipeline->detector, frame_data, width, height,
                                                      raw_dets, MAX_DETECTIONS_PER_FRAME);
        result.num_detections = filter_detections(raw_dets, num_raw, width, height,
                                                   result.detections, MAX_DETECTIONS_PER_FRAME);
        log_debug("Raw detections: %d, After filtering: %d", num_raw, result.num_detections);
    }

    if ((pipeline->enabled_stages & PIPELINE_ENABLE_POSE) && pipeline->pose_estimator && result.num_detections > 0) {
        result.num_poses = estimate_poses(pipeline->pose_estimator, frame_data, width, height,
                                           result.detections, result.num_detections,
                                           result.poses, MAX_POSES_PER_FRAME);
    }

    if ((pipeline->enabled_stages & PIPELINE_ENABLE_FACE) && pipeline->face_detector) {
        result.num_faces = detect_faces(pipeline->face_detector, pipeline->face_recognizer,
                                         frame_data, width, height,
                                         result.faces, MAX_FACES_PER_FRAME);
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
