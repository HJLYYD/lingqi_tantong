#include "tcn_action_predictor.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>
#include <errno.h>

#ifdef HAS_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include "ort_common.h"
#include "ort_inference_context.h"
#else
#error "tcn_action_predictor requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

/*
 * 1D-TCN action classes — auto-detected from model output size.
 *
 * The model (1D-TCN_Skeleton_INT8.onnx) outputs [1, N] where N is
 * auto-detected at load time (actual: 400 classes).
 *
 * Class labels for the first 60 classes are drawn from the NTU-RGB+D
 * taxonomy (common subset).  Classes beyond that return "action_N".
 */
#define TCN_MAX_KNOWN_CLASSES 60

static const char* KNOWN_ACTION_CLASSES[] = {
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "put something into pocket", "take something out of pocket",
    "jump up", "make a phone call/answer phone", "playing with phone/tablet",
    "typing on a keyboard", "pointing to something with finger", "taking a selfie",
    "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling", "touch head", "touch chest",
    "touch neck", "touch back", "touch cheek", "hit with fist", "hit with object",
    "push", "pull", "draw x", "draw circle clockwise", "draw circle counterclockwise",
    "draw triangle", "turn around clockwise", "turn around counterclockwise",
    "arm curl", "arm cross", "leg squat"
};

/* Generate action name for classes beyond the known taxonomy.
 * Buffer must be at least 16 bytes. */
static const char* tcn_get_action_name_buf(int action_id, char* buf, size_t bufsz) {
    if (action_id >= 0 && action_id < TCN_MAX_KNOWN_CLASSES) {
        return KNOWN_ACTION_CLASSES[action_id];
    }
    if (buf && bufsz > 0) {
        snprintf(buf, bufsz, "action_%d", action_id);
        return buf;
    }
    return "unknown";
}

/*
 * COCO-17 → OpenPose-18 keypoint index mapping.
 *
 * YOLOv8-pose outputs 17 COCO keypoints.  The 1D-TCN model expects
 * 18 keypoints in OpenPose format.
 *
 * OpenPose-18 joints:
 *   0:nose   1:neck    2:r_shoulder  3:r_elbow  4:r_wrist
 *   5:l_shoulder  6:l_elbow  7:l_wrist  8:r_hip  9:r_knee
 *   10:r_ankle  11:l_hip  12:l_knee  13:l_ankle  14:r_eye
 *   15:l_eye  16:r_ear  17:l_ear
 *
 * COCO-17 joints:
 *   0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
 *   5:l_shoulder 6:r_shoulder 7:l_elbow 8:r_elbow
 *   9:l_wrist 10:r_wrist 11:l_hip 12:r_hip
 *   13:l_knee 14:r_knee 15:l_ankle 16:r_ankle
 */
typedef struct {
    int src1;        /* primary COCO index, -1 = unused */
    int src2;        /* secondary COCO index, -1 = unused */
    float blend;     /* 1.0 = use src1 only; 0.5 = average src1+src2 */
} CocoToKptMapping;

static const CocoToKptMapping COCO_TO_OPENPOSE18[18] = {
    /* OpenPose 0: nose */
    {0,  -1, 1.0f},
    /* OpenPose 1: neck → midpoint of shoulders */
    {5,  6,  0.5f},
    /* OpenPose 2: right shoulder */
    {6,  -1, 1.0f},
    /* OpenPose 3: right elbow */
    {8,  -1, 1.0f},
    /* OpenPose 4: right wrist */
    {10, -1, 1.0f},
    /* OpenPose 5: left shoulder */
    {5,  -1, 1.0f},
    /* OpenPose 6: left elbow */
    {7,  -1, 1.0f},
    /* OpenPose 7: left wrist */
    {9,  -1, 1.0f},
    /* OpenPose 8: right hip */
    {12, -1, 1.0f},
    /* OpenPose 9: right knee */
    {14, -1, 1.0f},
    /* OpenPose 10: right ankle */
    {16, -1, 1.0f},
    /* OpenPose 11: left hip */
    {11, -1, 1.0f},
    /* OpenPose 12: left knee */
    {13, -1, 1.0f},
    /* OpenPose 13: left ankle */
    {15, -1, 1.0f},
    /* OpenPose 14: right eye */
    {2,  -1, 1.0f},
    /* OpenPose 15: left eye */
    {1,  -1, 1.0f},
    /* OpenPose 16: right ear */
    {4,  -1, 1.0f},
    /* OpenPose 17: left ear */
    {3,  -1, 1.0f},
};

TcnActionPredictor* tcn_action_predictor_create(const char* model_path,
                                                  int num_frames,
                                                  int num_keypoints,
                                                  int num_persons,
                                                  int num_classes,
                                                  float conf_thresh) {
    TcnActionPredictor* pred = (TcnActionPredictor*)calloc(1, sizeof(TcnActionPredictor));
    if (!pred) return NULL;

    pred->num_frames = (num_frames > 0 && num_frames <= TCN_MAX_FRAMES) ? num_frames : TCN_MAX_FRAMES;
    pred->num_keypoints = (num_keypoints > 0 && num_keypoints <= TCN_NUM_KEYPOINTS) ? num_keypoints : TCN_NUM_KEYPOINTS;
    pred->num_persons = (num_persons > 0 && num_persons <= TCN_MAX_PERSONS) ? num_persons : 1;
    pred->num_channels = TCN_NUM_CHANNELS;
    pred->num_classes = (num_classes > 0 && num_classes <= TCN_MAX_CLASSES) ? num_classes : TCN_MAX_CLASSES;
    pred->confidence_threshold = conf_thresh;
    pred->input_ndim = 5;   /* default 5D [1, C, T, V, M]; model load may override to 4D */
    pred->min_frames_to_predict = 30;  /* fast-start: run with only 30 frames (replicated) */
    pred->buffer_frames = 0;
    pred->buffer_person_id = -1;
    pred->model_loaded = false;

    /* Init mutexes for async prediction */
    if (pthread_mutex_init(&pred->mutex, NULL) != 0) {
        log_error("TcnActionPredictor: mutex init failed: %s", strerror(errno));
        free(pred);
        return NULL;
    }
    if (pthread_mutex_init(&pred->result_mutex, NULL) != 0) {
        pthread_mutex_destroy(&pred->mutex);
        free(pred);
        return NULL;
    }
    memset(&pred->latest_action, 0, sizeof(pred->latest_action));
    pred->has_new_action = false;

    if (!model_path || !tcn_action_predictor_load_model(pred, model_path)) {
        log_error("TcnActionPredictor: failed to load model %s", model_path ? model_path : "(null)");
        pthread_mutex_destroy(&pred->result_mutex);
        pthread_mutex_destroy(&pred->mutex);
        free(pred);
        return NULL;
    }

    return pred;
}

void tcn_action_predictor_destroy(TcnActionPredictor* predictor) {
    if (!predictor) return;
    ort_ctx_destroy(predictor->ctx);
    if (predictor->session) {
        ort_release_session(predictor->session);
    }
    free(predictor->prealloc_input);
    pthread_mutex_destroy(&predictor->mutex);
    pthread_mutex_destroy(&predictor->result_mutex);
    free(predictor);
}

bool tcn_action_predictor_load_model(TcnActionPredictor* predictor, const char* model_path) {
    if (!predictor || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    if (!ort_global_init()) {
        log_error("TcnActionPredictor: ORT runtime not initialized");
        return false;
    }

    /*
     * 1D-TCN uses standard 1D convolution operators (Conv1D, BN, ReLU) which
     * SpacemiT EP's IME hardware accelerator supports natively.  INT8-quantized
     * models benefit from RVV 1.0 + IME acceleration.
     *
     * Unlike ST-GCN (which uses graph convolution ops unsupported by IME),
     * 1D-TCN CAN run on SpacemiT EP.  We pass use_ep=true to leverage
     * hardware acceleration when available.
     */
    predictor->session = ort_create_session(model_path, 4, true);
    if (!predictor->session) {
        log_error("TcnActionPredictor: failed to create ONNX session for %s", model_path);
        return false;
    }

    const OrtApi* ort = ort_get_api();

    /* ── Detect input name + shape dynamically ── */
    OrtAllocator* allocator = NULL;
    OrtStatus* st_a = ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_a) ort->ReleaseStatus(st_a);

    /* Query number of inputs */
    size_t num_inputs = 0;
    OrtStatus* st_ni = ort->SessionGetInputCount(predictor->session, &num_inputs);
    if (st_ni) { ort->ReleaseStatus(st_ni); num_inputs = 1; }

    /* Store input name */
    strncpy(predictor->input_name, "input", sizeof(predictor->input_name) - 1);

    if (allocator && num_inputs > 0) {
        char* real_name = NULL;
        OrtStatus* s = ort->SessionGetInputName(predictor->session, 0, allocator, &real_name);
        if (s == NULL && real_name) {
            strncpy(predictor->input_name, real_name, sizeof(predictor->input_name) - 1);
            OrtStatus* sf = ort->AllocatorFree(allocator, real_name);
            if (sf) ort->ReleaseStatus(sf);
        } else {
            if (s) ort->ReleaseStatus(s);
        }
    }

    /* ── Auto-detect input shape from model signature ──
     * 1D-TCN expects [1, C, T, V] (4D) or [1, C, T, V, M] (5D).
     * Query the first input's shape and override the configured
     * num_frames / num_keypoints / num_persons to match. */
    if (allocator && num_inputs > 0) {
        OrtTypeInfo* type_info = NULL;
        OrtStatus* s_ti = ort->SessionGetInputTypeInfo(predictor->session, 0, &type_info);
        if (s_ti == NULL && type_info) {
            const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
            OrtStatus* s_cast = ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
            if (s_cast == NULL && tensor_info) {
                size_t nd = 0;
                { OrtStatus* _s = ort->GetDimensionsCount(tensor_info, &nd); if (_s) ort->ReleaseStatus(_s); }
                int64_t idims[5] = {0};
                size_t nd_read = (nd > 5) ? 5 : nd;
                { OrtStatus* _s = ort->GetDimensions(tensor_info, idims, nd_read); if (_s) ort->ReleaseStatus(_s); }

                int64_t model_C = 0, model_T = 0, model_V = 0, model_M = 1;
                int detected_ndim = 5;  /* default to 5D like ST-GCN */

                if (nd >= 5 && idims[0] == 1) {
                    /* batch-first 5D: [1, C, T, V, M] */
                    model_C = idims[1];
                    model_T = idims[2];
                    model_V = idims[3];
                    model_M = idims[4];
                    detected_ndim = 5;
                } else if (nd >= 4 && idims[0] == 1) {
                    /* batch-first 4D: [1, C, T, V] */
                    model_C = idims[1];
                    model_T = idims[2];
                    model_V = idims[3];
                    model_M = 1;
                    detected_ndim = 4;
                } else if (nd >= 4 && (idims[0] == 3 || idims[0] == 2)) {
                    /* channel-first 4D: [C, T, V] or [C, T, V, M] */
                    model_C = idims[0];
                    model_T = idims[1];
                    model_V = idims[2];
                    model_M = (nd >= 4) ? idims[3] : 1;
                    detected_ndim = (nd >= 4 && idims[3] > 1) ? 5 : 4;
                } else if (nd >= 3 && idims[0] > 1 && idims[0] <= 3) {
                    /* channel-first 3D: [C, T, V] */
                    model_C = idims[0];
                    model_T = idims[1];
                    model_V = idims[2];
                    model_M = 1;
                    detected_ndim = 4;
                } else {
                    log_warning("TCN: unexpected input shape with nd=%zu, "
                                "dims=[%lld,%lld,%lld,%lld,%lld] — using configured values",
                                nd,
                                (long long)idims[0], (long long)idims[1],
                                (long long)idims[2], (long long)idims[3],
                                (long long)idims[4]);
                }

                /* Store detected rank for inference-time shape matching */
                predictor->input_ndim = detected_ndim;

                if (model_T > 0 && model_T <= TCN_MAX_FRAMES &&
                    model_V > 0 && model_V <= TCN_NUM_KEYPOINTS &&
                    model_M > 0 && model_M <= TCN_MAX_PERSONS) {
                    if (predictor->num_frames != (int)model_T) {
                        log_info("TCN: overriding configured %d frames with model's %lld frames",
                                 predictor->num_frames, (long long)model_T);
                        predictor->num_frames = (int)model_T;
                    }
                    if (predictor->num_keypoints != (int)model_V) {
                        log_info("TCN: overriding configured %d keypoints with model's %lld keypoints",
                                 predictor->num_keypoints, (long long)model_V);
                        predictor->num_keypoints = (int)model_V;
                    }
                    if (predictor->num_persons != (int)model_M) {
                        log_info("TCN: overriding configured %d persons with model's %lld persons",
                                 predictor->num_persons, (long long)model_M);
                        predictor->num_persons = (int)model_M;
                    }
                    if (model_C > 0 && model_C != predictor->num_channels) {
                        log_info("TCN: overriding configured %d channels with model's %lld channels",
                                 predictor->num_channels, (long long)model_C);
                        predictor->num_channels = (int)model_C;
                    }
                } else if (model_T > 0) {
                    log_warning("TCN: model input shape exceeds buffer limits "
                                "(max T=%d V=%d M=%d) — using configured values",
                                TCN_MAX_FRAMES, TCN_NUM_KEYPOINTS, TCN_MAX_PERSONS);
                }
            }
            if (s_cast) ort->ReleaseStatus(s_cast);
        }
        if (s_ti) ort->ReleaseStatus(s_ti);
        ort->ReleaseTypeInfo(type_info);
    }

    /* ── Create inference context with MODEL-DETECTED dimensions ── */
    predictor->ctx = ort_ctx_create(predictor->session,
                                      predictor->num_frames,
                                      predictor->num_keypoints,
                                      predictor->num_channels);
    if (!predictor->ctx) {
        log_error("TcnActionPredictor: failed to create inference context");
        if (predictor->session) {
            ort_release_session(predictor->session);
            predictor->session = NULL;
        }
        return false;
    }
    predictor->ctx->input_name[0] = '\0';

    /* ── Detect output class count from model shape ── */
    size_t num_outputs = 0;
    OrtStatus* st_no = ort->SessionGetOutputCount(predictor->session, &num_outputs);
    if (st_no) { ort->ReleaseStatus(st_no); num_outputs = 1; }

    if (num_outputs > 0 && allocator) {
        OrtTypeInfo* type_info = NULL;
        OrtStatus* s_ti = ort->SessionGetOutputTypeInfo(predictor->session, 0, &type_info);
        if (s_ti == NULL && type_info) {
            const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
            OrtStatus* s_cast = ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
            if (s_cast == NULL && tensor_info) {
                size_t nd = 0;
                { OrtStatus* _s = ort->GetDimensionsCount(tensor_info, &nd); if (_s) ort->ReleaseStatus(_s); }
                int64_t odims[2] = {0};
                { OrtStatus* _s = ort->GetDimensions(tensor_info, odims, nd < 2 ? nd : 2); if (_s) ort->ReleaseStatus(_s); }

                int detected_classes = (nd >= 2 && odims[1] > 0) ? (int)odims[1] :
                                       (nd >= 1 && odims[0] > 0) ? (int)odims[0] : 0;

                if (detected_classes > 0 && detected_classes <= TCN_MAX_CLASSES) {
                    if (predictor->num_classes != detected_classes) {
                        log_info("TCN: overriding configured %d classes with model's actual %d classes",
                                 predictor->num_classes, detected_classes);
                        predictor->num_classes = detected_classes;
                    }
                }
            }
            if (s_cast) ort->ReleaseStatus(s_cast);
        }
        if (s_ti) ort->ReleaseStatus(s_ti);
        ort->ReleaseTypeInfo(type_info);
    }

    /* model_loaded only when ctx AND session are both valid */
    predictor->model_loaded = (predictor->ctx != NULL && predictor->session != NULL);

    /* ── Pre-allocate inference tensor ──
     * Size uses model-detected dimensions: C × T × V × M. */
    {
        size_t input_cnt = (size_t)predictor->num_channels * predictor->num_frames *
                            predictor->num_keypoints * predictor->num_persons;

        free(predictor->prealloc_input);
        predictor->prealloc_input = (float*)calloc(input_cnt, sizeof(float));

        if (predictor->prealloc_input) {
            predictor->prealloc_input_size = input_cnt;
            predictor->prealloc_valid = true;
        } else {
            log_error("TCN: failed to pre-allocate input tensor (%zu floats)", input_cnt);
            predictor->prealloc_valid = false;
        }
    }

    log_info("TCN action model loaded: %s (%.2f MB) classes=%d inputs=%zu "
             "shape=[1,%d,%d,%d,%d] input_name=[%s]",
             model_path, file_size / (1024.0 * 1024.0), predictor->num_classes,
             num_inputs,
             predictor->num_channels, predictor->num_frames,
             predictor->num_keypoints, predictor->num_persons,
             predictor->input_name);
    return true;
}

void tcn_action_predictor_push_pose(TcnActionPredictor* predictor,
                                     const PoseEstimation* pose,
                                     int img_width, int img_height) {
    if (!predictor || !pose) return;

    pthread_mutex_lock(&predictor->mutex);

    int T = predictor->num_frames;
    int V = predictor->num_keypoints;  /* model-expected keypoint count (typically 18 OpenPose) */
    int M = predictor->num_persons;
    int C = predictor->num_channels;

    /* Sliding window: shift out oldest frame if buffer is full */
    if (predictor->buffer_frames >= T) {
        int frame_size = C * V * M;
        memmove(predictor->skeleton_buffer,
                predictor->skeleton_buffer + frame_size,
                (size_t)(T - 1) * frame_size * sizeof(float));
        predictor->buffer_frames = T - 1;
    }

    int t = predictor->buffer_frames;
    float* frame_ptr = predictor->skeleton_buffer + (size_t)t * C * V * M;

    memset(frame_ptr, 0, (size_t)C * V * M * sizeof(float));

    float w = (img_width > 0) ? (float)img_width : 1.0f;
    float h = (img_height > 0) ? (float)img_height : 1.0f;

    int src_kpts = pose->num_keypoints;  /* COCO-17 from YOLOv8-pose */

    /* Map COCO-17 keypoints → model keypoints (OpenPose-18) using the
     * static mapping table.  For joints that map to COCO joints, use the
     * mapped position + confidence.  Unmapped joints stay zero. */
    for (int kpt_idx = 0; kpt_idx < V && kpt_idx < 18; kpt_idx++) {
        const CocoToKptMapping* m = &COCO_TO_OPENPOSE18[kpt_idx];
        int c1 = m->src1;
        int c2 = m->src2;

        float kx = 0.0f, ky = 0.0f, kconf = 0.0f;
        int valid_sources = 0;

        if (c1 >= 0 && c1 < src_kpts && pose->keypoints[c1].confidence > 0.0f) {
            kx += pose->keypoints[c1].x;
            ky += pose->keypoints[c1].y;
            kconf += pose->keypoints[c1].confidence;
            valid_sources++;
        }
        if (c2 >= 0 && c2 < src_kpts && pose->keypoints[c2].confidence > 0.0f && m->blend < 1.0f) {
            kx += pose->keypoints[c2].x;
            ky += pose->keypoints[c2].y;
            kconf += pose->keypoints[c2].confidence;
            valid_sources++;
        }

        if (valid_sources > 0 && kconf > 0.0f) {
            float inv = 1.0f / (float)valid_sources;
            /* Channel 0: normalized X */
            int idx = 0 * V * M + kpt_idx * M + 0;
            frame_ptr[idx] = (kx * inv) / w;
            /* Channel 1: normalized Y */
            idx = 1 * V * M + kpt_idx * M + 0;
            frame_ptr[idx] = (ky * inv) / h;
            /* Channel 2: keypoint confidence (average of source confidences) */
            idx = 2 * V * M + kpt_idx * M + 0;
            frame_ptr[idx] = kconf * inv;
        }
    }

    predictor->buffer_frames++;

    pthread_mutex_unlock(&predictor->mutex);
}

ActionResult tcn_action_predictor_predict(TcnActionPredictor* predictor) {
    ActionResult result;
    memset(&result, 0, sizeof(result));

    if (!predictor || !predictor->session) return result;

    const OrtApi* g_ort = ort_get_api();
    if (!g_ort) return result;

    int C = predictor->num_channels;
    int T = predictor->num_frames;
    int V = predictor->num_keypoints;
    int M = predictor->num_persons;
    size_t frame_size = (size_t)C * V * M;
    size_t tensor_count = (size_t)T * frame_size;

    /*
     * ── Split lock scope (same pattern as ST-GCN) ──
     *
     * Only hold the mutex while copying data FROM skeleton_buffer INTO the
     * pre-allocated tensor buffer (~1ms).  ORT inference runs on the
     * pre-allocated buffer OUTSIDE the lock, so push_pose() can continue
     * appending new poses concurrently.
     */

    /* ── Phase 1: Copy skeleton data under lock (fast, ~1ms) ── */
    float* input_tensor;
    bool input_dynamic = false;
    if (predictor->prealloc_valid && predictor->prealloc_input) {
        input_tensor = predictor->prealloc_input;
    } else {
        input_tensor = (float*)calloc(tensor_count, sizeof(float));
        input_dynamic = true;
        if (!input_tensor) return result;
    }
    memset(input_tensor, 0, tensor_count * sizeof(float));

    /* ── Lock: copy skeleton_buffer → prealloc tensor ── */
    pthread_mutex_lock(&predictor->mutex);

    int valid_frames = (predictor->buffer_frames < T) ? predictor->buffer_frames : T;

    if (valid_frames > 0) {
        /* Transpose skeleton_buffer from [T][C][V][M] to ONNX [C][T][V][M].
         *
         * FAST-START: When buffer is partial (< model's T), replicate available
         * frames to fill the full temporal window (tile/repeat).  This avoids
         * zero-padding which produces garbage predictions.
         *
         * Once buffer reaches num_frames, all T slots contain real data. */
        size_t ch_size = (size_t)T * V * M;
        size_t frame_src = (size_t)V * M;
        size_t frame_dst = (size_t)V * M;

        for (int c = 0; c < C; c++) {
            for (int t_dst = 0; t_dst < T; t_dst++) {
                /* Frame replication: t_src = t_dst % valid_frames.
                 * When valid_frames == T, this maps 1:1 (no replication). */
                int t_src = t_dst % valid_frames;
                size_t src_off = (size_t)t_src * C * frame_src + (size_t)c * frame_src;
                size_t dst_off = (size_t)c * ch_size + (size_t)t_dst * frame_dst;
                memcpy(input_tensor + dst_off,
                       predictor->skeleton_buffer + src_off,
                       frame_src * sizeof(float));
            }
        }
    }

    /* ── UNLOCK: tensor data is now in prealloc buffer ──
     * push_pose() can safely add new skeletons from here onward. */
    pthread_mutex_unlock(&predictor->mutex);

    /* ── Phase 2: ORT inference (no lock) ──
     * Works on prealloc buffer, no contention with push_pose(). */

    /* Determine the actual ONNX input shape using the rank detected at load time */
    int64_t input_shape[5];
    int input_ndim = predictor->input_ndim;

    input_shape[0] = 1;
    input_shape[1] = C;
    input_shape[2] = T;
    input_shape[3] = V;
    if (input_ndim >= 5) {
        input_shape[4] = M;
    }

    size_t input_bytes = tensor_count * sizeof(float);

    OrtValue* input_ort = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        predictor->ctx->memory_info, input_tensor, input_bytes,
        input_shape, input_ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ort);
    if (status) {
        g_ort->ReleaseStatus(status);
        if (input_dynamic) free(input_tensor);
        return result;
    }

    /* ── Run single-input inference ── */
    const char* input_names[1];
    OrtValue* input_vals[1];
    input_names[0] = predictor->input_name;
    input_vals[0] = input_ort;

    const char* output_names[1] = {NULL};
    OrtAllocator* out_allocator = NULL;
    bool output_name_from_allocator = false;

    OrtStatus* st_alloc = g_ort->GetAllocatorWithDefaultOptions(&out_allocator);
    if (st_alloc) g_ort->ReleaseStatus(st_alloc);

    if (out_allocator) {
        char* name_ptr = NULL;
        OrtStatus* s = g_ort->SessionGetOutputName(predictor->session, 0, out_allocator, &name_ptr);
        if (s) {
            g_ort->ReleaseStatus(s);
            output_names[0] = "output";
            output_name_from_allocator = false;
        } else {
            output_names[0] = name_ptr;
            output_name_from_allocator = true;
        }
    } else {
        output_names[0] = "output";
        output_name_from_allocator = false;
    }

    OrtValue* output_val = NULL;
    status = g_ort->Run(predictor->session, NULL,
                        input_names, (const OrtValue* const*)input_vals, 1,
                        output_names, 1, &output_val);
    g_ort->ReleaseValue(input_ort);

    if (input_dynamic) free(input_tensor);

    if (out_allocator && output_names[0] && output_name_from_allocator) {
        OrtStatus* st_f = g_ort->AllocatorFree(out_allocator, (void*)output_names[0]);
        if (st_f) g_ort->ReleaseStatus(st_f);
    }

    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("TCN inference failed: %s", msg ? msg : "unknown");
        g_ort->ReleaseStatus(status);
        return result;
    }

    /* ── Phase 3: Process output (no lock needed) ── */
    float* output_data = NULL;
    OrtStatus* st_mut = g_ort->GetTensorMutableData(output_val, (void**)&output_data);

    OrtTensorTypeAndShapeInfo* o_si = NULL;
    size_t o_elem = 0;
    if (g_ort->GetTensorTypeAndShape(output_val, &o_si) == NULL) {
        { OrtStatus* _s = g_ort->GetTensorShapeElementCount(o_si, &o_elem); if (_s) g_ort->ReleaseStatus(_s); }
        g_ort->ReleaseTensorTypeAndShapeInfo(o_si);
    }
    int actual_classes = UTILS_MIN((int)o_elem, predictor->num_classes);

    if (!st_mut && output_data) {
        int num_actions = 0;
        for (int c = 0; c < actual_classes && num_actions < TCN_MAX_ACTIONS; c++) {
            float score = output_data[c];
            if (score >= predictor->confidence_threshold) {
                result.actions[num_actions].action_id = c;
                char name_buf[LABEL_LEN];
                strncpy(result.actions[num_actions].action_name,
                        tcn_get_action_name_buf(c, name_buf, sizeof(name_buf)), MAX_LABEL_LEN - 1);
                result.actions[num_actions].action_name[MAX_LABEL_LEN - 1] = '\0';
                result.actions[num_actions].confidence = score;
                num_actions++;
            }
        }

        /* Sort by confidence descending */
        for (int i = 0; i < num_actions - 1; i++) {
            for (int j = i + 1; j < num_actions; j++) {
                if (result.actions[j].confidence > result.actions[i].confidence) {
                    ActionPrediction tmp = result.actions[i];
                    result.actions[i] = result.actions[j];
                    result.actions[j] = tmp;
                }
            }
        }

        result.num_actions = num_actions;

        /* Top-1 prediction */
        float max_score = -1.0f;
        int best_class = 0;
        for (int c = 0; c < actual_classes; c++) {
            if (output_data[c] > max_score) {
                max_score = output_data[c];
                best_class = c;
            }
        }
        result.predicted_action_id = best_class;
        result.predicted_confidence = max_score;
    }
    if (st_mut) g_ort->ReleaseStatus(st_mut);

    g_ort->ReleaseValue(output_val);
    return result;
}

/* ── Async prediction API ── */

void tcn_action_predictor_run_async(TcnActionPredictor* predictor) {
    if (!predictor) return;
    ActionResult r = tcn_action_predictor_predict(predictor);
    pthread_mutex_lock(&predictor->result_mutex);
    predictor->latest_action = r;
    predictor->has_new_action = true;
    pthread_mutex_unlock(&predictor->result_mutex);
}

bool tcn_action_predictor_get_latest(const TcnActionPredictor* predictor,
                                     ActionResult* out) {
    if (!predictor || !out) return false;
    TcnActionPredictor* rw = (TcnActionPredictor*)predictor;
    pthread_mutex_lock(&rw->result_mutex);
    bool has = rw->has_new_action;
    if (has) {
        *out = rw->latest_action;
        rw->has_new_action = false;
    }
    pthread_mutex_unlock(&rw->result_mutex);
    return has;
}

void tcn_action_predictor_reset(TcnActionPredictor* predictor) {
    if (!predictor) return;
    pthread_mutex_lock(&predictor->mutex);
    memset(predictor->skeleton_buffer, 0, sizeof(predictor->skeleton_buffer));
    predictor->buffer_frames = 0;
    predictor->buffer_person_id = -1;
    pthread_mutex_unlock(&predictor->mutex);
}

int tcn_action_predictor_get_buffer_fill(const TcnActionPredictor* predictor) {
    if (!predictor) return 0;
    TcnActionPredictor* rw = (TcnActionPredictor*)predictor;
    pthread_mutex_lock(&rw->mutex);
    int fill = rw->buffer_frames;
    pthread_mutex_unlock(&rw->mutex);
    return fill;
}

void tcn_action_predictor_set_min_frames(TcnActionPredictor* predictor, int min_frames) {
    if (!predictor) return;
    if (min_frames < 1) min_frames = 1;
    if (min_frames > predictor->num_frames) min_frames = predictor->num_frames;
    predictor->min_frames_to_predict = min_frames;
    log_info("TCN: min_frames_to_predict set to %d (model expects %d frames)",
             predictor->min_frames_to_predict, predictor->num_frames);
}

const char* tcn_get_action_name(int action_id) {
    if (action_id >= 0 && action_id < TCN_MAX_KNOWN_CLASSES) {
        return KNOWN_ACTION_CLASSES[action_id];
    }
    static char name_buf[32];
    snprintf(name_buf, sizeof(name_buf), "action_%d", action_id);
    return name_buf;
}
