#include "stgcn_action_recognizer.h"
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
#error "stgcn_action_recognizer requires HAS_ONNX_RUNTIME (real inference only - no heuristic fallback)"
#endif

/*
 * ST-GCN action classes — auto-detected from model output size.
 *
 * The model (stgcn.fp32.onnx) outputs [1, N] where N is the number of
 * action classes.  N is detected at load time from the output shape.
 *
 * For 7-class models, the labels below cover common NTU-RGB+D subsets.
 * If the model outputs more than STGCN_MAX_KNOWN_CLASSES, class names
 * are generated as "action_N".
 */
#define STGCN_MAX_KNOWN_CLASSES 60

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

/*
 * COCO-17 → NTU-RGB+D 25 keypoint index mapping.
 *
 * YOLOv8-pose outputs 17 COCO keypoints.  ST-GCN models trained on NTU-RGB+D
 * expect 25 keypoints.  This table maps each NTU index (0–24) to either:
 *   - a single COCO source (blend=1.0)
 *   - a weighted blend of two COCO sources (blend=0.5 averages them)
 *
 * NTU-RGB+D 25 joints:
 *   0:base_spine  1:mid_spine   2:neck       3:head        4:l_shoulder
 *   5:l_elbow     6:l_wrist     7:l_hand     8:r_shoulder  9:r_elbow
 *   10:r_wrist    11:r_hand     12:l_hip     13:l_knee     14:l_ankle
 *   15:l_foot     16:r_hip      17:r_knee    18:r_ankle    19:r_foot
 *   20:spine      21:l_hand_tip 22:l_thumb   23:r_hand_tip 24:r_thumb
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
} CocoToNtuMapping;

static const CocoToNtuMapping COCO_TO_NTU[25] = {
    /* NTU 0: base of spine → midpoint of hips */
    {11, 12, 0.5f},
    /* NTU 1: middle of spine → midpoint of shoulders and hips */
    {5,  11, 0.5f},   /* (l_shoulder+l_hip)/2 */
    /* NTU 2: neck → midpoint of shoulders */
    {5,  6,  0.5f},
    /* NTU 3: head → nose */
    {0,  -1, 1.0f},
    /* NTU 4: left shoulder */
    {5,  -1, 1.0f},
    /* NTU 5: left elbow */
    {7,  -1, 1.0f},
    /* NTU 6: left wrist */
    {9,  -1, 1.0f},
    /* NTU 7: left hand → same as wrist */
    {9,  -1, 1.0f},
    /* NTU 8: right shoulder */
    {6,  -1, 1.0f},
    /* NTU 9: right elbow */
    {8,  -1, 1.0f},
    /* NTU 10: right wrist */
    {10, -1, 1.0f},
    /* NTU 11: right hand → same as wrist */
    {10, -1, 1.0f},
    /* NTU 12: left hip */
    {11, -1, 1.0f},
    /* NTU 13: left knee */
    {13, -1, 1.0f},
    /* NTU 14: left ankle */
    {15, -1, 1.0f},
    /* NTU 15: left foot → same as ankle */
    {15, -1, 1.0f},
    /* NTU 16: right hip */
    {12, -1, 1.0f},
    /* NTU 17: right knee */
    {14, -1, 1.0f},
    /* NTU 18: right ankle */
    {16, -1, 1.0f},
    /* NTU 19: right foot → same as ankle */
    {16, -1, 1.0f},
    /* NTU 20: spine → midpoint of shoulders and hips */
    {5,  11, 0.5f},
    /* NTU 21: left hand tip → same as wrist */
    {9,  -1, 1.0f},
    /* NTU 22: left thumb → same as wrist */
    {9,  -1, 1.0f},
    /* NTU 23: right hand tip → same as wrist */
    {10, -1, 1.0f},
    /* NTU 24: right thumb → same as wrist */
    {10, -1, 1.0f},
};

STGCNActionRecognizer* stgcn_action_recognizer_create(const char* model_path,
                                                        int num_frames,
                                                        int num_keypoints,
                                                        int num_persons,
                                                        int num_classes,
                                                        float conf_thresh) {
    STGCNActionRecognizer* rec = (STGCNActionRecognizer*)calloc(1, sizeof(STGCNActionRecognizer));
    if (!rec) return NULL;

    rec->num_frames = (num_frames > 0 && num_frames <= STGCN_MAX_FRAMES) ? num_frames : 300;
    rec->num_keypoints = (num_keypoints > 0 && num_keypoints <= STGCN_NUM_KEYPOINTS) ? num_keypoints : 25;
    rec->num_persons = (num_persons > 0 && num_persons <= STGCN_MAX_PERSONS) ? num_persons : 1;
    rec->num_classes = (num_classes > 0 && num_classes <= STGCN_MAX_KNOWN_CLASSES) ? num_classes : 7;
    rec->confidence_threshold = conf_thresh;
    rec->buffer_frames = 0;
    rec->buffer_person_id = -1;
    rec->model_loaded = false;

    /* Init mutexes for async recognition */
    if (pthread_mutex_init(&rec->mutex, NULL) != 0) {
        log_error("STGCNActionRecognizer: mutex init failed: %s", strerror(errno));
        free(rec);
        return NULL;
    }
    if (pthread_mutex_init(&rec->result_mutex, NULL) != 0) {
        pthread_mutex_destroy(&rec->mutex);
        free(rec);
        return NULL;
    }
    memset(&rec->latest_action, 0, sizeof(rec->latest_action));
    rec->has_new_action = false;

    if (!model_path || !stgcn_action_recognizer_load_model(rec, model_path)) {
        log_error("STGCNActionRecognizer: failed to load model %s", model_path ? model_path : "(null)");
        free(rec);
        return NULL;
    }

    return rec;
}

void stgcn_action_recognizer_destroy(STGCNActionRecognizer* recognizer) {
    if (!recognizer) return;
    ort_ctx_destroy(recognizer->ctx);
    const OrtApi* g_ort = ort_get_api();
    if (recognizer->session && g_ort) {
        g_ort->ReleaseSession(recognizer->session);
    }
    free(recognizer->prealloc_pts);
    free(recognizer->prealloc_mot);
    free(recognizer->prealloc_padded);
    pthread_mutex_destroy(&recognizer->mutex);
    pthread_mutex_destroy(&recognizer->result_mutex);
    free(recognizer);
}

bool stgcn_action_recognizer_load_model(STGCNActionRecognizer* recognizer, const char* model_path) {
    if (!recognizer || !model_path) return false;

    size_t file_size = 0;
    if (ort_validate_onnx_file(model_path, &file_size) != 0) {
        return false;
    }

    if (!ort_global_init()) {
        log_error("STGCNActionRecognizer: ORT runtime not initialized");
        return false;
    }

    /* ST-GCN uses graph convolution ops (not standard CNN Conv2D) which
     * SpacemiT EP's IME hardware accelerator does not support.  Force CPU
     * EP even though the model is INT8-quantized — the quantization saves
     * memory bandwidth and CPU EP handles the dequantization efficiently
     * on general-purpose cores. */
    recognizer->session = ort_create_session(model_path, 4, false);
    if (!recognizer->session) {
        log_error("STGCNActionRecognizer: failed to create ONNX session for %s", model_path);
        return false;
    }

    const OrtApi* ort = ort_get_api();

    /* ── Detect input names + shapes dynamically (BEFORE ctx creation) ── */
    OrtAllocator* allocator = NULL;
    OrtStatus* st_a = ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_a) ort->ReleaseStatus(st_a);

    /* Query number of inputs */
    size_t num_inputs = 0;
    OrtStatus* st_ni = ort->SessionGetInputCount(recognizer->session, &num_inputs);
    if (st_ni) { ort->ReleaseStatus(st_ni); num_inputs = 1; }

    /* Store input names */
    strncpy(recognizer->input_name_pts, "pts", sizeof(recognizer->input_name_pts) - 1);
    strncpy(recognizer->input_name_mot, "mot", sizeof(recognizer->input_name_mot) - 1);

    if (allocator) {
        for (size_t ii = 0; ii < num_inputs && ii < 2; ii++) {
            char* real_name = NULL;
            OrtStatus* s = ort->SessionGetInputName(recognizer->session, ii, allocator, &real_name);
            if (s == NULL && real_name) {
                if (ii == 0) {
                    strncpy(recognizer->input_name_pts, real_name, sizeof(recognizer->input_name_pts) - 1);
                } else {
                    strncpy(recognizer->input_name_mot, real_name, sizeof(recognizer->input_name_mot) - 1);
                }
                OrtStatus* sf = ort->AllocatorFree(allocator, real_name);
                if (sf) ort->ReleaseStatus(sf);
            } else {
                if (s) ort->ReleaseStatus(s);
            }
        }
    }

    recognizer->has_mot_input = (num_inputs >= 2);

    /* ── Auto-detect input shape from model signature ──
     * ST-GCN expects [1, C, T, V, M] (5D NTU-RGB+D format).
     * Query the first input's shape and override the configured
     * num_frames / num_keypoints / num_persons to match. */
    if (allocator && num_inputs > 0) {
        OrtTypeInfo* type_info = NULL;
        OrtStatus* s_ti = ort->SessionGetInputTypeInfo(recognizer->session, 0, &type_info);
        if (s_ti == NULL && type_info) {
            const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
            OrtStatus* s_cast = ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
            if (s_cast == NULL && tensor_info) {
                size_t nd = 0;
                { OrtStatus* _s = ort->GetDimensionsCount(tensor_info, &nd); if (_s) ort->ReleaseStatus(_s); }
                int64_t idims[5] = {0};
                size_t nd_read = (nd > 5) ? 5 : nd;
                { OrtStatus* _s = ort->GetDimensions(tensor_info, idims, nd_read); if (_s) ort->ReleaseStatus(_s); }

                /* Canonical ST-GCN shape: [1, C, T, V, M] or [C, T, V, M].
                 * Index 0 may be batch (1) or channels (3).  Detect by size:
                 *   dim[0]==1  → batch  → T=dim[2], V=dim[3], M=dim[4]
                 *   dim[0]==3  → channels → T=dim[1], V=dim[2], M=dim[3] */
                int64_t model_T = 0, model_V = 0, model_M = 1;

                if (nd >= 5 && idims[0] == 1) {
                    /* batch-first: [1, C, T, V, M] */
                    model_T = idims[2];
                    model_V = idims[3];
                    model_M = idims[4];
                } else if (nd >= 4 && (idims[0] == 3 || idims[0] == 2)) {
                    /* channel-first: [C, T, V, M] — mot input may have C=2 */
                    model_T = idims[1];
                    model_V = idims[2];
                    model_M = (nd >= 4) ? idims[3] : 1;
                } else if (nd >= 4) {
                    /* Unknown layout — assume [1, C, T, V] or [1, C, T, V] with implicit M=1 */
                    model_T = idims[2];
                    model_V = idims[3];
                    model_M = (nd >= 5) ? idims[4] : 1;
                }

                if (model_T > 0 && model_T <= STGCN_MAX_FRAMES &&
                    model_V > 0 && model_V <= STGCN_NUM_KEYPOINTS &&
                    model_M > 0 && model_M <= STGCN_MAX_PERSONS) {
                    if (recognizer->num_frames != (int)model_T) {
                        log_info("ST-GCN: overriding configured %d frames with model's %lld frames",
                                 recognizer->num_frames, (long long)model_T);
                        recognizer->num_frames = (int)model_T;
                    }
                    if (recognizer->num_keypoints != (int)model_V) {
                        log_info("ST-GCN: overriding configured %d keypoints with model's %lld keypoints",
                                 recognizer->num_keypoints, (long long)model_V);
                        recognizer->num_keypoints = (int)model_V;
                    }
                    if (recognizer->num_persons != (int)model_M) {
                        log_info("ST-GCN: overriding configured %d persons with model's %lld persons",
                                 recognizer->num_persons, (long long)model_M);
                        recognizer->num_persons = (int)model_M;
                    }
                } else if (model_T > 0) {
                    log_warning("ST-GCN: model input shape [%lld,%lld,%lld,%lld,%lld] exceeds buffer limits "
                                "(max T=%d V=%d M=%d) — using configured values",
                                (long long)idims[0], (long long)idims[1],
                                (long long)idims[2], (long long)idims[3],
                                (long long)idims[4],
                                STGCN_MAX_FRAMES, STGCN_NUM_KEYPOINTS, STGCN_MAX_PERSONS);
                }
            }
            if (s_cast) ort->ReleaseStatus(s_cast);
        }
        if (s_ti) ort->ReleaseStatus(s_ti);
        ort->ReleaseTypeInfo(type_info);
    }

    /* ── Create inference context with MODEL-DETECTED dimensions ── */
    recognizer->ctx = ort_ctx_create(recognizer->session,
                                      recognizer->num_frames,
                                      recognizer->num_keypoints,
                                      STGCN_NUM_CHANNELS);
    if (!recognizer->ctx) {
        log_error("STGCNActionRecognizer: failed to create inference context");
        if (recognizer->session && ort) {
            ort->ReleaseSession(recognizer->session);
            recognizer->session = NULL;
        }
        return false;
    }
    recognizer->ctx->input_name[0] = '\0';

    /* ── Detect output class count from model shape ── */
    size_t num_outputs = 0;
    OrtStatus* st_no = ort->SessionGetOutputCount(recognizer->session, &num_outputs);
    if (st_no) { ort->ReleaseStatus(st_no); num_outputs = 1; }

    if (num_outputs > 0 && allocator) {
        OrtTypeInfo* type_info = NULL;
        OrtStatus* s_ti = ort->SessionGetOutputTypeInfo(recognizer->session, 0, &type_info);
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

                if (detected_classes > 0 && detected_classes < STGCN_MAX_KNOWN_CLASSES) {
                    if (recognizer->num_classes != detected_classes) {
                        log_info("ST-GCN: overriding configured %d classes with model's actual %d classes",
                                 recognizer->num_classes, detected_classes);
                        recognizer->num_classes = detected_classes;
                    }
                }
            }
            if (s_cast) ort->ReleaseStatus(s_cast);
        }
        if (s_ti) ort->ReleaseStatus(s_ti);
        ort->ReleaseTypeInfo(type_info);
    }

    /* model_loaded only when ctx AND session are both valid */
    recognizer->model_loaded = (recognizer->ctx != NULL && recognizer->session != NULL);

    /* ── Pre-allocate inference tensors ──
     * Sizes use model-detected dimensions: T × C × V × M for pts, T × 2 × V × M for mot. */
    {
        size_t pts_cnt = (size_t)recognizer->num_frames * STGCN_NUM_CHANNELS *
                          recognizer->num_keypoints * recognizer->num_persons;
        size_t mot_cnt  = (size_t)recognizer->num_frames * 2 *
                          recognizer->num_keypoints * recognizer->num_persons;

        /* Free any prior pre-allocations (shouldn't happen, but be safe) */
        free(recognizer->prealloc_pts);
        free(recognizer->prealloc_mot);
        free(recognizer->prealloc_padded);

        recognizer->prealloc_pts = (float*)calloc(pts_cnt, sizeof(float));
        recognizer->prealloc_mot = (float*)calloc(mot_cnt, sizeof(float));
        recognizer->prealloc_padded = (float*)calloc(pts_cnt, sizeof(float));

        if (recognizer->prealloc_pts && recognizer->prealloc_mot && recognizer->prealloc_padded) {
            recognizer->prealloc_pts_size = pts_cnt;
            recognizer->prealloc_mot_size = mot_cnt;
            recognizer->prealloc_valid = true;
        } else {
            if (!recognizer->prealloc_pts) {
                log_error("ST-GCN: failed to pre-allocate pts tensor (%zu floats)", pts_cnt);
            }
            recognizer->prealloc_valid = (recognizer->prealloc_pts != NULL);
        }
    }

    log_info("ST-GCN action model loaded: %s (%.2f MB) classes=%d inputs=%zu (mot=%s) "
             "shape=[1,%d,%d,%d,%d] input_names=[%s,%s]",
             model_path, file_size / (1024.0 * 1024.0), recognizer->num_classes,
             num_inputs, recognizer->has_mot_input ? "yes" : "no",
             STGCN_NUM_CHANNELS, recognizer->num_frames,
             recognizer->num_keypoints, recognizer->num_persons,
             recognizer->input_name_pts, recognizer->input_name_mot);
    return true;
}

void stgcn_action_recognizer_push_pose(STGCNActionRecognizer* recognizer,
                                        const PoseEstimation* pose,
                                        int img_width, int img_height) {
    if (!recognizer || !pose) return;

    pthread_mutex_lock(&recognizer->mutex);

    int T = recognizer->num_frames;
    int V = recognizer->num_keypoints;  /* model-expected keypoint count (typically 25 NTU) */
    int M = recognizer->num_persons;
    int C = STGCN_NUM_CHANNELS;

    /* Sliding window: shift out oldest frame if buffer is full */
    if (recognizer->buffer_frames >= T) {
        int frame_size = C * V * M;
        memmove(recognizer->skeleton_buffer,
                recognizer->skeleton_buffer + frame_size,
                (size_t)(T - 1) * frame_size * sizeof(float));
        recognizer->buffer_frames = T - 1;
    }

    int t = recognizer->buffer_frames;
    float* frame_ptr = recognizer->skeleton_buffer + (size_t)t * C * V * M;

    memset(frame_ptr, 0, (size_t)C * V * M * sizeof(float));

    float w = (img_width > 0) ? (float)img_width : 1.0f;
    float h = (img_height > 0) ? (float)img_height : 1.0f;

    int src_kpts = pose->num_keypoints;  /* COCO-17 from YOLOv8-pose */

    /* Map COCO-17 keypoints → NTU-25 keypoints using the static mapping table.
     * For NTU joints that map to COCO joints, use the mapped position + confidence.
     * For joints without a COCO equivalent, the value stays zero (pre-cleared above). */
    for (int ntu_idx = 0; ntu_idx < V && ntu_idx < 25; ntu_idx++) {
        const CocoToNtuMapping* m = &COCO_TO_NTU[ntu_idx];
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
            int idx = 0 * V * M + ntu_idx * M + 0;
            frame_ptr[idx] = (kx * inv) / w;
            /* Channel 1: normalized Y */
            idx = 1 * V * M + ntu_idx * M + 0;
            frame_ptr[idx] = (ky * inv) / h;
            /* Channel 2: keypoint confidence (average of source confidences) */
            idx = 2 * V * M + ntu_idx * M + 0;
            frame_ptr[idx] = kconf * inv;
        }
    }

    recognizer->buffer_frames++;

    pthread_mutex_unlock(&recognizer->mutex);
}

ActionResult stgcn_action_recognizer_recognize(STGCNActionRecognizer* recognizer) {
    ActionResult result;
    memset(&result, 0, sizeof(result));

    if (!recognizer || !recognizer->session) return result;

    const OrtApi* g_ort = ort_get_api();
    if (!g_ort) return result;

    int T = recognizer->num_frames;
    int V = recognizer->num_keypoints;
    int M = recognizer->num_persons;
    int C = STGCN_NUM_CHANNELS;
    size_t frame_size = (size_t)C * V * M;
    size_t tensor_count = (size_t)T * frame_size;

    /*
     * ── CRITICAL FIX: Split lock scope ──
     *
     * Previously `mutex` was held for the ENTIRE recognize() call (~400ms
     * for ORT inference on CPU EP), which blocked push_pose() — the
     * inference thread couldn't feed new poses to the ST-GCN buffer
     * while action recognition was running.  This caused ~400ms gaps
     * in the skeleton buffer timeline every time ST-GCN ran.
     *
     * FIX: Only hold the mutex while copying data FROM skeleton_buffer
     * INTO the pre-allocated tensor buffers (~1ms).  ORT inference runs
     * on the pre-allocated buffers OUTSIDE the lock, so push_pose() can
     * continue appending new poses concurrently.
     *
     * Thread safety:
     *   - prealloc_pts/mot/padded are owned exclusively by recognize()
     *     (only called from ST-GCN thread)
     *   - skeleton_buffer is shared with push_pose() (protected by mutex)
     *   - ORT Run reads from prealloc buffers (no lock contention)
     */

    /* ── Phase 1: Copy skeleton data under lock (fast, ~1ms) ── */
    float* pts_tensor;
    bool pts_dynamic = false;
    if (recognizer->prealloc_valid && recognizer->prealloc_pts) {
        pts_tensor = recognizer->prealloc_pts;
    } else {
        pts_tensor = (float*)calloc(tensor_count, sizeof(float));
        pts_dynamic = true;
        if (!pts_tensor) return result;
    }
    memset(pts_tensor, 0, tensor_count * sizeof(float));

    /* mot tensor preparation */
    const int MOT_C = 2;
    size_t mot_frame_size = (size_t)MOT_C * V * M;
    size_t mot_count = (size_t)T * mot_frame_size;

    float* mot_tensor = NULL;
    bool mot_dynamic = false;
    if (recognizer->has_mot_input) {
        if (recognizer->prealloc_valid && recognizer->prealloc_mot) {
            mot_tensor = recognizer->prealloc_mot;
        } else {
            mot_tensor = (float*)calloc(mot_count, sizeof(float));
            mot_dynamic = true;
        }
        if (mot_tensor) memset(mot_tensor, 0, mot_count * sizeof(float));
    }

    /* ── Lock: copy skeleton_buffer → prealloc tensors ── */
    pthread_mutex_lock(&recognizer->mutex);

    int valid_frames = (recognizer->buffer_frames < T) ? recognizer->buffer_frames : T;

    if (valid_frames > 0) {
        /* Transpose skeleton_buffer from [T][C][V][M] to ONNX [C][T][V][M].
         * Right-align: pad oldest frames with zeros. */
        int offset = T - valid_frames;
        size_t ch_size = (size_t)T * V * M;
        size_t frame_src = (size_t)V * M;
        size_t frame_dst = (size_t)V * M;

        for (int c = 0; c < C; c++) {
            for (int t_src = 0; t_src < valid_frames; t_src++) {
                int t_dst = offset + t_src;
                size_t src_off = (size_t)t_src * C * frame_src + (size_t)c * frame_src;
                size_t dst_off = (size_t)c * ch_size + (size_t)t_dst * frame_dst;
                memcpy(pts_tensor + dst_off,
                       recognizer->skeleton_buffer + src_off,
                       frame_src * sizeof(float));
            }
        }
    }

    /* Compute motion features (mot) under lock */
    if (mot_tensor && recognizer->has_mot_input && valid_frames > 0) {
        float* padded_pts;
        bool padded_dynamic = false;
        if (recognizer->prealloc_valid && recognizer->prealloc_padded) {
            padded_pts = recognizer->prealloc_padded;
            memset(padded_pts, 0, tensor_count * sizeof(float));
        } else {
            padded_pts = (float*)calloc(tensor_count, sizeof(float));
            padded_dynamic = true;
        }
        if (padded_pts) {
            int offset = T - valid_frames;
            memcpy(padded_pts + (size_t)offset * frame_size,
                   recognizer->skeleton_buffer,
                   (size_t)valid_frames * frame_size * sizeof(float));

            size_t mot_ch_stride = (size_t)T * V * M;
            size_t frm_stride = (size_t)V * M;

            for (int t = 0; t < T - 1; t++) {
                const float* cur = padded_pts + (size_t)t * frame_size;
                const float* nxt = padded_pts + (size_t)(t + 1) * frame_size;

                float* mot_dx = mot_tensor + (size_t)0 * mot_ch_stride + (size_t)t * frm_stride;
                float* mot_dy = mot_tensor + (size_t)1 * mot_ch_stride + (size_t)t * frm_stride;

                for (int v = 0; v < V; v++) {
                    int ptx = 0 * V * M + v * M + 0;
                    int pty = 1 * V * M + v * M + 0;
                    int idx = v * M + 0;

                    mot_dx[idx] = nxt[ptx] - cur[ptx];
                    mot_dy[idx] = nxt[pty] - cur[pty];
                }
            }
            if (padded_dynamic) free(padded_pts);
        }
    }

    /* ── UNLOCK: tensor data is now in prealloc buffers ──
     * push_pose() can safely add new skeletons from here onward. */
    pthread_mutex_unlock(&recognizer->mutex);

    /* ── Phase 2: ORT inference (no lock, ~400ms) ──
     * Works on prealloc buffers, no contention with push_pose(). */

    /* Create pts OrtValue */
    int64_t pts_shape[5] = {1, C, T, V, M};
    size_t pts_bytes = tensor_count * sizeof(float);

    OrtValue* pts_ort = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        recognizer->ctx->memory_info, pts_tensor, pts_bytes,
        pts_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pts_ort);
    if (status) {
        g_ort->ReleaseStatus(status);
        if (pts_dynamic) free(pts_tensor);
        if (mot_dynamic) free(mot_tensor);
        return result;
    }

    /* Create mot OrtValue */
    OrtValue* mot_ort = NULL;
    if (mot_tensor && recognizer->has_mot_input) {
        int64_t mot_shape[5] = {1, MOT_C, T, V, M};
        status = g_ort->CreateTensorWithDataAsOrtValue(
            recognizer->ctx->memory_info, mot_tensor, mot_count * sizeof(float),
            mot_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &mot_ort);
        if (status) {
            g_ort->ReleaseStatus(status);
            mot_ort = NULL;
        }
    }

    /* ── Run dual-input inference ── */
    const char* input_names[2];
    OrtValue* input_vals[2];
    int num_inputs_to_run = 1;

    input_names[0] = recognizer->input_name_pts;
    input_vals[0] = pts_ort;

    if (mot_ort && recognizer->has_mot_input) {
        input_names[1] = recognizer->input_name_mot;
        input_vals[1] = mot_ort;
        num_inputs_to_run = 2;
    }

    const char* output_names[1] = {NULL};
    OrtAllocator* allocator = NULL;
    bool output_name_from_allocator = false;

    OrtStatus* st_alloc = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (st_alloc) g_ort->ReleaseStatus(st_alloc);

    if (allocator) {
        char* name_ptr = NULL;
        OrtStatus* s = g_ort->SessionGetOutputName(recognizer->session, 0, allocator, &name_ptr);
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
    status = g_ort->Run(recognizer->session, NULL,
                        input_names, (const OrtValue* const*)input_vals, num_inputs_to_run,
                        output_names, 1, &output_val);
    g_ort->ReleaseValue(pts_ort);
    if (mot_ort) g_ort->ReleaseValue(mot_ort);

    if (pts_dynamic) free(pts_tensor);
    if (mot_dynamic) free(mot_tensor);

    if (allocator && output_names[0] && output_name_from_allocator) {
        OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)output_names[0]);
        if (st_f) g_ort->ReleaseStatus(st_f);
    }

    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("ST-GCN inference failed: %s", msg ? msg : "unknown");
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
    int actual_classes = UTILS_MIN((int)o_elem, recognizer->num_classes);

    if (!st_mut && output_data) {
        int num_actions = 0;
        for (int c = 0; c < actual_classes && num_actions < STGCN_MAX_ACTIONS; c++) {
            float score = output_data[c];
            if (score >= recognizer->confidence_threshold) {
                result.actions[num_actions].action_id = c;
                strncpy(result.actions[num_actions].action_name,
                        stgcn_get_action_name(c), MAX_STRING_LEN - 1);
                result.actions[num_actions].action_name[MAX_STRING_LEN - 1] = '\0';
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

/* ── Async recognition API ── */

void stgcn_action_recognizer_run_async(STGCNActionRecognizer* recognizer) {
    if (!recognizer) return;
    ActionResult r = stgcn_action_recognizer_recognize(recognizer);
    /* recognize() handles its own mutex; store result under result_mutex */
    pthread_mutex_lock(&recognizer->result_mutex);
    recognizer->latest_action = r;
    recognizer->has_new_action = true;
    pthread_mutex_unlock(&recognizer->result_mutex);
}

bool stgcn_action_recognizer_get_latest(const STGCNActionRecognizer* recognizer,
                                        ActionResult* out) {
    if (!recognizer || !out) return false;
    /* Cast away const for mutex — internal synchronization only */
    STGCNActionRecognizer* rw = (STGCNActionRecognizer*)recognizer;
    pthread_mutex_lock(&rw->result_mutex);
    bool has = rw->has_new_action;
    if (has) {
        *out = rw->latest_action;
        rw->has_new_action = false;
    }
    pthread_mutex_unlock(&rw->result_mutex);
    return has;
}

void stgcn_action_recognizer_reset(STGCNActionRecognizer* recognizer) {
    if (!recognizer) return;
    pthread_mutex_lock(&recognizer->mutex);
    memset(recognizer->skeleton_buffer, 0, sizeof(recognizer->skeleton_buffer));
    recognizer->buffer_frames = 0;
    recognizer->buffer_person_id = -1;
    pthread_mutex_unlock(&recognizer->mutex);
}

const char* stgcn_get_action_name(int action_id) {
    if (action_id >= 0 && action_id < STGCN_MAX_KNOWN_CLASSES) {
        return KNOWN_ACTION_CLASSES[action_id];
    }
    return "unknown";
}
