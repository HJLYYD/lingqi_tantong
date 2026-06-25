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

STGCNActionRecognizer* stgcn_action_recognizer_create(const char* model_path,
                                                        int num_frames,
                                                        int num_keypoints,
                                                        int num_persons,
                                                        int num_classes,
                                                        float conf_thresh) {
    STGCNActionRecognizer* rec = (STGCNActionRecognizer*)calloc(1, sizeof(STGCNActionRecognizer));
    if (!rec) return NULL;

    rec->num_frames = (num_frames > 0 && num_frames <= STGCN_MAX_FRAMES) ? num_frames : 30;
    rec->num_keypoints = (num_keypoints > 0 && num_keypoints <= STGCN_NUM_KEYPOINTS) ? num_keypoints : 14;
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

    recognizer->session = ort_create_session(model_path, 4, true);
    if (!recognizer->session) {
        log_error("STGCNActionRecognizer: failed to create ONNX session for %s", model_path);
        return false;
    }

    recognizer->ctx = ort_ctx_create(recognizer->session,
                                      recognizer->num_frames,
                                      recognizer->num_keypoints,
                                      STGCN_NUM_CHANNELS);
    if (!recognizer->ctx) {
        log_error("STGCNActionRecognizer: failed to create inference context");
        const OrtApi* ort = ort_get_api();
        if (recognizer->session && ort) {
            ort->ReleaseSession(recognizer->session);
            recognizer->session = NULL;
        }
        return false;
    }
    recognizer->ctx->input_name[0] = '\0';

    const OrtApi* ort = ort_get_api();

    /* ── Detect input names dynamically ── */
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
     * Eliminates per-frame malloc/free in stgcn_action_recognizer_recognize().
     * Sizes are fixed at creation time: T × C × V × M for pts, T × 2 × V × M for mot. */
    {
        size_t pts_cnt = (size_t)recognizer->num_frames * STGCN_NUM_CHANNELS *
                          recognizer->num_keypoints * recognizer->num_persons;
        size_t mot_cnt  = (size_t)recognizer->num_frames * 2 *
                          recognizer->num_keypoints * recognizer->num_persons;

        recognizer->prealloc_pts = (float*)calloc(pts_cnt, sizeof(float));
        recognizer->prealloc_mot = (float*)calloc(mot_cnt, sizeof(float));
        recognizer->prealloc_padded = (float*)calloc(pts_cnt, sizeof(float));

        if (recognizer->prealloc_pts && recognizer->prealloc_mot && recognizer->prealloc_padded) {
            recognizer->prealloc_pts_size = pts_cnt;
            recognizer->prealloc_mot_size = mot_cnt;
            recognizer->prealloc_valid = true;
        } else {
            /* At least pts must succeed for inference to work.
             * mot and padded are optional — will use dynamic alloc as fallback. */
            if (!recognizer->prealloc_pts) {
                log_error("ST-GCN: failed to pre-allocate pts tensor (%zu floats)", pts_cnt);
            }
            recognizer->prealloc_valid = (recognizer->prealloc_pts != NULL);
        }
    }

    log_info("ST-GCN action model loaded: %s (%.2f MB) classes=%d inputs=%zu (mot=%s) input_names=[%s,%s]",
             model_path, file_size / (1024.0 * 1024.0), recognizer->num_classes,
             num_inputs, recognizer->has_mot_input ? "yes" : "no",
             recognizer->input_name_pts, recognizer->input_name_mot);
    return true;
}

void stgcn_action_recognizer_push_pose(STGCNActionRecognizer* recognizer,
                                        const PoseEstimation* pose,
                                        int img_width, int img_height) {
    if (!recognizer || !pose) return;

    pthread_mutex_lock(&recognizer->mutex);

    int T = recognizer->num_frames;
    int V = recognizer->num_keypoints;
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

    int num_kp = (pose->num_keypoints < V) ? pose->num_keypoints : V;
    for (int v = 0; v < num_kp; v++) {
        /* Channel 0: normalized X */
        int idx = 0 * V * M + v * M + 0;
        frame_ptr[idx] = pose->keypoints[v].x / w;
        /* Channel 1: normalized Y */
        idx = 1 * V * M + v * M + 0;
        frame_ptr[idx] = pose->keypoints[v].y / h;
        /* Channel 2: keypoint confidence */
        idx = 2 * V * M + v * M + 0;
        frame_ptr[idx] = pose->keypoints[v].confidence;
    }

    recognizer->buffer_frames++;

    pthread_mutex_unlock(&recognizer->mutex);
}

ActionResult stgcn_action_recognizer_recognize(STGCNActionRecognizer* recognizer) {
    ActionResult result;
    memset(&result, 0, sizeof(result));

    if (!recognizer || !recognizer->session) return result;

    pthread_mutex_lock(&recognizer->mutex);

    const OrtApi* g_ort = ort_get_api();
    if (!g_ort) { pthread_mutex_unlock(&recognizer->mutex); return result; }

    int T = recognizer->num_frames;
    int V = recognizer->num_keypoints;
    int M = recognizer->num_persons;
    int C = STGCN_NUM_CHANNELS;
    size_t frame_size = (size_t)C * V * M;
    size_t tensor_count = (size_t)T * frame_size;

    /* ── Prepare pts input tensor ──
     * Use pre-allocated buffer when available; fall back to dynamic
     * allocation only if pre-allocation failed at load time. */
    float* pts_tensor;
    bool pts_dynamic = false;
    if (recognizer->prealloc_valid && recognizer->prealloc_pts) {
        pts_tensor = recognizer->prealloc_pts;
        memset(pts_tensor, 0, tensor_count * sizeof(float));
    } else {
        pts_tensor = (float*)calloc(tensor_count, sizeof(float));
        pts_dynamic = true;
    }
    if (!pts_tensor) { pthread_mutex_unlock(&recognizer->mutex); return result; }

    int valid_frames = (recognizer->buffer_frames < T) ? recognizer->buffer_frames : T;
    if (valid_frames > 0) {
        /* Right-align: pad with zeros at the beginning */
        int offset = T - valid_frames;
        memcpy(pts_tensor + (size_t)offset * frame_size,
               recognizer->skeleton_buffer,
               (size_t)valid_frames * frame_size * sizeof(float));
    }

    /* Model expects 4D [1,C,T,V] — no M (person) dimension.
     * Since M=1, memory layout is identical; we just declare rank=4. */
    int64_t pts_shape[4] = {1, C, T, V};
    size_t pts_bytes = tensor_count * sizeof(float);

    OrtValue* pts_ort = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        recognizer->ctx->memory_info, pts_tensor, pts_bytes,
        pts_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pts_ort);
    if (status) {
        g_ort->ReleaseStatus(status);
        if (pts_dynamic) free(pts_tensor);
        pthread_mutex_unlock(&recognizer->mutex);
        return result;
    }

    /* ── Prepare mot input tensor ──
     * mot channel count is 2 (dx, dy) for ST-GCN standard. */
    const int MOT_C = 2;
    size_t mot_frame_size = (size_t)MOT_C * V * M;
    size_t mot_count = (size_t)T * mot_frame_size;

    float* mot_tensor = NULL;
    bool mot_dynamic = false;
    if (recognizer->has_mot_input) {
        if (recognizer->prealloc_valid && recognizer->prealloc_mot) {
            mot_tensor = recognizer->prealloc_mot;
            memset(mot_tensor, 0, mot_count * sizeof(float));
        } else {
            mot_tensor = (float*)calloc(mot_count, sizeof(float));
            mot_dynamic = true;
        }
    }

    OrtValue* mot_ort = NULL;
    if (mot_tensor && recognizer->has_mot_input) {
        if (valid_frames > 0) {
            /* Compute mot from pts in the sliding window */
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
                /* Compute motion: mot[t] = pts[t+1] - pts[t]
                 * Only C=3 channel: x, y, confidence.  Motion in x, y only. */
                for (int t = 0; t < T - 1; t++) {
                    const float* cur = padded_pts + (size_t)t * frame_size;
                    const float* nxt = padded_pts + (size_t)(t + 1) * frame_size;
                    float* mot = mot_tensor + (size_t)t * mot_frame_size;
                    for (int v = 0; v < V; v++) {
                        int mot_idx_0 = 0 * V * M + v * M + 0;
                        int mot_idx_1 = 1 * V * M + v * M + 0;
                        int pts_idx_x = 0 * V * M + v * M + 0;
                        int pts_idx_y = 1 * V * M + v * M + 0;
                        mot[mot_idx_0] = nxt[pts_idx_x] - cur[pts_idx_x];
                        mot[mot_idx_1] = nxt[pts_idx_y] - cur[pts_idx_y];
                    }
                }
                if (padded_dynamic) free(padded_pts);
            }
        }

        int64_t mot_shape[4] = {1, MOT_C, T, V};
        status = g_ort->CreateTensorWithDataAsOrtValue(
            recognizer->ctx->memory_info, mot_tensor, mot_count * sizeof(float),
            mot_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &mot_ort);
        if (status) {
            g_ort->ReleaseStatus(status);
            mot_ort = NULL;
        }
    }
    if (pts_dynamic) free(pts_tensor);
    if (mot_dynamic) free(mot_tensor);

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

    if (allocator && output_names[0] && output_name_from_allocator) {
        OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)output_names[0]);
        if (st_f) g_ort->ReleaseStatus(st_f);
    }

    if (status) {
        const char* msg = g_ort->GetErrorMessage(status);
        log_error("ST-GCN inference failed: %s", msg ? msg : "unknown");
        g_ort->ReleaseStatus(status);
        pthread_mutex_unlock(&recognizer->mutex);
        return result;
    }

    float* output_data = NULL;
    OrtStatus* st_mut = g_ort->GetTensorMutableData(output_val, (void**)&output_data);

    /* Verify output tensor size */
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
    pthread_mutex_unlock(&recognizer->mutex);
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
