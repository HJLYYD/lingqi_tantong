# LingQi TanTong -- Detailed Bug-Fix Implementation Plan

## Overview

This plan addresses 10 bugs (A through J) in a C11 real-time AI vision pipeline
running on SpacemiT K1 RISC-V (riscv64, Clang 19, RVV 1.0).  The fixes are
ordered from simplest/local to most complex/spanning.

---

## Bug A: Partial-body flags overwritten by struct copy

**File:** `src/inference_pipeline.c`
**Function:** `filter_detections` (lines 262--426)
**Severity:** Logic -- partial-body flag is always false in output.

### Root Cause

Lines 387--401 assign metadata fields (`is_partial_body`,
`num_visible_keypoints`, confidence boost) on `output[filtered]`.  Line 407
then does `output[filtered++] = *det;` which is a whole-struct copy from the
input detection `*det` -- the input detections come from `poses_to_detections`
(lines 437--455) which calls `detection_init` / memset and never touches those
fields, so they are always zero.  The metadata written on lines 387--401 is
immediately overwritten.

The three code paths within the `kpt_filter_enabled` block:

| Lines | Tier passed           | Writes to output[filtered]          |
|-------|-----------------------|-------------------------------------|
| 387--390 | Side-body          | is_partial_body=true, nvis, +boost  |
| 393--396 | Upper-body         | is_partial_body=true, nvis, +boost  |
| 399--401 | Full-body          | is_partial_body=false, nvis=17      |

All are erased by line 407.

### Fix

Replace the single line 407 with a three-step sequence: copy, re-apply metadata,
then advance the index.

**Old (line 407):**

```c
        output[filtered++] = *det;
```

**New (replace entire line 407):**

```c
        output[filtered] = *det;

        /* Re-apply metadata that was set on lines 387-401 above,
         * because the struct copy from input det overwrites it.
         * The three code paths above already set the correct values
         * on output[filtered] -- but the metadata was set BEFORE the
         * copy, so we need to re-set it AFTER the copy.
         *
         * We use a local helper to avoid duplicating the three-tier
         * conditional.  Each of the three paths above now needs to
         * SAVE their decisions so we can replay them here.
         *
         * SIMPLEST APPROACH: move the three-path setting AFTER the copy. */
```

Wait -- re-reading more carefully, the simplest fix is to move the metadata
assignment AFTER the copy, not before.  Let me restructure:

**Old (lines 387--407):**

```c
                        /* Side-body passed -- mark as partial */
                        output[filtered].is_partial_body = true;
                        output[filtered].num_visible_keypoints =
                            (n_left >= n_right && n_left >= 3) ? n_left : n_right;
                        output[filtered].confidence += TRACKING_PARTIAL_BODY_CONF_BOOST;
                    } else {
                        /* Upper-body passed -- mark as partial */
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
```

**New (restructured):**

```c
                        /* Side-body passed -- mark as partial */
                        is_partial = true;
                        nvis = (n_left >= n_right && n_left >= 3) ? n_left : n_right;
                        apply_boost = true;
                    } else {
                        /* Upper-body passed -- mark as partial */
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

        /* Copy detection first, then apply metadata AFTER the copy
         * so the partial-body / keypoint fields are not overwritten. */
        output[filtered] = *det;
        output[filtered].is_partial_body = is_partial;
        output[filtered].num_visible_keypoints = nvis;
        if (apply_boost) {
            output[filtered].confidence += TRACKING_PARTIAL_BODY_CONF_BOOST;
        }
        filtered++;
```

Requires three new local variables declared at the top of the `for` loop body:

```c
        bool is_partial = false;
        int nvis = 0;
        bool apply_boost = false;
```

### Verification

After the fix, add a unit test that:
1. Creates a Detection with `is_partial_body = false` (as `poses_to_detections` does).
2. Runs `filter_detections` with a side-body pose that passes tier-3.
3. Asserts that the output `is_partial_body == true` and `num_visible_keypoints > 0`.

In integration, log the `is_partial_body` field in the per-frame summary and
verify that partial-body detections appear when people are partially occluded.

### RISC-V memory-model safety

No atomic/shared memory involved -- purely local stack variables and a
single-threaded write to `output[filtered]`.  No barriers needed.

---

## Bug B: Multi-person poses interleaved in ST-GCN buffer

**File:** `src/inference_pipeline.c`
**Function:** `inference_pipeline_process_frame` (lines 795--810)
**Severity:** Logic -- ST-GCN temporal semantics corrupted in multi-person scenes.

### Root Cause

Lines 801--803 iterate over ALL poses in the frame and push each one into the
ST-GCN sliding window:

```c
        for (int p = 0; p < out_result->num_poses; p++) {
            stgcn_action_recognizer_push_pose(pipeline->action_recognizer, &out_result->poses[p], width, height);
        }
```

The ST-GCN expects ONE skeleton per timestep in its buffer.  When three people
are visible, three unrelated poses are pushed sequentially into the buffer at
the same frame index `t`, destroying the temporal sequence the model was
trained on (typically NTU-RGB+D where each frame has exactly one skeleton).

### Fix

Select only the primary person's pose.  "Primary" is defined as:
1. The pose matched to the detection with the highest tracking confidence, OR
2. The pose with the highest keypoint confidence score.

**Old (lines 801--803):**

```c
        for (int p = 0; p < out_result->num_poses; p++) {
            stgcn_action_recognizer_push_pose(pipeline->action_recognizer, &out_result->poses[p], width, height);
        }
```

**New:**

```c
        /* ST-GCN expects ONE skeleton per timestep.  Select the primary
         * (highest-confidence) pose to avoid corrupting the temporal buffer
         * with interleaved multi-person skeletons. */
        int best_pose_idx = -1;
        float best_pose_conf = -1.0f;
        for (int p = 0; p < out_result->num_poses; p++) {
            if (out_result->poses[p].confidence > best_pose_conf) {
                best_pose_conf = out_result->poses[p].confidence;
                best_pose_idx = p;
            }
        }
        if (best_pose_idx >= 0) {
            stgcn_action_recognizer_push_pose(pipeline->action_recognizer,
                &out_result->poses[best_pose_idx], width, height);
        }
```

### Alternative considered

Push one pose per timestep by cycling through persons round-robin.  This would
preserve multi-person action recognition but requires a per-person ST-GCN
instance or multi-person model -- out of scope for this fix.

### Verification

1. Create a synthetic frame with 2+ poses at different confidences.
2. Run `inference_pipeline_process_frame`.
3. Assert `stgcn_action_recognizer_push_pose` is called exactly once with the
   highest-confidence pose.

---

## Bug C: ST-GCN string literal freed via AllocatorFree

**File:** `src/stgcn_action_recognizer.c`
**Function:** `stgcn_action_recognizer_recognize` (lines 285--517)
**Severity:** Crash -- undefined behavior on `AllocatorFree` of a string literal.

### Root Cause

Lines 431--439:
```c
        if (s) {
            g_ort->ReleaseStatus(s);
            output_names[0] = "output";       // <-- string literal
        } else {
            output_names[0] = name_ptr;        // <-- allocator-owned
        }
    } else {
        output_names[0] = "output";            // <-- string literal
    }
```

Line 448--451:
```c
    if (allocator && output_names[0]) {
        OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)output_names[0]);
        if (st_f) g_ort->ReleaseStatus(st_f);
    }
```

When `output_names[0]` holds `"output"` (a string literal in `.rodata`),
`AllocatorFree` tries to free non-heap memory, causing a crash.

### Fix

Track whether the name was allocator-allocated with a boolean flag, and only
call `AllocatorFree` when the flag is true.

**Add a local variable:**

```c
    bool output_name_from_allocator = false;
```

**In the allocator branch (lines 428--439), modify:**

```c
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
```

**In the AllocatorFree call (lines 448--451), change condition:**

```c
    if (allocator && output_names[0] && output_name_from_allocator) {
        OrtStatus* st_f = g_ort->AllocatorFree(allocator, (void*)output_names[0]);
        if (st_f) g_ort->ReleaseStatus(st_f);
    }
```

### RISC-V weak-memory consideration

`output_name_from_allocator` is a local stack variable in a single-threaded
function (the mutex is held throughout).  No ordering concern.

---

## Bug D: ST-GCN model_loaded set with NULL ctx

**File:** `src/stgcn_action_recognizer.c`
**Function:** `stgcn_action_recognizer_load_model` (lines 105--236)
**Severity:** Logic -- downstream code may assume model is ready when `ctx` is NULL.

### Root Cause

Line 202 sets `recognizer->model_loaded = true;` unconditionally, but lines
124--130 may leave `recognizer->ctx == NULL` if `ort_ctx_create` fails:

```c
    recognizer->ctx = ort_ctx_create(recognizer->session,
                                      recognizer->num_frames,
                                      recognizer->num_keypoints,
                                      STGCN_NUM_CHANNELS);
    if (recognizer->ctx) {
        recognizer->ctx->input_name[0] = '\0';
    }
    // ... many lines ...
    recognizer->model_loaded = true;   // line 202 -- ctx may be NULL!
```

### Fix

Move `model_loaded = true` to only after confirming `ctx != NULL`.  There are
two approaches:

**Approach A (minimal):** Check ctx before setting the flag.

**Old:**
```c
    recognizer->model_loaded = true;
```

**New:**
```c
    /* Only mark model loaded if context was successfully created.
     * ctx is required for inference (provides memory_info, allocator, etc.). */
    recognizer->model_loaded = (recognizer->ctx != NULL);
```

**Approach B (recommended -- fail early):** Return false when ctx creation fails.

WARNING: If we return false here, the recognizer is destroyed in the caller
(`stgcn_action_recognizer_create` line 81--85).  This is the correct behavior
since inference cannot proceed without a context.  Currently the code silently
proceeds, and `recognize()` will crash at line 333 when dereferencing
`recognizer->ctx->memory_info`.

**Add after line 130:**

```c
    if (!recognizer->ctx) {
        log_error("STGCNActionRecognizer: failed to create inference context");
        const OrtApi* ort = ort_get_api();
        if (recognizer->session && ort) {
            ort->ReleaseSession(recognizer->session);
            recognizer->session = NULL;
        }
        return false;
    }
```

This follows the pattern in `yolov5_face_detector_load_model` (lines 137--143).

With this change, line 202 (`recognizer->model_loaded = true;`) is only reached
when `ctx != NULL`, so it is correct as-is.  However, adding the safety check
`(recognizer->ctx != NULL)` is still good defensive practice.

### Combined fix

1. Add early-return check after line 130.
2. Change line 202 to `recognizer->model_loaded = true;` (no change needed if
   we fix it with the early return -- but add the guard anyway).

---

## Bug E: ONNX Session leak on ort_ctx_create failure

**File:** `src/yolov8_pose_estimator.c`
**Function:** `yolov8_pose_estimator_load_model` (lines 94--142)
**Severity:** Resource leak.

### Root Cause

Lines 134--138:
```c
    est->ctx = ort_ctx_create(est->session, est->input_width, est->input_height, 3);
    if (!est->ctx) {
        log_error("YOLOv8Pose: failed to create inference context");
        return false;        // <-- est->session is leaked here
    }
```

### Fix

**Old (lines 134--138):**

```c
    est->ctx = ort_ctx_create(est->session, est->input_width, est->input_height, 3);
    if (!est->ctx) {
        log_error("YOLOv8Pose: failed to create inference context");
        return false;
    }
```

**New (follow the pattern from yolov5_face_detector.c lines 137--143):**

```c
    est->ctx = ort_ctx_create(est->session, est->input_width, est->input_height, 3);
    if (!est->ctx) {
        log_error("YOLOv8Pose: failed to create inference context");
        const OrtApi* ort = ort_get_api();
        if (est->session && ort) {
            ort->ReleaseSession(est->session);
            est->session = NULL;
        }
        return false;
    }
```

---

## Bug F: ITG3205 gyro scale 8x too low

**File:** `src/k1_imu.c`
**Location:** Line 34
**Severity:** Gyroscope readings are 1/8 of true value (250/2000).

### Root Cause

```c
#define GYRO_SCALE   ((250.0f * (float)M_PI / 180.0f) / 32768.0f)
```

The ITG3205 datasheet specifies:
- Fixed full-scale range: +/-2000 deg/s (NOT configurable)
- Sensitivity: 14.375 LSB/(deg/s)
- The formula is: rad/s = raw * (pi/180) / 14.375

The current code assumes +/-250 deg/s (a different gyro's configurable range),
which gives a scale 8x too small.

### Fix

**Old:**
```c
#define GYRO_SCALE   ((250.0f * (float)M_PI / 180.0f) / 32768.0f) /* ±250dps → rad/s */
```

**New (option 1 -- datasheet formula):**
```c
#define GYRO_SCALE   (((float)M_PI / 180.0f) / 14.375f)  /* ±2000dps → rad/s (ITG3205 fixed) */
```

**New (option 2 -- conceptually clearer, same result):**
```c
#define GYRO_SCALE   ((2000.0f * (float)M_PI / 180.0f) / 32768.0f) /* ±2000dps → rad/s */
```

Both give identical result: `2000 * pi/180 / 32768 = pi/180 / 14.375`.
Option 1 is preferred because it directly matches the datasheet's stated
sensitivity (14.375 LSB/(deg/s)), making the relationship to the hardware
explicit.

### Verification

1. Read raw register values from ITG3205 with known angular velocity.
2. Convert to rad/s using both old and new formulas.
3. Assert new values are 8x larger than old values.
4. With gyro at rest, bias should be near zero after calibration (unchanged
   behavior since bias is sampled and subtracted -- calibration is unaffected).

---

## Bug G: ESP32 IMU dt hardcoded to 0.01s for 1Hz data

**File:** `src/imu_handler.c`
**Functions:** `imu_handler_feed_k1_imu` (line 317), `imu_handler_feed_external_raw` (line 350)
**Severity:** Madgwick filter diverges due to 100x incorrect integration step.

### Root Cause

Both `imu_handler_feed_k1_imu` (line 317) and `imu_handler_feed_external_raw`
(line 350) use:
```c
float dt = 0.01f;  /* 100Hz */
```

The ESP32 IMU data arrives at ~1 Hz (one CoAP packet per frame).  Using dt=0.01
for 1 Hz data makes the Madgwick filter under-integrate gyro by 100x and
over-weight the accelerometer correction by 100x, causing divergence.

The K1 local IMU (GY85 over I2C) actually runs at ~100 Hz, so dt=0.01 is
correct for the K1 feed path -- but the same hardcoded value is used there too,
which is brittle.  Both paths should use actual timestamps.

### Fix

Add a per-filter `last_timestamp` field to `MadgwickFilter` and compute `dt`
from actual timestamp differences.

**Step 1: Extend MadgwickFilter struct (in `core_types.h` or local to imu_handler.c)**

```c
typedef struct {
    float qw, qx, qy, qz;
    float beta;
    float sample_freq;
    float integral_fb[3];
    bool  initialized;
    double last_timestamp;   /* NEW: for dt computation */
} MadgwickFilter;
```

**Step 2: Update `madgwick_init` to initialize `last_timestamp = 0.0`**

Set `mf->last_timestamp = 0.0;` in the memset (already zero via calloc).

**Step 3: Update `imu_handler_feed_external_raw` (ESP32 path)**

**Old (line 346--352):**

```c
void imu_handler_feed_external_raw(IMUHandler* h, const float accel[3], const float gyro[3]) {
    if (!h || !accel || !gyro) return;

    /* Madgwick 更新相机滤波器 */
    float dt = 0.01f;  /* 100Hz */
    madgwick_update(&h->cam_filter, gyro[0], gyro[1], gyro[2],
                    accel[0], accel[1], accel[2], dt);
```

**New:**

```c
void imu_handler_feed_external_raw(IMUHandler* h, const float accel[3], const float gyro[3]) {
    if (!h || !accel || !gyro) return;

    /* Compute actual dt from wall-clock time.
     * ESP32 data arrives at ~1Hz, but dt=0.01 would cause filter divergence.
     * Use monotonic clock for cross-platform compatibility. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double now = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;

    float dt;
    if (h->cam_filter.last_timestamp > 0.0) {
        dt = (float)(now - h->cam_filter.last_timestamp);
        /* Clamp to prevent insane values on clock jumps */
        if (dt < 0.001f) dt = 0.001f;
        if (dt > 10.0f)  dt = 0.01f;   /* first sample or long pause: default */
    } else {
        dt = 0.01f;  /* first sample: assume 100Hz */
    }
    h->cam_filter.last_timestamp = now;

    madgwick_update(&h->cam_filter, gyro[0], gyro[1], gyro[2],
                    accel[0], accel[1], accel[2], dt);
```

**Step 4: Update `imu_handler_feed_k1_imu` (K1 local IMU path)**

Same pattern using `h->k1_filter.last_timestamp`:

**Old (line 313--319):**

```c
void imu_handler_feed_k1_imu(IMUHandler* h, const IMUData* data) {
    if (!h || !data) return;

    /* Madgwick 更新 K1 滤波器 */
    float dt = 0.01f;  /* 100Hz */
    madgwick_update(&h->k1_filter, data->gyro_x, data->gyro_y, data->gyro_z,
                    data->accel_x, data->accel_y, data->accel_z, dt);
```

**New:**

```c
void imu_handler_feed_k1_imu(IMUHandler* h, const IMUData* data) {
    if (!h || !data) return;

    /* Compute actual dt from data timestamp (set by k1_imu_read_sample
     * which uses CLOCK_MONOTONIC).  K1 IMU typically runs at ~100Hz. */
    float dt;
    if (h->k1_filter.last_timestamp > 0.0) {
        dt = (float)(data->timestamp - h->k1_filter.last_timestamp);
        if (dt < 0.001f) dt = 0.001f;
        if (dt > 10.0f)  dt = 0.01f;
    } else {
        dt = 0.01f;
    }
    h->k1_filter.last_timestamp = data->timestamp;

    madgwick_update(&h->k1_filter, data->gyro_x, data->gyro_y, data->gyro_z,
                    data->accel_x, data->accel_y, data->accel_z, dt);
```

### Verification

1. Feed synthetic IMU data at 1 Hz with known angular velocity.
2. With fixed dt=0.01, the orientation quaternion diverges rapidly.
3. With computed dt=1.0, the orientation converges correctly.
4. Log actual dt values in debug output to confirm correct values.

---

## Bug H: Acceleration threshold never triggers for ADXL345

**File:** `src/imu_handler.c`
**Function:** `madgwick_update` (line 40)
**Severity:** Logic -- acceleration-based gyro-only fallback never activates.

### Root Cause

Line 40:
```c
if (a_norm < 0.5f || a_norm > 3.0f * 9.81f) {
```

ADXL345 at +/-2g range outputs max +/-19.6 m/s^2 (2 * 9.81).  The threshold
check `a_norm > 29.4 m/s^2` can never be reached.  The purpose of this check is
to detect high-acceleration states (fast motion, impact) where the accelerometer
is not a reliable gravity reference, and fall back to gyro-only integration.

### Fix

Change the upper threshold to 2.0g (19.6 m/s^2) to match the sensor's range,
and change the lower threshold to 0.5g (4.9 m/s^2) to also flag free-fall /
zero-g conditions.

**Old (line 40):**

```c
if (a_norm < 0.5f || a_norm > 3.0f * 9.81f) {
```

**New:**

```c
/* ADXL345 at ±2g: max output is 2*9.81 = 19.6 m/s².
 * Thresholds: < 0.5g (free-fall / microgravity) or > 2.0g (high acceleration)
 * trigger gyro-only fallback (accelerometer unreliable as gravity reference). */
if (a_norm < 0.5f * 9.81f || a_norm > 2.0f * 9.81f) {
```

### Note on Espressif ESP32 ADXL345 config

If the ESP32 side also uses ADXL345 at +/-2g, the same fix applies there.  The
ESP32-side IMU handler is in `imu_handler.c` function `madgwick_update` which
is shared -- the fix in `imu_handler.c` covers both the K1 filter and the
camera filter (since both use the same `madgwick_update` function).

### Verification

1. Read ADXL345 at rest (a_norm ~= 9.81 m/s^2).  Filter runs normally.
2. Simulate free-fall by zeroing acceleration.  With old threshold,
   `a_norm=0 < 0.5` is actually already handled (the lower check works).
   The main fix is the upper threshold.
3. Simulate impact (> 2g, e.g. 18 m/s^2).  With old threshold at 29.4,
   gyro-only fallback never activates.  With new threshold at 19.6, it
   activates correctly.

---

## Bug I: Result storage -- tracked objects never saved

**Files:** `src/system_controller.c`, `include/result_manager.h`, `src/result_manager.c`
**Functions:** Post-process threads, `result_manager_add_tracked_object`
**Severity:** Data loss -- session reports show zero tracked objects.

### Root Cause

`result_manager_add_tracked_object` (defined in `result_manager.c` line 103)
is declared in `result_manager.h` line 56 but is never called anywhere in the
codebase.  After tracking completes in the post-process thread, the tracked
objects are processed for display but never stored in the result manager.

### Fix

Add calls to `result_manager_add_tracked_object` in both places where tracking
results are processed:

1. **`k1_postprocess_thread`** (system_controller.c, around line 608--622)
2. **`system_controller_process_video`** (system_controller.c, around line 1367--1381)

**In `k1_postprocess_thread`, add after the existing per-object loop (after line 622):**

```c
        /* ── Save tracked objects to result manager ── */
        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            /* Only save confirmed tracks to avoid noise from transient detections */
            if (obj->frames_seen >= 3) {
                result_manager_add_tracked_object(sc->result_manager,
                    obj->track_id,
                    obj->height_meters,
                    obj->frames_seen,
                    obj->has_pose ? 1 : 0);
            }
        }
```

**In `system_controller_process_video`, add after the existing per-object loop (after line 1381):**

```c
        /* ── Save tracked objects to result manager ── */
        for (int i = 0; i < tracking.num_tracked; i++) {
            TrackedObject* obj = &tracking.tracked_objects[i];
            if (obj->frames_seen >= 3) {
                result_manager_add_tracked_object(sc->result_manager,
                    obj->track_id,
                    obj->height_meters,
                    obj->frames_seen,
                    obj->has_pose ? 1 : 0);
            }
        }
```

### Additional: per-frame result saving

The result manager already has `result_manager_save_frame` for saving
individual frame images.  To add per-frame metadata (poses, faces, actions),
define a new function:

**New function in `result_manager.h` / `result_manager.c`:**

```c
/**
 * Save per-frame inference metadata as JSON lines (.jsonl).
 * Each line is a JSON object with frame number, detections, poses, faces, actions.
 */
int result_manager_save_frame_metadata(const ResultManager* rm,
                                        const char* session_id,
                                        const InferenceResult* result,
                                        int frame_num);
```

Implementation writes to `{output_dir}/frames/{session_id}_metadata.jsonl`,
appending one JSON line per frame.

### Verification

After the fix, session reports should show non-zero tracked object counts.
The JSON report (`reports/session_*/report.json`) should have populated
`tracked_objects` array.

---

## Bug J: Spatial trajectories never saved before shutdown

**Files:** `src/system_controller.c`, `src/result_manager.c`, `include/result_manager.h`
**Functions:** `result_manager_end_session`, `spatial_engine_get_trajectory`
**Severity:** Data loss -- 3D spatial trajectories are computed but never persisted.

### Root Cause

`spatial_engine_get_trajectory` (spatial_engine.h line 88) is never called in
the codebase.  At shutdown, `spatial_engine_clear_trajectories` or
`spatial_engine_destroy` frees trajectory data without saving it.

### Fix

Extend `result_manager_end_session` to:
1. Get active track IDs from the spatial engine.
2. For each track, get the trajectory via `spatial_engine_get_trajectory`.
3. Write trajectories to the JSON report.

**Step 1: Pass spatial engine to result_manager_end_session**

The current signature is:
```c
const char* result_manager_end_session(ResultManager* rm);
```

Change to:
```c
const char* result_manager_end_session(ResultManager* rm, SpatialLocalizationEngine* spatial);
```

If `spatial` is non-NULL, iterate active tracks and save trajectories.

**Step 2: Update all callers**

There are two callers:
1. `system_controller_process_realtime_k1` (line 966):
   ```c
   result_manager_end_session(sc->result_manager);
   ```
   Change to:
   ```c
   result_manager_end_session(sc->result_manager, sc->spatial_engine);
   ```

2. `system_controller_process_video` (line 1448):
   ```c
   result_manager_end_session(sc->result_manager);
   ```
   Change to:
   ```c
   result_manager_end_session(sc->result_manager, sc->spatial_engine);
   ```

**Step 3: Implement trajectory saving in result_manager_end_session**

```c
const char* result_manager_end_session(ResultManager* rm, SpatialLocalizationEngine* spatial) {
    if (!rm || !rm->current_session) {
        log_warning("No active session to end");
        return "";
    }

    SessionResult* session = rm->current_session;

    /* ── Save spatial trajectories before ending the session ── */
    if (spatial) {
        int track_ids[SPATIAL_MAX_PERSONS];
        int num_active = spatial_engine_get_active_tracks(spatial, track_ids, SPATIAL_MAX_PERSONS);

        /* Store trajectory summary in session tracked_objects */
        for (int i = 0; i < num_active && session->num_tracked_objects < RM_MAX_TRACKED_OBJS; i++) {
            int tid = track_ids[i];
            int traj_count = 0;
            const SpatialPosition* traj = spatial_engine_get_trajectory(spatial, tid, &traj_count);
            if (traj && traj_count > 0) {
                TrackedObjectSummary* obj = &session->tracked_objects[session->num_tracked_objects++];
                obj->track_id = tid;
                obj->position_count = traj_count;
                /* Height not directly available here; caller should have set it */
            }
        }
    }

    get_iso_time(session->end_time, sizeof(session->end_time));
    session->active = false;

    /* ... rest of existing end_session logic ... */
```

**Step 4: Add trajectory data to JSON report**

Extend the JSON report at `save_json_report` to include trajectory points
for each tracked object.  For each object, add:

```json
      "trajectory": [
        {"x": 0.5, "y": 1.2, "z": 3.0},
        {"x": 0.6, "y": 1.3, "z": 3.1}
      ]
```

This requires storing trajectory data in `TrackedObjectSummary`.  Since
trajectories can be up to 300 points, we have two options:

**Option A (small):** Store only trajectory summary (start/end position,
total distance).  Modify `TrackedObjectSummary`:

```c
typedef struct {
    int track_id;
    float height_meters;
    int position_count;
    int pose_count;
    float start_x, start_y, start_z;  /* NEW */
    float end_x, end_y, end_z;        /* NEW */
    float total_distance_m;           /* NEW */
} TrackedObjectSummary;
```

**Option B (full):** Store the full trajectory in a separate file
(suggested: `{output_dir}/trajectories/{session_id}_track_{id}.csv`).

Recommend Option A for the in-memory summary (it's already part of the
session report), plus Option B for full trajectory export.

**New helper function:**

```c
int result_manager_save_trajectory_csv(const ResultManager* rm,
                                        const char* session_id,
                                        int track_id,
                                        const SpatialPosition* traj,
                                        int count);
```

This writes a CSV with columns: `index, x, y, z, confidence, world_x, world_z`.

### RISC-V memory-model safety

`spatial_engine_get_trajectory` returns a pointer to the engine's internal
trajectory buffer.  At session-end time, all pipeline threads have been joined,
so there is no concurrent access.  No barriers needed.

---

## Dependency order for implementation

```
Bug C  (stgcn free literal)     -- standalone, no deps
Bug D  (stgcn model_loaded)     -- standalone, no deps
Bug E  (pose session leak)       -- standalone, no deps
Bug F  (gyro scale)              -- standalone, no deps
Bug H  (accel threshold)         -- standalone, no deps

Bug G  (IMU dt)                  -- modifies MadgwickFilter struct
  depends on: nothing structural

Bug A  (partial-body overwrite)  -- modifies filter_detections logic
  depends on: understanding Detection struct (already known)

Bug B  (ST-GCN interleave)       -- modifies push_pose call site
  depends on: Bug D (so recognizer works at all)

Bug I  (result storage)          -- adds calls to result_manager
  depends on: understanding TrackingResult, TrackedObject

Bug J  (trajectory save)         -- modifies result_manager_end_session signature
  depends on: Bug I (both touch session end)
  affects: system_controller.c (both realtime + video paths)
```

Recommended implementation order: C, D, E, F, H, G, A, B, I, J.

---

## Clang 19 / RISC-V specific notes

1. **`volatile` for shared flags**: The pipeline already uses `volatile bool`
   for `running` and `shutdown` flags across threads.  All new code follows the
   same pattern where applicable.

2. **`__sync_synchronize()` barriers**: Already used in the pipeline ring buffer
   operations.  The bug fixes don't introduce new inter-thread data that would
   need barriers -- all new shared data is accessed under existing mutexes or
   at session-end when threads are joined.

3. **No VLA or alloca**: All new code uses fixed-size arrays or heap allocation
   via `calloc`/`malloc`, consistent with existing code.

4. **-Werror compliance**: No new warnings introduced.  The string literal fix
   (Bug C) avoids the `-Wfree-nonheap-object` warning that would fire with
   newer Clang versions.
