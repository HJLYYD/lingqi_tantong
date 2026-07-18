#include "keypoint_validator.h"
#include "logger.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct KeypointValidator {
    KeypointValidatorConfig config;
    int frame_count;
};

/* ── Helpers ── */

static inline bool kpt_valid(const PoseEstimation* pose, int idx, float min_conf) {
    return (idx < pose->num_keypoints &&
            pose->keypoints[idx].confidence >= min_conf &&
            pose->keypoints[idx].x >= 0.0f &&
            pose->keypoints[idx].y >= 0.0f);
}

static inline float kpt_dist(const Keypoint* a, const Keypoint* b) {
    float dx = a->x - b->x;
    float dy = a->y - b->y;
    return sqrtf(dx * dx + dy * dy);
}

static inline float clamp_score(float val, float lo, float hi) {
    if (val <= lo) return 0.0f;
    if (val >= hi) return 1.0f;
    return (val - lo) / (hi - lo);
}

/* ── Lifecycle ── */

KeypointValidator* keypoint_validator_create(const KeypointValidatorConfig* config) {
    KeypointValidator* kv = (KeypointValidator*)calloc(1, sizeof(KeypointValidator));
    if (!kv) return NULL;

    if (config) {
        kv->config = *config;
    } else {
        kv->config.min_keypoints       = KV_DEFAULT_MIN_KEYPOINTS;
        kv->config.kpt_conf_threshold  = KV_DEFAULT_KPT_CONF_THRESHOLD;
        kv->config.validity_threshold  = KV_DEFAULT_VALIDITY_THRESHOLD;
        kv->config.in_bbox_ratio       = KV_DEFAULT_IN_BBOX_RATIO;
        kv->config.symmetry_tolerance  = KV_DEFAULT_SYMMETRY_TOLERANCE;
        kv->config.debug_frame_interval = 0;
    }

    kv->frame_count = 0;
    log_info("KeypointValidator created: min_kpts=%d conf=%.2f threshold=%.2f",
             kv->config.min_keypoints, (double)kv->config.kpt_conf_threshold,
             (double)kv->config.validity_threshold);
    return kv;
}

void keypoint_validator_destroy(KeypointValidator* kv) {
    free(kv);
}

/* ── Count valid keypoints ── */

int keypoint_validator_count_valid(const PoseEstimation* pose, float min_conf) {
    if (!pose || pose->num_keypoints <= 0) return 0;
    int count = 0;
    for (int i = 0; i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f &&
            pose->keypoints[i].y >= 0.0f) {
            count++;
        }
    }
    return count;
}

/* ── Full anatomical validation ── */

KeypointValidityResult keypoint_validator_validate(
    KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox)
{
    KeypointValidityResult r;
    memset(&r, 0, sizeof(r));

    if (!kv || !pose || !bbox) {
        return r;
    }

    r.total_keypoints = pose->num_keypoints;
    float min_conf = kv->config.kpt_conf_threshold;

    /* ══════════════════════════════════════════════════════════════
     * Check 1: Minimum valid keypoint count (weight: 0.15)
     * ══════════════════════════════════════════════════════════════ */
    r.valid_keypoint_count = keypoint_validator_count_valid(pose, min_conf);
    float score_count = clamp_score((float)r.valid_keypoint_count,
                                     (float)kv->config.min_keypoints * 0.5f,
                                     (float)kv->config.min_keypoints * 1.5f);
    r.failure_reasons[0] = score_count;

    /* ── NEW: Hard threshold on minimum keypoint count ──
     * Raised from min_keypoints/2 to min_keypoints.  Objects that
     * produce scattered keypoints won't reach this bar. */
    if (r.valid_keypoint_count < kv->config.min_keypoints) {
        r.anatomical_score = score_count * 0.15f;
        r.is_valid_human = false;
        r.num_failures = 1;
        return r;
    }

    /* ── NEW: Mandatory anatomical gate ──
     * At least one of: both shoulders, both hips, both knees, OR
     * nose + one shoulder must be visible.  Objects that produce
     * scattered keypoints without any paired limb structure are
     * rejected outright. */
    {
        bool has_paired_shoulders = kpt_valid(pose, KPT_LEFT_SHOULDER, min_conf) &&
                                    kpt_valid(pose, KPT_RIGHT_SHOULDER, min_conf);
        bool has_paired_hips     = kpt_valid(pose, KPT_LEFT_HIP, min_conf) &&
                                    kpt_valid(pose, KPT_RIGHT_HIP, min_conf);
        bool has_paired_knees    = kpt_valid(pose, KPT_LEFT_KNEE, min_conf) &&
                                    kpt_valid(pose, KPT_RIGHT_KNEE, min_conf);
        bool has_nose_shoulder   = kpt_valid(pose, KPT_NOSE, min_conf) &&
                                   (kpt_valid(pose, KPT_LEFT_SHOULDER, min_conf) ||
                                    kpt_valid(pose, KPT_RIGHT_SHOULDER, min_conf));

        if (!has_paired_shoulders && !has_paired_hips &&
            !has_paired_knees && !has_nose_shoulder) {
            /* No paired limb structure — not a human */
            r.anatomical_score = 0.0f;
            r.is_valid_human = false;
            r.num_failures = 6;
            return r;
        }
    }

    /* ── NEW: Head-at-top enforcement ──
     * If nose is visible, it MUST be in the top portion of the bbox.
     * Real humans always have their head at the top of the bounding box;
     * objects do not. */
    if (kpt_valid(pose, KPT_NOSE, min_conf)) {
        float nose_y = pose->keypoints[KPT_NOSE].y;
        float bbox_h = bbox_height(bbox);
        float nose_rel_y = (nose_y - bbox->y_min) / UTILS_MAX(bbox_h, 1.0f);
        if (nose_rel_y > KV_MANDATORY_HEAD_AT_TOP) {
            /* Nose is NOT at the top of the bbox → not a human */
            r.anatomical_score = 0.0f;
            r.is_valid_human = false;
            r.num_failures = 6;
            return r;
        }
    }

    /* ══════════════════════════════════════════════════════════════
     * Check 2: Shoulder pair validity (weight: 0.25)
     * ══════════════════════════════════════════════════════════════ */
    float score_shoulders = 0.0f;
    bool has_both_shoulders = kpt_valid(pose, KPT_LEFT_SHOULDER, min_conf) &&
                              kpt_valid(pose, KPT_RIGHT_SHOULDER, min_conf);

    if (has_both_shoulders) {
        const Keypoint* ls = &pose->keypoints[KPT_LEFT_SHOULDER];
        const Keypoint* rs = &pose->keypoints[KPT_RIGHT_SHOULDER];

        /* Shoulder width should be reasonable relative to bbox width */
        float shoulder_w = fabsf(ls->x - rs->x);
        float bbox_w = bbox_width(bbox);
        float sw_ratio = (bbox_w > 1.0f) ? (shoulder_w / bbox_w) : 0.0f;

        /* Human shoulders are typically 0.15–0.90 of bbox width */
        float score_sw = clamp_score(sw_ratio, 0.10f, 0.90f);

        /* Shoulder vertical alignment — should be roughly horizontal */
        float shoulder_dy = fabsf(ls->y - rs->y);
        float bbox_h = bbox_height(bbox);
        float sym_tol = kv->config.symmetry_tolerance * bbox_h;
        float score_sym = (sym_tol > 0.0f) ? clamp_score(shoulder_dy / sym_tol, 0.0f, 2.0f) : 0.5f;
        score_sym = 1.0f - score_sym;  /* invert: small dy → high score */

        /* Shoulder width within image (not degenerate) */
        float score_degen = (shoulder_w > 3.0f) ? 1.0f : 0.0f;

        score_shoulders = score_sw * 0.4f + score_sym * 0.4f + score_degen * 0.2f;
    }
    r.failure_reasons[1] = score_shoulders;

    /* ══════════════════════════════════════════════════════════════
     * Check 3: Torso proportion (weight: 0.20)
     *   shoulder-hip vertical distance / shoulder width
     * ══════════════════════════════════════════════════════════════ */
    float score_torso = 0.0f;
    bool has_hips = kpt_valid(pose, KPT_LEFT_HIP, min_conf) ||
                    kpt_valid(pose, KPT_RIGHT_HIP, min_conf);

    if (has_both_shoulders && has_hips) {
        float shoulder_mid_y = (pose->keypoints[KPT_LEFT_SHOULDER].y +
                                pose->keypoints[KPT_RIGHT_SHOULDER].y) * 0.5f;
        float hip_mid_y;
        if (kpt_valid(pose, KPT_LEFT_HIP, min_conf) &&
            kpt_valid(pose, KPT_RIGHT_HIP, min_conf)) {
            hip_mid_y = (pose->keypoints[KPT_LEFT_HIP].y +
                         pose->keypoints[KPT_RIGHT_HIP].y) * 0.5f;
        } else if (kpt_valid(pose, KPT_LEFT_HIP, min_conf)) {
            hip_mid_y = pose->keypoints[KPT_LEFT_HIP].y;
        } else {
            hip_mid_y = pose->keypoints[KPT_RIGHT_HIP].y;
        }

        float shoulder_w = fabsf(pose->keypoints[KPT_LEFT_SHOULDER].x -
                                  pose->keypoints[KPT_RIGHT_SHOULDER].x);
        float torso_len = hip_mid_y - shoulder_mid_y;  /* positive = hips below shoulders */

        if (shoulder_w > 1.0f && torso_len > 0.0f) {
            float torso_ratio = torso_len / shoulder_w;
            score_torso = clamp_score(torso_ratio, KV_TORSO_RATIO_MIN, KV_TORSO_RATIO_MAX);
        }
    }
    r.failure_reasons[2] = score_torso;

    /* ══════════════════════════════════════════════════════════════
     * Check 4: Limb proportion (weight: 0.15)
     *   upper leg (hip→knee) / lower leg (knee→ankle)
     * ══════════════════════════════════════════════════════════════ */
    float score_limbs = 0.0f;
    int limb_pairs = 0;

    /* Right leg */
    if (kpt_valid(pose, KPT_RIGHT_HIP, min_conf) &&
        kpt_valid(pose, KPT_RIGHT_KNEE, min_conf) &&
        kpt_valid(pose, KPT_RIGHT_ANKLE, min_conf)) {
        float upper = kpt_dist(&pose->keypoints[KPT_RIGHT_HIP],
                                &pose->keypoints[KPT_RIGHT_KNEE]);
        float lower = kpt_dist(&pose->keypoints[KPT_RIGHT_KNEE],
                                &pose->keypoints[KPT_RIGHT_ANKLE]);
        if (lower > 1.0f) {
            float ratio = upper / lower;
            score_limbs += clamp_score(ratio, KV_LIMB_RATIO_MIN, KV_LIMB_RATIO_MAX);
            limb_pairs++;
        }
    }

    /* Left leg */
    if (kpt_valid(pose, KPT_LEFT_HIP, min_conf) &&
        kpt_valid(pose, KPT_LEFT_KNEE, min_conf) &&
        kpt_valid(pose, KPT_LEFT_ANKLE, min_conf)) {
        float upper = kpt_dist(&pose->keypoints[KPT_LEFT_HIP],
                                &pose->keypoints[KPT_LEFT_KNEE]);
        float lower = kpt_dist(&pose->keypoints[KPT_LEFT_KNEE],
                                &pose->keypoints[KPT_LEFT_ANKLE]);
        if (lower > 1.0f) {
            float ratio = upper / lower;
            score_limbs += clamp_score(ratio, KV_LIMB_RATIO_MIN, KV_LIMB_RATIO_MAX);
            limb_pairs++;
        }
    }

    if (limb_pairs > 0) {
        score_limbs /= (float)limb_pairs;
    } else {
        /* No limb data — zero score (missing data is NOT evidence of being human) */
        score_limbs = 0.0f;
    }
    r.failure_reasons[3] = score_limbs;

    /* ══════════════════════════════════════════════════════════════
     * Check 5: Keypoints within bbox (weight: 0.10)
     * ══════════════════════════════════════════════════════════════ */
    int kpts_in_bbox = 0;
    int kpts_valid_total = 0;
    for (int i = 0; i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f &&
            pose->keypoints[i].y >= 0.0f) {
            kpts_valid_total++;
            if (pose->keypoints[i].x >= bbox->x_min &&
                pose->keypoints[i].x <= bbox->x_max &&
                pose->keypoints[i].y >= bbox->y_min &&
                pose->keypoints[i].y <= bbox->y_max) {
                kpts_in_bbox++;
            }
        }
    }
    float in_bbox_ratio = (kpts_valid_total > 0) ?
        (float)kpts_in_bbox / (float)kpts_valid_total : 0.0f;
    float score_bbox = clamp_score(in_bbox_ratio,
                                    kv->config.in_bbox_ratio * 0.5f,
                                    kv->config.in_bbox_ratio);
    r.failure_reasons[4] = score_bbox;

    /* ══════════════════════════════════════════════════════════════
     * Check 6: Left-right symmetry (weight: 0.15)
     *   Paired keypoints (shoulders, hips, knees, ankles, eyes, ears)
     *   should be at roughly the same Y coordinate.
     * ══════════════════════════════════════════════════════════════ */
    static const int sym_pairs[][2] = {
        {KPT_LEFT_SHOULDER,  KPT_RIGHT_SHOULDER},
        {KPT_LEFT_HIP,       KPT_RIGHT_HIP},
        {KPT_LEFT_KNEE,      KPT_RIGHT_KNEE},
        {KPT_LEFT_ANKLE,     KPT_RIGHT_ANKLE},
        {KPT_LEFT_EYE,       KPT_RIGHT_EYE},
        {KPT_LEFT_EAR,       KPT_RIGHT_EAR},
    };
    const int num_sym_pairs = sizeof(sym_pairs) / sizeof(sym_pairs[0]);

    float sym_score_sum = 0.0f;
    int sym_pairs_valid = 0;
    float bbox_h = bbox_height(bbox);
    float max_sym_dy = kv->config.symmetry_tolerance * bbox_h;

    for (int p = 0; p < num_sym_pairs; p++) {
        int iL = sym_pairs[p][0];
        int iR = sym_pairs[p][1];
        if (kpt_valid(pose, iL, min_conf) && kpt_valid(pose, iR, min_conf)) {
            float dy = fabsf(pose->keypoints[iL].y - pose->keypoints[iR].y);
            if (max_sym_dy > 0.0f) {
                /* 0 dy → 1.0, >tolerance dy → 0.0 */
                float s = 1.0f - clamp_score(dy, 0.0f, max_sym_dy * 2.0f);
                if (s < 0.0f) s = 0.0f;
                sym_score_sum += s;
                sym_pairs_valid++;
            }
        }
    }
    float score_symmetry = (sym_pairs_valid > 0) ?
        sym_score_sum / (float)sym_pairs_valid : 0.0f;
    r.failure_reasons[5] = score_symmetry;

    /* ══════════════════════════════════════════════════════════════
     * Check 7: Head above shoulders (weight: implicit in shoulders)
     *   If nose is visible, it MUST be above shoulder mid-point.
     *   IF both nose and shoulders are visible and nose is BELOW
     *   shoulders, this is a strong non-human signal.
     * ══════════════════════════════════════════════════════════════ */
    if (kpt_valid(pose, KPT_NOSE, min_conf) && has_both_shoulders) {
        float shoulder_mid_y = (pose->keypoints[KPT_LEFT_SHOULDER].y +
                                pose->keypoints[KPT_RIGHT_SHOULDER].y) * 0.5f;
        float nose_y = pose->keypoints[KPT_NOSE].y;
        r.failure_reasons[6] = (nose_y < shoulder_mid_y) ? 1.0f : 0.0f;
    } else {
        r.failure_reasons[6] = 0.0f;  /* no head data → no evidence (zero, not neutral) */
    }

    /* ══════════════════════════════════════════════════════════════
     * Weighted composite score
     * ══════════════════════════════════════════════════════════════ */
    r.anatomical_score =
        r.failure_reasons[0] * 0.15f +   /* min keypoint count */
        r.failure_reasons[1] * 0.25f +   /* shoulder validity */
        r.failure_reasons[2] * 0.20f +   /* torso proportion */
        r.failure_reasons[3] * 0.15f +   /* limb proportion */
        r.failure_reasons[4] * 0.10f +   /* keypoints in bbox */
        r.failure_reasons[5] * 0.15f;    /* symmetry */
    /* Head check (r.failure_reasons[6]) is diagnostic only, not in score
     * because it's implicitly covered by shoulder+torso checks. */

    r.is_valid_human = (r.anatomical_score >= kv->config.validity_threshold);

    /* Count checks that failed significantly */
    r.num_failures = 0;
    float check_thresholds[6] = {0.3f, 0.2f, 0.2f, 0.2f, 0.3f, 0.2f};
    for (int c = 0; c < 6; c++) {
        if (r.failure_reasons[c] < check_thresholds[c]) r.num_failures++;
    }

    /* ── Diagnostic logging ── */
    kv->frame_count++;
    if (kv->config.debug_frame_interval > 0 &&
        kv->frame_count % kv->config.debug_frame_interval == 0) {
        log_info("KeypointValidator frame %d: score=%.2f valid=%d kpts=%d/%d "
                 "checks=[%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]",
                 kv->frame_count, (double)r.anatomical_score, r.is_valid_human,
                 r.valid_keypoint_count, r.total_keypoints,
                 (double)r.failure_reasons[0], (double)r.failure_reasons[1],
                 (double)r.failure_reasons[2], (double)r.failure_reasons[3],
                 (double)r.failure_reasons[4], (double)r.failure_reasons[5]);
    }

    return r;
}

/* ── Quick check ── */

bool keypoint_validator_quick_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox)
{
    if (!kv || !pose || !bbox) return false;

    float min_conf = kv->config.kpt_conf_threshold;

    /* Fast path 1: minimum keypoint count */
    int n_valid = keypoint_validator_count_valid(pose, min_conf);
    if (n_valid < kv->config.min_keypoints) return false;

    /* Fast path 2: at least ONE paired limb set visible */
    bool has_shoulders = kpt_valid(pose, KPT_LEFT_SHOULDER, min_conf) &&
                         kpt_valid(pose, KPT_RIGHT_SHOULDER, min_conf);
    bool has_hips     = kpt_valid(pose, KPT_LEFT_HIP, min_conf) &&
                         kpt_valid(pose, KPT_RIGHT_HIP, min_conf);
    bool has_knees    = kpt_valid(pose, KPT_LEFT_KNEE, min_conf) &&
                         kpt_valid(pose, KPT_RIGHT_KNEE, min_conf);

    if (!has_shoulders && !has_hips && !has_knees) {
        /* No paired limb set visible — can't validate anatomy */
        return false;
    }

    /* Fast path 3: shoulder symmetry if both visible */
    if (has_shoulders) {
        float dy = fabsf(pose->keypoints[KPT_LEFT_SHOULDER].y -
                          pose->keypoints[KPT_RIGHT_SHOULDER].y);
        float bbox_h = bbox_height(bbox);
        if (bbox_h > 0.0f && dy > kv->config.symmetry_tolerance * bbox_h * 3.0f) {
            return false;  /* severely asymmetric → not human */
        }
    }

    /* Fast path 4: at least 50% of valid keypoints in bbox */
    int in_bbox = 0;
    for (int i = 0; i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f && pose->keypoints[i].y >= 0.0f) {
            if (pose->keypoints[i].x >= bbox->x_min &&
                pose->keypoints[i].x <= bbox->x_max &&
                pose->keypoints[i].y >= bbox->y_min &&
                pose->keypoints[i].y <= bbox->y_max) {
                in_bbox++;
            }
        }
    }
    if (n_valid > 0 && (float)in_bbox / (float)n_valid < 0.4f) {
        return false;  /* most keypoints outside bbox → scattered noise */
    }

    return true;
}

/* ── Partial-Body Quick Checks ── */

bool keypoint_validator_upper_body_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox)
{
    if (!kv || !pose || !bbox) return false;

    float min_conf = kv->config.kpt_conf_threshold;

    /* Count upper-body keypoints (indices 0-10: nose through wrists) */
    int n_upper = 0;
    bool has_nose = false, has_lshoulder = false, has_rshoulder = false;
    for (int i = 0; i <= 10 && i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f && pose->keypoints[i].y >= 0.0f) {
            n_upper++;
            if (i == 0) has_nose = true;
            if (i == 5) has_lshoulder = true;
            if (i == 6) has_rshoulder = true;
        }
    }

    /* v2.6: Minimum 3 upper-body keypoints (was 4) — allows half-body
     * where only head + shoulders are visible (e.g. behind obstacle). */
    if (n_upper < 3) return false;

    /* At least one shoulder must be visible */
    if (!has_lshoulder && !has_rshoulder) return false;

    /* Head-above-shoulders check: only if nose + at least one shoulder visible */
    if (has_nose && (has_lshoulder || has_rshoulder)) {
        float sh_y = has_lshoulder ? pose->keypoints[5].y : pose->keypoints[6].y;
        if (pose->keypoints[0].y >= sh_y) return false; /* nose below shoulder */
    }

    /* Shoulder symmetry if both visible — relaxed for half-body */
    if (has_lshoulder && has_rshoulder) {
        float dy = fabsf(pose->keypoints[5].y - pose->keypoints[6].y);
        float bbox_h = bbox_height(bbox);
        if (bbox_h > 0.0f && dy > kv->config.symmetry_tolerance * bbox_h * 8.0f) {
            return false; /* extremely asymmetric shoulders — not human */
        }
    }

    /* In-bbox check: at least 50% of visible upper-body kpts in bbox */
    int in_bbox = 0, total = 0;
    for (int i = 0; i <= 10 && i < pose->num_keypoints; i++) {
        if (pose->keypoints[i].confidence >= min_conf &&
            pose->keypoints[i].x >= 0.0f && pose->keypoints[i].y >= 0.0f) {
            total++;
            if (pose->keypoints[i].x >= bbox->x_min &&
                pose->keypoints[i].x <= bbox->x_max &&
                pose->keypoints[i].y >= bbox->y_min &&
                pose->keypoints[i].y <= bbox->y_max) {
                in_bbox++;
            }
        }
    }
    if (total > 0 && (float)in_bbox / (float)total < 0.4f) return false;

    return true;
}

bool keypoint_validator_side_body_check(
    const KeypointValidator* kv,
    const PoseEstimation* pose,
    const BoundingBox* bbox)
{
    if (!kv || !pose || !bbox) return false;

    float min_conf = kv->config.kpt_conf_threshold;

    /* Check left side chain: eye(1) → ear(3) → shoulder(5) → elbow(7) → wrist(9) → hip(11) → knee(13) → ankle(15) */
    static const int left_chain[] = {1, 3, 5, 7, 9, 11, 13, 15};
    int n_left = 0;
    for (int i = 0; i < 8; i++) {
        int idx = left_chain[i];
        if (idx < pose->num_keypoints &&
            pose->keypoints[idx].confidence >= min_conf &&
            pose->keypoints[idx].x >= 0.0f && pose->keypoints[idx].y >= 0.0f) {
            n_left++;
        }
    }

    /* Check right side chain: eye(2) → ear(4) → shoulder(6) → elbow(8) → wrist(10) → hip(12) → knee(14) → ankle(16) */
    static const int right_chain[] = {2, 4, 6, 8, 10, 12, 14, 16};
    int n_right = 0;
    for (int i = 0; i < 8; i++) {
        int idx = right_chain[i];
        if (idx < pose->num_keypoints &&
            pose->keypoints[idx].confidence >= min_conf &&
            pose->keypoints[idx].x >= 0.0f && pose->keypoints[idx].y >= 0.0f) {
            n_right++;
        }
    }

    /* At least 3 keypoints on one side */
    if (n_left < 3 && n_right < 3) return false;

    /* Check vertical ordering on the stronger side */
    if (n_left >= n_right && n_left >= 3) {
        /* Verify shoulder → hip → knee vertical order */
        if (pose->keypoints[5].confidence >= min_conf &&
            pose->keypoints[11].confidence >= min_conf) {
            if (pose->keypoints[5].y >= pose->keypoints[11].y) return false; /* shoulder below hip */
        }
        if (pose->keypoints[11].confidence >= min_conf &&
            pose->keypoints[13].confidence >= min_conf) {
            if (pose->keypoints[11].y >= pose->keypoints[13].y) return false; /* hip below knee */
        }
    } else if (n_right >= 3) {
        if (pose->keypoints[6].confidence >= min_conf &&
            pose->keypoints[12].confidence >= min_conf) {
            if (pose->keypoints[6].y >= pose->keypoints[12].y) return false;
        }
        if (pose->keypoints[12].confidence >= min_conf &&
            pose->keypoints[14].confidence >= min_conf) {
            if (pose->keypoints[12].y >= pose->keypoints[14].y) return false;
        }
    }

    return true;
}

/* ── Body aspect ratio check ── */

bool keypoint_validator_check_body_aspect(const BoundingBox* bbox,
                                           int frame_w, int frame_h) {
    if (!bbox || frame_w <= 0 || frame_h <= 0) return false;

    float bw = bbox_width(bbox);
    float bh = bbox_height(bbox);

    /* Degenerate box → reject */
    if (bw < 5.0f || bh < 5.0f) return false;

    /* Height/width aspect ratio must be human-like */
    float aspect = bh / UTILS_MAX(bw, 1.0f);
    if (aspect < KV_BBOX_ASPECT_MIN || aspect > KV_BBOX_ASPECT_MAX) {
        return false;
    }

    /* Area ratio must be reasonable for a person at typical distances */
    float frame_area = (float)(frame_w * frame_h);
    float area_ratio = (bw * bh) / UTILS_MAX(frame_area, 1.0f);
    if (area_ratio < KV_BBOX_AREA_MIN_RATIO || area_ratio > KV_BBOX_AREA_MAX_RATIO) {
        return false;
    }

    return true;
}

/* ── Detection consensus check ── */

float keypoint_validator_detection_consensus(const Detection* a, const Detection* b,
                                              float min_iou) {
    if (!a || !b) return 0.0f;
    float iou = bbox_iou(&a->bbox, &b->bbox);
    if (iou < min_iou) return 0.0f;
    /* Return the higher confidence as consensus strength */
    return (a->confidence > b->confidence) ? a->confidence : b->confidence;
}

/* ── OKS computation for pose NMS ── */

/*
 * COCO per-keypoint sigmas for OKS.
 * Source: COCO dataset keypoint evaluation metric.
 */
static const float COCO_KPT_SIGMAS[17] = {
    0.026f, /* nose */
    0.025f, /* left_eye */
    0.025f, /* right_eye */
    0.035f, /* left_ear */
    0.035f, /* right_ear */
    0.079f, /* left_shoulder */
    0.079f, /* right_shoulder */
    0.072f, /* left_elbow */
    0.072f, /* right_elbow */
    0.062f, /* left_wrist */
    0.062f, /* right_wrist */
    0.107f, /* left_hip */
    0.107f, /* right_hip */
    0.087f, /* left_knee */
    0.087f, /* right_knee */
    0.089f, /* left_ankle */
    0.089f  /* right_ankle */
};

float keypoint_validator_compute_oks(const PoseEstimation* a, const PoseEstimation* b) {
    if (!a->has_bbox || !b->has_bbox) return 0.0f;

    float area_a = bbox_area(&a->bbox);
    float area_b = bbox_area(&b->bbox);
    float s2 = sqrtf((area_a + area_b) * 0.5f);
    if (s2 < 1.0f) return 0.0f;

    float oks_sum = 0.0f;
    int valid = 0;
    int n_kpts = (a->num_keypoints < b->num_keypoints) ? a->num_keypoints : b->num_keypoints;
    if (n_kpts > 17) n_kpts = 17;

    for (int k = 0; k < n_kpts; k++) {
        if (a->keypoints[k].confidence < 0.3f || b->keypoints[k].confidence < 0.3f)
            continue;
        float dx = a->keypoints[k].x - b->keypoints[k].x;
        float dy = a->keypoints[k].y - b->keypoints[k].y;
        float k2 = COCO_KPT_SIGMAS[k] * COCO_KPT_SIGMAS[k];
        float e = (dx * dx + dy * dy) / (2.0f * s2 * k2 + 1e-6f);
        oks_sum += expf(-e);
        valid++;
    }

    if (valid == 0) return 0.0f;
    return oks_sum / (float)valid;
}
