#include "yolo_postprocess.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifdef HAS_ONNX_RUNTIME
#include "ort_common.h"
#endif

void yolo_softmax_stable(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = (sum > 1e-12f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

float yolo_dfl_decode_position(const float* reg_data, int pix, int hw, float dists_out[4]) {
    float peaks[4];
    for (int coord = 0; coord < 4; coord++) {
        float bins[16];
        int base = coord * 16;
        for (int b = 0; b < 16; b++) {
            bins[b] = reg_data[(base + b) * hw + pix];
        }
        yolo_softmax_stable(bins, 16);
        float max_bin = bins[0];
        float val = 0.0f;
        for (int b = 0; b < 16; b++) {
            if (bins[b] > max_bin) max_bin = bins[b];
            val += bins[b] * (float)b;
        }
        peaks[coord] = max_bin;
        dists_out[coord] = val;
    }
    float dfl_conf = sqrtf(sqrtf(peaks[0] * peaks[1] * peaks[2] * peaks[3]));
    return dfl_conf;
}

void yolo_preprocess(const uint8_t* image_data, int width, int height,
                     float* out_tensor, int target_w, int target_h,
                     float* out_scale, int* out_pad_x, int* out_pad_y,
                     int* out_crop_x, int* out_crop_y) {
    float src_ar = (float)width / (float)UTILS_MAX(height, 1);
    float dst_ar = (float)target_w / (float)UTILS_MAX(target_h, 1);
    float ar_ratio = src_ar / UTILS_MAX(dst_ar, 0.01f);
    bool need_crop = (ar_ratio < 0.7f || ar_ratio > 1.4f);

    const uint8_t* src_ptr = image_data;
    int src_w = width, src_h = height;
    int crop_x = 0, crop_y = 0;

    uint8_t* crop_buf = NULL;
    if (need_crop) {
        int crop_w, crop_h;
        if (src_ar < dst_ar) {
            crop_w = width;
            crop_h = (int)((float)width / dst_ar + 0.5f);
            crop_x = 0;
            crop_y = (height - crop_h) / 2;
        } else {
            crop_h = height;
            crop_w = (int)((float)height * dst_ar + 0.5f);
            crop_x = (width - crop_w) / 2;
            crop_y = 0;
        }
        if (crop_y < 0) crop_y = 0;
        if (crop_x < 0) crop_x = 0;
        if (crop_x + crop_w > width)  crop_w = width - crop_x;
        if (crop_y + crop_h > height) crop_h = height - crop_y;

        crop_buf = (uint8_t*)malloc((size_t)crop_w * crop_h * 3);
        if (crop_buf) {
            for (int y = 0; y < crop_h; y++) {
                for (int x = 0; x < crop_w; x++) {
                    int si = ((crop_y + y) * width + (crop_x + x)) * 3;
                    int di = (y * crop_w + x) * 3;
                    crop_buf[di + 0] = image_data[si + 0];
                    crop_buf[di + 1] = image_data[si + 1];
                    crop_buf[di + 2] = image_data[si + 2];
                }
            }
            src_ptr = crop_buf;
            src_w = crop_w;
            src_h = crop_h;
        }
    }

    if (out_crop_x) *out_crop_x = crop_x;
    if (out_crop_y) *out_crop_y = crop_y;

    uint8_t* padded = (uint8_t*)malloc((size_t)target_w * target_h * 3);
    if (!padded) { free(crop_buf); return; }

    utils_letterbox(src_ptr, src_w, src_h, padded, target_w, target_h, 3, out_scale, out_pad_x, out_pad_y);
    free(crop_buf);

    int pixels = target_w * target_h;
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_idx = (y * target_w + x) * 3;
            int dst_r = 0 * pixels + y * target_w + x;
            int dst_g = 1 * pixels + y * target_w + x;
            int dst_b = 2 * pixels + y * target_w + x;

            out_tensor[dst_r] = padded[src_idx + 0] / 255.0f;
            out_tensor[dst_g] = padded[src_idx + 1] / 255.0f;
            out_tensor[dst_b] = padded[src_idx + 2] / 255.0f;
        }
    }

    free(padded);
}

void yolo_map_to_original(float mx, float my, float scale, int pad_x, int pad_y,
                          int crop_x, int crop_y, float* ox, float* oy) {
    *ox = (mx - (float)pad_x) / scale + (float)crop_x;
    *oy = (my - (float)pad_y) / scale + (float)crop_y;
}

#ifdef HAS_ONNX_RUNTIME
int yolo_detect_xquant_split(size_t num_outputs, OrtValue** output_vals,
                             int group_indices[3][3], int mode) {
    if (num_outputs < 3 || num_outputs % 3 != 0) return 0;

    const OrtApi* ort = ort_get_api();
    if (!ort) return 0;

    int out_hw[12] = {0}, out_c[12] = {0};
    int valid = 0;

    for (size_t oi = 0; oi < num_outputs && oi < 12; oi++) {
        OrtTensorTypeAndShapeInfo* si = NULL;
        if (ort->GetTensorTypeAndShape(output_vals[oi], &si)) continue;
        size_t nd = 0;
        { OrtStatus* _s = ort->GetDimensionsCount(si, &nd); if (_s) ort->ReleaseStatus(_s); }
        int64_t dims[4] = {0};
        { OrtStatus* _s = ort->GetDimensions(si, dims, nd); if (_s) ort->ReleaseStatus(_s); }
        ort->ReleaseTensorTypeAndShapeInfo(si);
        if (nd < 3) continue;
        if (dims[0] != 1) continue;
        out_c[oi] = (int)dims[1];
        int h = (int)dims[2], w = (nd >= 4) ? (int)dims[3] : 1;
        out_hw[oi] = h * w;
        valid++;
    }
    if (valid != (int)num_outputs) return 0;

    int num_groups = 0;
    bool used[12] = {false};

    for (size_t i = 0; i < num_outputs && num_groups < 3; i++) {
        if (used[i]) continue;
        int hw_i = out_hw[i];
        int group[3] = {(int)i, -1, -1};
        int found = 1;
        for (size_t j = i + 1; j < num_outputs && found < 3; j++) {
            if (used[j]) continue;
            if (out_hw[j] == hw_i) {
                group[found++] = (int)j;
                used[j] = true;
            }
        }
        if (found == 3) {
            used[i] = true;
            if (mode == 0) {
                int reg_idx = -1, cls_idx = -1, ext_idx = -1;
                for (int k = 0; k < 3; k++) {
                    int ch = out_c[group[k]];
                    if (ch >= 60 && ch <= 70) reg_idx = group[k];
                    else if (ch >= 10) cls_idx = group[k];
                    else ext_idx = group[k];
                }
                if (reg_idx >= 0 && cls_idx >= 0) {
                    group_indices[num_groups][0] = reg_idx;
                    group_indices[num_groups][1] = cls_idx;
                    group_indices[num_groups][2] = ext_idx >= 0 ? ext_idx : -1;
                    num_groups++;
                }
            } else {
                int reg_idx = -1, cls_idx = -1, kpt_idx = -1;
                for (int k = 0; k < 3; k++) {
                    int ch = out_c[group[k]];
                    if (ch >= 60 && ch <= 70) reg_idx = group[k];
                    else if (ch >= 40 && ch <= 60) kpt_idx = group[k];
                    else cls_idx = group[k];
                }
                if (reg_idx >= 0 && kpt_idx >= 0) {
                    group_indices[num_groups][0] = reg_idx;
                    group_indices[num_groups][1] = cls_idx >= 0 ? cls_idx : -1;
                    group_indices[num_groups][2] = kpt_idx;
                    num_groups++;
                }
            }
        }
    }

    if (num_groups != (int)(num_outputs / 3)) return 0;
    return num_groups;
}
#endif

static void swap_bytes(void* a, void* b, uint8_t* buf, size_t size) {
    memcpy(buf, a, size);
    memcpy(a, b, size);
    memcpy(b, buf, size);
}

static float default_bbox_sim(const void* a, const void* b) {
    return bbox_iou((const BoundingBox*)a, (const BoundingBox*)b);
}

int yolo_nms_suppress(void* items, int num_items, size_t item_size,
                      float conf_threshold, float iou_threshold,
                      int max_output, yolo_similarity_fn sim_fn, int conf_offset) {
    if (num_items <= 0 || !items || item_size == 0) return 0;
    if (max_output <= 0) return 0;

    uint8_t* swap_buf = (uint8_t*)malloc(item_size);
    if (!swap_buf) return 0;

    uint8_t* ptr = (uint8_t*)items;

    for (int i = 0; i < num_items - 1; i++) {
        for (int j = i + 1; j < num_items; j++) {
            float ci = *(float*)(ptr + (size_t)i * item_size + (size_t)conf_offset);
            float cj = *(float*)(ptr + (size_t)j * item_size + (size_t)conf_offset);
            if (ci < cj) {
                swap_bytes(ptr + (size_t)i * item_size,
                           ptr + (size_t)j * item_size,
                           swap_buf, item_size);
            }
        }
    }

    int filtered = 0;
    for (int i = 0; i < num_items; i++) {
        float conf = *(float*)(ptr + (size_t)i * item_size + (size_t)conf_offset);
        if (conf >= conf_threshold) {
            if (filtered != i) {
                memcpy(ptr + (size_t)filtered * item_size,
                       ptr + (size_t)i * item_size, item_size);
            }
            filtered++;
        }
    }
    num_items = filtered;

    yolo_similarity_fn sim = sim_fn ? sim_fn : default_bbox_sim;

    bool* suppressed = (bool*)calloc((size_t)num_items, sizeof(bool));
    if (!suppressed) {
        free(swap_buf);
        return num_items;
    }

    int keep = 0;
    for (int i = 0; i < num_items; i++) {
        if (suppressed[i]) continue;
        if (keep < max_output) {
            if (keep != i) {
                memcpy(ptr + (size_t)keep * item_size,
                       ptr + (size_t)i * item_size, item_size);
            }
            keep++;
        }
        for (int j = i + 1; j < num_items; j++) {
            if (suppressed[j]) continue;
            float s = sim(ptr + (size_t)i * item_size,
                         ptr + (size_t)j * item_size);
            if (s > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    free(suppressed);
    free(swap_buf);
    return keep;
}