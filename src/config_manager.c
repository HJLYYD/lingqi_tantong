#define _GNU_SOURCE
#include "config_manager.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * YAML Configuration Subset Notes:
 *
 * This module implements a minimal YAML parser supporting only the following subset:
 *   - Top-level key: value pairs
 *   - Nested objects (indent >= 2 spaces)
 *   - String values (quoted and bare strings)
 *   - Integer and floating-point values
 *   - Boolean values (true/false/yes/no)
 *   - Single-line comments (#)
 *
 * Unsupported standard YAML features:
 *   - Anchors & aliases: &name, *name, << merge
 *   - Multi-line strings: |, >, |-, >-
 *   - Deep nesting: map-of-map (>2 levels)
 *   - List indentation: sub-structures under - item
 *   - Flow style: {} inline maps, [] inline lists
 *   - Type tags: !!str, !!int, etc.
 *   - Multi-document: --- separator
 *
 * For full YAML support, consider integrating libyaml (https://github.com/yaml/libyaml).
 * The current subset satisfies the configuration needs of this project's default.yaml.
 */

static void config_set_defaults(ConfigManager* cm) {
    config_set_bool(cm, "system.use_onnx", true);
    config_set_bool(cm, "system.use_spacemit_ep", true);
    config_set_int(cm, "system.spacemit_ep_intra_threads", 3);
    config_set_int(cm, "system.max_frames", 0);
    config_set_int(cm, "system.worker_threads", 4);
    config_set_string(cm, "system.log_level", "INFO");

    config_set_string(cm, "detection.backend", "ai_accel");
    config_set_string(cm, "detection.model_path", "models/Human Recognition/yolo11n.q.onnx");
    config_set_float(cm, "detection.confidence_threshold", 0.20f);  /* DFL peakiness range: 0.18-0.70 */
    config_set_float(cm, "detection.iou_threshold", 0.45f);         /* tighter NMS */
    config_set_int(cm, "detection.input_size", 640);
    config_set_int(cm, "detection.input_size.0", 640);
    config_set_int(cm, "detection.input_size.1", 640);
    config_set_int(cm, "detection.classes", 80);

    config_set_string(cm, "pose.backend", "ai_accel");
    config_set_string(cm, "pose.model_path", "models/Action Prediction/Skeleton Recognition/yolov8n-pose.q.onnx");
    config_set_float(cm, "pose.confidence_threshold", 0.15f);
    config_set_float(cm, "pose.iou_threshold", 0.45f);
    config_set_int(cm, "pose.input_size.0", 640);
    config_set_int(cm, "pose.input_size.1", 640);
    config_set_int(cm, "pose.num_keypoints", 17);

    config_set_string(cm, "face.detection_backend", "ai_accel");
    config_set_string(cm, "face.recognition_backend", "ai_accel");
    config_set_string(cm, "face.detection_model_path", "models/Face Recognition/yolov5n-face_cut.q.onnx");
    config_set_string(cm, "face.recognition_model_path", "models/Face Recognition/arcface_mobilefacenet_cut.q.onnx");
    config_set_int(cm, "face.face_min_size", 24);
    config_set_float(cm, "face.confidence_threshold", 0.5f);
    config_set_int(cm, "face.input_size.0", 320);
    config_set_int(cm, "face.input_size.1", 320);
    config_set_int(cm, "face.embedding_dim", 128);  /* ArcFace MobileFaceNet-cuted output [1,128] */
    config_set_float(cm, "face.similarity_threshold", 0.55f);

    config_set_string(cm, "action.backend", "ai_accel");
    config_set_string(cm, "action.model_path", "models/Action Prediction/Skeleton-based Action Prediction/stgcn.fp32.onnx");
    config_set_int(cm, "action.num_frames", 30);    /* ST-GCN temporal window; match default.yaml */
    config_set_int(cm, "action.num_keypoints", 14); /* ST-GCN model joint count */
    config_set_int(cm, "action.num_persons", 1);
    config_set_int(cm, "action.num_classes", 7);    /* auto-detected at load time; 7-class NTU RGB+D subset */
    config_set_float(cm, "action.confidence_threshold", 0.5f);

    config_set_int(cm, "tracking.max_lost", 30);
    config_set_float(cm, "tracking.min_iou", 0.50f);
    config_set_float(cm, "tracking.max_distance", 5.0f);
    config_set_int(cm, "tracking.max_track_history", 300);
    config_set_float(cm, "tracking.kalman_process_noise", 0.05f);
    config_set_float(cm, "tracking.kalman_measurement_noise", 0.1f);

    config_set_float(cm, "spatial.fx", 960.0f);
    config_set_float(cm, "spatial.fy", 960.0f);
    config_set_float(cm, "spatial.cx", 960.0f);
    config_set_float(cm, "spatial.cy", 540.0f);
    config_set_float(cm, "spatial.avg_human_height", 1.7f);

    config_set_bool(cm, "visualization.show_info_bar", true);
    config_set_bool(cm, "visualization.corner_markers", true);
    config_set_bool(cm, "visualization.crosshair", true);
    config_set_int(cm, "visualization.render_size.0", 1920);
    config_set_int(cm, "visualization.render_size.1", 1080);

    config_set_int(cm, "video.camera_width", 1920);
    config_set_int(cm, "video.camera_height", 1080);
    config_set_float(cm, "video.camera_fps", 30.0f);
    config_set_string(cm, "video.camera_device", "/dev/video0");
    config_set_string(cm, "video.camera_format", "MJPEG");

    config_set_string(cm, "arrow.uart_device_A", "/dev/ttyS0");
    config_set_string(cm, "arrow.uart_device_C", "/dev/ttyS1");
    config_set_int(cm, "arrow.baudrate", 3000000);

    config_set_int(cm, "performance.openmp_threads", 6);
    config_set_bool(cm, "performance.realtime_scheduling", true);
}

ConfigManager* config_manager_create(const char* config_path) {
    ConfigManager* cm = (ConfigManager*)calloc(1, sizeof(ConfigManager));
    if (!cm) return NULL;

    config_set_defaults(cm);

    if (config_path) {
        strncpy(cm->config_path, config_path, MAX_PATH_LEN - 1);
        config_load_from_file(cm, config_path);
    }

    log_info("ConfigManager initialized with %d top-level keys", cm->num_sections);
    return cm;
}

void config_manager_destroy(ConfigManager* cm) {
    free(cm);
}

static ConfigSection* find_section(ConfigManager* cm, const char* section_name, bool create) {
    for (int i = 0; i < cm->num_sections; i++) {
        if (strcmp(cm->sections[i].section, section_name) == 0) {
            return &cm->sections[i];
        }
    }
    if (create && cm->num_sections < MAX_CONFIG_DEPTH) {
        ConfigSection* sec = &cm->sections[cm->num_sections++];
        strncpy(sec->section, section_name, MAX_SECTION_LEN - 1);
        sec->num_entries = 0;
        return sec;
    }
    return NULL;
}

static ConfigEntry* find_entry(ConfigSection* sec, const char* key) {
    for (int i = 0; i < sec->num_entries; i++) {
        if (strcmp(sec->entries[i].key, key) == 0) {
            return &sec->entries[i];
        }
    }
    return NULL;
}

static ConfigEntry* get_or_create_entry(ConfigManager* cm, const char* key_path) {
    char section_name[MAX_SECTION_LEN];
    char key_name[MAX_KEY_LEN];

    const char* dot = strchr(key_path, '.');
    if (!dot) return NULL;

    int sec_len = (int)(dot - key_path);
    if (sec_len >= MAX_SECTION_LEN) sec_len = MAX_SECTION_LEN - 1;
    memcpy(section_name, key_path, sec_len);
    section_name[sec_len] = '\0';

    strncpy(key_name, dot + 1, MAX_KEY_LEN - 1);
    key_name[MAX_KEY_LEN - 1] = '\0';

    ConfigSection* sec = find_section(cm, section_name, true);
    if (!sec) return NULL;

    ConfigEntry* entry = find_entry(sec, key_name);
    if (!entry && sec->num_entries < MAX_CONFIG_KEYS) {
        entry = &sec->entries[sec->num_entries++];
        strncpy(entry->key, key_name, MAX_KEY_LEN - 1);
        entry->key[MAX_KEY_LEN - 1] = '\0';
    }
    return entry;
}

int config_get_int(const ConfigManager* cm, const char* key_path, int default_val) {
    char section_name[MAX_SECTION_LEN];
    char key_name[MAX_KEY_LEN];
    const char* dot = strchr(key_path, '.');
    if (!dot) return default_val;

    int sec_len = (int)(dot - key_path);
    if (sec_len >= MAX_SECTION_LEN) sec_len = MAX_SECTION_LEN - 1;
    memcpy(section_name, key_path, sec_len);
    section_name[sec_len] = '\0';
    strncpy(key_name, dot + 1, MAX_KEY_LEN - 1);
    key_name[MAX_KEY_LEN - 1] = '\0';

    for (int i = 0; i < cm->num_sections; i++) {
        if (strcmp(cm->sections[i].section, section_name) == 0) {
            for (int j = 0; j < cm->sections[i].num_entries; j++) {
                if (strcmp(cm->sections[i].entries[j].key, key_name) == 0) {
                    if (cm->sections[i].entries[j].type == CONFIG_TYPE_INT)
                        return cm->sections[i].entries[j].value.int_val;
                }
            }
        }
    }
    return default_val;
}

float config_get_float(const ConfigManager* cm, const char* key_path, float default_val) {
    char section_name[MAX_SECTION_LEN];
    char key_name[MAX_KEY_LEN];
    const char* dot = strchr(key_path, '.');
    if (!dot) return default_val;

    int sec_len = (int)(dot - key_path);
    if (sec_len >= MAX_SECTION_LEN) sec_len = MAX_SECTION_LEN - 1;
    memcpy(section_name, key_path, sec_len);
    section_name[sec_len] = '\0';
    strncpy(key_name, dot + 1, MAX_KEY_LEN - 1);
    key_name[MAX_KEY_LEN - 1] = '\0';

    for (int i = 0; i < cm->num_sections; i++) {
        if (strcmp(cm->sections[i].section, section_name) == 0) {
            for (int j = 0; j < cm->sections[i].num_entries; j++) {
                if (strcmp(cm->sections[i].entries[j].key, key_name) == 0) {
                    if (cm->sections[i].entries[j].type == CONFIG_TYPE_FLOAT)
                        return cm->sections[i].entries[j].value.float_val;
                }
            }
        }
    }
    return default_val;
}

bool config_get_bool(const ConfigManager* cm, const char* key_path, bool default_val) {
    char section_name[MAX_SECTION_LEN];
    char key_name[MAX_KEY_LEN];
    const char* dot = strchr(key_path, '.');
    if (!dot) return default_val;

    int sec_len = (int)(dot - key_path);
    if (sec_len >= MAX_SECTION_LEN) sec_len = MAX_SECTION_LEN - 1;
    memcpy(section_name, key_path, sec_len);
    section_name[sec_len] = '\0';
    strncpy(key_name, dot + 1, MAX_KEY_LEN - 1);
    key_name[MAX_KEY_LEN - 1] = '\0';

    for (int i = 0; i < cm->num_sections; i++) {
        if (strcmp(cm->sections[i].section, section_name) == 0) {
            for (int j = 0; j < cm->sections[i].num_entries; j++) {
                if (strcmp(cm->sections[i].entries[j].key, key_name) == 0) {
                    if (cm->sections[i].entries[j].type == CONFIG_TYPE_BOOL)
                        return cm->sections[i].entries[j].value.bool_val;
                }
            }
        }
    }
    return default_val;
}

const char* config_get_string(const ConfigManager* cm, const char* key_path, const char* default_val) {
    char section_name[MAX_SECTION_LEN];
    char key_name[MAX_KEY_LEN];
    const char* dot = strchr(key_path, '.');
    if (!dot) return default_val;

    int sec_len = (int)(dot - key_path);
    if (sec_len >= MAX_SECTION_LEN) sec_len = MAX_SECTION_LEN - 1;
    memcpy(section_name, key_path, sec_len);
    section_name[sec_len] = '\0';
    strncpy(key_name, dot + 1, MAX_KEY_LEN - 1);
    key_name[MAX_KEY_LEN - 1] = '\0';

    for (int i = 0; i < cm->num_sections; i++) {
        if (strcmp(cm->sections[i].section, section_name) == 0) {
            for (int j = 0; j < cm->sections[i].num_entries; j++) {
                if (strcmp(cm->sections[i].entries[j].key, key_name) == 0) {
                    if (cm->sections[i].entries[j].type == CONFIG_TYPE_STRING)
                        return cm->sections[i].entries[j].value.string_val;
                }
            }
        }
    }
    return default_val;
}

void config_set_int(ConfigManager* cm, const char* key_path, int val) {
    ConfigEntry* entry = get_or_create_entry(cm, key_path);
    if (entry) {
        entry->type = CONFIG_TYPE_INT;
        entry->value.int_val = val;
    }
}

void config_set_float(ConfigManager* cm, const char* key_path, float val) {
    ConfigEntry* entry = get_or_create_entry(cm, key_path);
    if (entry) {
        entry->type = CONFIG_TYPE_FLOAT;
        entry->value.float_val = val;
    }
}

void config_set_bool(ConfigManager* cm, const char* key_path, bool val) {
    ConfigEntry* entry = get_or_create_entry(cm, key_path);
    if (entry) {
        entry->type = CONFIG_TYPE_BOOL;
        entry->value.bool_val = val;
    }
}

void config_set_string(ConfigManager* cm, const char* key_path, const char* val) {
    ConfigEntry* entry = get_or_create_entry(cm, key_path);
    if (entry) {
        entry->type = CONFIG_TYPE_STRING;
        strncpy(entry->value.string_val, val ? val : "", MAX_VALUE_LEN - 1);
        entry->value.string_val[MAX_VALUE_LEN - 1] = '\0';
    }
}

void config_load_defaults(ConfigManager* cm) {
    config_set_defaults(cm);
}

int config_load_from_file(ConfigManager* cm, const char* path) {
    if (!cm || !path) {
        log_warning("Config load: NULL config or path");
        return -1;
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        log_info("Config file not found: %s, using defaults", path);
        return -1;
    }

    char line[512];
    char section_stack[MAX_CONFIG_DEPTH][MAX_SECTION_LEN];
    int section_depth = 0;
    int parsed_count = 0;

    while (fgets(line, sizeof(line), f)) {
        char* p = line;

        while (*p == ' ' || *p == '\t') p++;

        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;

        int indent = (int)(p - line);

        char trimmed[512];
        size_t len = strlen(p);
        while (len > 0 && (p[len - 1] == '\n' || p[len - 1] == '\r')) len--;
        memcpy(trimmed, p, len);
        trimmed[len] = '\0';

        char key_part[256];
        char value_part[256];
        memset(key_part, 0, sizeof(key_part));
        memset(value_part, 0, sizeof(value_part));

        char* colon = strchr(trimmed, ':');
        if (!colon) continue;

        int key_len = (int)(colon - trimmed);
        if (key_len >= (int)sizeof(key_part)) key_len = sizeof(key_part) - 1;
        memcpy(key_part, trimmed, key_len);
        key_part[key_len] = '\0';

        const char* val_start = colon + 1;
        while (*val_start == ' ' || *val_start == '\t') val_start++;
        strncpy(value_part, val_start, sizeof(value_part) - 1);

        if (value_part[0] == '\0') {
            int target_depth = indent / 2;
            if (target_depth >= MAX_CONFIG_DEPTH) target_depth = MAX_CONFIG_DEPTH - 1;

            section_depth = target_depth;
            if (section_depth >= 0 && section_depth < MAX_CONFIG_DEPTH) {
                strncpy(section_stack[section_depth], key_part, MAX_SECTION_LEN - 1);
                section_stack[section_depth][MAX_SECTION_LEN - 1] = '\0';
            }
            for (int d = section_depth + 1; d < MAX_CONFIG_DEPTH; d++) {
                section_stack[d][0] = '\0';
            }
            continue;
        }

        char full_key[MAX_KEY_LEN * 2];
        full_key[0] = '\0';
        size_t fk_off = 0;
        for (int d = 0; d <= section_depth && d < MAX_CONFIG_DEPTH; d++) {
            if (section_stack[d][0] == '\0') continue;
            int written;
            if (fk_off == 0) {
                written = snprintf(full_key + fk_off, sizeof(full_key) - fk_off, "%s", section_stack[d]);
            } else {
                written = snprintf(full_key + fk_off, sizeof(full_key) - fk_off, ".%s", section_stack[d]);
            }
            if (written < 0 || (size_t)written >= sizeof(full_key) - fk_off) {
                full_key[sizeof(full_key) - 1] = '\0';
                fk_off = sizeof(full_key) - 1;
                break;
            }
            fk_off += (size_t)written;
        }
        if (fk_off > 0 && fk_off < sizeof(full_key) - 1) {
            int written = snprintf(full_key + fk_off, sizeof(full_key) - fk_off, ".%s", key_part);
            if (written < 0 || (size_t)written >= sizeof(full_key) - fk_off) {
                full_key[sizeof(full_key) - 1] = '\0';
            }
        } else if (fk_off == 0) {
            strncpy(full_key, key_part, sizeof(full_key) - 1);
            full_key[sizeof(full_key) - 1] = '\0';
        }

        if (full_key[0] == '\0') continue;

        if (value_part[0] == '"' || value_part[0] == '\'') {
            char str_val[MAX_VALUE_LEN];
            int v_len = (int)strlen(value_part);
            int start = 1;
            int end = v_len - 1;
            if (end > start && (value_part[end] == '"' || value_part[end] == '\'')) {
                end--;
            }
            int copy_len = end - start + 1;
            if (copy_len >= MAX_VALUE_LEN) copy_len = MAX_VALUE_LEN - 1;
            memcpy(str_val, value_part + start, copy_len);
            str_val[copy_len] = '\0';
            config_set_string(cm, full_key, str_val);
            parsed_count++;
        }
        else if (strcmp(value_part, "true") == 0 || strcmp(value_part, "True") == 0 ||
                 strcmp(value_part, "TRUE") == 0 || strcmp(value_part, "yes") == 0) {
            config_set_bool(cm, full_key, true);
            parsed_count++;
        }
        else if (strcmp(value_part, "false") == 0 || strcmp(value_part, "False") == 0 ||
                 strcmp(value_part, "FALSE") == 0 || strcmp(value_part, "no") == 0) {
            config_set_bool(cm, full_key, false);
            parsed_count++;
        }
        else if (value_part[0] == '[') {
            char* bracket = strchr(value_part, ']');
            if (bracket) {
                *bracket = '\0';
                char* saveptr = NULL;
                char* token = strtok_r(value_part + 1, ",", &saveptr);
                int idx = 0;
                while (token && idx < 8) {
                    while (*token == ' ') token++;
                    char indexed_key[MAX_KEY_LEN * 2 + 16];
                    snprintf(indexed_key, sizeof(indexed_key), "%s.%d", full_key, idx);
                    if (strchr(token, '.')) {
                        config_set_float(cm, indexed_key, (float)atof(token));
                    } else {
                        config_set_int(cm, indexed_key, atoi(token));
                    }
                    token = strtok_r(NULL, ",", &saveptr);
                    idx++;
                }
                parsed_count++;
            }
        }
        else {
            if (strchr(value_part, '.')) {
                config_set_float(cm, full_key, (float)atof(value_part));
            } else {
                char* endptr;
                long int_val = strtol(value_part, &endptr, 10);
                if (*endptr == '\0' || *endptr == ' ' || *endptr == '\n' || *endptr == '\r') {
                    config_set_int(cm, full_key, (int)int_val);
                } else {
                    config_set_string(cm, full_key, value_part);
                }
            }
            parsed_count++;
        }
    }

    fclose(f);
    log_info("YAML config loaded from %s: %d keys parsed", path, parsed_count);
    return 0;
}
