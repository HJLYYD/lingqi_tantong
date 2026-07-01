/*
 * web_server.c — Embedded HTTP + WebSocket server implementation
 *
 * Uses Mongoose event-driven architecture:
 *   - MG_EV_HTTP_MSG  → route HTTP requests (static files, API)
 *   - MG_EV_WS_MSG    → handle incoming WebSocket messages (echo/heartbeat)
 *   - MG_EV_CLOSE     → cleanup disconnected clients
 *   - mg_timer_add()  → periodic broadcast of frame ring buffer
 *
 * Thread model:
 *   This file's mg_mgr_poll() loop runs in its own pthread.
 *   web_server_push_frame() is called from the PostProcess thread
 *   (or main thread in offline mode) and uses pthread_mutex to
 *   safely write into the frame ring buffer.
 */

#define _GNU_SOURCE   /* for asprintf, open_memstream */
#include "web_server.h"
#include "system_controller.h"
#include "config_manager.h"
#include "pipeline_state.h"
#include "logger.h"
#include "json_writer.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Internal helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Mark a connection as WebSocket so the timer knows to broadcast to it */
static void mark_ws(struct mg_connection *c) {
    c->data[0] = 'W';
}

static bool is_ws(struct mg_connection *c) {
    return c->data[0] == 'W';
}

/* ── Simple JSON string building helpers (no external deps) ── */

static int json_append(char *buf, int len, int written, const char *fmt, ...) {
    if (written >= len) return written;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + written, (size_t)(len - written), fmt, ap);
    va_end(ap);
    /* BUGFIX: vsnprintf returns the number of bytes that WOULD have been
     * written on truncation — far exceeding remaining space.  Clamp to len
     * so subsequent calls don't write past the buffer end. */
    return (n < 0) ? written : ((written + n > len) ? len : written + n);
}

/* ── Build /api/status JSON response (real data from SystemController) ── */
static int build_status_json(WebServer *ws, char *buf, int len) {
    int w = 0;
    w = json_append(buf, len, w, "{");
    w = json_append(buf, len, w, "\"state\":\"%s\",",
                    ws->controller
                        ? psm_state_name(psm_get(&ws->controller->state_machine))
                        : "unknown");
    w = json_append(buf, len, w, "\"version\":\"2.0.0\",");
    w = json_append(buf, len, w, "\"platform\":\"");
#ifdef PLATFORM_RISCV64
    w = json_append(buf, len, w, "K1");
#else
    w = json_append(buf, len, w, "host");
#endif
    w = json_append(buf, len, w, "\",");
    w = json_append(buf, len, w, "\"server\":\"mongoose\",");
    w = json_append(buf, len, w, "\"frontend\":\"react\"");
    if (ws->controller) {
        w = json_append(buf, len, w, ",\"frame_count\":%d", ws->controller->frame_count);
        w = json_append(buf, len, w, ",\"active_tracks\":%d",
                        ws->controller->tracking_manager
                            ? ws->controller->tracking_manager->num_tracks : 0);
        w = json_append(buf, len, w, ",\"errors\":%d", ws->controller->detection_count);
    }
    w = json_append(buf, len, w, "}");
    return w;
}

/* ── Build /api/config JSON from ConfigManager ──
 * Uses JsonW streaming writer + open_memstream() for dynamic allocation.
 * Returns heap-allocated string; caller must free(). */
static char* build_config_json(WebServer *ws, int *out_len) {
    char *buf = NULL;
    size_t size = 0;
    FILE *f = open_memstream(&buf, &size);
    if (!f) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    JsonW jw;
    jw_init(&jw, f, 0);  /* compact mode */

    jw_obj_begin(&jw);

    if (ws->controller && ws->controller->config) {
        ConfigManager *cm = ws->controller->config;

        for (int s = 0; s < cm->num_sections; s++) {
            ConfigSection *sec = &cm->sections[s];
            if (sec->num_entries == 0) continue;

            jw_key(&jw, sec->section);
            jw_obj_begin(&jw);

            for (int e = 0; e < sec->num_entries; e++) {
                ConfigEntry *ent = &sec->entries[e];
                jw_key(&jw, ent->key);

                switch (ent->type) {
                case CONFIG_TYPE_INT:
                    jw_int(&jw, ent->value.int_val);
                    break;
                case CONFIG_TYPE_FLOAT:
                    jw_float(&jw, ent->value.float_val, 6);
                    break;
                case CONFIG_TYPE_BOOL:
                    jw_bool(&jw, ent->value.bool_val);
                    break;
                case CONFIG_TYPE_STRING:
                    jw_str(&jw, ent->value.string_val);
                    break;
                default:
                    jw_null(&jw);
                    break;
                }
            }

            jw_obj_end(&jw);
        }
    }

    jw_obj_end(&jw);
    jw_flush(&jw);
    fclose(f);

    if (out_len) *out_len = (int)size;
    return buf;
}

/* ── Build /api/models JSON ──
 * Uses JsonW + open_memstream for dynamic allocation. */
static char* build_models_json(int *out_len) {
    char *buf = NULL;
    size_t size = 0;
    FILE *f = open_memstream(&buf, &size);
    if (!f) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    JsonW jw;
    jw_init(&jw, f, 0);  /* compact */

    jw_obj_begin(&jw);
    jw_key(&jw, "models");
    jw_arr_begin(&jw);

    const char *model_dirs[] = {"models", "/usr/share/lingqi_tantong/models", NULL};

    for (int d = 0; model_dirs[d]; d++) {
        DIR *dir = opendir(model_dirs[d]);
        if (!dir) continue;

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            const char *name = entry->d_name;
            size_t nl = strlen(name);
            if (nl < 5) continue;
            if (strcmp(name + nl - 5, ".onnx") != 0) continue;

            char path[512];
            snprintf(path, sizeof(path), "%s/%s", model_dirs[d], name);

            const char *mtype = "unknown";
            if (strstr(name, "yolov8") || strstr(name, "yolo11")) mtype = "pose";
            else if (strstr(name, "yolov5")) mtype = "face";
            else if (strstr(name, "stgcn")) mtype = "action";
            else if (strstr(name, "arcface") || strstr(name, "mobilefacenet")) mtype = "face_recog";

            jw_obj_begin(&jw);
            jw_kstr(&jw, "name", name);
            jw_kstr(&jw, "path", path);
            jw_kstr(&jw, "type", mtype);
            jw_obj_end(&jw);
        }
        closedir(dir);
    }

    jw_arr_end(&jw);
    jw_obj_end(&jw);
    jw_flush(&jw);
    fclose(f);

    if (out_len) *out_len = (int)size;
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════
 * WebSocket command handler (Phase B)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Minimal JSON string extraction: find value for key in simple flat JSON */
#if 0  /* unused — kept for reference */
static const char *json_get_str(const char *json, int len, const char *key) {
    (void)len;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    return p;  /* points to start of value */
}

static int json_get_int(const char *json, int len, const char *key, int def) {
    (void)len;
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    return atoi(p);
}
#endif  /* unused dead code */

/* Forward-declare for command handler */
static void send_ws_response(struct mg_connection *c, const char *id,
                              const char *status, const char *data_json);

static void handle_ws_command(WebServer *ws, struct mg_connection *c,
                               const char *data, int len) {
    /* data is now null-terminated (caller guarantees it) */

    /* Extract cmd and id from JSON */
    const char *cmd = NULL;
    const char *id  = NULL;

    /* Very simple JSON parsing: find "cmd":"..." */
    {
        const char *p = strstr(data, "\"cmd\":\"");
        if (!p) {
            /* Might be a ping/heartbeat — echo back */
            if (len < 256) {
                mg_ws_send(c, data, (size_t)len, WEBSOCKET_OP_TEXT);
            }
            return;
        }
        cmd = p + 7; /* skip "cmd":" */
    }

    {
        const char *p = strstr(data, "\"id\":\"");
        if (p) id = p + 6; /* skip "id":" */
    }

    /* Default response ID */
    char id_buf[64] = "0";
    if (id) {
        strncpy(id_buf, id, sizeof(id_buf) - 1);
        const char *end = strchr(id_buf, '"');
        if (end) *((char *)end) = '\0';
    }

    /* Extract command string */
    char cmd_buf[64] = {0};
    {
        const char *end = strchr(cmd, '"');
        size_t cl = end ? (size_t)(end - cmd) : 0;
        if (cl >= sizeof(cmd_buf)) cl = sizeof(cmd_buf) - 1;
        memcpy(cmd_buf, cmd, cl);
    }

    log_info("[WS] Command: %s (id=%s)", cmd_buf, id_buf);

    /* ── Command dispatch ── */
    if (strcmp(cmd_buf, "get_status") == 0) {
        char buf[512];
        build_status_json(ws, buf, sizeof(buf));
        send_ws_response(c, id_buf, "ok", buf);
        return;
    }

    if (strcmp(cmd_buf, "get_config") == 0) {
        int cfg_len = 0;
        char *cfg_json = build_config_json(ws, &cfg_len);
        send_ws_response(c, id_buf, "ok", cfg_json ? cfg_json : "{}");
        free(cfg_json);
        return;
    }

    if (strcmp(cmd_buf, "get_models") == 0) {
        int m_len = 0;
        char *m_json = build_models_json(&m_len);
        send_ws_response(c, id_buf, "ok", m_json ? m_json : "{\"models\":[]}");
        free(m_json);
        return;
    }

    if (strcmp(cmd_buf, "start_pipeline") == 0) {
        if (!ws->controller) {
            send_ws_response(c, id_buf, "error", "{\"message\":\"No controller\"}");
            return;
        }
        PipelineState st = psm_get(&ws->controller->state_machine);
        if (st != PIPELINE_STATE_IDLE) {
            send_ws_response(c, id_buf, "error",
                             "{\"message\":\"Pipeline not idle\"}");
            return;
        }
        int rc = system_controller_start_async(ws->controller,
                                               PIPELINE_MODE_REALTIME);
        send_ws_response(c, id_buf, rc == 0 ? "ok" : "error",
                         rc == 0 ? "{\"message\":\"Pipeline starting\"}"
                                 : "{\"message\":\"Start failed\"}");
        return;
    }

    if (strcmp(cmd_buf, "stop_pipeline") == 0) {
        if (!ws->controller) {
            send_ws_response(c, id_buf, "error", "{\"message\":\"No controller\"}");
            return;
        }
        int rc = system_controller_stop_async(ws->controller);
        send_ws_response(c, id_buf, rc == 0 ? "ok" : "error",
                         rc == 0 ? "{\"message\":\"Pipeline stopped\"}"
                                 : "{\"message\":\"Stop failed\"}");
        return;
    }

    if (strcmp(cmd_buf, "restart_pipeline") == 0) {
        if (!ws->controller) {
            send_ws_response(c, id_buf, "error", "{\"message\":\"No controller\"}");
            return;
        }
        system_controller_stop_async(ws->controller);
        int rc = system_controller_start_async(ws->controller,
                                               PIPELINE_MODE_REALTIME);
        send_ws_response(c, id_buf, rc == 0 ? "ok" : "error",
                         rc == 0 ? "{\"message\":\"Pipeline restarting\"}"
                                 : "{\"message\":\"Restart failed\"}");
        return;
    }

    if (strcmp(cmd_buf, "set_config") == 0) {
        if (!ws->controller || !ws->controller->config) {
            send_ws_response(c, id_buf, "error", "{\"message\":\"No config\"}");
            return;
        }
        /* Parse key and value from params */
        const char *key = NULL;
        const char *val = NULL;
        {
            const char *p = strstr(data, "\"key\":\"");
            if (p) key = p + 7;
            p = strstr(data, "\"value\":");
            if (p) val = p + 8;
        }
        if (!key || !val) {
            send_ws_response(c, id_buf, "error",
                             "{\"message\":\"Missing key or value\"}");
            return;
        }
        char key_buf[128] = {0};
        {
            const char *end = strchr(key, '"');
            size_t kl = end ? (size_t)(end - key) : 0;
            if (kl >= sizeof(key_buf)) kl = sizeof(key_buf) - 1;
            memcpy(key_buf, key, kl);
        }
        /* Set value based on format */
        if (*val == '"') {
            char val_buf[256] = {0};
            val++;
            const char *end = strchr(val, '"');
            size_t vl = end ? (size_t)(end - val) : 0;
            if (vl >= sizeof(val_buf)) vl = sizeof(val_buf) - 1;
            memcpy(val_buf, val, vl);
            config_set_string(ws->controller->config, key_buf, val_buf);
        } else if (strncmp(val, "true", 4) == 0) {
            config_set_bool(ws->controller->config, key_buf, true);
        } else if (strncmp(val, "false", 5) == 0) {
            config_set_bool(ws->controller->config, key_buf, false);
        } else if (strchr(val, '.')) {
            config_set_float(ws->controller->config, key_buf, (float)atof(val));
        } else {
            config_set_int(ws->controller->config, key_buf, atoi(val));
        }
        log_info("[WS] Config set: %s = <value>", key_buf);
        /* Persist to YAML so settings survive restart */
        if (ws->controller->config->config_path[0]) {
            config_save_to_file(ws->controller->config, ws->controller->config->config_path);
        }
        send_ws_response(c, id_buf, "ok", "{\"message\":\"Config updated\"}");
        return;
    }

    if (strcmp(cmd_buf, "reset_world_origin") == 0) {
        if (ws->controller) {
            system_controller_reset_world_origin(ws->controller);
        }
        send_ws_response(c, id_buf, "ok", "{\"message\":\"World origin reset\"}");
        return;
    }

    if (strcmp(cmd_buf, "ping") == 0) {
        send_ws_response(c, id_buf, "ok", "{\"pong\":true}");
        return;
    }

    /* Unknown command */
    log_warn("[WS] Unknown command: %s", cmd_buf);
    send_ws_response(c, id_buf, "error", "{\"message\":\"Unknown command\"}");
}

static void send_ws_response(struct mg_connection *c, const char *id,
                              const char *status, const char *data_json) {
    char *buf = NULL;
    if (data_json && data_json[0] == '{') {
        asprintf(&buf,
                 "{\"type\":\"response\",\"id\":\"%s\",\"status\":\"%s\",\"data\":%s}",
                 id ? id : "0", status, data_json);
    } else {
        asprintf(&buf,
                 "{\"type\":\"response\",\"id\":\"%s\",\"status\":\"%s\"}",
                 id ? id : "0", status);
    }
    if (buf) {
        mg_ws_send(c, buf, strlen(buf), WEBSOCKET_OP_TEXT);
        free(buf);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * POST route handlers
 * ═══════════════════════════════════════════════════════════════════════ */

static void handle_post_config(WebServer *ws, struct mg_connection *c,
                                struct mg_http_message *hm) {
    if (!ws->controller || !ws->controller->config) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"No config manager\"}");
        return;
    }

    /* Parse key and value from JSON body */
    /* body format: {"key":"detection.confidence_threshold","value":0.5} */
    if (!hm->body.buf || hm->body.len < 10) {
        mg_http_reply(c, 400, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"Empty or invalid body\"}");
        return;
    }
    const char *body = hm->body.buf;
    char key_buf[128] = {0};
    const char *kp = strstr(body, "\"key\":\"");
    if (kp) {
        kp += 7;
        const char *ke = strchr(kp, '"');
        size_t kl = ke ? (size_t)(ke - kp) : 0;
        if (kl >= sizeof(key_buf)) kl = sizeof(key_buf) - 1;
        memcpy(key_buf, kp, kl);
    } else {
        mg_http_reply(c, 400, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"Missing key\"}");
        return;
    }

    const char *vp = strstr(body, "\"value\":");
    if (!vp) {
        mg_http_reply(c, 400, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"Missing value\"}");
        return;
    }
    vp += 8;

    /* Set based on type */
    if (*vp == '"') {
        char val_buf[256] = {0};
        vp++;
        const char *ve = strchr(vp, '"');
        size_t vl = ve ? (size_t)(ve - vp) : 0;
        if (vl >= sizeof(val_buf)) vl = sizeof(val_buf) - 1;
        memcpy(val_buf, vp, vl);
        config_set_string(ws->controller->config, key_buf, val_buf);
    } else if (strncmp(vp, "true", 4) == 0) {
        config_set_bool(ws->controller->config, key_buf, true);
    } else if (strncmp(vp, "false", 5) == 0) {
        config_set_bool(ws->controller->config, key_buf, false);
    } else if (strchr(vp, '.')) {
        config_set_float(ws->controller->config, key_buf, (float)atof(vp));
    } else {
        config_set_int(ws->controller->config, key_buf, atoi(vp));
    }

    /* Persist to YAML so settings survive restart */
    if (ws->controller->config->config_path[0]) {
        config_save_to_file(ws->controller->config, ws->controller->config->config_path);
    }

    mg_http_reply(c, 200, "Content-Type: application/json\r\n"
                       "Access-Control-Allow-Origin: *\r\n",
                  "{\"ok\":true,\"key\":\"%s\"}", key_buf);
}

static void handle_pipeline_start(WebServer *ws, struct mg_connection *c) {
    if (!ws->controller) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"No controller\"}");
        return;
    }
    if (!psm_is_idle(&ws->controller->state_machine)) {
        mg_http_reply(c, 409, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"Pipeline not idle\"}");
        return;
    }
    int rc = system_controller_start_async(ws->controller, PIPELINE_MODE_REALTIME);
    mg_http_reply(c, rc == 0 ? 200 : 500,
                  "Content-Type: application/json\r\n"
                  "Access-Control-Allow-Origin: *\r\n",
                  rc == 0 ? "{\"ok\":true,\"state\":\"starting\"}"
                          : "{\"error\":\"Start failed\"}");
}

static void handle_pipeline_stop(WebServer *ws, struct mg_connection *c) {
    if (!ws->controller) {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n"
                           "Access-Control-Allow-Origin: *\r\n",
                      "{\"error\":\"No controller\"}");
        return;
    }
    int rc = system_controller_stop_async(ws->controller);
    mg_http_reply(c, rc == 0 ? 200 : 500,
                  "Content-Type: application/json\r\n"
                  "Access-Control-Allow-Origin: *\r\n",
                  rc == 0 ? "{\"ok\":true,\"state\":\"stopped\"}"
                          : "{\"error\":\"Stop failed\"}");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Status broadcast timer
 * ═══════════════════════════════════════════════════════════════════════ */

/* Status broadcast timer — called from mg_mgr_poll thread (same thread as
 * ev_handler), so ws_clients[] access is safe without frame_mutex. */
static void status_broadcast_timer_fn(void *arg) {
    WebServer *ws = (WebServer *)arg;
    if (!ws->controller) return;

    char buf[512];
    PipelineState st = psm_get(&ws->controller->state_machine);
    int fc = ws->controller->frame_count;
    int active = ws->controller->tracking_manager
                     ? ws->controller->tracking_manager->num_tracks : 0;

    snprintf(buf, sizeof(buf),
             "{\"type\":\"event\",\"event\":\"pipeline_status\","
             "\"data\":{\"state\":\"%s\",\"frame_count\":%d,"
             "\"active_tracks\":%d}}",
             psm_state_name(st), fc, active);

    for (int i = 0; i < ws->ws_count; i++) {
        struct mg_connection *client = ws->ws_clients[i];
        if (client && is_ws(client)) {
            mg_ws_send(client, buf, strlen(buf), WEBSOCKET_OP_TEXT);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Mongoose event handler
 * ═══════════════════════════════════════════════════════════════════════ */

static void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
    WebServer *ws = (WebServer *)c->fn_data;

    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
        ws->http_requests++;

        /* ── WebSocket upgrade ── */
        if (mg_match(hm->uri, mg_str("/ws"), NULL)) {
            mg_ws_upgrade(c, hm, NULL);
            mark_ws(c);
            /* Register client */
            pthread_mutex_lock(&ws->frame_mutex);
            if (ws->ws_count < WS_MAX_CLIENTS) {
                ws->ws_clients[ws->ws_count++] = c;
                ws->clients_served++;
                log_info("[Web] WebSocket client connected (%d total)", ws->ws_count);
            }
            pthread_mutex_unlock(&ws->frame_mutex);

            /* Send hello message */
            char hello[256];
            snprintf(hello, sizeof(hello),
                     "{\"type\":\"hello\",\"session_id\":\"realtime\","
                     "\"version\":\"2.0.0\",\"platform\":\"%s\"}",
#ifdef PLATFORM_RISCV64
                     "K1"
#else
                     "host"
#endif
            );
            mg_ws_send(c, hello, strlen(hello), WEBSOCKET_OP_TEXT);
            return;
        }

        /* ── API: status ── */
        if (mg_match(hm->uri, mg_str("/api/status"), NULL)) {
            char buf[1024];
            int len = build_status_json(ws, buf, sizeof(buf));
            mg_http_reply(c, 200, "Content-Type: application/json\r\n"
                               "Access-Control-Allow-Origin: *\r\n",
                          "%.*s", len, buf);
            return;
        }

        /* ── API: config (GET — real data from ConfigManager) ── */
        if (mg_match(hm->uri, mg_str("/api/config"), NULL)) {
            if (mg_match(hm->method, mg_str("GET"), NULL)) {
                int cfg_len = 0;
                char *cfg_json = build_config_json(ws, &cfg_len);
                mg_http_reply(c, 200, "Content-Type: application/json\r\n"
                                   "Access-Control-Allow-Origin: *\r\n",
                              "%.*s", cfg_len, cfg_json ? cfg_json : "{}");
                free(cfg_json);
                return;
            }
            /* POST /api/config */
            if (mg_match(hm->method, mg_str("POST"), NULL)) {
                handle_post_config(ws, c, hm);
                return;
            }
        }

        /* ── API: pipeline start ── */
        if (mg_match(hm->uri, mg_str("/api/pipeline/start"), NULL)) {
            handle_pipeline_start(ws, c);
            return;
        }

        /* ── API: pipeline stop ── */
        if (mg_match(hm->uri, mg_str("/api/pipeline/stop"), NULL)) {
            handle_pipeline_stop(ws, c);
            return;
        }

        /* ── API: models ── */
        if (mg_match(hm->uri, mg_str("/api/models"), NULL)) {
            int m_len = 0;
            char *m_json = build_models_json(&m_len);
            mg_http_reply(c, 200, "Content-Type: application/json\r\n"
                               "Access-Control-Allow-Origin: *\r\n",
                          "%.*s", m_len, m_json ? m_json : "{\"models\":[]}");
            free(m_json);
            return;
        }

        /* ── API: latest JPEG frame ──
         * CRITICAL: JPEG is binary (contains \x00 bytes).
         * mg_http_reply() uses vsnprintf internally → "%.*s" stops
         * at the first null byte, truncating the image → browser
         * shows broken-image icon.  Use mg_printf + mg_send instead. */
        if (mg_match(hm->uri, mg_str("/api/frame.jpg"), NULL)) {
            if (ws->controller && ws->controller->latest_jpeg && ws->controller->latest_jpeg_len > 0) {
                pthread_mutex_lock(&ws->controller->jpeg_mutex);
                size_t jpeg_len = ws->controller->latest_jpeg_len;
                mg_printf(c,
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: %zu\r\n"
                    "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "\r\n",
                    jpeg_len);
                mg_send(c, ws->controller->latest_jpeg, jpeg_len);
                pthread_mutex_unlock(&ws->controller->jpeg_mutex);
            } else {
                mg_http_reply(c, 204, "", "");  /* No content yet */
            }
            return;
        }

        /* ── Serve static frontend files ── */
        struct mg_http_serve_opts opts = {
            .root_dir = ws->web_root,
            .extra_headers = "Access-Control-Allow-Origin: *\r\n"
        };
        mg_http_serve_dir(c, hm, &opts);
        return;
    }

    /* ── WebSocket message received ── */
    if (ev == MG_EV_WS_MSG) {
        struct mg_ws_message *wm = (struct mg_ws_message *)ev_data;
        if (wm->data.len > 0 && wm->data.buf && wm->data.len < 4096) {
            /* Copy + null-terminate before parsing — Mongoose does not
             * guarantee a null terminator on WebSocket text frames. */
            char ws_buf[4096];
            size_t copy_len = wm->data.len < sizeof(ws_buf)-1
                              ? wm->data.len : sizeof(ws_buf)-1;
            memcpy(ws_buf, wm->data.buf, copy_len);
            ws_buf[copy_len] = '\0';
            handle_ws_command(ws, c, ws_buf, (int)copy_len);
        }
        /* Let Mongoose manage recv buffer — don't clear it here or
         * subsequent WS frames in the same TCP packet will be lost. */
        return;
    }

    /* ── Connection closed ── */
    if (ev == MG_EV_CLOSE) {
        if (is_ws(c)) {
            pthread_mutex_lock(&ws->frame_mutex);
            /* Remove from client list */
            for (int i = 0; i < ws->ws_count; i++) {
                if (ws->ws_clients[i] == c) {
                    ws->ws_clients[i] = ws->ws_clients[ws->ws_count - 1];
                    ws->ws_count--;
                    log_info("[Web] WebSocket client disconnected (%d remain)", ws->ws_count);
                    break;
                }
            }
            pthread_mutex_unlock(&ws->frame_mutex);
        }
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Timer callback: broadcast pending frames to all WebSocket clients
 * ═══════════════════════════════════════════════════════════════════════ */

static void broadcast_timer_fn(void *arg) {
    WebServer *ws = (WebServer *)arg;

    pthread_mutex_lock(&ws->frame_mutex);

    /* Read all pending frames from the ring buffer */
    int read_idx = ws->frame_read_idx;
    while (read_idx != ws->frame_write_idx) {
        WsFrameSlot *slot = &ws->frame_ring[read_idx % WS_RING_SIZE];
        if (!slot->has_data) break;

        /* Broadcast to all connected WebSocket clients */
        for (int i = 0; i < ws->ws_count; i++) {
            struct mg_connection *client = ws->ws_clients[i];
            if (client && is_ws(client)) {
                /* 1. Send text JSON frame (metadata, no base64 JPEG) */
                mg_ws_send(client, slot->json, (size_t)slot->json_len,
                           WEBSOCKET_OP_TEXT);

                /* 2. Send binary JPEG frame if available (single combined frame)
                 *    Format: 'J'(0x4A) 'P'(0x50) frame_hi frame_lo | JPEG data */
                if (slot->has_jpeg && slot->jpeg_len > 0) {
                    /* Combine 4-byte header + JPEG into single binary frame */
                    size_t total = 4 + (size_t)slot->jpeg_len;
                    uint8_t combined[4 + WS_MAX_JPEG_LEN];
                    combined[0] = 'J';
                    combined[1] = 'P';
                    combined[2] = (uint8_t)((slot->frame_index >> 8) & 0xFF);
                    combined[3] = (uint8_t)(slot->frame_index & 0xFF);
                    memcpy(combined + 4, slot->jpeg_data, (size_t)slot->jpeg_len);
                    mg_ws_send(client, combined, total, WEBSOCKET_OP_BINARY);
                }
            }
        }

        slot->has_data = false;
        slot->has_jpeg = false;
        read_idx++;
        ws->frames_broadcast++;
    }
    ws->frame_read_idx = read_idx;

    pthread_mutex_unlock(&ws->frame_mutex);
    /* Binary frame flush happens on next mg_mgr_poll() iteration.
     * Poll interval reduced to 50ms below — max 50ms extra latency for JPEG frames. */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Server thread function
 * ═══════════════════════════════════════════════════════════════════════ */

static void *server_thread_fn(void *arg) {
    WebServer *ws = (WebServer *)arg;

    log_info("[Web] Server thread starting on %s (root=%s)",
             ws->listen_addr, ws->web_root);

    /* Set up frame broadcast timer (200ms) */
    mg_timer_add(&ws->mgr, WS_BROADCAST_INTERVAL_MS, MG_TIMER_REPEAT,
                 broadcast_timer_fn, ws);

    /* Set up status broadcast timer (500ms) — Phase B */
    mg_timer_add(&ws->mgr, WS_STATUS_INTERVAL_MS, MG_TIMER_REPEAT,
                 status_broadcast_timer_fn, ws);

    /* Create HTTP listener */
    mg_http_listen(&ws->mgr, ws->listen_addr, ev_handler, ws);

    log_info("[Web] Listening on %s", ws->listen_addr);

    /* Event loop — 50ms poll for lower binary-frame latency */
    while (ws->running) {
        mg_mgr_poll(&ws->mgr, 50);
    }

    log_info("[Web] Server thread exiting (broadcast=%d, clients=%d, http=%d)",
             ws->frames_broadcast, ws->clients_served, ws->http_requests);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════ */

int web_server_init(WebServer *ws, const char *listen_addr, const char *web_root) {
    if (!ws || !listen_addr || !web_root) return -1;

    memset(ws, 0, sizeof(*ws));
    mg_mgr_init(&ws->mgr);

    strncpy(ws->listen_addr, listen_addr, sizeof(ws->listen_addr) - 1);
    ws->listen_addr[sizeof(ws->listen_addr) - 1] = '\0';

    strncpy(ws->web_root, web_root, sizeof(ws->web_root) - 1);
    ws->web_root[sizeof(ws->web_root) - 1] = '\0';

    if (pthread_mutex_init(&ws->frame_mutex, NULL) != 0) {
        mg_mgr_free(&ws->mgr);
        return -1;
    }

    /* Verify web_root exists (non-fatal warning) */
    if (access(ws->web_root, R_OK) != 0) {
        log_warn("[Web] Web root not accessible: %s (frontend may not load)", ws->web_root);
    }

    return 0;
}

int web_server_start(WebServer *ws) {
    if (!ws || ws->running) return -1;

    ws->running = true;
    ws->shutdown = false;

    if (pthread_create(&ws->thread, NULL, server_thread_fn, ws) != 0) {
        log_error("[Web] Failed to create server thread");
        ws->running = false;
        return -1;
    }

    log_info("[Web] Server started on %s", ws->listen_addr);
    return 0;
}

int web_server_push_frame(WebServer *ws, const char *frame_json, int frame_index) {
    return web_server_push_frame_jpeg(ws, frame_json, frame_index, NULL, 0);
}

int web_server_push_frame_jpeg(WebServer *ws, const char *frame_json,
                                int frame_index,
                                const uint8_t* jpeg_data, int jpeg_len) {
    if (!ws || !frame_json || !ws->running) return -1;

    int json_len = (int)strlen(frame_json);
    if (json_len >= WS_MAX_FRAME_JSON_LEN) {
        log_warn("[Web] Frame JSON too large: %d bytes (max %d)", json_len,
                 WS_MAX_FRAME_JSON_LEN);
        return -1;
    }

    pthread_mutex_lock(&ws->frame_mutex);

    int write_idx = ws->frame_write_idx;
    int next_idx  = write_idx + 1;

    /* Check if ring buffer is full (read hasn't caught up) */
    if (next_idx - ws->frame_read_idx > WS_RING_SIZE) {
        /* Drop oldest unread frame */
        ws->frame_read_idx++;
        log_debug("[Web] Frame ring buffer full, dropping oldest frame");
    }

    WsFrameSlot *slot = &ws->frame_ring[write_idx % WS_RING_SIZE];
    memcpy(slot->json, frame_json, (size_t)(json_len + 1));
    slot->json_len  = json_len;
    slot->frame_index = frame_index;
    slot->has_data   = true;

    /* Attach raw JPEG for binary WebSocket send */
    if (jpeg_data && jpeg_len > 0 && jpeg_len <= WS_MAX_JPEG_LEN) {
        memcpy(slot->jpeg_data, jpeg_data, (size_t)jpeg_len);
        slot->jpeg_len  = jpeg_len;
        slot->has_jpeg  = true;
    } else {
        slot->has_jpeg = false;
    }

    ws->frame_write_idx = next_idx;

    pthread_mutex_unlock(&ws->frame_mutex);
    return 0;
}

void web_server_stop(WebServer *ws) {
    if (!ws || !ws->running) return;

    log_info("[Web] Stopping server...");
    ws->running = false;

    /* Wait for server thread to exit */
    pthread_join(ws->thread, NULL);

    /* Cleanup Mongoose */
    mg_mgr_free(&ws->mgr);
    pthread_mutex_destroy(&ws->frame_mutex);

    ws->shutdown = true;
    log_info("[Web] Server stopped (broadcast=%d frames, served=%d clients)",
             ws->frames_broadcast, ws->clients_served);
}

/* ── Pipeline control integration (Phase B) ── */

void web_server_set_controller(WebServer *ws, SystemController *sc) {
    if (!ws) return;
    ws->controller = sc;
    log_info("[Web] Controller attached");
}

void web_server_broadcast_event(WebServer *ws, const char *event,
                                const char *json_data) {
    if (!ws || !event || !json_data) return;

    char buf[16384];
    snprintf(buf, sizeof(buf),
             "{\"type\":\"event\",\"event\":\"%s\",\"data\":%s}",
             event, json_data);

    pthread_mutex_lock(&ws->frame_mutex);
    for (int i = 0; i < ws->ws_count; i++) {
        struct mg_connection *client = ws->ws_clients[i];
        if (client && is_ws(client)) {
            mg_ws_send(client, buf, strlen(buf), WEBSOCKET_OP_TEXT);
        }
    }
    pthread_mutex_unlock(&ws->frame_mutex);
}

/* ── Convenience: create + start with defaults ── */

WebServer *web_server_create_default(int port) {
    if (port <= 0 || port > 65535) port = 8080;

    WebServer *ws = (WebServer *)calloc(1, sizeof(WebServer));
    if (!ws) return NULL;

    char addr[64];
    snprintf(addr, sizeof(addr), "http://0.0.0.0:%d", port);

    /* Determine web root: check install prefix first, then local */
    const char *web_root = "/usr/share/lingqi_tantong/web";
    if (access(web_root, R_OK) != 0) {
        web_root = "web";  /* fallback: local dev build */
    }

    if (web_server_init(ws, addr, web_root) != 0) {
        free(ws);
        return NULL;
    }

    if (web_server_start(ws) != 0) {
        free(ws);
        return NULL;
    }

    return ws;
}

void web_server_destroy_default(WebServer *ws) {
    if (!ws) return;
    web_server_stop(ws);
    free(ws);
}
