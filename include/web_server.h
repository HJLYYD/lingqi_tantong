/*
 * web_server.h — Embedded HTTP + WebSocket server for LingQi TanTong
 *
 * Uses Mongoose (cesanta/mongoose) single-file embedded networking library.
 * Serves:
 *   - Static React SPA frontend (GET /app/... wildcard)
 *   - REST API endpoints (GET /api/status, /api/config)
 *   - WebSocket real-time frame stream (ws://host:port/ws)
 *
 * Architecture:
 *   WebServer runs in its own pthread. The inference pipeline's PostProcess
 *   thread writes completed FrameMessage JSON into a thread-safe ring buffer.
 *   A Mongoose timer polls the buffer every ~200ms and broadcasts to all
 *   connected WebSocket clients.
 *
 * Integration:
 *   1. web_server_start()  — after pipeline init, before main loop
 *   2. web_server_push_frame() — called by PostProcess thread each frame
 *   3. web_server_stop()   — during cleanup
 */

#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include "mongoose.h"
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

/* Forward declaration to avoid circular include */
typedef struct SystemController SystemController;

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ── */
#define WS_MAX_CLIENTS         32          /* max concurrent WebSocket clients */
#define WS_RING_SIZE           32          /* frame ring buffer capacity */
#define WS_BROADCAST_INTERVAL_MS  200     /* timer period for frame broadcast */
#define WS_STATUS_INTERVAL_MS  500         /* timer period for status broadcast */
#define WS_MAX_FRAME_JSON_LEN  16384       /* max serialized frame JSON size */
#define WS_MAX_JPEG_LEN        8192        /* max raw JPEG thumbnail size */

/* ── WebSocket frame ring buffer entry ── */
typedef struct {
    char   json[WS_MAX_FRAME_JSON_LEN];
    int    json_len;
    int    frame_index;
    bool   has_data;
    /* Optional raw JPEG for binary WebSocket send (no base64 overhead) */
    uint8_t jpeg_data[WS_MAX_JPEG_LEN];
    int     jpeg_len;
    bool    has_jpeg;
} WsFrameSlot;

/* ── Web server state ── */
typedef struct {
    struct mg_mgr          mgr;                         /* Mongoose event manager */
    struct mg_connection  *ws_clients[WS_MAX_CLIENTS];  /* active WebSocket connections */
    int                    ws_count;

    /* ── Frame ring buffer (PostProcess thread writes, server thread reads) ── */
    WsFrameSlot            frame_ring[WS_RING_SIZE];
    volatile int           frame_write_idx;   /* PostProcess thread writes here */
    int                    frame_read_idx;    /* server timer reads here */
    pthread_mutex_t        frame_mutex;       /* protects frame_ring[] */

    /* ── Server config (set before start) ── */
    char                   listen_addr[64];   /* e.g. "http://0.0.0.0:8080" */
    char                   web_root[256];     /* static file root directory */

    /* ── Threading ── */
    pthread_t              thread;
    volatile bool          running;
    volatile bool          shutdown;

    /* ── Pipeline control (Phase B) ── */
    SystemController*      controller;     /* set via web_server_set_controller() */

    /* ── Statistics ── */
    volatile int           frames_broadcast;
    volatile int           clients_served;
    volatile int           http_requests;
} WebServer;

/* ── API ── */

/**
 * Initialize web server struct. Does NOT start serving.
 *
 * @param ws           pointer to caller-allocated WebServer
 * @param listen_addr  e.g. "http://0.0.0.0:8080" or "http://localhost:8080"
 * @param web_root     directory containing static frontend files (web/)
 * @return 0 on success, -1 on error
 */
int  web_server_init(WebServer* ws, const char* listen_addr, const char* web_root);

/**
 * Start the web server in a background pthread.
 * Returns immediately; the server runs in its own thread.
 *
 * @param ws  initialized WebServer (call web_server_init first)
 * @return 0 on success, -1 on error
 */
int  web_server_start(WebServer* ws);

/**
 * Push a completed frame JSON into the ring buffer for WebSocket broadcast.
 * Called by the PostProcess thread after each frame finishes.
 * Thread-safe: uses pthread_mutex to protect the ring buffer.
 *
 * @param ws         running WebServer
 * @param frame_json  null-terminated JSON string
 * @param frame_index monotonic frame number
 * @return 0 on success, -1 if buffer is full
 */
int  web_server_push_frame(WebServer* ws, const char* frame_json, int frame_index);

/**
 * Push a completed frame JSON + optional raw JPEG into the ring buffer.
 * The JPEG is sent as a binary WebSocket frame (no base64 overhead) after
 * the text JSON frame, reducing bandwidth by ~33%.
 *
 * @param ws          running WebServer
 * @param frame_json  null-terminated JSON string (no base64 JPEG needed)
 * @param frame_index monotonic frame number
 * @param jpeg_data   raw JPEG bytes (can be NULL)
 * @param jpeg_len    JPEG byte count (0 if none)
 * @return 0 on success, -1 if buffer is full
 */
int  web_server_push_frame_jpeg(WebServer* ws, const char* frame_json,
                                int frame_index,
                                const uint8_t* jpeg_data, int jpeg_len);

/**
 * Stop the web server thread and release all connections.
 * Blocks until the server thread exits.
 *
 * @param ws  running WebServer
 */
void web_server_stop(WebServer* ws);

/**
 * Convenience: create + start web server with default settings.
 * Listens on 0.0.0.0:8080, serves web/ from install prefix.
 *
 * @param port  TCP port (0 = use default 8080)
 * @return heap-allocated WebServer, or NULL on error
 */
WebServer* web_server_create_default(int port);

/**
 * Convenience: stop + destroy a default-created WebServer.
 */
void web_server_destroy_default(WebServer* ws);

/* ── Pipeline control integration (Phase B) ── */

/**
 * Set the SystemController reference for command handling.
 * Must be called before starting the pipeline (after web_server_init).
 */
void web_server_set_controller(WebServer* ws, SystemController* sc);

/**
 * Broadcast a JSON event to all connected WebSocket clients.
 * Used for pipeline status change notifications.
 *
 * @param ws      running WebServer
 * @param event   event name (e.g. "pipeline_status")
 * @param json    pre-formatted JSON data payload
 */
void web_server_broadcast_event(WebServer* ws, const char* event,
                                const char* json);

#ifdef __cplusplus
}
#endif

#endif /* WEB_SERVER_H */
