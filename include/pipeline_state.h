/**
 * pipeline_state.h — Five-state pipeline state machine
 *
 * Replaces the raw `volatile sig_atomic_t running` with explicit state tracking,
 * enabling GUI-driven pipeline lifecycle management.
 *
 * States:
 *   IDLE     → STARTING → RUNNING → STOPPING → IDLE
 *                ↓            ↓          ↓
 *              ERROR        ERROR      ERROR
 */

#ifndef PIPELINE_STATE_H
#define PIPELINE_STATE_H

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── State enum ── */
typedef enum {
    PIPELINE_STATE_IDLE     = 0,
    PIPELINE_STATE_STARTING = 1,
    PIPELINE_STATE_RUNNING  = 2,
    PIPELINE_STATE_STOPPING = 3,
    PIPELINE_STATE_ERROR    = 4,
} PipelineState;

/* ── State machine ── */
typedef struct {
    PipelineState state;
    char           error_message[256];
    int            error_code;            /* 0 = no error */
    uint64_t       state_entry_time_ms;   /* timestamp of last transition */
    pthread_mutex_t state_mutex;
    pthread_cond_t  state_change_cond;
} PipelineStateMachine;

/* ── API ── */

/** Initialize the state machine in IDLE state. */
void psm_init(PipelineStateMachine *psm);

/** Destroy mutex/cond resources. */
void psm_destroy(PipelineStateMachine *psm);

/**
 * Attempt a state transition.
 * Returns true on success, false if the transition is invalid.
 * Valid transitions:
 *   IDLE     → STARTING
 *   STARTING → RUNNING
 *   STARTING → ERROR
 *   RUNNING  → STOPPING
 *   RUNNING  → ERROR
 *   STOPPING → IDLE
 *   STOPPING → ERROR
 *   ERROR    → IDLE                     (recovery)
 */
bool psm_transition(PipelineStateMachine *psm, PipelineState to,
                    const char *reason);

/** Get current state (thread-safe). */
PipelineState psm_get(const PipelineStateMachine *psm);

/** Get state as a human-readable string. */
const char *psm_state_name(PipelineState s);

/**
 * Block until the state machine reaches `target` or timeout.
 * Returns true if target reached, false on timeout.
 * timeout_ms = 0 means wait forever.
 */
bool psm_wait_for(PipelineStateMachine *psm, PipelineState target,
                  int timeout_ms);

/** Check if the state machine is in a running-like state. */
static inline bool psm_is_running(const PipelineStateMachine *psm) {
    PipelineState s = psm_get(psm);
    return s == PIPELINE_STATE_STARTING || s == PIPELINE_STATE_RUNNING;
}

/** Check if the state machine is idle (ready to start). */
static inline bool psm_is_idle(const PipelineStateMachine *psm) {
    return psm_get(psm) == PIPELINE_STATE_IDLE;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINE_STATE_H */
