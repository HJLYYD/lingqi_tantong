/**
 * pipeline_state.c — Five-state pipeline state machine implementation
 */

#include "pipeline_state.h"
#include "utils.h"      /* utils_get_time_ms */
#include "logger.h"     /* log_info / log_warn / log_error */
#include <string.h>
#include <time.h>

/* ── Transition table ──
 * transition_allowed[from][to] = true if valid
 */
static bool transition_allowed[5][5] = {
    /* to → IDLE,  STARTING, RUNNING, STOPPING, ERROR */
    /* IDLE     */ { false,  true,     false,   false,    false },
    /* STARTING */ { false,  false,    true,    true,     true  },  /* allow cancel during startup */
    /* RUNNING  */ { false,  false,    false,   true,     true  },
    /* STOPPING */ { true,   false,    false,   false,    true  },
    /* ERROR    */ { true,   false,    false,   false,    false },
};

void psm_init(PipelineStateMachine *psm) {
    if (!psm) return;
    memset(psm, 0, sizeof(*psm));
    psm->state = PIPELINE_STATE_IDLE;
    psm->state_entry_time_ms = utils_get_time_ms();
    psm->error_code = 0;
    psm->error_message[0] = '\0';
    pthread_mutex_init(&psm->state_mutex, NULL);
    pthread_cond_init(&psm->state_change_cond, NULL);
}

void psm_destroy(PipelineStateMachine *psm) {
    if (!psm) return;
    pthread_mutex_destroy(&psm->state_mutex);
    pthread_cond_destroy(&psm->state_change_cond);
}

bool psm_transition(PipelineStateMachine *psm, PipelineState to,
                    const char *reason) {
    if (!psm) return false;

    pthread_mutex_lock(&psm->state_mutex);

    PipelineState from = psm->state;

    if (!transition_allowed[from][to]) {
        pthread_mutex_unlock(&psm->state_mutex);
        log_warn("[StateMachine] Invalid transition: %s → %s (reason: %s)",
                 psm_state_name(from), psm_state_name(to),
                 reason ? reason : "none");
        return false;
    }

    psm->state = to;
    psm->state_entry_time_ms = utils_get_time_ms();
    if (to == PIPELINE_STATE_ERROR && reason) {
        strncpy(psm->error_message, reason, sizeof(psm->error_message) - 1);
        psm->error_message[sizeof(psm->error_message) - 1] = '\0';
    }
    if (to == PIPELINE_STATE_IDLE) {
        psm->error_code = 0;
        psm->error_message[0] = '\0';
    }

    pthread_cond_broadcast(&psm->state_change_cond);
    pthread_mutex_unlock(&psm->state_mutex);

    log_info("[StateMachine] %s → %s (reason: %s)",
             psm_state_name(from), psm_state_name(to),
             reason ? reason : "none");

    return true;
}

PipelineState psm_get(const PipelineStateMachine *psm) {
    if (!psm) return PIPELINE_STATE_ERROR;

    pthread_mutex_lock((pthread_mutex_t *)&psm->state_mutex);
    PipelineState s = psm->state;
    pthread_mutex_unlock((pthread_mutex_t *)&psm->state_mutex);
    return s;
}

const char *psm_state_name(PipelineState s) {
    switch (s) {
        case PIPELINE_STATE_IDLE:     return "idle";
        case PIPELINE_STATE_STARTING: return "starting";
        case PIPELINE_STATE_RUNNING:  return "running";
        case PIPELINE_STATE_STOPPING: return "stopping";
        case PIPELINE_STATE_ERROR:    return "error";
        default:                      return "unknown";
    }
}

bool psm_wait_for(PipelineStateMachine *psm, PipelineState target,
                  int timeout_ms) {
    if (!psm) return false;

    pthread_mutex_lock(&psm->state_mutex);

    if (psm->state == target) {
        pthread_mutex_unlock(&psm->state_mutex);
        return true;
    }

    if (timeout_ms <= 0) {
        /* Wait forever */
        while (psm->state != target) {
            pthread_cond_wait(&psm->state_change_cond, &psm->state_mutex);
        }
        pthread_mutex_unlock(&psm->state_mutex);
        return true;
    }

    /* Timed wait */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec  += 1;
        ts.tv_nsec -= 1000000000L;
    }

    int rc = 0;
    while (psm->state != target && rc == 0) {
        rc = pthread_cond_timedwait(&psm->state_change_cond,
                                    &psm->state_mutex, &ts);
    }

    bool ok = (psm->state == target);
    pthread_mutex_unlock(&psm->state_mutex);
    return ok;
}
