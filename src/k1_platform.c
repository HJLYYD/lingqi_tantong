#define _GNU_SOURCE

#include "k1_platform.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <errno.h>

struct K1Platform {
    bool is_k1;
    uint32_t capabilities;
    int cpu_count;
    int tcm_size;
    int tcm_fd;
    void* tcm_mapped;
    size_t tcm_mapped_size;
};

static K1Platform* g_k1_plat = NULL;
static pthread_once_t g_k1_plat_once = PTHREAD_ONCE_INIT;

static void detect_k1_capabilities(K1Platform* plat);  /* forward decl */

static void k1_platform_init_once(void) {
    K1Platform* plat = (K1Platform*)calloc(1, sizeof(K1Platform));
    if (!plat) return;

    plat->tcm_fd = -1;
    plat->tcm_mapped = NULL;
    plat->tcm_mapped_size = 0;

    detect_k1_capabilities(plat);
    g_k1_plat = plat;

    /* Log detected capabilities (runs once, thread-safe) */
    const char* caps_str[9] = {"RVV1.0","AI-IME","TCM","VPU","JPU","GPU","MPP","Spacengine","SpacemiT_EP"};
    log_info("K1 Platform: %s, %d CPUs, Capabilities:", plat->is_k1 ? "DETECTED" : "NOT DETECTED", plat->cpu_count);
    for (int i = 0; i < 9; i++) {
        if (plat->capabilities & (1 << i)) {
            log_info("  [+] %s", caps_str[i]);
        }
    }
}

static void detect_k1_capabilities(K1Platform* plat) {
    plat->capabilities = 0;
    plat->tcm_size = 0;
    plat->cpu_count = K1_CPU_CORES;

    /* Detect is_k1 from device-tree — only set true on confirmed K1 hardware */
    plat->is_k1 = false;
    FILE* fp = fopen("/proc/device-tree/compatible", "r");
    if (fp) {
        char buf[256] = {0};
        size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
        fclose(fp);
        if (n > 0 && strstr(buf, "spacemit,k1")) {
            plat->is_k1 = true;
        }
    }

    plat->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (plat->cpu_count <= 0) plat->cpu_count = K1_CPU_CORES;
    if (plat->cpu_count > K1_CPU_CORES) plat->cpu_count = K1_CPU_CORES;

    if (plat->is_k1) {
        plat->capabilities |= K1_CAP_RVV_1_0;
        plat->capabilities |= K1_CAP_SPACEMIT_EP;
    }

    /* TCM is reserved for SpacemiT EP — app layer should not use it.
     * We probe /dev/tcm for diagnostics only; do NOT advertise K1_CAP_TCM
     * because k1_tcm_alloc() always returns NULL (TCM is EP-exclusive). */
    int tcm_fd = open("/dev/tcm", O_RDWR);
    if (tcm_fd >= 0) {
        plat->tcm_size = K1_TCM_SIZE;
        close(tcm_fd);
        log_info("K1 TCM present (%zu MB, reserved for SpacemiT EP)",
                 (size_t)(K1_TCM_SIZE / (1024 * 1024)));
    }

    if (access("/dev/video0", F_OK) == 0) {
        plat->capabilities |= K1_CAP_VPU;
    }

    /* JPU hardware presence (informational only — JPEG decode uses libjpeg-turbo) */
    if (access("/dev/jpu", F_OK) == 0 || access("/dev/jpu0", F_OK) == 0) {
        plat->capabilities |= K1_CAP_JPU;
    }

    if (access("/dev/dri/renderD128", F_OK) == 0) {
        plat->capabilities |= K1_CAP_GPU;
    }
}

K1Platform* k1_platform_init(void) {
    pthread_once(&g_k1_plat_once, k1_platform_init_once);
    return g_k1_plat;
}

void k1_platform_destroy(K1Platform* plat) {
    if (!plat) return;

    if (plat->tcm_mapped) {
        munmap(plat->tcm_mapped, plat->tcm_mapped_size);
        plat->tcm_mapped = NULL;
    }
    if (plat->tcm_fd >= 0) {
        close(plat->tcm_fd);
        plat->tcm_fd = -1;
    }

    /* BUGFIX: null the global pointer BEFORE freeing, so concurrent
     * accessors in other threads see NULL instead of freed memory. */
    if (g_k1_plat == plat) g_k1_plat = NULL;
    free(plat);
}

bool k1_platform_is_k1(void) {
    return g_k1_plat ? g_k1_plat->is_k1 : false;
}

bool k1_platform_has_cap(K1Capability cap) {
    return g_k1_plat ? (g_k1_plat->capabilities & cap) != 0 : false;
}

uint32_t k1_platform_get_caps(void) {
    return g_k1_plat ? g_k1_plat->capabilities : 0;
}

int k1_platform_cpu_count(void) {
    return g_k1_plat ? g_k1_plat->cpu_count : 0;
}

int k1_platform_cluster_core_count(K1ClusterType cluster) {
    (void)cluster;
    if (!g_k1_plat || !g_k1_plat->is_k1) return 0;
    return K1_CLUSTER0_CORES;
}

int k1_platform_cluster_first_core(K1ClusterType cluster) {
    if (!g_k1_plat || !g_k1_plat->is_k1) return -1;
    return (cluster == K1_CLUSTER_AI) ? K1_CLUSTER0_OFFSET : K1_CLUSTER1_OFFSET;
}

int k1_platform_get_tcm_size(void) {
    return g_k1_plat ? g_k1_plat->tcm_size : 0;
}

bool k1_pin_thread_to_cpu(int cpu_id) {
    if (cpu_id < 0) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        log_warning("Failed to pin thread to CPU %d: %s", cpu_id, strerror(ret));
        return false;
    }
    return true;
}

bool k1_pin_thread_to_cluster(K1ClusterType cluster) {
    if (!g_k1_plat || !g_k1_plat->is_k1) return false;

    int first = k1_platform_cluster_first_core(cluster);
    int count = k1_platform_cluster_core_count(cluster);
    if (first < 0 || count <= 0) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < count; i++) {
        CPU_SET(first + i, &cpuset);
    }

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        log_warning("Failed to pin thread to cluster %d: %s", cluster, strerror(ret));
        return false;
    }
    return true;
}

bool k1_pin_current_to_cpu(int cpu_id) {
    return k1_pin_thread_to_cpu(cpu_id);
}

bool k1_pin_current_to_cluster(K1ClusterType cluster) {
    return k1_pin_thread_to_cluster(cluster);
}

void* k1_tcm_alloc(size_t size) {
    if (!g_k1_plat || !g_k1_plat->tcm_mapped || size > (size_t)g_k1_plat->tcm_size) {
        return NULL;
    }

    return g_k1_plat->tcm_mapped;
}

void k1_tcm_free(void* ptr, size_t size) {
    (void)ptr;
    (void)size;
}

bool k1_tcm_is_available(void) {
    return g_k1_plat && g_k1_plat->tcm_mapped && (g_k1_plat->capabilities & K1_CAP_TCM);
}

double k1_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

double k1_get_time_ms(void) {
    return k1_get_time_us() / 1000.0;
}

bool k1_is_cluster0_core(int cpu_id) {
    return cpu_id >= K1_CLUSTER0_OFFSET && cpu_id < K1_CLUSTER0_OFFSET + K1_CLUSTER0_CORES;
}

bool k1_is_cluster1_core(int cpu_id) {
    return cpu_id >= K1_CLUSTER1_OFFSET && cpu_id < K1_CLUSTER1_OFFSET + K1_CLUSTER1_CORES;
}

int k1_get_current_cpu(void) {
    return sched_getcpu();
}

void k1_thread_yield(void) {
    sched_yield();
}

void k1_memory_fence(void) {
    __sync_synchronize();
}