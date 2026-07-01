#ifndef K1_PLATFORM_H
#define K1_PLATFORM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K1_CPU_CORES              8
#define K1_CLUSTER0_CORES         4
#define K1_CLUSTER1_CORES         4
#define K1_CLUSTER0_OFFSET        0
#define K1_CLUSTER1_OFFSET        4
#define K1_TCM_SIZE               (512 * 1024)

#define K1_CPU_CLUSTER0_AI        0
#define K1_CPU_CLUSTER0_INFERENCE 1
#define K1_CPU_CLUSTER0_DETECT    2   /* legacy */
#define K1_CPU_CLUSTER0_POSE      3
#define K1_CPU_CLUSTER0_STGCN      3   /* ST-GCN on Cluster0 (needs RVV), share CPU3 with unused POSE */
#define K1_CPU_CLUSTER1_CAPTURE   4
#define K1_CPU_CLUSTER1_VPU       5
#define K1_CPU_CLUSTER1_VIZ       6
#define K1_CPU_CLUSTER1_STGCN     7   /* REMOVED: Cluster1 causes SIGILL in ORT MLAS RVV kernels */

typedef enum {
    K1_CLUSTER_AI       = 0,
    K1_CLUSTER_IO       = 1,
    K1_CLUSTER_COUNT    = 2
} K1ClusterType;

typedef enum {
    K1_CAP_RVV_1_0      = (1 << 0),
    K1_CAP_AI_IME       = (1 << 1),
    K1_CAP_TCM          = (1 << 2),
    K1_CAP_VPU          = (1 << 3),
    K1_CAP_JPU          = (1 << 4),
    K1_CAP_GPU          = (1 << 5),
    K1_CAP_MPP          = (1 << 6),
    K1_CAP_SPACENGINE   = (1 << 7),
    K1_CAP_SPACEMIT_EP  = (1 << 8),
} K1Capability;

typedef struct K1Platform K1Platform;

K1Platform* k1_platform_init(void);
void k1_platform_destroy(K1Platform* plat);

bool k1_platform_is_k1(void);
bool k1_platform_has_cap(K1Capability cap);
uint32_t k1_platform_get_caps(void);

int  k1_platform_cpu_count(void);
int  k1_platform_cluster_core_count(K1ClusterType cluster);
int  k1_platform_cluster_first_core(K1ClusterType cluster);
int  k1_platform_get_tcm_size(void);

bool k1_pin_thread_to_cpu(int cpu_id);
bool k1_pin_thread_to_cluster(K1ClusterType cluster);
bool k1_pin_current_to_cpu(int cpu_id);
bool k1_pin_current_to_cluster(K1ClusterType cluster);

void* k1_tcm_alloc(size_t size);
void  k1_tcm_free(void* ptr, size_t size);
bool  k1_tcm_is_available(void);

double k1_get_time_us(void);
double k1_get_time_ms(void);

bool k1_is_cluster0_core(int cpu_id);
bool k1_is_cluster1_core(int cpu_id);
int  k1_get_current_cpu(void);

void k1_thread_yield(void);
void k1_memory_fence(void);

#ifdef __cplusplus
}
#endif

#endif