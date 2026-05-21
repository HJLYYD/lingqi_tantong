#ifndef SPACEMIT_ORT_BRIDGE_H
#define SPACEMIT_ORT_BRIDGE_H

struct OrtSessionOptions;

#ifdef __cplusplus
extern "C" {
#endif

int spacemit_ort_session_options_init(struct OrtSessionOptions* session_options);

#ifdef __cplusplus
}
#endif

#endif