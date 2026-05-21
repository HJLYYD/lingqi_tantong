#ifndef MODEL_EXPORT_H
#define MODEL_EXPORT_H

#include "core_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EXPORT_OBJ  = 0,
    EXPORT_GLTF = 1,
    EXPORT_JSON = 2
} ExportFormat;

int model_export_spatial_data(
    const SpatialPosition* positions, int num_positions,
    const char* output_path,
    ExportFormat format
);

int model_export_obstacles(
    const BoundingBox* obstacles, int num_obstacles,
    const char* output_path,
    ExportFormat format
);

int model_export_scene(
    const SpatialPosition* person_positions, int num_persons,
    const BoundingBox* obstacles,         int num_obstacles,
    const char* output_path,
    ExportFormat format
);

#ifdef __cplusplus
}
#endif

#endif