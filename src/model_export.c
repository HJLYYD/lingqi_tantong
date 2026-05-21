#include "model_export.h"
#include "logger.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int export_obj_persons(const SpatialPosition* positions, int num_positions, FILE* f) {
    if (!positions || !f || num_positions <= 0) return 0;

    int vertex_offset = 0;
    int person_count = 0;

    for (int i = 0; i < num_positions; i++) {
        const SpatialPosition* p = &positions[i];
        if (!p->is_valid) continue;

        float center_x = p->world_x;
        float center_z = p->world_z;
        float height = UTILS_MAX(p->estimated_height, 1.6f);
        float half_w = 0.25f;
        float half_d = 0.15f;

        int group = person_count + 1;
        fprintf(f, "o Person_%d\n", group);

        float v[8][3] = {
            {center_x - half_w, 0.0f,        center_z - half_d},
            {center_x + half_w, 0.0f,        center_z - half_d},
            {center_x + half_w, 0.0f,        center_z + half_d},
            {center_x - half_w, 0.0f,        center_z + half_d},
            {center_x - half_w, height,      center_z - half_d},
            {center_x + half_w, height,      center_z - half_d},
            {center_x + half_w, height,      center_z + half_d},
            {center_x - half_w, height,      center_z + half_d}
        };

        for (int vi = 0; vi < 8; vi++) {
            fprintf(f, "v %.4f %.4f %.4f\n", v[vi][0], v[vi][1], v[vi][2]);
        }

        int global_vertex_offset = vertex_offset;
        int global_normal_offset = vertex_offset / 8 * 6;

        fprintf(f, "vn 0 0 -1\nvn 1 0 0\nvn 0 0 1\nvn -1 0 0\nvn 0 1 0\nvn 0 -1 0\n");

        int faces[6][4] = {
            {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 4, 8, 7},
            {4, 1, 5, 8}, {5, 6, 7, 8}, {1, 4, 3, 2}
        };
        int vn_map[] = {1, 2, 3, 4, 5, 6};

        for (int fi = 0; fi < 6; fi++) {
            fprintf(f, "f %d//%d %d//%d %d//%d %d//%d\n",
                global_vertex_offset + faces[fi][0], global_normal_offset + vn_map[fi],
                global_vertex_offset + faces[fi][1], global_normal_offset + vn_map[fi],
                global_vertex_offset + faces[fi][2], global_normal_offset + vn_map[fi],
                global_vertex_offset + faces[fi][3], global_normal_offset + vn_map[fi]);
        }

        vertex_offset += 8;
        person_count++;
    }

    return person_count;
}

static int export_obj_obstacles(const BoundingBox* obstacles, int num_obstacles, FILE* f) {
    if (!obstacles || !f || num_obstacles <= 0) return 0;

    int count = 0;
    float obstacle_height = 2.5f;

    for (int i = 0; i < num_obstacles; i++) {
        const BoundingBox* b = &obstacles[i];
        float cx = (b->x_min + b->x_max) * 0.5f;
        float cz = (b->y_min + b->y_max) * 0.5f;
        float w  = b->x_max - b->x_min;
        float d  = b->y_max - b->y_min;

        fprintf(f, "o Obstacle_%d\n", i + 1);

        fprintf(f, "v %.4f 0        %.4f\n", cx - w/2, cz - d/2);
        fprintf(f, "v %.4f 0        %.4f\n", cx + w/2, cz - d/2);
        fprintf(f, "v %.4f 0        %.4f\n", cx + w/2, cz + d/2);
        fprintf(f, "v %.4f 0        %.4f\n", cx - w/2, cz + d/2);
        fprintf(f, "v %.4f %.4f     %.4f\n", cx - w/2, obstacle_height, cz - d/2);
        fprintf(f, "v %.4f %.4f     %.4f\n", cx + w/2, obstacle_height, cz - d/2);
        fprintf(f, "v %.4f %.4f     %.4f\n", cx + w/2, obstacle_height, cz + d/2);
        fprintf(f, "v %.4f %.4f     %.4f\n", cx - w/2, obstacle_height, cz + d/2);

        int base = count * 8 + 1;
        fprintf(f, "f %d %d %d %d\n", base, base+1, base+2, base+3);
        fprintf(f, "f %d %d %d %d\n", base+5, base+6, base+7, base+8);
        fprintf(f, "f %d %d %d %d\n", base, base+1, base+5, base+4);
        fprintf(f, "f %d %d %d %d\n", base+1, base+2, base+6, base+5);
        fprintf(f, "f %d %d %d %d\n", base+2, base+3, base+7, base+6);
        fprintf(f, "f %d %d %d %d\n", base+3, base, base+4, base+7);

        count++;
    }

    return count;
}

int model_export_spatial_data(
    const SpatialPosition* positions, int num_positions,
    const char* output_path,
    ExportFormat format
) {
    return model_export_scene(positions, num_positions, NULL, 0, output_path, format);
}

int model_export_obstacles(
    const BoundingBox* obstacles, int num_obstacles,
    const char* output_path,
    ExportFormat format
) {
    return model_export_scene(NULL, 0, obstacles, num_obstacles, output_path, format);
}

int model_export_scene(
    const SpatialPosition* person_positions, int num_persons,
    const BoundingBox* obstacles,         int num_obstacles,
    const char* output_path,
    ExportFormat format
) {
    if (!output_path) {
        log_error("Output path is NULL");
        return -1;
    }

    if (format == EXPORT_OBJ) {
        FILE* f = fopen(output_path, "w");
        if (!f) {
            log_error("Cannot open OBJ output file: %s", output_path);
            return -1;
        }

        fprintf(f, "# LingQi TanTong C - 3D Scene Export\n");
        fprintf(f, "# Generated scene reconstruction\n");
        fprintf(f, "mtllib scene.mtl\n");

        int person_count = 0;
        if (person_positions && num_persons > 0) {
            person_count = export_obj_persons(person_positions, num_persons, f);
        }

        int obstacle_count = 0;
        if (obstacles && num_obstacles > 0) {
            obstacle_count = export_obj_obstacles(obstacles, num_obstacles, f);
        }

        fclose(f);
        log_info("OBJ scene exported: %s (%d persons, %d obstacles)",
                 output_path, person_count, obstacle_count);
        return person_count + obstacle_count;
    }

    if (format == EXPORT_JSON) {
        FILE* f = fopen(output_path, "w");
        if (!f) {
            log_error("Cannot open JSON output file: %s", output_path);
            return -1;
        }

        fprintf(f, "{\n  \"scene\": {\n");
        fprintf(f, "    \"persons\": [\n");

        int person_count = 0;
        if (person_positions && num_persons > 0) {
            for (int i = 0; i < num_persons; i++) {
                const SpatialPosition* p = &person_positions[i];
                if (!p->is_valid) continue;

                if (person_count > 0) fprintf(f, ",\n");
                fprintf(f, "      {\"id\": %d, \"x\": %.4f, \"z\": %.4f, "
                        "\"height\": %.4f, \"confidence\": %.4f}",
                        i, p->world_x, p->world_z,
                        p->estimated_height, p->confidence);
                person_count++;
            }
        }

        fprintf(f, "\n    ],\n    \"obstacles\": [\n");

        int obstacle_count = 0;
        if (obstacles && num_obstacles > 0) {
            for (int i = 0; i < num_obstacles; i++) {
                const BoundingBox* b = &obstacles[i];

                if (obstacle_count > 0) fprintf(f, ",\n");
                fprintf(f, "      {\"id\": %d, \"x_min\": %.4f, \"x_max\": %.4f, "
                        "\"z_min\": %.4f, \"z_max\": %.4f}",
                        i, b->x_min, b->x_max, b->y_min, b->y_max);
                obstacle_count++;
            }
        }

        fprintf(f, "\n    ]\n  }\n}\n");

        fclose(f);
        log_info("JSON scene exported: %s (%d persons, %d obstacles)",
                 output_path, person_count, obstacle_count);
        return person_count + obstacle_count;
    }

    if (format == EXPORT_GLTF) {
        /*
         * GLTF 格式导出未实现，静默回退到 OBJ 格式。
         * TODO: 实现 GLTF 2.0 导出 (使用 cgltf 库或手写 JSON)
         * GLTF 规范: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
         */
        log_warning("GLTF export not yet implemented, falling back to OBJ format");

        char obj_path[MAX_STRING_LEN];
        snprintf(obj_path, sizeof(obj_path), "%s.obj", output_path);
        return model_export_scene(person_positions, num_persons,
                                  obstacles, num_obstacles, obj_path, EXPORT_OBJ);
    }

    log_error("Unsupported export format: %d", format);
    return -1;
}