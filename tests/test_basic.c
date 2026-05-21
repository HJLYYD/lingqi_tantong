#include "core_types.h"
#include "logger.h"
#include "utils.h"
#include "config_manager.h"
#include "yolov8_detector.h"
#include "tracking_manager.h"
#include "spatial_engine.h"
#include "inference_pipeline.h"
#include "result_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("Running test: " #name " ... "); \
    test_##name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED\n  Assertion failed: %s at line %d\n", #cond, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, eps) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAILED\n  Float assertion failed: %f != %f at line %d\n", (a), (b), __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

TEST(bounding_box_basic) {
    BoundingBox bbox = {10.0f, 20.0f, 50.0f, 80.0f};
    ASSERT_FLOAT_EQ(bbox_center_x(&bbox), 30.0f, 1e-6f);
    ASSERT_FLOAT_EQ(bbox_center_y(&bbox), 50.0f, 1e-6f);
    ASSERT_FLOAT_EQ(bbox_width(&bbox), 40.0f, 1e-6f);
    ASSERT_FLOAT_EQ(bbox_height(&bbox), 60.0f, 1e-6f);
    ASSERT_FLOAT_EQ(bbox_area(&bbox), 2400.0f, 1e-6f);
}

TEST(bounding_box_iou) {
    BoundingBox a = {0.0f, 0.0f, 10.0f, 10.0f};
    BoundingBox b = {5.0f, 5.0f, 15.0f, 15.0f};
    float iou = bbox_iou(&a, &b);
    ASSERT_FLOAT_EQ(iou, 25.0f / 175.0f, 1e-6f);

    BoundingBox c = {20.0f, 20.0f, 30.0f, 30.0f};
    iou = bbox_iou(&a, &c);
    ASSERT_FLOAT_EQ(iou, 0.0f, 1e-6f);
}

TEST(spatial_position_distance) {
    SpatialPosition a = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .depth_confidence = 1.0f, .is_valid = true};
    SpatialPosition b = {.x = 3.0f, .y = 4.0f, .z = 0.0f, .depth_confidence = 1.0f, .is_valid = true};
    ASSERT_FLOAT_EQ(spatial_distance(&a, &b), 5.0f, 1e-6f);
    ASSERT_FLOAT_EQ(spatial_distance_from_origin(&b), 5.0f, 1e-6f);
}

TEST(trajectory_buffer) {
    TrajectoryBuffer buf;
    memset(&buf, 0, sizeof(buf));

    for (int i = 0; i < MAX_TRAJECTORY_LEN + 10; i++) {
        SpatialPosition pos = {.x = (float)i, .y = (float)i * 2, .z = (float)i * 3, .depth_confidence = 1.0f, .is_valid = true};
        trajectory_buffer_append(&buf, &pos);
    }

    ASSERT(buf.count == MAX_TRAJECTORY_LEN);
    ASSERT_FLOAT_EQ(buf.positions[0].x, 10.0f, 1e-6f);
    ASSERT_FLOAT_EQ(buf.positions[MAX_TRAJECTORY_LEN - 1].x, (float)(MAX_TRAJECTORY_LEN + 9), 1e-6f);
}

TEST(config_manager_basic) {
    ConfigManager* cm = config_manager_create(NULL);
    ASSERT(cm != NULL);

    config_set_int(cm, "test.value", 42);
    ASSERT(config_get_int(cm, "test.value", 0) == 42);

    config_set_float(cm, "test.ratio", 3.14f);
    ASSERT_FLOAT_EQ(config_get_float(cm, "test.ratio", 0.0f), 3.14f, 1e-6f);

    config_set_bool(cm, "test.flag", true);
    ASSERT(config_get_bool(cm, "test.flag", false) == true);

    config_set_string(cm, "test.name", "hello");
    ASSERT(strcmp(config_get_string(cm, "test.name", ""), "hello") == 0);

    config_manager_destroy(cm);
}

TEST(yolov8_detector_create) {
    YOLOv8Detector* det = yolov8_detector_create(NULL, 640, 640, 0.4f, 0.45f, false);
    ASSERT(det != NULL);
    ASSERT(det->input_width == 640);
    ASSERT(det->input_height == 640);
    ASSERT_FLOAT_EQ(det->confidence_threshold, 0.4f, 1e-6f);
    yolov8_detector_destroy(det);
}

TEST(tracker_create_update) {
    ObjectTracker* tracker = object_tracker_create(2, 0.3f, 5.0f, 300);
    ASSERT(tracker != NULL);
    ASSERT(tracker->max_lost == 2);
    ASSERT_FLOAT_EQ(tracker->min_iou, 0.3f, 1e-6f);

    Detection dets[2];
    memset(dets, 0, sizeof(dets));
    dets[0].bbox = (BoundingBox){10.0f, 10.0f, 50.0f, 100.0f};
    dets[0].confidence = 0.8f;
    dets[1].bbox = (BoundingBox){100.0f, 100.0f, 150.0f, 200.0f};
    dets[1].confidence = 0.7f;

    SpatialPosition pos[2] = {
        {.x = 1.0f, .y = 2.0f, .z = 3.0f, .depth_confidence = 1.0f, .is_valid = true},
        {.x = 4.0f, .y = 5.0f, .z = 6.0f, .depth_confidence = 1.0f, .is_valid = true}
    };

    TrackingResult result = object_tracker_update(tracker, dets, 2, pos, 2, 1);
    ASSERT(result.num_tracked == 0);

    result = object_tracker_update(tracker, dets, 2, pos, 2, 2);
    ASSERT(result.num_tracked == 0);

    result = object_tracker_update(tracker, dets, 2, pos, 2, 3);
    ASSERT(result.num_tracked == 2);

    object_tracker_destroy(tracker);
}

TEST(spatial_engine_depth) {
    SpatialLocalizationEngine* engine = spatial_engine_create(NULL, NULL, 500.0f, 1.7f);
    ASSERT(engine != NULL);

    Detection det;
    memset(&det, 0, sizeof(det));
    det.bbox = (BoundingBox){100.0f, 100.0f, 200.0f, 400.0f};

    spatial_engine_initialize_coordinate_system(engine, 1080, 1920, &det);
    ASSERT(engine->world_initialized == true);

    SpatialResult result = spatial_engine_calculate_position(engine, &det, 1920, 1080, NULL, 0, 0);
    ASSERT(result.depth > 0.0f);
    ASSERT(result.confidence > 0.0f);

    float height = spatial_engine_calculate_height(engine, &det, NULL);
    ASSERT(height > 0.0f);

    spatial_engine_destroy(engine);
}

TEST(utils_clamp) {
    ASSERT(UTILS_CLAMP(5, 0, 10) == 5);
    ASSERT(UTILS_CLAMP(-5, 0, 10) == 0);
    ASSERT(UTILS_CLAMP(15, 0, 10) == 10);
    ASSERT(UTILS_MIN(3, 5) == 3);
    ASSERT(UTILS_MAX(3, 5) == 5);
}

TEST(utils_sort) {
    float arr[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    int indices[8];
    utils_sort_float_desc(arr, indices, 8);

    ASSERT_FLOAT_EQ(arr[0], 9.0f, 1e-6f);
    ASSERT_FLOAT_EQ(arr[1], 6.0f, 1e-6f);
    ASSERT_FLOAT_EQ(arr[2], 5.0f, 1e-6f);
    ASSERT_FLOAT_EQ(arr[7], 1.0f, 1e-6f);
}

TEST(inference_pipeline_create) {
    AIInferencePipeline* pipeline = inference_pipeline_create(false);
    ASSERT(pipeline != NULL);
    ASSERT(pipeline->enabled_stages == PIPELINE_ENABLE_ALL);
    inference_pipeline_destroy(pipeline);
}

TEST(result_manager_session) {
    ResultManager* rm = result_manager_create("test_output");
    ASSERT(rm != NULL);

    const char* session_id = result_manager_start_session(rm, "test.mp4");
    ASSERT(strlen(session_id) > 0);

    result_manager_update_session_stats(rm, 100, 10, 25.5f, 40.0f);
    result_manager_add_tracked_object(rm, 1, 1.75f, 50, 10);
    result_manager_add_error(rm, "TestError", "Test error message");

    const char* ended_id = result_manager_end_session(rm);
    ASSERT(strcmp(ended_id, session_id) == 0);

    result_manager_destroy(rm);
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    printf("============================================================\n");
    printf("LingQi TanTong System - C Version Unit Tests\n");
    printf("============================================================\n\n");

    logger_init(NULL, LOG_LEVEL_WARNING);

    RUN_TEST(bounding_box_basic);
    RUN_TEST(bounding_box_iou);
    RUN_TEST(spatial_position_distance);
    RUN_TEST(trajectory_buffer);
    RUN_TEST(config_manager_basic);
    RUN_TEST(yolov8_detector_create);
    RUN_TEST(tracker_create_update);
    RUN_TEST(spatial_engine_depth);
    RUN_TEST(utils_clamp);
    RUN_TEST(utils_sort);
    RUN_TEST(inference_pipeline_create);
    RUN_TEST(result_manager_session);

    printf("\n============================================================\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("============================================================\n");

    logger_close();

    return tests_failed > 0 ? 1 : 0;
}
