cmake_minimum_required(VERSION 3.16)
project(LingQiTanTong C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(USE_ONNX_RUNTIME "Enable ONNX Runtime for real model inference" ON)
option(USE_SPACENGINE_AI "Enable SpacemiT RISC-V AI instruction acceleration" OFF)
option(USE_OPENMP "Enable OpenMP parallel processing" ON)
option(ENABLE_RVV_OPT "Enable RISC-V Vector (RVV 1.0) optimizations" OFF)
option(ENABLE_K1_PIPELINE "Enable K1 dual-cluster pipeline parallelism" ON)
option(ENABLE_K1_TCM "Enable K1 TCM (Tightly Coupled Memory) for AI weights" ON)
option(ENABLE_K1_VPU "Enable K1 VPU hardware video/JPEG acceleration" ON)
option(ENABLE_K1_JPU "Enable K1 JPU hardware JPEG decode" ON)

set(MUSE_PI_ARCH "rv64gcv0p7" CACHE STRING "Target RISC-V arch: rv64gcv0p7 (K1) or rv64gcv1p0")
set(BIANBU_SYSROOT "" CACHE PATH "Bianbu OS sysroot path (for cross-compilation)")
set(ONNX_RUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation directory")
set(SPACENGINE_DIR "" CACHE PATH "Path to SpacemiT Spacengine SDK")
set(K1_MPP_DIR "" CACHE PATH "Path to K1 MPP (Media Process Platform) SDK")
set(K1_JPU_DIR "" CACHE PATH "Path to K1 JPU hardware JPEG decode library")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" OR CMAKE_C_COMPILER MATCHES "riscv64")
    set(BUILDING_FOR_RISCV64 TRUE)
else()
    set(BUILDING_FOR_RISCV64 FALSE)
endif()

if(BUILDING_FOR_RISCV64)
    add_compile_definitions(PLATFORM_MUSE_PI_PRO)
    add_compile_definitions(PLATFORM_RISCV64)

    set(DETECTED_RISCV_ARCH "rv64gcv")

    if(WIN32)
        set(DEV_NULL "NUL")
    else()
        set(DEV_NULL "/dev/null")
    endif()

    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -march=rv64gcv1p0 -E -x c ${DEV_NULL}
        RESULT_VARIABLE TEST_RV64GCV1P0_RESULT
        ERROR_QUIET OUTPUT_QUIET
    )
    if(TEST_RV64GCV1P0_RESULT EQUAL 0)
        set(DETECTED_RISCV_ARCH "rv64gcv1p0")
        set(RVV_VERSION "1.0")
        message(STATUS "RVV 1.0 detected (K1 native)")
    else()
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -march=rv64gcv -E -x c ${DEV_NULL}
            RESULT_VARIABLE TEST_RV64GCV_RESULT
            ERROR_QUIET OUTPUT_QUIET
        )
        if(NOT TEST_RV64GCV_RESULT EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -march=rv64gcv0p7 -E -x c ${DEV_NULL}
                RESULT_VARIABLE TEST_RV64GCV0P7_RESULT
                ERROR_QUIET OUTPUT_QUIET
            )
            if(TEST_RV64GCV0P7_RESULT EQUAL 0)
                set(DETECTED_RISCV_ARCH "rv64gcv0p7")
                set(RVV_VERSION "0.7")
            else()
                set(DETECTED_RISCV_ARCH "rv64gc")
                set(RVV_VERSION "NONE")
                message(WARNING "V extension not detected in compiler, disabling RVV")
            endif()
        endif()
    endif()

    set(RISCV_COMMON_FLAGS "-O3 -Wall -Wextra -fomit-frame-pointer -ffast-math -flto")

    if(CMAKE_C_COMPILER MATCHES "riscv64-linux-gnu" OR CMAKE_CROSSCOMPILING)
        if(NOT CMAKE_C_COMPILER MATCHES "riscv64-linux-gnu")
            string(REPLACE " -flto" "" RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS}")
        endif()
    endif()

    set(RISCV_ARCH_FLAGS "-march=${DETECTED_RISCV_ARCH}")

    if(DETECTED_RISCV_ARCH STREQUAL "rv64gcv1p0")
        set(RISCV_ARCH_FLAGS "${RISCV_ARCH_FLAGS} -mtune=x60")
        set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} -fprefetch-loop-arrays -funroll-loops")
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0")
            set(K1_EXTRA_FLAGS "-mrvv-vector-bits=256")
        else()
            set(K1_EXTRA_FLAGS "")
            message(STATUS "GCC ${CMAKE_C_COMPILER_VERSION} < 14, -mrvv-vector-bits not available")
        endif()
        if(ENABLE_K1_PIPELINE)
            add_compile_definitions(HAS_K1_PIPELINE)
            if(K1_EXTRA_FLAGS)
                set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} ${K1_EXTRA_FLAGS}")
            endif()
            message(STATUS "K1 dual-cluster pipeline parallelism enabled")
        endif()
        if(ENABLE_K1_TCM)
            add_compile_definitions(HAS_K1_TCM)
            message(STATUS "K1 TCM memory management enabled")
        endif()
        if(ENABLE_K1_VPU)
            add_compile_definitions(HAS_K1_VPU)
            message(STATUS "K1 VPU hardware acceleration enabled")
        endif()
        if(ENABLE_K1_JPU)
            add_compile_definitions(HAS_K1_JPU)
            message(STATUS "K1 JPU hardware JPEG decode enabled")
        endif()
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RISCV_ARCH_FLAGS} ${RISCV_COMMON_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -flto -ffast-math")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3")

    message(STATUS "Detected RISC-V arch: ${DETECTED_RISCV_ARCH} (RVV ${RVV_VERSION})")
    message(STATUS "RISC-V C flags: ${RISCV_ARCH_FLAGS} ${RISCV_COMMON_FLAGS}")

    if(BIANBU_SYSROOT)
        set(CMAKE_SYSROOT "${BIANBU_SYSROOT}")
        set(CMAKE_FIND_ROOT_PATH "${BIANBU_SYSROOT}")
    endif()

    if(CMAKE_C_COMPILER MATCHES "riscv64" OR BUILDING_FOR_RISCV64)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic -lpthread -lm -ldl")
    endif()

    if(USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_C_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
            add_compile_definitions(HAS_OPENMP)
            message(STATUS "OpenMP enabled for RISC-V multi-core")
        else()
            message(WARNING "OpenMP not found, using single-core mode")
        endif()
    endif()

    if(ENABLE_RVV_OPT)
        add_compile_definitions(HAS_RVV_OPT)
        message(STATUS "RVV ${RVV_VERSION} vector optimization enabled")
    endif()
else()
    message(WARNING "Building for non-RISC-V platform (dev/testing only)")
    message(WARNING "Target platform is SpacemiT K1 Muse Pi Pro (riscv64)")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -Wextra")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 -flto")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3")
    add_compile_definitions(PLATFORM_HOST_DEV)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

set(CORE_SOURCES
    src/logger.c
    src/utils.c
    src/config_manager.c
    src/ort_common.c
    src/core_types.c
    src/k1_platform.c
)

set(DATA_SOURCES
    src/video_processor.c
    src/imu_handler.c
    src/model_store.c
    src/arrow_receiver.c
)

set(MODEL_SOURCES
    src/yolov8_detector.c
    src/yolov8_pose_estimator.c
    src/scrfd_detector.c
    src/arcface_recognizer.c
)

set(BUSINESS_SOURCES
    src/inference_pipeline.c
    src/tracking_manager.c
    src/spatial_engine.c
    src/result_manager.c
)

set(FUSION_SOURCES
    src/mahony_filter.c
    src/kcp_lite.c
)

set(PRESENTATION_SOURCES
    src/visualizer.c
    src/ar_renderer.c
    src/video_writer.c
    src/model_export.c
)

set(UTILS_SOURCES
    src/system_controller.c
)

set(AI_ACCEL_SOURCES
    src/ai_accel_adapter.c
)

set(MAIN_SOURCE
    src/main.c
)

set(BENCHMARK_SOURCE
    src/benchmark.c
)

set(ALL_LIB_SOURCES
    ${CORE_SOURCES}
    ${DATA_SOURCES}
    ${MODEL_SOURCES}
    ${BUSINESS_SOURCES}
    ${FUSION_SOURCES}
    ${PRESENTATION_SOURCES}
    ${UTILS_SOURCES}
    ${AI_ACCEL_SOURCES}
)

set(ALL_LIB_CXX_SOURCES "")

add_library(lingqi_tantong_core STATIC ${ALL_LIB_SOURCES} ${ALL_LIB_CXX_SOURCES})

if(ALL_LIB_CXX_SOURCES)
    target_link_libraries(lingqi_tantong_core stdc++)
endif()

add_executable(lingqi_tantong ${MAIN_SOURCE})
target_link_libraries(lingqi_tantong lingqi_tantong_core)

if(UNIX)
    target_link_libraries(lingqi_tantong m pthread)
endif()

if(ALL_LIB_CXX_SOURCES)
    target_link_libraries(lingqi_tantong stdc++)
endif()

if(BUILDING_FOR_RISCV64)
    target_link_libraries(lingqi_tantong atomic)
endif()

if(USE_ONNX_RUNTIME AND ONNX_RUNTIME_DIR)
    target_include_directories(lingqi_tantong_core PRIVATE "${ONNX_RUNTIME_DIR}/include")

    if(EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
        target_link_libraries(lingqi_tantong "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
    elseif(EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.a")
        target_link_libraries(lingqi_tantong "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.a")
    elseif(EXISTS "/usr/lib/libonnxruntime.so")
        target_link_libraries(lingqi_tantong onnxruntime)
    else()
        message(FATAL_ERROR "ONNX Runtime library not found at ${ONNX_RUNTIME_DIR}/lib/")
    endif()

    if(EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
        target_link_libraries(lingqi_tantong "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
        message(STATUS "SpacemiT RISC-V MLAS (optimized math) enabled")
    elseif(EXISTS "/usr/lib/libonnxruntime_mlas.so")
        target_link_libraries(lingqi_tantong onnxruntime_mlas)
        message(STATUS "SpacemiT RISC-V MLAS (optimized math) enabled")
    endif()

    add_compile_definitions(HAS_ONNX_RUNTIME)
    message(STATUS "ONNX Runtime enabled: ${ONNX_RUNTIME_DIR}")

    find_file(SPACEMIT_ORT_ENV_H
        NAMES spacemit_ort_env.h
        PATHS "${ONNX_RUNTIME_DIR}/include"
              "${ONNX_RUNTIME_DIR}/include/onnxruntime"
              "/usr/local/include/onnxruntime"
              "/usr/local/include"
        NO_DEFAULT_PATH
    )

    if(SPACEMIT_ORT_ENV_H AND CMAKE_CXX_COMPILER)
        enable_language(CXX)
        target_sources(lingqi_tantong_core PRIVATE src/spacemit_ort_bridge.cpp)
        target_compile_definitions(lingqi_tantong_core PRIVATE HAS_SPACEMIT_EP)
        target_link_libraries(lingqi_tantong_core stdc++)
        message(STATUS "SpacemiT Execution Provider enabled (ONNX Runtime)")
    else()
        if(SPACEMIT_ORT_ENV_H)
            message(STATUS "SpacemiT EP header found, but no CXX compiler - EP disabled")
        else()
            message(STATUS "spacemit_ort_env.h not found, using CPU-only ONNX Runtime")
        endif()
    endif()
elseif(USE_ONNX_RUNTIME)
    message(STATUS "ONNX Runtime dir not set: heuristic fallback mode")
    message(STATUS "  Set -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.2.0.2 for real inference")
endif()

if(USE_SPACENGINE_AI AND BUILDING_FOR_RISCV64)
    if(SPACENGINE_DIR)
        target_include_directories(lingqi_tantong_core PRIVATE "${SPACENGINE_DIR}/include")
        target_link_directories(lingqi_tantong PRIVATE "${SPACENGINE_DIR}/lib")
        target_link_libraries(lingqi_tantong spacengine_rt)

        target_compile_definitions(lingqi_tantong_core PRIVATE HAS_SPACENGINE_AI)
        message(STATUS "SpacemiT RISC-V AI acceleration enabled: ${SPACENGINE_DIR}")
    else()
        message(FATAL_ERROR "SPACENGINE_DIR must be set when USE_SPACENGINE_AI=ON")
    endif()
elseif(USE_SPACENGINE_AI)
    message(WARNING "Spacengine AI acceleration only available on RISC-V Muse Pi Pro")
endif()

if(BUILDING_FOR_RISCV64 AND DETECTED_RISCV_ARCH STREQUAL "rv64gcv1p0")
    if(ENABLE_K1_VPU AND K1_MPP_DIR)
        target_include_directories(lingqi_tantong_core PRIVATE "${K1_MPP_DIR}/include")
        target_link_directories(lingqi_tantong PRIVATE "${K1_MPP_DIR}/lib")
        target_link_libraries(lingqi_tantong mpp)
        add_compile_definitions(HAS_K1_MPP)
        message(STATUS "K1 MPP hardware video pipeline: ${K1_MPP_DIR}")
    elseif(ENABLE_K1_VPU)
        message(STATUS "K1 VPU: MPP SDK not set, video acceleration disabled")
        message(STATUS "  Set -DK1_MPP_DIR=/path/to/k1x-mpp for hardware video")
    endif()

    if(ENABLE_K1_JPU AND K1_JPU_DIR)
        target_include_directories(lingqi_tantong_core PRIVATE "${K1_JPU_DIR}/include")
        target_link_directories(lingqi_tantong PRIVATE "${K1_JPU_DIR}/lib")
        target_link_libraries(lingqi_tantong jpu)
        message(STATUS "K1 JPU hardware JPEG decode: ${K1_JPU_DIR}")
    elseif(ENABLE_K1_JPU)
        message(STATUS "K1 JPU: SDK not set, JPEG hardware decode disabled")
        message(STATUS "  Set -DK1_JPU_DIR=/path/to/k1x-jpu for HW JPEG decode")
    endif()
endif()

add_executable(test_basic tests/test_basic.c)
target_link_libraries(test_basic lingqi_tantong_core)

if(UNIX)
    target_link_libraries(test_basic m pthread)
endif()

if(BUILDING_FOR_RISCV64)
    target_link_libraries(test_basic atomic)
endif()

install(TARGETS lingqi_tantong DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models/ DESTINATION share/lingqi_tantong/models)
install(FILES configs/default.yaml DESTINATION etc/lingqi_tantong/config.yaml)

message(STATUS "============================================")
message(STATUS "LingQi TanTong Build Configuration")
message(STATUS "============================================")
message(STATUS "Target Platform:  ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C Compiler:        ${CMAKE_C_COMPILER}")
message(STATUS "C Flags:           ${CMAKE_C_FLAGS}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime:      ${USE_ONNX_RUNTIME}")
if(SPACEMIT_ORT_ENV_H AND CMAKE_CXX_COMPILER)
    message(STATUS "SpacemiT EP:       YES (RVV 1.0 + IME hardware acceleration)")
else()
    message(STATUS "SpacemiT EP:       NO (CPU-only)")
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "riscv")
    if(EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
        message(STATUS "RISC-V MLAS:       YES")
    else()
        message(STATUS "RISC-V MLAS:       NO (lib not found)")
    endif()
endif()
message(STATUS "Spacengine AI accel: ${USE_SPACENGINE_AI}")
message(STATUS "OpenMP:            ${USE_OPENMP}")
message(STATUS "RVV Optimize:      ${ENABLE_RVV_OPT}")
message(STATUS "K1 Pipeline:       ${ENABLE_K1_PIPELINE}")
message(STATUS "K1 TCM:            ${ENABLE_K1_TCM}")
message(STATUS "K1 VPU/MPP:        ${ENABLE_K1_VPU}")
message(STATUS "K1 JPU:            ${ENABLE_K1_JPU}")
message(STATUS "============================================")