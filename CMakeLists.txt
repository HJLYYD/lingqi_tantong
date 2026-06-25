cmake_minimum_required(VERSION 3.16)

# On K1 (riscv64), prefer Clang — better X60 support, no misaligned-vector bug.
# Must be BEFORE project() so compiler is locked in before all flag checks.
if(NOT CMAKE_C_COMPILER AND NOT CMAKE_CXX_COMPILER)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_HOST_ARCH MATCHES "riscv64")
        find_program(_CLANG NAMES clang-20 clang-19 clang-18 clang)
        find_program(_CLANGXX NAMES clang++-20 clang++-19 clang++-18 clang++)
        if(_CLANG AND _CLANGXX)
            set(CMAKE_C_COMPILER "${_CLANG}" CACHE FILEPATH "C compiler")
            set(CMAKE_CXX_COMPILER "${_CLANGXX}" CACHE FILEPATH "C++ compiler")
            set(_K1_USE_CLANG TRUE)
        endif()
    endif()
endif()

project(LingQiTanTong C)

if(_K1_USE_CLANG)
    message(STATUS "K1: using Clang (better X60 support, no SIGBUS risk)")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(USE_ONNX_RUNTIME "Enable ONNX Runtime for real model inference" ON)
option(USE_OPENMP "Enable OpenMP parallel processing" ON)
option(ENABLE_K1_PIPELINE "Enable K1 dual-cluster pipeline parallelism" ON)
option(ENABLE_K1_TCM "Enable K1 TCM (Tightly Coupled Memory) for AI weights" ON)
option(ENABLE_K1_VPU "Enable K1 VPU hardware video/JPEG acceleration" ON)
option(ENABLE_K1_JPU "Enable K1 JPU hardware JPEG decode" ON)
option(ENABLE_SDL2_DISPLAY "Enable SDL2 desktop window display (auto if available)" ON)

set(MUSE_PI_ARCH "rv64gcv0p7" CACHE STRING "Target RISC-V arch: rv64gcv0p7 (K1) or rv64gcv1p0")
set(BIANBU_SYSROOT "" CACHE PATH "Bianbu OS sysroot path (for cross-compilation)")
set(ONNX_RUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation directory")
set(K1_MPP_DIR "" CACHE PATH "Path to K1 MPP (Media Process Platform) SDK")
set(K1_JPU_DIR "" CACHE PATH "Path to K1 JPU hardware JPEG decode library")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" OR CMAKE_C_COMPILER MATCHES "riscv64")
    set(BUILDING_FOR_RISCV64 TRUE)
else()
    set(BUILDING_FOR_RISCV64 FALSE)
endif()

# RVV optimizations: auto-enabled on RISC-V targets.
# Must be after BUILDING_FOR_RISCV64 is determined.
if(BUILDING_FOR_RISCV64)
    set(ENABLE_RVV_OPT ON CACHE BOOL "Enable RISC-V Vector (RVV 1.0) optimizations" FORCE)
else()
    set(ENABLE_RVV_OPT OFF CACHE BOOL "Enable RISC-V Vector (RVV 1.0) optimizations" FORCE)
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

    set(RISCV_COMMON_FLAGS "-Wall -Wextra -fomit-frame-pointer")
    set(RISCV_RELEASE_FLAGS "-O3 -ffast-math")
    option(ENABLE_LTO "Enable Link-Time Optimization" ON)

    # Clang: always use version-matched lld linker (GNU ld is too strict for K1 .so deps)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        find_program(LLD_EXECUTABLE NAMES ld.lld-20 ld.lld-19 ld.lld-18 ld.lld-17 ld.lld)
        if(LLD_EXECUTABLE)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --ld-path=${LLD_EXECUTABLE} -Wl,--allow-shlib-undefined -Wl,--export-dynamic")
            message(STATUS "K1 linker: ${LLD_EXECUTABLE}")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -Wl,--allow-shlib-undefined -Wl,--export-dynamic")
        endif()
    endif()

    if(ENABLE_LTO)
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            set(RISCV_RELEASE_FLAGS "${RISCV_RELEASE_FLAGS} -flto -fopt-info-vec-optimized")
        elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
            find_program(LLVM_AR NAMES llvm-ar-20 llvm-ar-19 llvm-ar-18 llvm-ar-17 llvm-ar)
            find_program(LLVM_RANLIB NAMES llvm-ranlib-20 llvm-ranlib-19 llvm-ranlib-18 llvm-ranlib-17 llvm-ranlib)
            if(LLVM_AR AND LLVM_RANLIB)
                set(CMAKE_AR "${LLVM_AR}" CACHE FILEPATH "AR" FORCE)
                set(CMAKE_RANLIB "${LLVM_RANLIB}" CACHE FILEPATH "RANLIB" FORCE)
                set(RISCV_RELEASE_FLAGS "${RISCV_RELEASE_FLAGS} -flto=thin")
                message(STATUS "K1 Clang LTO (thin) enabled with llvm-ar + lld")
            else()
                message(STATUS "K1 Clang LTO disabled (llvm-ar not found — install llvm)")
            endif()
        endif()
    else()
        message(STATUS "LTO disabled (ENABLE_LTO=OFF)")
    endif()

    # Debug builds: no optimization, no LTO, no fast-math
    # (GCC 13.2 on RISC-V can emit illegal instructions with -ffast-math -flto)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
        message(STATUS "Debug build: -O0 -g (no LTO/fast-math for RISC-V safety)")
    else()
        set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} ${RISCV_RELEASE_FLAGS}")
    endif()

    if(CMAKE_C_COMPILER MATCHES "riscv64-linux-gnu" OR CMAKE_CROSSCOMPILING)
        if(NOT CMAKE_C_COMPILER MATCHES "riscv64-linux-gnu")
            string(REPLACE " -flto" "" RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS}")
        endif()
    endif()

    set(RISCV_ARCH_FLAGS "-march=${DETECTED_RISCV_ARCH}")

    if(DETECTED_RISCV_ARCH STREQUAL "rv64gcv1p0")
        # -mcpu=spacemit-x60 sets both -march and -mtune for X60 (GCC 14+).
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -mcpu=spacemit-x60 -E -x c ${DEV_NULL}
            RESULT_VARIABLE TEST_MCPU_X60_RESULT
            ERROR_QUIET OUTPUT_QUIET
        )
        if(TEST_MCPU_X60_RESULT EQUAL 0)
            set(RISCV_ARCH_FLAGS "-mcpu=spacemit-x60")
            message(STATUS "K1 -mcpu=spacemit-x60 supported by GCC ${CMAKE_C_COMPILER_VERSION}")
        else()
            set(RISCV_ARCH_FLAGS "-march=${DETECTED_RISCV_ARCH}")
            message(STATUS "GCC ${CMAKE_C_COMPILER_VERSION} does NOT support -mcpu=spacemit-x60; falling back to -march=${DETECTED_RISCV_ARCH}")
        endif()

        # -fprefetch-loop-arrays is unsupported on RISC-V backend; probe.
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} -march=${DETECTED_RISCV_ARCH} -fprefetch-loop-arrays -Werror -E -x c ${DEV_NULL}
            RESULT_VARIABLE TEST_PREFETCH_RESULT
            ERROR_QUIET OUTPUT_QUIET
        )
        if(TEST_PREFETCH_RESULT EQUAL 0)
            set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} -fprefetch-loop-arrays")
        endif()
        set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} -funroll-loops")

        # ── GCC auto-vectorization safety on X60 ────────────────────
        # X60 lacks Zicclsm (misaligned vector load/store hardware).
        # GCC 14 may emit misaligned vector instructions → SIGBUS.
        # GCC 15+ is safe.  Alternatives:
        #   - Upgrade OpenSBI to ≥v1.6 (software emulation)
        #   - Use Clang:   cmake -DCMAKE_C_COMPILER=clang
        #   - Disable vec:  add -fno-tree-vectorize to CFLAGS
        # Ref: community.milkv.io/t/spacemit-k1-m1-is-not-quite-rva22-compliant/2870
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS "15.0")
            message(STATUS "GCC ${CMAKE_C_COMPILER_VERSION} < 15 on K1 — auto-disabling tree-vectorize to avoid SIGBUS (X60 has no Zicclsm)")
            set(RISCV_COMMON_FLAGS "${RISCV_COMMON_FLAGS} -fno-tree-vectorize")
        endif()

        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0")
            set(K1_EXTRA_FLAGS "-mrvv-vector-bits=zvl")
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
            message(STATUS "K1 TCM memory management enabled")
        endif()
        if(ENABLE_K1_VPU)
            message(STATUS "K1 VPU hardware acceleration enabled")
        endif()
        if(ENABLE_K1_JPU)
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
        # find_package(OpenMP) fails on some RISC-V sysroots even when
        # -fopenmp works; probe directly with the compiler.
        # Clang on riscv64 needs special handling: libomp is not packaged
        # for riscv64 on Ubuntu/Bianbu, and omp.h lives in GCC's private
        # include dir.  Use libgomp + GCC's omp.h.
        if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
            set(_OPENMP_FLAG "-fopenmp=libgomp")
            # Find GCC's omp.h (Clang doesn't search GCC private include dirs)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-runtime-dir
                OUTPUT_VARIABLE _CLANG_RT_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            # Probe typical GCC include paths for omp.h
            set(_GCC_OMP_INCLUDE "")
            foreach(_GCC_VER 14 13 15)
                set(_TRY_PATH "/usr/lib/gcc/riscv64-linux-gnu/${_GCC_VER}/include")
                if(EXISTS "${_TRY_PATH}/omp.h")
                    set(_GCC_OMP_INCLUDE "${_TRY_PATH}")
                    break()
                endif()
            endforeach()
            if(_GCC_OMP_INCLUDE)
                set(_OPENMP_INCLUDE_FLAG "-isystem${_GCC_OMP_INCLUDE}")
            endif()
        else()
            set(_OPENMP_FLAG "-fopenmp")
            set(_OPENMP_INCLUDE_FLAG "")
        endif()
        execute_process(
            COMMAND ${CMAKE_C_COMPILER} ${_OPENMP_FLAG} ${_OPENMP_INCLUDE_FLAG} -x c ${DEV_NULL} -c -o ${DEV_NULL}
            RESULT_VARIABLE TEST_OPENMP_RESULT
            ERROR_QUIET OUTPUT_QUIET
        )
        if(TEST_OPENMP_RESULT EQUAL 0)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_OPENMP_FLAG} ${_OPENMP_INCLUDE_FLAG}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_OPENMP_FLAG}")
            add_compile_definitions(HAS_OPENMP)
            message(STATUS "OpenMP enabled (${_OPENMP_FLAG}${_OPENMP_INCLUDE_FLAG})")
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
    src/ort_inference_context.c
    src/core_types.c
    src/k1_platform.c
    src/k1_imu.c
    src/k1_jpu.c
    src/yolo_postprocess.c
    src/keypoint_validator.c
)

set(DATA_SOURCES
    src/video_processor.c
    src/imu_handler.c
    src/model_store.c
    src/coap_receiver.c
)

set(MODEL_SOURCES
    src/yolov8_pose_estimator.c
    src/yolov5_face_detector.c
    src/arcface_recognizer.c
    src/stgcn_action_recognizer.c
)

set(BUSINESS_SOURCES
    src/inference_pipeline.c
    src/tracking_manager.c
    src/spatial_engine.c
    src/result_manager.c
)

set(PRESENTATION_SOURCES
    src/visualizer.c
    src/ar_renderer.c
    src/video_writer.c
    src/display_output.c
)

set(UTILS_SOURCES
    src/system_controller.c
)

set(MAIN_SOURCE
    src/main.c
)

set(ALL_LIB_SOURCES
    ${CORE_SOURCES}
    ${DATA_SOURCES}
    ${MODEL_SOURCES}
    ${BUSINESS_SOURCES}
    ${PRESENTATION_SOURCES}
    ${UTILS_SOURCES}
)

set(ALL_LIB_CXX_SOURCES "")

add_library(lingqi_tantong_core STATIC ${ALL_LIB_SOURCES} ${ALL_LIB_CXX_SOURCES})

if(ALL_LIB_CXX_SOURCES)
    target_link_libraries(lingqi_tantong_core PRIVATE stdc++)
endif()

add_executable(lingqi_tantong ${MAIN_SOURCE})
target_link_libraries(lingqi_tantong PRIVATE lingqi_tantong_core)

if(UNIX)
    target_link_libraries(lingqi_tantong PRIVATE m pthread)
endif()

if(ALL_LIB_CXX_SOURCES)
    target_link_libraries(lingqi_tantong PRIVATE stdc++)
endif()

if(BUILDING_FOR_RISCV64)
    target_link_libraries(lingqi_tantong PRIVATE atomic)
endif()

if(USE_ONNX_RUNTIME)
    # ── Auto-detect ONNX Runtime installation ──
    # Priority: 1) user-specified ONNX_RUNTIME_DIR  2) SpacemiT ORT in /opt
    #            3) system-installed in /usr         4) /usr/local
    if(NOT ONNX_RUNTIME_DIR)
        # Probe /opt for SpacemiT ONNX Runtime (spacemit-ort.riscv64.*)
        file(GLOB ORT_CANDIDATES
            "/opt/spacemit-ort.riscv64.*"
            "/opt/spacemit-ort*")
        foreach(ORT_CAND IN LISTS ORT_CANDIDATES)
            if(IS_DIRECTORY "${ORT_CAND}" AND EXISTS "${ORT_CAND}/include/onnxruntime_c_api.h")
                set(ONNX_RUNTIME_DIR "${ORT_CAND}")
                message(STATUS "Auto-detected SpacemiT ORT at: ${ONNX_RUNTIME_DIR}")
                break()
            endif()
        endforeach()
    endif()

    if(NOT ONNX_RUNTIME_DIR)
        # Fall back to system-installed ONNX Runtime
        if(EXISTS "/usr/include/onnxruntime/onnxruntime_c_api.h")
            set(ONNX_RUNTIME_SYSTEM_INSTALL TRUE)
            message(STATUS "ONNX Runtime: using system install (/usr)")
        elseif(EXISTS "/usr/local/include/onnxruntime_c_api.h")
            set(ONNX_RUNTIME_SYSTEM_INSTALL TRUE)
            message(STATUS "ONNX Runtime: using system install (/usr/local)")
        else()
            # Final fallback: check if header is directly in /usr/include
            if(EXISTS "/usr/include/onnxruntime_c_api.h")
                set(ONNX_RUNTIME_SYSTEM_INSTALL TRUE)
                message(STATUS "ONNX Runtime: using system install (/usr/include)")
            endif()
        endif()
    endif()

    # Header check
    if(NOT ONNX_RUNTIME_DIR AND NOT ONNX_RUNTIME_SYSTEM_INSTALL)
        message(FATAL_ERROR
            "USE_ONNX_RUNTIME=ON but ONNX Runtime not found.\n"
            "  Auto-probed /opt/spacemit-ort*/ /usr/include /usr/local/include — all empty.\n"
            "  Install SpacemiT ONNX Runtime 2.0.2:\n"
            "    https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.2.tar.gz\n"
            "  Or set manually: cmake -DONNX_RUNTIME_DIR=/path/to/ort")
    endif()

    if(ONNX_RUNTIME_DIR)
        target_include_directories(lingqi_tantong_core PRIVATE "${ONNX_RUNTIME_DIR}/include")
    endif()

    # Library detection
    set(ORT_LIB_FOUND FALSE)
    if(ONNX_RUNTIME_DIR AND EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
        target_link_libraries(lingqi_tantong_core PUBLIC "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
        set(ORT_LIB_FOUND TRUE)
        message(STATUS "ONNX Runtime lib: ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
    elseif(ONNX_RUNTIME_DIR AND EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.a")
        target_link_libraries(lingqi_tantong_core PUBLIC "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.a")
        set(ORT_LIB_FOUND TRUE)
        message(STATUS "ONNX Runtime lib: ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.a (static)")
    else()
        find_library(ORT_LIB NAMES onnxruntime
            PATHS /usr/lib /usr/lib/riscv64-linux-gnu /usr/local/lib
            NO_DEFAULT_PATH)
        if(ORT_LIB)
            target_link_libraries(lingqi_tantong_core PUBLIC ${ORT_LIB})
            set(ORT_LIB_FOUND TRUE)
            message(STATUS "ONNX Runtime lib: ${ORT_LIB}")
        endif()
    endif()

    if(NOT ORT_LIB_FOUND)
        message(FATAL_ERROR
            "USE_ONNX_RUNTIME=ON but libonnxruntime.so not found.\n"
            "  Searched: ${ONNX_RUNTIME_DIR}/lib/, /usr/lib/, /usr/lib/riscv64-linux-gnu/\n"
            "  Install spacemit-onnxruntime: sudo apt install spacemit-onnxruntime\n"
            "  Or set: -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.x.x.x")
    endif()

    # SpacemiT MLAS (RISC-V optimized math)
    if(ONNX_RUNTIME_DIR AND EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
        target_link_libraries(lingqi_tantong_core PUBLIC "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
        message(STATUS "RISC-V MLAS: ${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so")
    elseif(EXISTS "/usr/lib/libonnxruntime_mlas.so")
        target_link_libraries(lingqi_tantong_core PUBLIC onnxruntime_mlas)
        message(STATUS "RISC-V MLAS: system lib")
    else()
        message(STATUS "RISC-V MLAS: not found (OK — using generic ORT math)")
    endif()

    # SpacemiT EP shared lib (libspacemit_ep.so)
    set(SPACEMIT_EP_FOUND FALSE)
    if(ONNX_RUNTIME_DIR)
        find_library(SPACEMIT_EP_LIB
            NAMES spacemit_ep
            PATHS "${ONNX_RUNTIME_DIR}/lib"
            NO_DEFAULT_PATH)
        if(SPACEMIT_EP_LIB)
            target_link_libraries(lingqi_tantong_core PUBLIC "${SPACEMIT_EP_LIB}")
            set(SPACEMIT_EP_FOUND TRUE)
            message(STATUS "SpacemiT EP lib: ${SPACEMIT_EP_LIB}")
        else()
            file(GLOB SPACEMIT_EP_LIBS_FALLBACK
                "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_providers_spacemit*.so"
                "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_spacemit*.so")
            foreach(SPM_LIB IN LISTS SPACEMIT_EP_LIBS_FALLBACK)
                target_link_libraries(lingqi_tantong_core PUBLIC "${SPM_LIB}")
                set(SPACEMIT_EP_FOUND TRUE)
                message(STATUS "SpacemiT EP lib (fallback): ${SPM_LIB}")
            endforeach()
        endif()
    endif()
    if(NOT SPACEMIT_EP_FOUND)
        find_library(SPACEMIT_EP_LIB
            NAMES spacemit_ep
            PATHS /usr/lib /usr/lib/riscv64-linux-gnu /usr/local/lib
            NO_DEFAULT_PATH)
        if(SPACEMIT_EP_LIB)
            target_link_libraries(lingqi_tantong_core PUBLIC "${SPACEMIT_EP_LIB}")
            set(SPACEMIT_EP_FOUND TRUE)
            message(STATUS "SpacemiT EP lib (system): ${SPACEMIT_EP_LIB}")
        endif()
    endif()

    if(NOT SPACEMIT_EP_FOUND)
        if(BUILDING_FOR_RISCV64)
            message(WARNING
                "libspacemit_ep.so not found — SpacemiT EP unavailable.\n"
                "  K1 AI acceleration (RVV 1.0 + IME) will NOT work.\n"
                "  Install: sudo apt install spacemit-onnxruntime\n"
                "  Or set: -DONNX_RUNTIME_DIR=/path/to/spacemit-ort.riscv64.x.x.x")
        else()
            message(STATUS "SpacemiT EP lib: not found (host build — OK)")
        endif()
    endif()

    add_compile_definitions(HAS_ONNX_RUNTIME)
    if(ONNX_RUNTIME_DIR)
        message(STATUS "ONNX Runtime: ${ONNX_RUNTIME_DIR}")
    else()
        message(STATUS "ONNX Runtime: system install")
    endif()

    # SpacemiT EP header (spacemit_ort_env.h) — C++ bridge
    find_file(SPACEMIT_ORT_ENV_H
        NAMES spacemit_ort_env.h
        PATHS "${ONNX_RUNTIME_DIR}/include"
              "${ONNX_RUNTIME_DIR}/include/onnxruntime"
              "/usr/local/include/onnxruntime"
              "/usr/local/include"
              "/usr/include/onnxruntime"
              "/usr/include"
        NO_DEFAULT_PATH
    )

    if(SPACEMIT_ORT_ENV_H)
        enable_language(CXX)
        target_sources(lingqi_tantong_core PRIVATE src/spacemit_ort_bridge.cpp)
        target_compile_definitions(lingqi_tantong_core PRIVATE HAS_SPACEMIT_EP)
        target_link_libraries(lingqi_tantong_core PRIVATE stdc++)
        list(APPEND ALL_LIB_CXX_SOURCES src/spacemit_ort_bridge.cpp)
        message(STATUS "SpacemiT EP: YES (${SPACEMIT_ORT_ENV_H})")
    else()
        if(BUILDING_FOR_RISCV64)
            message(WARNING
                "spacemit_ort_env.h not found — SpacemiT EP C++ bridge disabled.\n"
                "  K1 AI acceleration (RVV 1.0 + IME) will NOT work.\n"
                "  Searched: ${ONNX_RUNTIME_DIR}/include/, /usr/local/include/onnxruntime/")
        else()
            message(STATUS "SpacemiT EP header: not found (host build — OK)")
        endif()
    endif()
endif()

find_library(TURBOJPEG_LIB NAMES turbojpeg)
find_path(TURBOJPEG_INC NAMES turbojpeg.h
    PATHS /usr/include /usr/local/include /usr/include/riscv64-linux-gnu)
if(TURBOJPEG_LIB AND TURBOJPEG_INC)
    target_include_directories(lingqi_tantong_core PRIVATE ${TURBOJPEG_INC})
    target_link_libraries(lingqi_tantong PRIVATE ${TURBOJPEG_LIB})
    add_compile_definitions(HAS_LIBJPEG_TURBO)
    message(STATUS "libjpeg-turbo found: ${TURBOJPEG_LIB}")
else()
    message(FATAL_ERROR
        "libjpeg-turbo is REQUIRED for JPEG decode (Arrow UART frames).\n"
        "  Install on Bianbu/Debian: sudo apt install libturbojpeg0-dev\n"
        "  Install on RHEL/Fedora:   sudo dnf install turbojpeg-devel")
endif()

# check_include_file fails on some RISC-V sysroots; probe with shell echo.
execute_process(
    COMMAND sh -c "echo '#include <linux/videodev2.h>' | ${CMAKE_C_COMPILER} -E -x c - > /dev/null 2>&1"
    RESULT_VARIABLE TEST_V4L2_RESULT
)
if(TEST_V4L2_RESULT EQUAL 0)
    set(HAVE_V4L2 TRUE)
    add_compile_definitions(HAS_V4L2)
    message(STATUS "V4L2 enabled (Linux video capture)")
else()
    set(HAVE_V4L2 FALSE)
    if(BUILDING_FOR_RISCV64)
        message(WARNING
            "V4L2 header <linux/videodev2.h> NOT found - local camera capture disabled.\n"
            "  Arrow UART JPEG path still works, but /dev/video0 will not be used.\n"
            "  To enable: sudo apt install linux-libc-dev\n"
            "  Then re-run cmake.")
    endif()
endif()

if(BUILDING_FOR_RISCV64 AND DETECTED_RISCV_ARCH STREQUAL "rv64gcv1p0")
    # ── VPU / MPP auto-detection ──
    if(ENABLE_K1_VPU)
        if(K1_MPP_DIR)
            # User-specified path
            target_include_directories(lingqi_tantong_core PRIVATE "${K1_MPP_DIR}/include")
            target_link_directories(lingqi_tantong PRIVATE "${K1_MPP_DIR}/lib")
            target_link_libraries(lingqi_tantong PRIVATE mpp)
            target_compile_definitions(lingqi_tantong_core PRIVATE HAS_K1_VPU)
            message(STATUS "K1 VPU/MPP: ${K1_MPP_DIR}")
        else()
            # Auto-detect system-installed libmpp.so
            find_library(K1_MPP_LIB NAMES mpp spacemit_mpp
                PATHS /usr/lib /usr/lib/riscv64-linux-gnu NO_DEFAULT_PATH)
            if(K1_MPP_LIB)
                target_link_libraries(lingqi_tantong PRIVATE "${K1_MPP_LIB}" dl)
                target_compile_definitions(lingqi_tantong_core PRIVATE HAS_K1_VPU)
                message(STATUS "K1 VPU/MPP: auto-detected (${K1_MPP_LIB})")
            else()
                message(FATAL_ERROR
                    "ENABLE_K1_VPU=ON but libmpp.so not found.\n"
                    "  Auto-probed: /usr/lib/, /usr/lib/riscv64-linux-gnu/ — not found.\n"
                    "  Install: sudo apt install k1x-mpp\n"
                    "  Or set: -DK1_MPP_DIR=/path/to/k1x-mpp\n"
                    "  Or disable VPU: -DENABLE_K1_VPU=OFF")
            endif()
        endif()
    endif()

    # ── JPU auto-detection ──
    if(ENABLE_K1_JPU)
        if(K1_JPU_DIR)
            # NOTE: Do NOT link libjpu.so directly — it has an unresolved
            # "LogMsg" symbol.  k1_jpu.c loads it at runtime via dlopen/dlsym.
            target_include_directories(lingqi_tantong_core PRIVATE "${K1_JPU_DIR}/include")
            target_compile_definitions(lingqi_tantong_core PRIVATE HAS_K1_JPU_LIB)
            message(STATUS "K1 JPU: ${K1_JPU_DIR}")
        else()
            find_library(K1_JPU_LIB NAMES jpu PATHS /usr/lib /usr/lib/riscv64-linux-gnu NO_DEFAULT_PATH)
            if(K1_JPU_LIB)
                # Do NOT direct-link libjpu.so — libspacemit_mpp.so also exports
                # AsrJpuDecOpen, and the global linker resolves it to the wrong
                # implementation (→ segfault).  k1_jpu.c dlopens libjpu.so with
                # RTLD_LOCAL and binds the correct symbols via dlsym.
                # LogMsg is satisfied by our stub + --export-dynamic.
                target_compile_definitions(lingqi_tantong_core PRIVATE HAS_K1_JPU_LIB)
                message(STATUS "K1 JPU: auto-detected (${K1_JPU_LIB})")
            else()
                message(FATAL_ERROR
                    "ENABLE_K1_JPU=ON but libjpu.so not found.\n"
                    "  Auto-probed: /usr/lib/, /usr/lib/riscv64-linux-gnu/ — not found.\n"
                    "  Install: sudo apt install k1x-jpu\n"
                    "  Or set: -DK1_JPU_DIR=/path/to/k1x-jpu\n"
                    "  Or disable JPU: -DENABLE_K1_JPU=OFF")
            endif()
        endif()
    endif()
endif()

# ── SDL2 desktop window display ──────────────────────────────────────────
set(HAS_SDL2_DISPLAY FALSE)
if(ENABLE_SDL2_DISPLAY)
    find_package(SDL2 QUIET)
    if(SDL2_FOUND)
        set(HAS_SDL2_DISPLAY TRUE)
        target_include_directories(lingqi_tantong_core PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(lingqi_tantong PRIVATE ${SDL2_LIBRARIES})
        target_compile_definitions(lingqi_tantong_core PRIVATE HAS_SDL2)
        message(STATUS "SDL2 display:       YES (desktop window)")
    else()
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(SDL2_PC sdl2)
            if(SDL2_PC_FOUND)
                set(HAS_SDL2_DISPLAY TRUE)
                target_include_directories(lingqi_tantong_core PRIVATE ${SDL2_PC_INCLUDE_DIRS})
                target_link_libraries(lingqi_tantong PRIVATE ${SDL2_PC_LIBRARIES})
                target_compile_definitions(lingqi_tantong_core PRIVATE HAS_SDL2)
                message(STATUS "SDL2 display:       YES (pkg-config)")
            endif()
        endif()
    endif()
    if(NOT HAS_SDL2_DISPLAY)
        message(STATUS "SDL2 display:       NO — fbdev fallback (sudo apt install libsdl2-dev for window)")
    endif()
endif()
if(HAS_SDL2_DISPLAY)
    target_sources(lingqi_tantong_core PRIVATE src/display_sdl2.c)
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
if(SPACEMIT_ORT_ENV_H)
    message(STATUS "SpacemiT EP:       YES (RVV 1.0 + IME hardware acceleration)")
else()
    message(STATUS "SpacemiT EP:       NO (CPU-only)")
endif()
if(TURBOJPEG_LIB)
    message(STATUS "libjpeg-turbo:     ${TURBOJPEG_LIB}")
endif()
if(HAVE_V4L2)
    message(STATUS "V4L2:              YES (Linux video capture)")
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "riscv")
    if(EXISTS "${ONNX_RUNTIME_DIR}/lib/libonnxruntime_mlas.so" OR EXISTS "/usr/lib/libonnxruntime_mlas.so")
        message(STATUS "RISC-V MLAS:       YES")
    else()
        message(STATUS "RISC-V MLAS:       NO (lib not found)")
    endif()
endif()
message(STATUS "OpenMP:            ${USE_OPENMP}")
message(STATUS "RVV Optimize:      ${ENABLE_RVV_OPT}")
message(STATUS "K1 Pipeline:       ${ENABLE_K1_PIPELINE}")
message(STATUS "K1 TCM:            ${ENABLE_K1_TCM}")
message(STATUS "K1 VPU/MPP:        ${ENABLE_K1_VPU}")
message(STATUS "K1 JPU:            ${ENABLE_K1_JPU}")
message(STATUS "============================================")