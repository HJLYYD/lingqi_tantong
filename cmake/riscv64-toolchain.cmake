set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

set(TOOLCHAIN_PREFIX riscv64-linux-gnu)

set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_AR            ${TOOLCHAIN_PREFIX}-ar)
set(CMAKE_RANLIB        ${TOOLCHAIN_PREFIX}-ranlib)
set(CMAKE_STRIP         ${TOOLCHAIN_PREFIX}-strip)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(MUSE_PI_ARCH "rv64gcv1p0" CACHE STRING "Target RISC-V arch: rv64gcv1p0 (K1 X60) or rv64gcv0p7 (legacy)")

if(MUSE_PI_ARCH STREQUAL "rv64gcv1p0")
    set(RISCV_ARCH_FLAGS "-mcpu=spacemit-x60")
else()
    set(RISCV_ARCH_FLAGS "-march=rv64gcv_zicsr_zifencei -menable-experimental-extensions")
endif()

set(COMMON_FLAGS "${RISCV_ARCH_FLAGS} -mabi=lp64d -O3 -flto -ffast-math -fomit-frame-pointer -fPIC -fno-common -Wall -Wextra")

set(CMAKE_C_FLAGS_INIT   "${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS}")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-latomic -lpthread -lm -ldl -Wl,--gc-sections -Wl,-z,now")

set(BIANBU_SYSROOT "" CACHE PATH "Path to Bianbu OS sysroot for cross-compilation")

if(BIANBU_SYSROOT)
    set(CMAKE_SYSROOT "${BIANBU_SYSROOT}")
    set(ENV{PKG_CONFIG_PATH} "")
    set(ENV{PKG_CONFIG_SYSROOT_DIR} "${BIANBU_SYSROOT}")
    set(ENV{PKG_CONFIG_LIBDIR} "${BIANBU_SYSROOT}/usr/lib/pkgconfig:${BIANBU_SYSROOT}/usr/share/pkgconfig")

    set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} --sysroot=${BIANBU_SYSROOT}")
    set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} --sysroot=${BIANBU_SYSROOT}")
    set(CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT} --sysroot=${BIANBU_SYSROOT}")
endif()

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_CROSSCOMPILING_EMULATOR "" CACHE FILEPATH "QEMU user mode emulator for testing")

message(STATUS "RISC-V 64 cross-compilation toolchain configured for SpacemiT K1 Muse Pi Pro")
message(STATUS "  Architecture: ${MUSE_PI_ARCH}")
message(STATUS "  C Compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "  Sysroot:      ${BIANBU_SYSROOT}")