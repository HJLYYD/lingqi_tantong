cmake_minimum_required(VERSION 3.16)

# ESP-IDF 版本要求: >= 5.1 (ESP32P4 支持从 v5.1 开始)
# 在构建前请确认 IDF 环境:
#   $ idf.py --version
#   ESP-IDF v5.1.x 或更高

set(EXTRA_COMPONENT_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/components
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(lingqi_tantong_p4)