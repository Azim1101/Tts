cmake_minimum_required(VERSION 3.22.1)
project(dhvaani LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Layout produced by scripts/fetch_mnn.sh:
#   src/main/cpp/mnn/include/MNN/...
#   src/main/cpp/mnn/libs/<ABI>/libMNN.so, libMNN_Express.so, libc++_shared.so
set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/mnn")
set(MNN_LIB_DIR "${MNN_ROOT}/libs/${ANDROID_ABI}")

if(NOT EXISTS "${MNN_ROOT}/include/MNN/expr/Module.hpp")
    message(FATAL_ERROR
        "MNN headers not found at ${MNN_ROOT}/include.\n"
        "Run:  bash scripts/fetch_mnn.sh   from the repository root.")
endif()

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${MNN_LIB_DIR}/libMNN.so")

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB_DIR}/libMNN_Express.so")

add_library(dhvaani SHARED
    dhvaani.cpp
    dhvaani_jni.cpp
)

target_include_directories(dhvaani PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MNN_ROOT}/include
)

target_compile_options(dhvaani PRIVATE
    -O3 -ffast-math -fvisibility=hidden -Wall
)

# NEON dot-product / fp16 paths on arm64 give a large speedup on modern SoCs.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(dhvaani PRIVATE -march=armv8.2-a+fp16+dotprod)
endif()

target_link_libraries(dhvaani
    MNN
    MNN_Express
    android
    log
)
