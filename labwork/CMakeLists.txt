cmake_minimum_required(VERSION 3.0)
project(labwork)

include_directories(".")
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message("Correct CUDA location not found")
endif()

list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets)

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})

set(SOURCE_FILES src/jpegloader.cu include/jpegloader.h src/timer.cpp include/timer.h src/labwork.cu include/labwork.h)
cuda_add_executable(labwork ${SOURCE_FILES})
target_link_libraries(labwork ${JPEG_LIBRARIES})
