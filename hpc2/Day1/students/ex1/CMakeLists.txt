cmake_minimum_required(VERSION 3.0)
set(EXECUTABLE_NAME Exercise1)
set(PROJECT_NAME Exercise1)

project(${EXECUTABLE_NAME} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_30")
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message("CUDA location not found")
endif()

if (WIN32)
    include_directories(include "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_52)
endif(WIN32)
if (LINUX)
    include_directories("." include "/usr/local/cuda/samples/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_30)
endif(LINUX)

set(HEADERS Exercise.hpp include/chronoGPU.hpp  )
set(SOURCES main.cpp Exercise.cu include/chronoGPU.cu )

cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})