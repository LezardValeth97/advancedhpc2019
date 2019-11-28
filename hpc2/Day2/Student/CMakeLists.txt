cmake_minimum_required(VERSION 3.0)
set(EXECUTABLE_NAME Exercise1)
set(PROJECT_NAME Exercise1)

project(${EXECUTABLE_NAME} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
	
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)
if (NOT CUDA_FOUND)
    message("CUDA location not found")
endif()

set(CUDA_VERBOSE_BUILD ON)

if (WIN32)
    include_directories(utils "D:/CUDA Samples/v10.0/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_52)
endif(WIN32)
if (LINUX)
    include_directories(utils "/usr/local/cuda/samples/common/inc")
    list(APPEND CUDA_NVCC_FLAGS -std=c++11 -O3 -Wno-deprecated-gpu-targets --gpu-architecture=sm_30)
endif(LINUX)

set(HEADERS H_Matrix.cuh D_Matrix.cuh utils/chronoGPU.hpp utils/chronoCPU.hpp utils/common.hpp )
set(SOURCES main.cu H_Matrix.cu D_Matrix.cu Student.cu utils/chronoCPU.cpp utils/chronoGPU.cu )

cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})