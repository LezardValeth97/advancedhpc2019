cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 )

# set CUDA
set(CUDA_STANDARD 11)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD  ON)

# add CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

include_directories(src student ${CUDA_SAMPLES_INC})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS} -I./src -I./student")  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I./src -I./student")

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cu src/utils/Exercise.h
    src/exercise1/Exercise1.cu src/exercise1/Exercise1.h
    src/utils/chronoGPU.hpp src/utils/chronoGPU.cu
    src/utils/chronoCPU.hpp src/utils/chronoCPU.cpp
    student/exo1/student.h student/exo1/student.cu
)

add_executable(exo2 
    src/exercise2/main.cu src/utils/Exercise.h
    src/exercise2/Exercise2.cu src/exercise2/Exercise2.h
    src/utils/chronoGPU.hpp src/utils/chronoGPU.cu
    src/utils/chronoCPU.hpp src/utils/chronoCPU.cpp
    student/exo2/student.h student/exo2/student.cu
)