NVCC=/usr/local/cuda/bin/nvcc
SAMPLES=/usr/local/cuda/samples/common/inc

INCLUDES=-I${SAMPLES} -I./src -I./student
CFLAGS=-std=c++11

all: Exercise1 Exercise2

Exercise1: src/exercise1/main.o src/exercise1/Exercise1.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo1/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise1/main.o: src/exercise1/main.cu src/exercise1/Exercise1.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/utils/chronoGPU.o: src/utils/chronoGPU.cu src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/utils/chronoCPU.o: src/utils/chronoCPU.cpp src/utils/chronoCPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise1/Exercise1.o: src/exercise1/Exercise1.cu src/exercise1/Exercise1.h src/utils/Exercise.h src/utils/StudentWork.h student/exo1/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo1/student.o: student/exo1/student.cu student/exo1/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

Exercise2: src/exercise2/main.o src/exercise2/Exercise2.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo2/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise2/main.o: src/exercise2/main.cu src/exercise2/Exercise2.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise2/Exercise2.o: src/exercise2/Exercise2.cu src/exercise2/Exercise2.h src/utils/Exercise.h src/utils/StudentWork.h student/exo2/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo2/student.o: student/exo2/student.cu student/exo2/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@
