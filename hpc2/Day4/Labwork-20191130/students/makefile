NVCC=/usr/local/cuda/bin/nvcc
SAMPLES=/usr/local/cuda/samples/common/inc

INCLUDES=-I${SAMPLES} -I./src -I./student
CFLAGS=-std=c++11

all: Exercise1 Exercise2 Exercise3 Exercise4

src/utils/chronoGPU.o: src/utils/chronoGPU.cu src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/utils/chronoCPU.o: src/utils/chronoCPU.cpp src/utils/chronoCPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/utils/ppm.o: src/utils/ppm.cpp src/utils/ppm.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

###
Exercise1: src/exercise1/main.o src/exercise1/Exercise1.o src/utils/ppm.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo1/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise1/main.o: src/exercise1/main.cu src/exercise1/Exercise1.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise1/Exercise1.o: src/exercise1/Exercise1.cu src/exercise1/Exercise1.h src/utils/Exercise.h src/utils/StudentWork.h student/exo1/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo1/student.o: student/exo1/student.cu student/exo1/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

####
Exercise2: src/exercise2/main.o src/exercise2/Exercise2.o src/utils/ppm.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo1/student.o student/exo2/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise2/main.o: src/exercise2/main.cu src/exercise2/Exercise2.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise2/Exercise2.o: src/exercise2/Exercise2.cu src/exercise2/Exercise2.h src/utils/Exercise.h src/utils/StudentWork.h student/exo1/student.h student/exo2/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo2/student.o: student/exo2/student.cu student/exo1/student.h student/exo2/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

####
Exercise3: src/exercise3/main.o src/exercise3/Exercise3.o src/utils/ppm.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo1/student.o student/exo2/student.o student/exo3/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise3/main.o: src/exercise3/main.cu src/exercise3/Exercise3.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise3/Exercise3.o: src/exercise3/Exercise3.cu src/exercise3/Exercise3.h src/utils/Exercise.h src/utils/StudentWork.h student/exo1/student.h student/exo2/student.h student/exo3/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo3/student.o: student/exo3/student.cu student/exo1/student.h student/exo2/student.h student/exo3/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

####
Exercise4: src/exercise4/main.o src/exercise4/Exercise4.o src/utils/ppm.o src/utils/chronoGPU.o src/utils/chronoCPU.o student/exo1/student.o student/exo2/student.o student/exo3/student.o student/exo4/student.o
	${NVCC} ${CFLAGS} ${INCLUDES} $^ -o $@

src/exercise4/main.o: src/exercise4/main.cu src/exercise4/Exercise4.h src/utils/chronoGPU.hpp src/utils/chronoGPU.hpp
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

src/exercise4/Exercise4.o: src/exercise4/Exercise4.cu src/exercise4/Exercise4.h src/utils/Exercise.h src/utils/StudentWork.h student/exo1/student.h student/exo2/student.h student/exo3/student.h student/exo4/student.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@

student/exo4/student.o: student/exo4/student.cu student/exo1/student.h student/exo2/student.h student/exo3/student.h student/exo4/student.h src/utils/StudentWork.h
	${NVCC} ${CFLAGS} ${INCLUDES} -c $< -o $@
