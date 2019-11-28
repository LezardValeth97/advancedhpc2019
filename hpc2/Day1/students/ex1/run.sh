/usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++-6 -Iinclude -std=c++11 Exercise.cu include/chronoGPU.cu main.cpp
./a.out
