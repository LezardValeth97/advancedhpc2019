/usr/local/cuda/bin/nvcc -ccbin=/usr/bin/g++-6 -Iinclude -std=c++11 Student.cu main.cu D_Matrix.cu H_Matrix.cu utils/chronoGPU.cu utils/chronoCPU.cpp 
