#pragma warning( disable : 4244 ) 

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <curand_kernel.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <utils/chronoCPU.hpp>
#include <utils/chronoGPU.hpp>

#include <exercise1/Exercise1.h>


int main(int argc, const char**argv) 
{
    // find and start a device ...
    std::cout<<"Find the device ..." << std::endl;
    int bestDevice = findCudaDevice(argc, argv);
    checkCudaErrors( cudaSetDevice( bestDevice ) );

    // run exercise 1
    Exercise1("Exercise 1")
        .parseCommandLine(argc, argv)
        .evaluate();

    // bye
    return 0;
}
