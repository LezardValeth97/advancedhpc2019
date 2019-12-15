#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise3/Exercise3.h>


int main(int argc, const char**argv) 
{
    // find and start a device ...
    std::cout<<"Find the device ..." << std::endl;
    int bestDevice = findCudaDevice(argc, argv);
    checkCudaErrors( cudaSetDevice( bestDevice ) );

    // launch the exercise 3
    Exercise3("Exercise 3").parseCommandLine(argc, argv).evaluate(true);

    // bye
    return 0;
}
