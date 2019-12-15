#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise4/Exercise4.h>


int main(int argc, const char**argv) 
{
    // find and start a device ...
    std::cout<<"Find the device ..." << std::endl;
    int bestDevice = findCudaDevice(argc, argv);
    checkCudaErrors( cudaSetDevice( bestDevice ) );

    // launch the exercise 4
    Exercise4("Exercise 4").parseCommandLine(argc, argv).evaluate(true);

    // bye
    return 0;
}
