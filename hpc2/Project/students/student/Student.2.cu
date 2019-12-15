#include <Student.2.h>
#include <thrust/transform.h>
#include <math.h>
#include <utils/ChronoGPU.hpp>

namespace
{    
    // TODO: add your functor here
}


float student2(
    thrust::device_vector<uchar3>& d_input,
    thrust::device_vector<uchar>& d_V, 
    thrust::device_vector<uchar3>& d_output
) {    
    ChronoGPU chr;
    chr.start();
    // TODO: add your implementation here
    chr.stop();
    return chr.elapsedTime();
}