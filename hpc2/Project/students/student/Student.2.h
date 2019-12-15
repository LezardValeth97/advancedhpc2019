#pragma once
#include <utils/ppm.hpp>
#include <thrust/device_vector.h>


float student2(
    thrust::device_vector<uchar3>& d_input,
    thrust::device_vector<uchar>& d_V, 
    thrust::device_vector<uchar3>& d_output
);