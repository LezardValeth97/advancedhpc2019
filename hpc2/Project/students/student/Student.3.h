#pragma once

#include <utils/ppm.hpp>
#include <thrust/device_vector.h>

float student3(
    const unsigned width, const unsigned height,
    thrust::device_vector<uchar>& d_V, 
    thrust::device_vector<uchar>& d_Vfiltered, 
    const unsigned d
);