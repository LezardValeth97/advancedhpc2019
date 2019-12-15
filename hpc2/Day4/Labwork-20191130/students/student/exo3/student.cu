#include "student.h"

namespace {
	
	#define SIZE_HISTO 256
}

bool StudentWork3::isImplemented() const {
	return false;
}


thrust::device_vector<unsigned> 
StudentWork3::evalCdf_thrust(const thrust::device_vector<unsigned>& histo) 
{
	// v1 : thrust, just an inclusive scan
	thrust::device_vector<unsigned> cdf(256);
	return cdf;
}

thrust::device_vector<unsigned> 
StudentWork3::evalCdf_cuda(const thrust::device_vector<unsigned>& histo) 
{
	// v2 : CUDA
	thrust::device_vector<unsigned> cdf(256);
	return cdf;
}


