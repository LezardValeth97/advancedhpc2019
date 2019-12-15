#include "student.h"
#include <thrust/functional.h>
#include <algorithm>

namespace {

	#define SIZE_HISTO 256

	// this device function will work for thrust (no shared)
	// need to write a second one for shared memory (cdf)
	__device__ 
	float equalize(const float V, unsigned const*const cdf, const unsigned size) 
	{
		const unsigned val = unsigned( V*256.f );
		return 255.f / 256.f * (float(cdf[val]) / float(size)) ;
	}

}

bool StudentWork4::isImplemented() const {
	return true;
}


thrust::device_vector<float> 
StudentWork4::equalize_thrust(
	const thrust::device_vector<unsigned>& cdf,
	const thrust::device_vector<float>& d_V
) {
	// v1 : thrust, just an inclusive scan
	thrust::device_vector<float> result(d_V.size());
	return result;
}

thrust::device_vector<float> 
StudentWork4::equalize_cuda(
	const thrust::device_vector<unsigned>& cdf,
	const thrust::device_vector<float>& d_V
) {
	// v2 : CUDA
	thrust::device_vector<float> result(d_V.size());
	return result;
}


