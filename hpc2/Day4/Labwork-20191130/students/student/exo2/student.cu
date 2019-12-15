#include "student.h"
#include <thrust/for_each.h>

namespace {
	
}

bool StudentWork2::isImplemented() const {
	return false;
}


thrust::device_vector<unsigned> StudentWork2::histogram(
	const thrust::device_vector<float>&d_V
)
{
	// need to SCAN the values, and to add them into a histogram set to 0.
	// Need to use atomic operations to do that (else it is a loop on histogram value + transform_reduce)
	// NB: you can obtain the data pointer from a device_vector<T> vec using:
	//   T *ptr = vec.begin().base().get();
	thrust::device_vector<unsigned> histo(256);
	return histo;
}

thrust::device_vector<unsigned> StudentWork2::histogram_fast(
	const thrust::device_vector<float>&d_V
)
{
	// need to SCAN the values, and to add them into a histogram set to 0.
	// Need to use atomic operations to do that (else it is a loop on histogram value + transform_reduce)
	// NB: you can obtain the data pointer from a device_vector<T> vec using:
	//   T *ptr = vec.begin().base().get();
	thrust::device_vector<unsigned> histo(256);
	return histo;
}
