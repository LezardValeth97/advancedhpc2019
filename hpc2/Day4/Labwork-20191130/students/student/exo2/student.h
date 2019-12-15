#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <exo1/student.h>
#include <helper_math.h>


class StudentWork2 : public StudentWork1
{
public:

	// question about histogram ... V[i] is a float in [0..1]. It should be cast to uchar
	thrust::device_vector<unsigned> histogram(const thrust::device_vector<float>& d_V);

	// question about histogram using CUDA and BLOCKING pattern
	thrust::device_vector<unsigned> histogram_fast(const thrust::device_vector<float>& d_V);

	bool isImplemented() const ;

	StudentWork2() = default; 
	StudentWork2(const StudentWork2&) = default;
	~StudentWork2() = default;
	StudentWork2& operator=(const StudentWork2&) = default;
};