#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <helper_math.h>


class StudentWork1 : public StudentWork
{
public:

	void rgb2hsv(
		const thrust::device_vector<uchar3>&rgb,
		thrust::device_vector<float>&H,
		thrust::device_vector<float>&S,
		thrust::device_vector<float>&V
	);
	void hsv2rgb(
		const thrust::device_vector<float>&H,
		const thrust::device_vector<float>&S,
		const thrust::device_vector<float>&V,
		thrust::device_vector<uchar3>&RGB
	);

	bool isImplemented() const ;

	StudentWork1() = default; 
	StudentWork1(const StudentWork1&) = default;
	~StudentWork1() = default;
	StudentWork1& operator=(const StudentWork1&) = default;
};