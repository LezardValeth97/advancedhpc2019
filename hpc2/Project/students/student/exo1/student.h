#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <helper_math.h>

typedef unsigned char uchar;

class StudentWork1 : public StudentWork
{
public:

	void rgb2h(
		const thrust::device_vector<uchar3>&rgb,
		thrust::device_vector<uchar>&V
	);

	void median(
		const thrust::device_vector<uchar>& d_V,
		thrust::device_vector<uchar>& d_V_median,
		const unsigned width,
		const unsigned height,
		const unsigned filter_size
	);

	void apply_filter(
		const thrust::device_vector<uchar3>&RGB_old,
		const thrust::device_vector<uchar>&V_new,
		thrust::device_vector<uchar3>&RGB_new
	);

	bool isImplemented() const ;

	StudentWork1() = default; 
	StudentWork1(const StudentWork1&) = default;
	~StudentWork1() = default;
	StudentWork1& operator=(const StudentWork1&) = default;
};