#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <exo3/student.h>
#include <helper_math.h>


class StudentWork4 : public StudentWork3
{
public:

	thrust::device_vector<float> equalize_thrust(
		const thrust::device_vector<unsigned>& cdf,
		const thrust::device_vector<float>& d_V
	);
	thrust::device_vector<float> equalize_cuda(
		const thrust::device_vector<unsigned>& cdf,
		const thrust::device_vector<float>& d_V
	);
	
	bool isImplemented() const ;

	StudentWork4() = default; 
	StudentWork4(const StudentWork4&) = default;
	~StudentWork4() = default;
	StudentWork4& operator=(const StudentWork4&) = default;
};