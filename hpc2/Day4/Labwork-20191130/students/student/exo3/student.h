#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <exo2/student.h>
#include <helper_math.h>


class StudentWork3 : public StudentWork2
{
public:

	thrust::device_vector<unsigned> evalCdf_thrust(const thrust::device_vector<unsigned>& histo);
	thrust::device_vector<unsigned> evalCdf_cuda(const thrust::device_vector<unsigned>& histo);
	
	bool isImplemented() const ;

	StudentWork3() = default; 
	StudentWork3(const StudentWork3&) = default;
	~StudentWork3() = default;
	StudentWork3& operator=(const StudentWork3&) = default;
};