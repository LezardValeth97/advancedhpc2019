#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>

class StudentWork2 : public StudentWork {
public:

	bool isImplemented() const ;

	StudentWork2() = default; 
	StudentWork2(const StudentWork2&) = default;
	~StudentWork2() = default;
	StudentWork2& operator=(const StudentWork2&) = default;

    thrust::device_vector<unsigned> radixSortBase2(const thrust::device_vector<unsigned>&);

};
