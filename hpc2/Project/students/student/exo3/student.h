#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <exo2/student.h>
#include <helper_math.h>


class StudentWork3 : public StudentWork1
{
public:

	// allows you to parse some command line parameters ... add them in private section !
	StudentWork3& parseCommandLine(const int argc, const char**argv) ;

	// Second method ... 
	void justDoIt(
		const thrust::device_vector<uchar3>& d_RGB_in, 
		thrust::device_vector<uchar3>& d_RGB_out, 
		const unsigned width,
		const unsigned height 
	);

	bool isImplemented() const ;

	StudentWork3() = default; 
	StudentWork3(const StudentWork3&) = default;
	~StudentWork3() = default;
	StudentWork3& operator=(const StudentWork3&) = default;
};