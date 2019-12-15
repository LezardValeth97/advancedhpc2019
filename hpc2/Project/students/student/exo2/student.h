#pragma once
#include <thrust/device_vector.h>
#include <utils/StudentWork.h>
#include <exo1/student.h>
#include <helper_math.h>


class StudentWork2 : public StudentWork1
{
public:

	// allows you to parse some command line parameters ... add them in private section !
	StudentWork2& parseCommandLine(const int argc, const char**argv) ;
	
	// First method ... 
	void justDoIt(
		const thrust::device_vector<uchar3>& d_RGB_in, 
		thrust::device_vector<uchar3>& d_RGB_out, 
		const unsigned width,
		const unsigned height 
	);

	bool isImplemented() const ;

	StudentWork2() = default; 
	StudentWork2(const StudentWork2&) = default;
	~StudentWork2() = default;
	StudentWork2& operator=(const StudentWork2&) = default;
};