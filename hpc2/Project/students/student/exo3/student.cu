#include "student.h"

namespace {
	
}

bool StudentWork3::isImplemented() const {
	return false;
}

StudentWork3& StudentWork3::parseCommandLine(const int argc, const char**argv) {
	// add here the scan of your parameters ...
	// but return *this !
	return *this;
}


void StudentWork3::justDoIt(
	const thrust::device_vector<uchar3>& d_RGB_in, 
	thrust::device_vector<uchar3>& d_RGB_out, 
	const unsigned width,
	const unsigned height 
) {
}


