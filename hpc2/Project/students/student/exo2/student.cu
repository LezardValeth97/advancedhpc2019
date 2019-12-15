#include "student.h"
#include <thrust/for_each.h>

namespace {
	// add what you need here ...
}

bool StudentWork2::isImplemented() const {
	return false;
}

StudentWork2& StudentWork2::parseCommandLine(const int argc, const char**argv) {
	// add here the scan of your parameters ...
	// but return *this !
	return *this;
}

void 
StudentWork2::justDoIt(
	const thrust::device_vector<uchar3>& d_RGB_in, 
	thrust::device_vector<uchar3>& d_RGB_out, 
	const unsigned width,
	const unsigned height 
) {
}
