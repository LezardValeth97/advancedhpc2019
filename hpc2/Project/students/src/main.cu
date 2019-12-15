#pragma warning( disable : 4244 ) 
#include <iostream>
#include <algorithm>

#include <utils/chronoCPU.hpp>
#include <utils/chronoGPU.hpp>


#include <helper_cuda.h>
#include <helper_string.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <random>

#include <median.h>
#include <utils/ppm.hpp>

#include <Student.1.h>
#include <Student.2.h>
#include <Student.3.h>

namespace {
		
	// ==========================================================================================
	void usage( const char*prg ) {
		#ifdef WIN32
		const char*last_slash = strrchr(prg, '\\');
		#else
		const char*last_slash = strrchr(prg, '/');
		#endif
		std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
			<< " -i <image.ppm> [-f <image_output_filtered.ppm>] [-r <image_input_reference.ppm>] [-d <size>] [--thrust]"<< std::endl
			<< "\twhere <image_input.ppm> is the input image," << std::endl
			<< "\t<image_output_filtered.ppm> the output filtered image,"<<std::endl
			<< "\t<image_input_reference.ppm> the reference image," << std::endl
			<< "\tand <size> the width of the median filter,"
			<< "\tand where --thrust requests a thrust calculation for exercise 3."
			<< std::endl;
	}

	// ==========================================================================================
	void usageAndExit( const char*prg, const int code ) {
		usage(prg);
		exit( code );
	}
	
	// ==========================================================================================
    void displayHelpIfNeeded(const int argc, const char**argv) 
    {
        if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
            usageAndExit(argv[0], EXIT_SUCCESS);
        }
	}
	
	// ==========================================================================================
    char* getInputFileFromCmdLine(const int argc, const char**argv) 
    {
        if( checkCmdLineFlag(argc, argv, "i") ) {
            char* value;
			getCmdLineArgumentString(argc, argv, "i", &value);
			std::cout << "\tfind command line parameter -i=" << value << std::endl;
			return value;
		}
        return nullptr;
	}
	
	// ==========================================================================================
    char* getOutputFileFromCmdLine(const int argc, const char**argv, const char*const input,const unsigned d, const bool useThrust) 
    {
        if( checkCmdLineFlag(argc, argv, "o") ) {
            char* value;
            getCmdLineArgumentString(argc, argv, "o", &value);
			std::cout << "\tfind command line parameter -o=" << value << std::endl;
			return value;
		}
		std::string name(input);
		name.erase(name.size() - 4, 4).append("_filtered-");
		if( useThrust )
			name.append("thrust-");
		name.append(std::to_string(d)).append(".ppm");
		return strdup( name.c_str() );
	}
	
	// ==========================================================================================
    char* getRefFileFromCmdLine(const int argc, const char**argv) 
    {
        if( checkCmdLineFlag(argc, argv, "r") ) {
            char* value;
            getCmdLineArgumentString(argc, argv, "r", &value);
			std::cout << "\tfind command line parameter -r=" << value << std::endl;
			return value;
		}
        return nullptr;
	}
	
	
	// ==========================================================================================
    unsigned getFilterSizeFromCmdLine(const int argc, const char**argv) 
    {
        const unsigned minD = 4;
        const unsigned maxD = 64;
        unsigned size = 16;
        if( checkCmdLineFlag(argc, argv, "d") ) {
            int value = getCmdLineArgumentInt(argc, argv, "d");
            std::cout << "\tfind command line parameter -d=" << value << std::endl;
            if( value >= minD && value <= maxD )
                size = unsigned(value);
            else
                std::cerr << "\tWarning: parameter must be greater to " << minD << " and lesser than " << maxD << std::endl;
		}
        return size;
	}
	
	// ==========================================================================================
    unsigned getUseThrustFromCmdLine(const int argc, const char**argv) 
    {
        return checkCmdLineFlag(argc, argv, "thrust") || checkCmdLineFlag(argc, argv, "t");
	}
	
	// ==========================================================================================
	void exercise1( thrust::device_vector<uchar3>& d_input, thrust::device_vector<uchar>& d_V) 
	{
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
		std::cout << "Exercise 1 ... RGB -> HSV" << std::endl;
		// call student part
		const float elapsedTime = student1(d_input, d_V);
		std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
	}

	bool operator!=(const uchar3 a, const uchar3 b) {
		return a.x != b.x || a.y != b.y || a.z != b.z;
	}

	struct isNotEqualFunctor : public thrust::unary_function<thrust::tuple<uchar3,uchar3>,uchar> 
	{
		__device__ uchar operator()(const thrust::tuple<uchar3,uchar3> &t) 
		{
			const int3 a = make_int3(thrust::get<0>(t).x, thrust::get<0>(t).y, thrust::get<0>(t).z);
			const int3 b = make_int3(thrust::get<1>(t).x, thrust::get<1>(t).y, thrust::get<1>(t).z);
			return uchar(abs(a.x-b.x) + abs(a.y-b.y) + abs(a.z-b.z));
		}
	};
	// ==========================================================================================
	void exercise2( thrust::device_vector<uchar>& d_V, thrust::device_vector<uchar3>& d_input ) 
	{
		std::cout << std::endl;
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
		std::cout << "Exercise 2 ... HSV->RGB" << std::endl;
		// call student part
		thrust::device_vector<uchar3> d_output(d_V.size());
		float elapsedTime = student2(d_input, d_V, d_output);
		std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
		// compare with the input ...
		auto begin = thrust::make_zip_iterator( thrust::make_tuple( d_input.begin(), d_output.begin() ) );
		const uchar diff = thrust::transform_reduce(
			begin, begin+d_input.size(),
			isNotEqualFunctor(),
			uchar(0),
			thrust::maximum<uchar>()
		);
		if( diff > 0 ) 
			std::cerr<<"Your implementation fails ... try again!"<<std::endl;
		else	
			std::cerr<<"Your implementation seems to work!" << std::endl;
	}

	// ==========================================================================================
	void exercise3( 
		thrust::device_vector<uchar>& d_V, 
		thrust::device_vector<uchar3>& d_input,
		const unsigned d, 
		const bool useThrust,
		const unsigned width, const unsigned height,
		const char*const output_filter, const char*const input_reference 
	) {
		std::cout << std::endl;
		std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
		// call student part
		thrust::device_vector<uchar> d_Vfiltered(d_V.size());
		if( useThrust ){
			std::cout << "Exercise 3 ... H->H' with Thrust" << std::endl;
			const float elapsedTime = median_thrust(width, height, d_V, d_Vfiltered, d);
			std::cout << "The Thrust implementation runs in " << elapsedTime << " ms."<<std::endl;
		}	
		else {
			std::cout << "Exercise 3 ... H->H' with Cuda" << std::endl;
			const float elapsedTime = student3(width, height, d_V, d_Vfiltered, d);
			std::cout << "your implementation runs in " << elapsedTime << " ms."<<std::endl;
		}
		if( output_filter == nullptr && input_reference == nullptr )
			return;
		// transform HSV -> RGB		
		std::cout << "Transform HSV' to RGB"<<std::endl;
		thrust::device_vector<uchar3> d_output(d_V.size());
		student2(d_input, d_Vfiltered, d_output);
		PPMBitmap filtered( width, height );
		uchar3*const o_ptr = reinterpret_cast<uchar3*>(filtered.getPtr());
		thrust::copy_n(d_output.begin(), d_output.size(), o_ptr);
		// save the result
		if( output_filter != nullptr )
		{
			std::cout << "Save result to \""<<output_filter<<"\""<<std::endl;
			filtered.saveTo(output_filter);
		}
		if( input_reference != nullptr ) 
		{
			std::cout <<"Compare your result with file \""<<input_reference<<"\" ..."<<std::endl;
			PPMBitmap reference(input_reference);
			if( reference.getWidth() != width || reference.getHeight() != height ) 
			{
				std::cerr << "Reference file has bad resolution ... cannot compare it!" << std::endl;
				return ;
			}
			uchar3*const h_ref = reinterpret_cast<uchar3*>(reference.getPtr());
			bool errors = false;
			for(unsigned y=0; y<height; ++y)
				for(unsigned x=0; x<width; ++x) 
				{
					const unsigned offset = x + y*width;
					const uchar3 f_pixel = o_ptr[offset];
					const uchar3 r_pixel = h_ref[offset];
					if( f_pixel != r_pixel ) 
						{std::cerr<<"exercise 3: bad result for pixel "<<x<<","<<y<<std::endl; errors = true;}
				}
			if(!errors)
				std::cout << "Good jobs, your algorithm computes the correct result!"<< std::endl;
		}
	}
}

// ==========================================================================================
int main( int ac, const char**av)
{
	// find and start a device ...
	std::cout<<"Find the device ..." << std::endl;
	int bestDevice = findCudaDevice(ac, av);
	checkCudaErrors( cudaSetDevice( bestDevice ) );
	
	// parse the command line
	displayHelpIfNeeded(ac, av);
	const char *const input = getInputFileFromCmdLine(ac, av);
	if( input == nullptr )
		usageAndExit( av[0], EXIT_FAILURE );
	const unsigned d = getFilterSizeFromCmdLine(ac, av);
	const bool useThrust = getUseThrustFromCmdLine(ac, av);
	const char *const output_filter = getOutputFileFromCmdLine(ac, av, input, d, useThrust);
	const char *const input_reference = getRefFileFromCmdLine(ac, av);

	// display some piece of information
	std::cout << "Original  image: " << input << std::endl;
	std::cout << "Filtered  image: " << output_filter << std::endl;
	std::cout << "Reference image: " << (input_reference==nullptr ? "null" : input_reference) << std::endl;
	std::cout << "Filter width:    " << d << std::endl;
	std::cout << "Use thrust:      " << useThrust << std::endl;

	// read the input image, build temporary buffers
	PPMBitmap in( input );
	std::cout << "Image loaded. Width = "<<in.getWidth() <<", height = "<<in.getHeight()<<std::endl;
	thrust::device_vector<uchar> d_V(in.getWidth() * in.getHeight());
	thrust::device_vector<uchar3> d_input(d_V.size());
	thrust::copy_n(reinterpret_cast<uchar3*>(in.getPtr()), d_input.size(), d_input.begin());
	// call first exercise
	exercise1(d_input, d_V);
	// call second exercise
	exercise2(d_V, d_input);
	// call third exercise
	exercise3(d_V, d_input, d, useThrust, in.getWidth(), in.getHeight(), output_filter, input_reference);

	return EXIT_SUCCESS;
}
