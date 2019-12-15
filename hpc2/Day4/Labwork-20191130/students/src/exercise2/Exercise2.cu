#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise2/Exercise2.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/equal.h>


namespace {
        
}

// ==========================================================================================
void Exercise2::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " -i=<image.ppm>"<< std::endl
        << "\twhere <image_input.ppm> is the input image." << std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise2::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise2::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise2& Exercise2::parseCommandLine(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "i") ) {
        char *value;
        getCmdLineArgumentString(argc, argv, "i", &value);
        std::cout << "Input file is " << value << std::endl;
        inputFileName = std::string(value);
    }
    else
        usageAndExit(argv[0], -1); 
    return *this;
}

void Exercise2::loadImage() 
{
    input = new PPMBitmap(inputFileName.c_str());
    const unsigned size = input->getWidth()*input->getHeight();
    uchar*ptr = input->getPtr();
    thrust::host_vector<uchar3> h_RGB(reinterpret_cast<uchar3*>(ptr), reinterpret_cast<uchar3*>(ptr+size*3));
    d_RGB_in = h_RGB;
    d_H.resize(size);
    d_S.resize(size);
    d_V.resize(size);
}

void Exercise2::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Convert the image, build the histogram" << std::endl;
    // build a host vector containing the reference
    loadImage();
    ChronoGPU chr;
    StudentWork2& worker = *reinterpret_cast<StudentWork2*>(student);
    worker.rgb2hsv( d_RGB_in, d_H, d_S, d_V );
    if( verbose )
        std::cout << "\tBuild histogram with Thrust" << std::endl;
    d_histo = worker.histogram( d_V ); // awake the GPU ...
    chr.start();
    d_histo = worker.histogram( d_V );
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime() << " ms" << std::endl;
    if( verbose )
        std::cout << "\tBuild histogram with CUDA and BLOCKING" << std::endl;
    chr.start();
    d_histo_fast = worker.histogram_fast( d_V );
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime() << " ms" << std::endl;
}


bool Exercise2::check_histo(const thrust::device_vector<unsigned>&histo) {
    if ( histo.size() != 256 )
        return false;
    const unsigned sum = thrust::reduce(histo.begin(), histo.end());
    return ( sum == input->getWidth() * input->getHeight() );
}

bool Exercise2::check() {
    return check_histo(d_histo) && check_histo(d_histo_fast) 
        && thrust::equal(d_histo.begin(), d_histo.end(), d_histo_fast.begin());
}
