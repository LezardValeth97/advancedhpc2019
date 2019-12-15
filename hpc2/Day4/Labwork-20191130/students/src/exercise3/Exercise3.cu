#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise3/Exercise3.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
}

// ==========================================================================================
void Exercise3::usage( const char*const prg ) {
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
void Exercise3::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise3::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise3& Exercise3::parseCommandLine(const int argc, const char**argv) 
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

void Exercise3::loadImage() 
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

void Exercise3::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Convert the image, build histogram, calculate the CDF" << std::endl;
    prepare(verbose);
    run_thrust(verbose);
    run_cuda(verbose);
}

void Exercise3::prepare(const bool verbose) 
{
    // build a host vector containing the reference
    loadImage();
    StudentWork3& worker = *reinterpret_cast<StudentWork3*>(student);
    worker.rgb2hsv( d_RGB_in, d_H, d_S, d_V );
    d_histo = worker.histogram_fast( d_V );
}

void Exercise3::run_thrust(const bool verbose)
{
    StudentWork3& worker = *reinterpret_cast<StudentWork3*>(student);
    ChronoGPU chr;
    if( verbose ) 
        std::cout << "\tBuild CDF with Thrust" << std::endl;
    for(int i=0;i<5;++i) d_cdf = worker.evalCdf_thrust(d_histo); // awake the GPU
    chr.start();
    for(int i=0;i<10;++i) d_cdf = worker.evalCdf_thrust(d_histo);
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime()/10.f << " ms" << std::endl;
}

void Exercise3::run_cuda(const bool verbose)
{
    ChronoGPU chr;
    StudentWork3& worker = *reinterpret_cast<StudentWork3*>(student);
    if( verbose ) 
        std::cout << "\tBuild CDF with CUDA" << std::endl;
    chr.start();
    for(int i=0;i<10;++i) d_cdf_fast = worker.evalCdf_cuda(d_histo);
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime()/10.f << " ms" << std::endl;
}
    
bool Exercise3::check_cdf(const thrust::host_vector<unsigned>& cdf) 
{
    if ( cdf.size() != 256 )
        return false;
    thrust::host_vector<unsigned> h_histo(d_histo);
    unsigned sum = 0;
    for(int i=0; i<256; ++i) {
        sum += h_histo[i];
        if( cdf[i] != sum ) return false;
    }
    return true;
}

bool Exercise3::check() {
    return check_cdf(d_cdf) && check_cdf(d_cdf_fast);
}

