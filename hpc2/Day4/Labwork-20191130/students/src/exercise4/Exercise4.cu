#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise4/Exercise4.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
}

// ==========================================================================================
void Exercise4::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " -i=<image.ppm> [-f=<image_output_basename.ppm>]"<< std::endl
        << "\twhere <image_input.ppm> is the input image," << std::endl
        << "\t<image_output_basename.ppm> is the basename of the output images."<<std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise4::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise4::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise4& Exercise4::parseCommandLine(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "i") ) {
        char *value;
        getCmdLineArgumentString(argc, argv, "i", &value);
        std::cout << "Input file is " << value << std::endl;
        inputFileName = std::string(value);
    }
    else
        usageAndExit(argv[0], -1); 
    if( checkCmdLineFlag(argc, argv, "i") ) {
        char*value;
        getCmdLineArgumentString(argc, argv, "i", &value);
        std::cout << "Output file is " << value << std::endl;
        outputFileName_rgb = std::string(value);
    }
    else {
        outputFileName_rgb = inputFileName;
    }
    buildOutputFileName();
    return *this;
}

void Exercise4::buildOutputFileName() 
{
    // hsv -> rgb
    outputFileName_rgb.erase( outputFileName_rgb.size() - 4, 4 ).append("_equalized.ppm");
}

void Exercise4::loadImage() 
{
    input = new PPMBitmap(inputFileName.c_str());
    const unsigned size = input->getWidth()*input->getHeight();
    uchar*ptr = input->getPtr();
    thrust::host_vector<uchar3> h_RGB(reinterpret_cast<uchar3*>(ptr), reinterpret_cast<uchar3*>(ptr+size*3));
    d_RGB_in = h_RGB;
    d_H.resize(size);
    d_S.resize(size);
    d_V.resize(size);
    d_RGB_out.resize(size);
}

void Exercise4::saveImage(
    const char*filename, 
    const thrust::host_vector<uchar3>&h_image, 
    const unsigned width, 
    const unsigned height
) {
    PPMBitmap output(input->getWidth(), input->getHeight());
    thrust::copy(h_image.begin(), h_image.end(), reinterpret_cast<uchar3*>(output.getPtr()));
    output.saveTo(filename);
    std::cout << "Image saved to " << filename << std::endl;
}

void Exercise4::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Convert the image, build histogram, calculate the CDF" << std::endl;
    prepare(verbose);
    run_thrust(verbose);
    run_cuda(verbose);
    run_final(verbose);
}

void Exercise4::prepare(const bool verbose) 
{
    // build a host vector containing the reference
    loadImage();
    StudentWork4& worker = *reinterpret_cast<StudentWork4*>(student);
    worker.rgb2hsv( d_RGB_in, d_H, d_S, d_V );
    d_histo = worker.histogram_fast( d_V );
    d_cdf = worker.evalCdf_cuda(d_histo);
}

void Exercise4::run_thrust(const bool verbose)
{
    StudentWork4& worker = *reinterpret_cast<StudentWork4*>(student);
    ChronoGPU chr;
    if( verbose ) 
        std::cout << "\tBuild CDF with Thrust" << std::endl;
    d_V_equalized = worker.equalize_thrust(d_cdf, d_V); // awake the GPU
    chr.start();
    for(int i=0;i<10;++i) d_V_equalized = worker.equalize_thrust(d_cdf, d_V);
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime()/10.f << " ms" << std::endl;
}

void Exercise4::run_cuda(const bool verbose)
{
    ChronoGPU chr;
    StudentWork4& worker = *reinterpret_cast<StudentWork4*>(student);
    if( verbose ) 
        std::cout << "\tBuild CDF with CUDA" << std::endl;
    chr.start();
    for(int i=0;i<10;++i) d_V_equalized_fast = worker.equalize_cuda(d_cdf, d_V);
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime()/10.f << " ms" << std::endl;
}

void Exercise4::run_final(const bool verbose) 
{
    StudentWork4& worker = *reinterpret_cast<StudentWork4*>(student);
    worker.hsv2rgb( d_H, d_S, d_V_equalized, d_RGB_out );
    saveImage(outputFileName_rgb.c_str(), d_RGB_out, input->getWidth(), input->getHeight());
}

bool Exercise4::check() 
{
    return thrust::equal(d_V_equalized.begin(), d_V_equalized.end(), d_V_equalized_fast.begin());
}

