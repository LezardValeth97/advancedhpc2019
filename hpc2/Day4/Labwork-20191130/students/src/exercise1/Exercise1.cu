#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise1/Exercise1.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
        
    __device__
    uchar3 HSV4RGB( const float H, const float S, const float V ) {
        return make_uchar3( H / 360.f * 256.f, S * 256.f, V * 256.f );
    }
        
    struct HSV2PpmFunctor : public thrust::unary_function<thrust::tuple<float,float,float>, uchar3> 
    {
        __device__ uchar3 operator()(thrust::tuple<float,float,float>&t) {
            return HSV4RGB(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
        }
    };
}

// ==========================================================================================
void Exercise1::usage( const char*const prg ) {
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
void Exercise1::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise1::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise1& Exercise1::parseCommandLine(const int argc, const char**argv) 
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
        outputFileName_hsv = std::string(value);
        outputFileName_rgb = std::string(value);
    }
    else {
        outputFileName_hsv = inputFileName;
        outputFileName_rgb = inputFileName;
    }
    buildOutputFileName();
    return *this;
}

void Exercise1::buildOutputFileName() 
{
    // rgb -> hsv
    outputFileName_hsv.erase( outputFileName_hsv.size() - 4, 4 ).append("_hsv.ppm");
    // hsv -> rgb
    outputFileName_rgb.erase( outputFileName_rgb.size() - 4, 4 ).append("_rgb.ppm");
}

void Exercise1::loadImage() 
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

void Exercise1::saveImage(
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

void Exercise1::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Convert the image" << std::endl;
    // build a host vector containing the reference
    loadImage();
    ChronoGPU chr;
    chr.start();
    reinterpret_cast<StudentWork1*>(student)->rgb2hsv( d_RGB_in, d_H, d_S, d_V );
    reinterpret_cast<StudentWork1*>(student)->hsv2rgb( d_H, d_S, d_V, d_RGB_out );
    chr.stop();
    if( verbose )
        std::cout << "\tStudent's Work Done in " << chr.elapsedTime() << " ms" << std::endl;
}


bool Exercise1::check() {
    buildHSV();
    saveImage(outputFileName_hsv.c_str(), d_HSV_out, input->getWidth(), input->getHeight());
    saveImage(outputFileName_rgb.c_str(), d_RGB_out, input->getWidth(), input->getHeight());
    return true;
}

void Exercise1::buildHSV() 
{
    d_HSV_out.resize(d_RGB_in.size());
    thrust::copy_n(
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(d_H.begin(), d_S.begin(), d_V.begin())
            ),
            HSV2PpmFunctor()
        ),
        d_RGB_in.size(),
        d_HSV_out.begin()
    );
}