#pragma once

#include <utils/Exercise.h>
#include <exo1/student.h>
#include <thrust/device_vector.h>
#include <utils/ppm.h>

class Exercise1 : public Exercise 
{
public:
    Exercise1(const std::string& name ) 
        : Exercise(name, new StudentWork1())
    {}

    Exercise1() = delete;
    Exercise1(const Exercise1&) = default;
    ~Exercise1() = default;
    Exercise1& operator= (const Exercise1&) = default;

    Exercise1& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();
    
    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);

    void loadImage();
    
    void saveImage(
        const char*filename, 
        const thrust::host_vector<uchar3>&, 
        const unsigned width, 
        const unsigned height
    );

    void buildOutputFileName();

    void buildHSV();

    PPMBitmap *input;
    thrust::device_vector<uchar3> d_RGB_in;
	thrust::device_vector<float> d_H;
	thrust::device_vector<float> d_S;
	thrust::device_vector<float> d_V;
    thrust::device_vector<uchar3> d_RGB_out;
    thrust::device_vector<uchar3> d_HSV_out;
    
    std::string inputFileName;
    
    std::string outputFileName_hsv;
    std::string outputFileName_rgb;
};