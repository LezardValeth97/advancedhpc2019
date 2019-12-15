#pragma once

#include <utils/Exercise.h>
#include <exo2/student.h>
#include <thrust/device_vector.h>
#include <utils/ppm.h>

class Exercise2 : public Exercise 
{
public:
    Exercise2(const std::string& name ) 
        : Exercise(name, new StudentWork2())
    {}

    Exercise2() = delete;
    Exercise2(const Exercise2&) = default;
    ~Exercise2() = default;
    Exercise2& operator= (const Exercise2&) = default;

    Exercise2& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);

    bool check();

    bool check_histo(const thrust::device_vector<unsigned>&histo);
    
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

    thrust::device_vector<unsigned> d_histo;
    thrust::device_vector<unsigned> d_histo_fast;
    
    std::string inputFileName;
    
    std::string outputFileName_hsv;
    std::string outputFileName_rgb;
};