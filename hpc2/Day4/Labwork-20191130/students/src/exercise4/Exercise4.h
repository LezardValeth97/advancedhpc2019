#pragma once

#include <utils/Exercise.h>
#include <exo4/student.h>
#include <thrust/device_vector.h>
#include <utils/ppm.h>

class Exercise4 : public Exercise 
{
public:
    Exercise4(const std::string& name ) 
        : Exercise(name, new StudentWork3())
    {}

    Exercise4() = delete;
    Exercise4(const Exercise4&) = default;
    ~Exercise4() = default;
    Exercise4& operator= (const Exercise4&) = default;

    Exercise4& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);
    void prepare(const bool verbose);
    void run_thrust(const bool verbose);
    void run_cuda(const bool verbose);
    void run_final(const bool verbose);
    
    bool check();
    bool check_cdf(const thrust::host_vector<unsigned>& cdf) ;

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


    PPMBitmap *input;
    thrust::device_vector<uchar3> d_RGB_in;
	thrust::device_vector<float> d_H;
	thrust::device_vector<float> d_S;
	thrust::device_vector<float> d_V;
    thrust::device_vector<uchar3> d_RGB_out;

    thrust::device_vector<unsigned> d_histo;
    thrust::device_vector<unsigned> d_cdf;
    thrust::device_vector<float> d_V_equalized;
    thrust::device_vector<float> d_V_equalized_fast;
    
    std::string inputFileName;
    
    std::string outputFileName_rgb;
};