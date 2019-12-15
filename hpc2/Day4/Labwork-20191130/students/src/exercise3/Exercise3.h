#pragma once

#include <utils/Exercise.h>
#include <exo3/student.h>
#include <thrust/device_vector.h>
#include <utils/ppm.h>

class Exercise3 : public Exercise 
{
public:
    Exercise3(const std::string& name ) 
        : Exercise(name, new StudentWork3())
    {}

    Exercise3() = delete;
    Exercise3(const Exercise3&) = default;
    ~Exercise3() = default;
    Exercise3& operator= (const Exercise3&) = default;

    Exercise3& parseCommandLine(const int argc, const char**argv) ;
    
private:

    void run(const bool verbose);
    void prepare(const bool verbose);
    void run_thrust(const bool verbose);
    void run_cuda(const bool verbose);
    
    bool check();
    bool check_cdf(const thrust::host_vector<unsigned>& cdf) ;

    void displayHelpIfNeeded(const int argc, const char**argv) ;
    void usage(const char*const);
    void usageAndExit(const char*const, const int);

    void loadImage();


    PPMBitmap *input;
    thrust::device_vector<uchar3> d_RGB_in;
	thrust::device_vector<float> d_H;
	thrust::device_vector<float> d_S;
	thrust::device_vector<float> d_V;

    thrust::device_vector<unsigned> d_histo;
    thrust::device_vector<unsigned> d_cdf;
    thrust::device_vector<unsigned> d_cdf_fast;
    
    std::string inputFileName;
};