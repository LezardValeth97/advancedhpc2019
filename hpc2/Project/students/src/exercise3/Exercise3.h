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

    PPMBitmap *input;
    thrust::device_vector<uchar3> d_RGB_in;
    thrust::device_vector<uchar3> d_RGB_out;

    std::string inputFileName;
    
    std::string outputFileName_rgb;
};