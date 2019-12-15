#pragma once

#include <utils/Exercise.h>
#include <exo2/student.h>
#include <thrust/device_vector.h>

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
    // size of the vector of integers
    int n;

    void run(const bool verbose);

    bool check();

    void displayHelpIfNeeded(const int argc, const char**argv) ;

    void createReference(const bool verbose);

    long long checkResult( const bool verbose ) ;

    thrust::device_vector<unsigned> d_input;
    thrust::device_vector<unsigned> d_student;
};