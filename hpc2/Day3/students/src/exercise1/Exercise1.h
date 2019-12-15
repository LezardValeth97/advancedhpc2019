#pragma once

#include <utils/Exercise.h>
#include <exo1/student.h>
#include <thrust/device_vector.h>

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
    // size of the vector of colored objects
    int n;

    void run(const bool verbose);

    bool check();

    void displayHelpIfNeeded(const int argc, const char**argv) ;

    void createReference(int N);

    bool checkResult( const long long truth ) ;

    thrust::device_vector<ColoredObject> d_vector;
    thrust::device_vector<ColoredObject> d_student;
};