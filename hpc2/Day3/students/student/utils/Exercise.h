#pragma once

#include <iostream>
#include <string>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <utils/chronoCPU.hpp>
#include <utils/chronoGPU.hpp>
#include <utils/StudentWork.h>

class Exercise 
{
public:

    const std::string name;

    Exercise(const std::string& name, StudentWork*student=nullptr) 
        : name(name), student(student)
    {}

    Exercise() = delete;
    Exercise(const Exercise&) = default;
    ~Exercise() = default;
    Exercise& operator= (const Exercise&) = default;
    
    void evaluate(const bool verbose=true) {
        if( !verifyConfiguration(verbose) )
            return ;
        if (verbose) 
            std::cout << "Run exercise " << name << "..." << std::endl;
        run(verbose);
        if ( check() ) 
            std::cout << "Well done: exercise " << name << " SEEMS TO WORK!" << std::endl;
        else
            std::cout << "Bad job: exercise " << name << " DOES NOT WORK!" << std::endl;
    }

    virtual Exercise& parseCommandLine(const int argc, const char**argv) {
        return *this;
    }

protected:
    StudentWork*student;

    void setStudentWork(StudentWork*const student) 
    {
        this->student = student;
    }

    bool verifyConfiguration(const bool verbose) const {
        if (student == nullptr) {
            std::cerr << "Exercise " << name << " not configurated correctly!" << std::endl;
            return false;
        }
        if ( !(student->isImplemented()) ) {
            std::cout << "Exercise " << name << " not implement yet..." << std::endl;
            return false;
        }
        return true;
    }

    virtual void run(const bool verbose) = 0;
    virtual bool check()  = 0;
    
    int getNFromCmdLine(const int argc, const char**argv, int N = 8, const int maxN = 29 ) const
    {
        if( checkCmdLineFlag(argc, argv, "n") ) {
            int value;
            getCmdLineArgumentValue(argc, argv, "n", &value);
            std::cout << "\tfind command line parameter -n=" << value << std::endl;
            if( value >= 1 && value <= maxN )
                N = value;
            else
                std::cerr << "\tWarning: parameter must be positive and lesser than " << maxN << std::endl;
        }
        return N;
    }
};