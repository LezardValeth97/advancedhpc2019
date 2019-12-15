#pragma warning( disable : 4244 ) 

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <utils/chronoCPU.hpp>
#include <utils/chronoGPU.hpp>
#include <curand_kernel.h>

#include <exo2/student.h>
#include <exercise2/Exercise2.h>

namespace {

    struct RandomUnsignedFunctor : thrust::unary_function<int,unsigned> {
        int m_seed;
        RandomUnsignedFunctor(int seed) : m_seed(seed) {}
        __device__ unsigned operator()(const int idx) { 
            curandState s;
            curand_init(m_seed+idx, 0, 0, &s);
            return curand(&s); 
        }
    };

    //template<int gold>
    class CheckFunctor : public thrust::unary_function<thrust::tuple<const int,const int>,long long>
    {
    public:
        __device__
        long long operator() (const thrust::tuple<const int, const int>& t) 
        {
            const int a = thrust::get<0>(t);
            const int b = thrust::get<1>(t);
            return static_cast<long long>(a != b);
        }
    };

}
    
void Exercise2::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        std::cout << "Usage: " << argv[0] << " [-h] [--help] [-n=xxx]" << std::endl;
        std::cout << "\twhere options -h and --help display this help," << std::endl;
        std::cout << "\t  and option -n=xxx sets the number of elements of arrays to xxx" << std::endl;
        exit(0);
    }
}

Exercise2& Exercise2::parseCommandLine(const int argc, const char**argv) 
{
    n = 1 << getNFromCmdLine(argc, argv, 4, 28);
    std::cout << "Do the exercise with N=" << n << std::endl;
    return *this;
}

void Exercise2::createReference(const bool verbose) 
{
    if( verbose )
        std::cout << "Build a device vector occupying " << (n>>18) << "Mb" << std::endl;
    d_input.resize(n);
    auto seed =
        std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();   
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        d_input.begin(),
        RandomUnsignedFunctor(seed)        
    );
}

long long Exercise2::checkResult( const bool verbose=true ) 
{
    thrust::device_vector<unsigned> d_sorted(d_input);
    ChronoGPU chr;
    chr.start();
    thrust::sort(d_sorted.begin(), d_sorted.end());
    chr.stop();
    if( verbose )
        std::cout << "\tReference calculated in " << chr.elapsedTime() << " ms" << std::endl;
    auto start_zipped = thrust::make_zip_iterator(
        thrust::make_tuple( 
            d_student.begin(), 
            d_sorted.begin()
        )
    ); 
    auto start = thrust::make_transform_iterator( start_zipped, CheckFunctor() );
    auto stop = start + n;
    return thrust::reduce( start, stop, 0ll );
}

void Exercise2::run(const bool verbose) {
    if( verbose )
        std::cout << std::endl << "Radix Sort using base 2 ..." << std::endl;
    createReference(verbose);
    ChronoGPU chr;
    chr.start();
    StudentWork2*work = reinterpret_cast<StudentWork2*>(student);
    d_student = work->radixSortBase2(d_input);
    chr.stop();
    if( verbose )
        std::cout << "\tDone in " << chr.elapsedTime() << " ms" << std::endl;
}

bool Exercise2::check()
{
    const long long nbErrors = checkResult();
    return ( nbErrors == 0 );
}


