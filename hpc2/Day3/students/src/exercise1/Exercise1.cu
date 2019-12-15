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
    struct RandomColorFunctor : thrust::unary_function<int,ColoredObject> {
        int m_seed;
        RandomColorFunctor(int seed) : m_seed(seed) {}
        __device__ ColoredObject operator()(const int idx) { 
            curandState s;
            curand_init(m_seed+idx, 0, 0, &s);
            return ColoredObject(
                curand_uniform(&s)<0.5f ? ColoredObject::Color::BLUE : ColoredObject::Color::RED,
                int(curand_uniform(&s) * (1<<24))        
            ); 
        }
    };

    class CheckFunctor : public thrust::unary_function<const ColoredObject,long long>
    {
    public:
        __device__
        long long operator() (const ColoredObject& co) const
        {
            return static_cast<long long>( co.color == ColoredObject::Color::BLUE ); 
        }
    };

}

Exercise1& Exercise1::parseCommandLine(const int argc, const char**argv) 
{
    n = 1 << getNFromCmdLine(argc, argv, 8, 26);
    std::cout << "Do the exercise with N=" << n << std::endl;
    return *this;
}

void Exercise1::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        std::cout << "Usage: " << argv[0] << " [-h] [--help] [-n=xxx]" << std::endl;
        std::cout << "\twhere options -h and --help display this help," << std::endl;
        std::cout << "\t  and option -n=xxx sets the number of elements of arrays to xxx" << std::endl;
        exit(0);
    }
}

void Exercise1::createReference(int N) 
{
    auto seed =
        std::chrono::duration_cast<std::chrono::minutes>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();    
    d_vector.resize(N);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        d_vector.begin(),
        RandomColorFunctor(seed)        
    );
}


void Exercise1::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "List Selection ..." << std::endl;
    // build a host vector containing the reference
    createReference( n );
    ChronoGPU chr;
    chr.start();
    d_student = reinterpret_cast<StudentWork1*>(student)->compactBlue(d_vector);
    chr.stop();
    if( verbose )
        std::cout << "\tStudent's Work Done in " << chr.elapsedTime() << " ms" << std::endl;
}


bool Exercise1::checkResult( const long long truth ) 
{
    const long long N = static_cast<long long>(d_student.size());
    if( truth != N ) return false;
    auto begin = thrust::make_transform_iterator( d_student.begin(), ::CheckFunctor());
    const long long nbSuccess = thrust::reduce( begin, begin+N);
    return truth == nbSuccess;
}

bool Exercise1::check() {
    // count the number of blue objects
    auto begin = thrust::make_transform_iterator(d_vector.begin(), ::CheckFunctor());
    const long long nbBlue = thrust::reduce(begin, begin+n);
    std::cout << "\tThe random vector contains " << nbBlue << " BLUE objects." << std::endl;
    return checkResult( nbBlue );
}