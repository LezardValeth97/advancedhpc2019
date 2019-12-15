#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <exo2/student.h>

// Exercise 2: radix sort

namespace {

	class FunctorN : public thrust::unary_function<int,int>{
		public :
			__device__ int operator()(const int &x){
				return 1- x ;
			}
	} ;

	class FunctorS : public thrust::unary_function<int,int>{
		const int size ;
		const unsigned* F ;
		const unsigned* up ;
		const unsigned* down ;
		public :

			__host__ __device__ FunctorS() = delete ;
			__host__ __device__ FunctorS(int Size, unsigned *Flag ,unsigned *u,unsigned *d):size(Size),F(Flag),up(d),down(u){}
			FunctorS(const FunctorS&) = default ;

			__device__ long long operator()(const int&i){
				return (F[i] == 1) ? size-up[i] : down[i] ;
			}
	} ;


	class FunctorB : public thrust::unary_function<int,int>{
		const int i ;
		public :

			__host__ __device__ FunctorB() = delete ;
			__host__ __device__ FunctorB(int bit):i(bit){}
			FunctorB(const FunctorB&) = default ;

			__device__ int operator()(const int&y){
				return (i > 0 ? y>>i : y)&0x01  ;
			}

	};

	void display(thrust::host_vector<unsigned> input){
		for (int i=0;i<input.size();i++){
			std::cout <<input[i] << " " ;
		}
		std::cout << std::endl ;
	}
		// Add here what you need ...
}

bool StudentWork2::isImplemented() const {
	return true;
}

thrust::device_vector<unsigned> StudentWork2::radixSortBase2( const thrust::device_vector<unsigned>& d_input ) 
{
	display(d_input) ;
	thrust::device_vector<unsigned> result = d_input ;
	thrust::device_vector<unsigned> resultF(d_input.size()) ;
	thrust::device_vector<unsigned> Flag(d_input.size()) ;
	thrust::device_vector<unsigned> Idown(d_input.size()) ;	
	thrust::device_vector<unsigned> Iup(d_input.size()) ;	
	thrust::counting_iterator<int> i(0) ;

	for (int nb = 0;nb < 32;nb++){

		thrust::transform(result.begin(),result.end(),Flag.begin(),FunctorB(nb)); 
		auto Iscan = thrust::make_transform_iterator(Flag.begin(),FunctorN()) ;
		thrust::exclusive_scan(thrust::device,Iscan,Iscan+result.size(),Idown.begin(),0)  ;

		thrust::inclusive_scan(thrust::device,Flag.rbegin(),Flag.rend(),Iup.rbegin()) ;

		auto Index = thrust::make_transform_iterator(i,FunctorS(d_input.size(),Flag.begin().base().get(),Idown.begin().base().get(),Iup.begin().base().get())) ;
 
		thrust::scatter(thrust::device,result.begin(),result.end(),Index,resultF.begin()) ;

		thrust::copy(thrust::device,resultF.begin(),resultF.end(),result.begin());
	} 
	return result;
}