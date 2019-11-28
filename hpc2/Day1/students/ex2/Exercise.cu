#include "Exercise.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include "include/chronoGPU.hpp"
using namespace std;

struct OddEvenGather: public thrust::unary_function<const int, int>
{
	const int n;
	OddEvenGather(int size): n(size){}
	__device__ int operator()(const int &x){
		if ((x*2) < n){
			return x*2;
		}
		else return (1 + (x*2) - n);
	}
};

void Exercise::Question1(const thrust::host_vector<int>& A,
						 thrust::host_vector<int>& OE ) const
{
  	// TODO: extract values at even and odd indices from A and put them into OE.
	// TODO: using GATHER
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();	
		thrust::device_vector<int> gpuA = A;
		thrust::device_vector<int> gpuOE(OE.size());
		thrust::counting_iterator<float> B(0.5);
		chrUP.stop();

		chrGPU.start();	
		auto begin_gather = thrust::make_transform_iterator(B, OddEvenGather(gpuA.size()));
		auto end_gather = thrust::make_transform_iterator(B+gpuA.size(), OddEvenGather(gpuA.size()));
		thrust::gather(thrust::device,
						begin_gather, end_gather,
						gpuA.begin(), gpuOE.begin());
		chrGPU.stop();
		chrDOWN.start();
		OE = gpuOE;
		chrDOWN.stop();
	}
	float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question2 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;
}

struct OddEvenScatter: public thrust::unary_function<const int, int>
{
	const int n;
	OddEvenScatter(int size): n(size){}
	__device__ int operator()(const int &x){
		if(x%2 == 0){
			return x/2;
		}
		else return(n+x)/2; 
	}
};

void Exercise::Question2(const thrust::host_vector<int>&A, 
						thrust::host_vector<int>&OE) const 
{
	// TODO: idem q1 using SCATTER
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();	
		thrust::device_vector<int> gpuA = A;
		thrust::device_vector<int> gpuOE(OE.size());
		thrust::counting_iterator<float> B(0.5);
		chrUP.stop();

		auto begin_scatter = thrust::make_transform_iterator(B, OddEvenScatter(gpuA.size()));
		// auto end_gather = thrust::make_transform_iterator(B+gpuA.size(), OddEvenScatter(gpuA.size()));
		chrGPU.start();
		thrust::scatter(thrust::device,
						gpuA.begin(), gpuA.end(),
						begin_scatter, gpuOE.begin());
		chrGPU.stop();
		chrDOWN.start();
		OE = gpuOE;
		chrDOWN.stop();
	}
	float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question2 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;
}




template <typename T>
void Exercise::Question3(const thrust::host_vector<T>& A,
						thrust::host_vector<T>&OE) const 
{
  // TODO: idem for big objects
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();
		thrust::device_vector<T> gpuA = A;
		thrust::device_vector<T> gpuOE(OE.size());
		thrust::counting_iterator<float> B(0.5);

		chrUP.stop();

		chrGPU.start();
		// thrust::gather(thrust::device,
		// 				thrust::make_transform_iterator(B, OddEvenGather(gpuA.size())),
		// 				thrust::make_transform_iterator(B+gpuA.size(), OddEvenGather(gpuA.size())),
		// 				gpuA.begin(), gpuOE.begin());		
		thrust::scatter(thrust::device, 
			gpuA.begin(), gpuA.end(),
			thrust::make_transform_iterator(B, OddEvenScatter(gpuA.size())),
			//thrust::make_transform_iterator(B + gpuA.size(), evenOddFunction(gpuA.size())),
			gpuOE.begin()
		);
		chrGPU.stop();

		chrDOWN.start();
		OE = gpuOE;
		chrDOWN.stop();
		// for(int i=0; i<100; ++i){
		// 	cout<<"[" << i << "] " <<gpuOE[i] << endl;
		// }		
	}
	float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question 3 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;
}


struct MyDataType {
	MyDataType(int i) : m_i(i) {}
	MyDataType() = default;
	~MyDataType() = default;
	int m_i;
	operator int() const { return m_i; }

	// TODO: add what you want ...
	int x[10];
};

// Warning: do not modify the following function ...
void Exercise::checkQuestion3() const {
	const size_t size = sizeof(MyDataType)*m_size;
	std::cout<<"Check exercice 3 with arrays of size "<<(size>>20)<<" Mb"<<std::endl;
	checkQuestion3withDataType(MyDataType(0));
}


