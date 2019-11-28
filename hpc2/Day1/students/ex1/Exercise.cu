#include "Exercise.hpp"
#include <chronoGPU.hpp>
#include<iostream>

#include<thrust/device_vector.h>
#include<thrust/sequence.h>
#include<thrust/fill.h>
#include<thrust/transform.h>
#include "include/chronoGPU.hpp"
using namespace std;

// display a vector
// void display(thrust::device_vector <float>& U, const string& name){
// 	//copy the data from GPU memory to CPU memory
// 	thrust::host_vector<float> V = U;
// 	// display each values
// 	for(int i=0; i<V.size(); ++i){
// 		cout<<name<<"[" << i << "] " <<V[i] << endl;
// 	}
// }

// function that adds two elements and return a new one
class AdditionFunctor : public thrust::binary_function<float,float,float>{
public:
	__host__ __device__ float operator()(const float &x, const float &y) const {
		return x + y;
	}
};

// void doAddtion(unsigned workSize){
// 	thrust::device_vector<float> result(workSize);
// 	thrust::device_vector<float> U(workSize);
// 	thrust::device_vector<float> V(workSize);

// 	// initilization of two given arrays
// 	// fill U with 1,2, 3 .... U.size()
// 	thrust::sequence(U.begin(), U.end(), 1.f);

// 	// fill V with 4,4, .... 4
// 	thrust::fill(V.begin(), V.end(), 4.f);

// 	// do a MAP (one-to-one operation)
// 	thrust::transform(U.begin(), U.end(), V.begin(), result.begin(), AdditionFunctor());

// 	// display the result (if not too big)
// 	if (workSize < 128) display(result, "result");
// }

void Exercise::Question1(const thrust::host_vector<int>& A,
						const thrust::host_vector<int>& B, 
						thrust::host_vector<int>&C) const
{
	// TODO: addition of two vectors using thrust
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();		
		thrust::device_vector<float> U = A;
		thrust::device_vector<float> V = B;
		thrust::device_vector<float> result(C.size());
		chrUP.stop();

		chrGPU.start();	
		thrust::transform(U.begin(), U.end(), V.begin(), result.begin(), AdditionFunctor());
		chrGPU.stop();
		chrDOWN.start();
		C = result;
		chrDOWN.stop();
	// for(int i=0; i<U.size(); ++i){
	// 	cout<<"[" << i << "] " <<U[i] << endl;
	// }
	// doAddtion(10); // add "big" vector (of size 10)
	// display(C, "result");
	}
	float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question1 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;	
}


void Exercise::Question2(thrust::host_vector<int>&A) const 
{
  	// TODO: addition using ad hoc iterators
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();  	
		thrust::device_vector<float> result(A.size());
		thrust::counting_iterator<float> U(1.f);
		thrust::constant_iterator<float> V(4.f);
		chrUP.stop();

		chrGPU.start();
		thrust::transform(U, U+A.size(), V, result.begin(), AdditionFunctor());
		chrGPU.stop();
		chrDOWN.start();
		A = result;
		chrDOWN.stop();	
	// for(int i=0; i<100; ++i){
	// 	cout<<"[" << i << "] " <<U[i] << endl;
	// }
	}
		float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question3 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;
}

typedef thrust::tuple<int, int, int> myInt3;
class additionFunctor3: public thrust::unary_function<myInt3, int>{
	public:
		__device__ int operator()(const myInt3 &tuple){
			return thrust::get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple);
		}
};

void Exercise::Question3(const thrust::host_vector<int>& A,
						const thrust::host_vector<int>& B, 
						const thrust::host_vector<int>& C, 
						thrust::host_vector<int>&D) const 
{
	// TODO
	ChronoGPU chrUP, chrDOWN, chrGPU;
	for(int i=3; i--;){
		chrUP.start();		
		thrust::device_vector<int> gpuA = A;
		thrust::device_vector<int> gpuB = B;
		thrust::device_vector<int> gpuC = C;
		thrust::device_vector<int> gpuD(A.size());
		chrUP.stop();

		chrGPU.start();
		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(gpuA.begin(), gpuB.begin(), gpuC.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(gpuA.end(), gpuB.end(), gpuC.end())), 
			gpuD.begin(), 
			additionFunctor3()
		);
		chrGPU.stop();
		chrDOWN.start();
		D = gpuD;
		chrDOWN.stop();				
		// for(int i=0; i<100; ++i){
		// 	cout<<"[" << i << "] " <<gpuD[i] << endl;
		// }
	}
	float elapsed = chrUP.elapsedTime() + chrGPU.elapsedTime() + chrDOWN.elapsedTime();
	cout<<"Question3 ended in: "<<elapsed<<endl;
	cout<<"UP time: "<<chrUP.elapsedTime()<<endl;
	cout<<"GPU time: "<<chrGPU.elapsedTime()<<endl;
	cout<<"DOWN time: "<<chrDOWN.elapsedTime()<<endl;	
}	
