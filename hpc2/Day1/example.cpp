#include <iostream>

#include<thrust/device_vector.h>
#include<thrust/sequence.h>
#include<thrust/fill.h>

using namespace std

// // display a vector
// void display(thrust::device_vector <float>& U, const string& name){
// 	//copy the data from GPU memory to CPU memory
// 	thrust::host_vector<float> V = U;
// 	// display each values
// 	for(int i=0; i<V.size(); ++i){
// 		cout<<name<<"[" << i << "] " <<V[i] << endl;
// 	}
// }

// // function that adds two elements and return a new one
// class AdditionFunctor : public thrust::binary_function<float,float,float>{
// public:
// 	__host__ __device__ float operator()(const float &x, const float &y){
// 		return x + y;
// 	}
// };

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

// /// main function 
// int main(void){
// 	doAddtion(10); // add "big" vector (of size 10)
// 	return 0;
// }



typedef thrust::tuple<float, float, float> myFloat3;
class AdditionFunctor3: public thrust::unary_function<myFloat3, float>{
public:
	__device__ float operation()(const myFloat3&tuple){
		return thrust::get<0>(tuple) + thrust::get<1>(tuple) + thrust::get<2>(tuple);
	}
};

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(X, Y, Z)),
						thrust::make_zip_iterator(thrust::make_tuple(X+size, Y+size, Z+size)),
						result.begin(),
						AdditionFunctor3() );


void doAddtionIterator(const unsigned workSize){
	thrust::device_vector<float> result(workSize);
	thrust::counting_iterator<float> U(1.f);
	thrust::constant_iterator<float> V(4.f);
	thrust::transform(U, U+workSize, V, result.begin(), AdditionFunctor);
	if (workSize < 128) display(result, "result");
}