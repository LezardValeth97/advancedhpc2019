#include "D_Matrix.cuh"
#include "H_Matrix.cuh"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

//////////////////////////////////////////////////////////////////////////////////
// Exercice 1
bool D_Matrix::Exo1IsDone() {
	return true;
}

// returns this times that ...
D_Matrix D_Matrix::operator+(const D_Matrix& that) const
{
	// do "d_val + that.d_val" 
	D_Matrix result(m_n);
	const int size = m_n*m_n;
	thrust::transform(d_val,
					 d_val + size, 
					 that.d_val,
					 result.d_val,
					 thrust::plus<int>());
	return result;
}



//////////////////////////////////////////////////////////////////////////////////
// Exercice 2
bool D_Matrix::Exo2IsDone() {
	return true;
}

struct scatterFunction: public thrust::unary_function<const int, int>{
	const int n;
	scatterFunction(int size): n(size){}
	__device__ int operator()(const int &x){
		int row = x/n;
		int col = x%n;
		return col*n + row;
	}
};

// define the Matrix::transpose function
D_Matrix D_Matrix::transpose() const
{
	D_Matrix result(m_n);
	const int size = m_n*m_n;

	thrust::counting_iterator<int>X(0);
	thrust::scatter(//thrust::device, 
			d_val, d_val+size,
			thrust::make_transform_iterator(X, scatterFunction(m_n)), 
			result.d_val
		);
	return result;
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 3
bool D_Matrix::Exo3IsDone() {
	return true;
}

struct diffusionFunction: public thrust::unary_function<long long, int>{
	const thrust::device_ptr<int> d_val;
	const int m_n;
	// const long long m_start;
	diffusionFunction(const thrust::device_ptr<int> val, const int n): d_val(val), m_n(n){}
	__device__ int operator()(const long long i){
		return d_val[i%m_n];
	}
};

void D_Matrix::diffusion(const int line, D_Matrix& result) const 
{
	const int size = m_n*m_n;
	// thrust::counting_iterator<int>X(0);
	thrust::copy_n(thrust::make_transform_iterator(
					thrust::make_counting_iterator(0ll), //0ll = 0 long long
					diffusionFunction(d_val+m_n*line, m_n)
					), 
				size,
				result.d_val
	);
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 4
bool D_Matrix::Exo4IsDone() {
	return true;
}

// returns this times that ...
D_Matrix D_Matrix::product1(const D_Matrix& that) const
{	
	D_Matrix result(m_n);
	// Transpose "that" matrix	
	D_Matrix T_that = that.transpose();
	D_Matrix diffusion_that(m_n);
	D_Matrix D(m_n);
	thrust::device_vector<int> key(m_n);
	thrust::device_vector<int> column(m_n);

	// For each column of result:
	for (int i = 0; i < m_n; ++i)
	{
		// Calculate diffusion of T_that[i]
		thrust::copy_n(
			thrust::make_transform_iterator(
				thrust::make_counting_iterator(0ll), //0ll = 0 long long
				diffusionFunction(T_that, i)
			), 
			m_n * m_n,
			diffusion_that.d_val				
		);

		// Make production of each line of this by i-th column of that
		thrust::transform(
			d_val, d_val + m_n*m_n,
			diffusion_that.d_val, D.d_val,
			thrust::placeholders::_1 * thrust::placeholders::_2
		);

		// { Reduction of each line of D, saved into vector ‘‘Column’’ }
		thrust::reduce_by_key(
			thrust::make_transform_iterator(
				thrust::make_counting_iterator(0ll),
				thrust::placeholders::_1 / m_n * m_n + i
				),
			thrust::make_transform_iterator(
				thrust::make_counting_iterator((long long) m_n * m_n),
				thrust::placeholders::_1 / m_n * m_n + i
				),
			D.d_val,
			key.begin(),
			column.begin()
		);

		thrust::scatter(column.begin(), column.end(), key.begin(), result.d_val);
	}
	return result;
}


//////////////////////////////////////////////////////////////////////////////////
// Exercice 5
bool D_Matrix::Exo5IsDone() {
	return false;
}
// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {
	return D_Matrix(m_n);
}
