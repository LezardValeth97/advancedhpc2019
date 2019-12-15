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

class TransposeFunctor1 : public thrust::unary_function<const int, int>{
        const int size1;
public:
        __host__ __device__ TransposeFunctor1() = delete;
        __host__ __device__ TransposeFunctor1(const int size)
                : size1(size)
        {}
        TransposeFunctor1(const TransposeFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
		int row =idx/size1;
                int col =idx%size1;
                return row + col*size1;
        }
};

class DiffusionFunctor1 : public thrust::unary_function<const int, int>{
        const int* ptr;
	const int m_n;
public:
        __host__ __device__ DiffusionFunctor1() = delete;
        __host__ __device__ DiffusionFunctor1(const int* ptr, const int size)
                : ptr(ptr), m_n(size)
        {}
        DiffusionFunctor1(const DiffusionFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
                return ptr[idx%m_n];
        }
};


class LineFunctor1 : public thrust::unary_function<const int, int>{
        const int m_n;
public:
        __host__ __device__ LineFunctor1() = delete;
        __host__ __device__ LineFunctor1(const int size)
                : m_n(size)
        {}
        LineFunctor1(const LineFunctor1&) = default;
        __host__ __device__ int operator()(const int &idx){
                return idx/m_n*m_n;
        }
};

struct MultiplyTupleFunctor1 : public thrust::unary_function< thrust::tuple<int,int>, int>{
        __device__ int operator() (const thrust::tuple<int,int>&t) const {return thrust::get<0>(t) * thrust::get<1>(t);}
};


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
	D_Matrix TBi(m_n);
	thrust::device_vector<int> key(m_n);
	thrust::device_vector<int> column(m_n);
	const int size = m_n*m_n;

	// For each column of result:
	for (int i = 0; i < m_n; ++i){
		T_that.diffusion(i, TBi);
		thrust::transform(
			d_val, d_val+size,
			TBi.d_val, D.d_val, thrust::multiplies<int>()
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
	return true;
}
// returns this times that ...
D_Matrix D_Matrix::product2(const D_Matrix& that) const {

	D_Matrix result(m_n);
        int size = m_n*m_n;
        auto iterator1 =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        LineFunctor1(m_n)
                );

	thrust::device_vector<int> column(m_n);

        thrust::device_vector<int> output_keys(m_n);

	thrust::fill(result.d_val, result.d_val+size, 0);

	D_Matrix tb = that.transpose();

        for (int i=0; i<m_n; ++i){


		auto iterator2 =
                thrust::make_transform_iterator(
                        thrust::make_counting_iterator<int>(0),
                        DiffusionFunctor1(tb.d_val.get()+m_n*i,m_n)
                );

		auto iterator3 =
                thrust::make_transform_iterator(
                        thrust::make_zip_iterator(thrust::make_tuple(d_val, iterator2)),
                        MultiplyTupleFunctor1()
                );

                //functor pour les ligne pour input

                thrust::reduce_by_key(iterator1, iterator1+size, iterator3, output_keys.begin(), column.begin());

                thrust::copy(
                        column.begin(),
                        column.end(),
                        //output_keys.begin(),
                        result.d_val+i*m_n
                );

        }
        return result.transpose();

}

