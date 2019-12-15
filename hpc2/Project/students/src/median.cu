#pragma warning( disable : 4244 ) 
#include <median.h>
#include <helper_cuda.h>
#include <utils/ChronoGPU.hpp>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include <algorithm>


namespace 
{
   
    size_t get_device_memory(bool display=false) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props,0);
        if( display ) 
            std::cout << "... device memory: "<< (props.totalGlobalMem>>20) << " Mb" << std::endl;
        return props.totalGlobalMem;
    }

    struct FillValues : public thrust::unary_function<size_t, int> {
        const thrust::device_ptr<uchar> values;
        const size_t width;
        const size_t height;
        const size_t size;
        const size_t nbNeighbors;

        FillValues(const thrust::device_ptr<uchar>&v, const size_t w, const size_t h, const size_t s)
        : values(v), width(w), height(h), size(s), nbNeighbors(s*s)
        {}

        __device__
        int operator() (const size_t& i) 
        {
            // for each pixel, we count nbNeighbors neighbors ...
            const size_t pixid = i / nbNeighbors; 
            // then the neighbors number is the rest 
            const size_t pos = i % nbNeighbors;
            // mirror effect on the image boundary ... 
            const size_t true_row = abs(static_cast<long long>(pixid/width + pos / size) - static_cast<long long>(size>>1));
            const size_t row = true_row < height ? true_row : 2*height-1 - true_row;
            const size_t true_col = abs(static_cast<long long>((pixid%width) + (pos % size)) - static_cast<long long>(size>>1));
            const size_t col = true_col < width  ? true_col : 2*width-1 - true_col; 

            return (pixid << 8) | unsigned(values[row * width + col]);
        }
    };

    struct countingIteratorFunctor : thrust::unary_function<size_t,size_t> {
        const size_t size;
        countingIteratorFunctor(const size_t s) : size(s)
        {}

        __device__
        size_t operator() (const size_t&id) {
            return size * id + size/2;
        }
    };

    void sort_method(
        const unsigned width, const unsigned height,
        thrust::device_vector<uchar>& d_V, 
        thrust::device_vector<uchar>& d_Vfiltered, 
        const unsigned filter_size
    ) {
        const size_t nbPixels = width * height;
        const size_t nbValues = filter_size*filter_size; 
        size_t d_mem = get_device_memory( true );
        const size_t block_size = size_t(((d_mem/16)/nbValues)*nbValues);
        for(size_t start = 0; start<nbValues*nbPixels; start += block_size) {
            const size_t true_size = min(block_size, nbValues*nbPixels-start);
            //std::cout<<"...... work from " << start << " to " << start+true_size << " (limit being " << (nbValues*nbPixels) << ")"<< std::endl;
            // out values into array
            thrust::device_vector<int> d_mean(true_size);
            thrust::transform(
                thrust::counting_iterator<size_t>(start),
                thrust::counting_iterator<size_t>(start+true_size),
                d_mean.begin(),
                FillValues(d_V.data(), width, height, filter_size)
            );
            // sort the data
            thrust::sort(
                d_mean.begin(),
                d_mean.end()
            );
            // extract the median ...
            auto begin = thrust::make_transform_iterator( thrust::make_counting_iterator(size_t(0)), countingIteratorFunctor(nbValues) );
            thrust::gather(
                begin, begin+(true_size / nbValues),
                d_mean.begin(),
                d_Vfiltered.begin()+(start/nbValues)
            );
        }
    }
}

float median_thrust(
    const unsigned width, const unsigned height,
    thrust::device_vector<uchar>& d_V, 
    thrust::device_vector<uchar>& d_Vfiltered, 
    const unsigned d
) {
    ChronoGPU chr;
    chr.start();
    ::sort_method(width, height, d_V, d_Vfiltered, d);
    chr.stop();
    return chr.elapsedTime();
}