#include <cuda_runtime.h>
#include <Student.3.h>
#include <helper_cuda.h>
#include <utils/ChronoGPU.hpp>
#include <algorithm>


namespace 
{
    __device__ void fill_shared_memory(
        const uchar*const d_V, 
        const int py,
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;
        // TODO
    }
    
    __device__ void update_histo(
        const uchar*const d_V, 
        const int py,
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        // need to remove the top line of previous pixel py-1, and to add the bottom one of current pixel
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;
        // TODO
    }

    __device__ void scan(const int py) 
    {
        extern __shared__ int s_mem[];
        // where data come from
        int *const s_Histo = &s_mem[0];
        // where data will go
        volatile int *const s_scan = &s_mem[256];

        // 256 threads ...
        // TODO
    }

    __device__ void apply_filter(        
        const uchar*const d_V,
        uchar*const d_Vfiltered,
        const int py,
        const unsigned width,
        const unsigned limit
    ) {
        extern __shared__ int s_mem[];
        const int *const s_cdf = &s_mem[256];
        // TODO
        // after scan, the histo is a CDF (cumulative distribution function)
        // use this property to extract the median value ;-)
        // in other word, for ONE thread write the following line:
        // d_Vfiltered[py*width+blockIdx.x] = threadIdx.x;
    }


    #ifdef CHECK
    __device__ void check_scan() 
    {
        extern __shared__ int s_mem[];
        const int *const s_scan = &s_mem[256];
        if( threadIdx.x>0 && s_scan[threadIdx.x-1]>s_scan[threadIdx.x] )
            printf("[%d/%d] bad values: %d\n", blockIdx.x, threadIdx.x, s_scan[threadIdx.x]);
    }
#endif

    __global__ void filter(
        const uchar*const d_V, 
        uchar*const d_Vfiltered, 
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        fill_shared_memory(d_V, 0, width, height, filter_size);

        // first pixel is specific (no update is needed): just scan and then apply filter
        scan(0); 
#ifdef CHECK
        check_scan();
#endif
        apply_filter(d_V, d_Vfiltered, 0, width, filter_size*filter_size/2);

        // others came after the first one, only updating the histo
        for(int py=1; py<height; ++py) 
        {
            update_histo(d_V, py, width, height, filter_size);
            scan(py); 
#ifdef CHECK
            check_scan();
#endif
            apply_filter(d_V, d_Vfiltered, py, width, filter_size*filter_size/2);
        }
    }
    
}

float student3(
    const unsigned width, const unsigned height,
    thrust::device_vector<uchar>& d_V, 
    thrust::device_vector<uchar>& d_Vfiltered, 
    const unsigned d
) {    
    ChronoGPU chr;
    chr.start();
    dim3 threads(256);    
    std::cout << "number of threads: "<< threads.x << "x" << threads.y << std::endl;
    uchar*const V = d_V.data().get();
    uchar*const F = d_Vfiltered.data().get();
    dim3 blocks(width); 
    ::filter<<<blocks, threads, sizeof(int)*512>>>(V, F, width, height, d);
    chr.stop();
    return chr.elapsedTime();
}