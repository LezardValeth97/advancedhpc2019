#include "student.h"

namespace {
    
    template<typename T>
    __device__ inline T max(const T&a, const T&b) { 
        return a<b ? b : a; 
    }

    struct RGB2VFunctor : public thrust::unary_function<uchar3,uchar>
    {
        __device__
        uchar operator() (const uchar3& RGB) {
            return max(RGB.x, max(RGB.y, RGB.z)); // return the Value, i.e. the max
        }
    };
    
    struct FilterFunctor : public thrust::binary_function<const uchar3,const uchar,uchar3>
    {
        __device__
        uchar3 operator() (const uchar3&u_rgb, const uchar V) 
        {
            const float3 RGB = make_float3( float(u_rgb.x), float(u_rgb.y), float(u_rgb.z));
            const float d = fmaxf(RGB.x, fmaxf(RGB.y, RGB.z)); // old value
            const float N = d > 0.f ? float(V) / d : 0.f; // ratio
            const float R = fminf(RGB.x * N, 255.f);
            const float G = fminf(RGB.y * N, 255.f);
            const float B = fminf(RGB.z * N, 255.f);
            return make_uchar3(R, G, B); // modify the value of a given pixel
        }
    };
    
    // first pixel, fill all the shared memory with its neighbours
    __device__ void fill_shared_memory(
        const uchar*const d_V, 
        const int py,
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;
        // we have exactly 256 threads
        s_Histo[threadIdx.x] = 0u;
        __syncthreads();
        const int startX = px - (filter_size>>1);
        const int startY = py - (filter_size>>1);

        for (unsigned tid = threadIdx.x; tid < filter_size * filter_size; tid += blockDim.x){
        // TODO: histogram with all neighbors
            int x = startX + tid % filter_size;
            int y = startY + tid / filter_size;
            int tidx, tidy;

            tidx = (x < 0) ? -x : ((x >= width) ? 2 * width - 1 - x : x);
            tidy = (y < 0) ? -y : ((y >= height) ? 2 * height - 1 - y : y);

            atomicAdd(&s_Histo[(unsigned)d_V[tidy * width + tidx]], 1); 
        }
        __syncthreads();
    }
    
    __device__ void update_histo(
        const uchar*const d_V, 
        const int py,
        const unsigned width, const unsigned height, 
        const unsigned filter_size
    ) {
        // need to remove the top line, and to add the bottom one
        extern __shared__ int s_Histo[];
        const int px = blockIdx.x;
        const int startX = px - (filter_size>>1);
        const int startY = py - (filter_size>>1);
        
        for(unsigned tid=threadIdx.x; tid < filter_size; tid+=blockDim.x ){
            // TODO: modify histogram, remove old top line, add new bottom one
            
            int x = startX + tid % filter_size;
            int tidx;

            tidx = (x < 0) ? -x : ((x >= width) ? 2 * width - 1 - x : x);

            if (startY - 1 >= 0){
                atomicSub(&s_Histo[(unsigned)d_V[(startY - 1) * width + tidx]], 1);
            }

            if (startY + filter_size - 1 < height){
                atomicAdd(&s_Histo[(unsigned)d_V[(startY + filter_size - 1) * width + tidx]], 1);
            }
        }
        __syncthreads();
    }

    __device__ void scan(const int py) 
    {
        extern __shared__ int s_mem[];
        const int *const s_Histo = &s_mem[0];
        volatile int *const s_scan = &s_mem[256];

        // 256 threads ...
        s_scan[threadIdx.x] = s_Histo[threadIdx.x];
        __syncthreads();
        // TODO: a scan into the current block (using shared memory)
        int x = 0;
        int y = 0;
        for (int i = 1; i < 256; i *= 2){
            if (threadIdx.x  >= i){
                x = s_scan[threadIdx.x];
                y = s_scan[threadIdx.x - i];
                y = y + x;
            }
            __syncthreads();

            if (threadIdx.x >= i){
                s_scan[threadIdx.x] = y;
            }
            __syncthreads();
        }
    }

    __device__ void apply_filter(        
        const uchar*const d_V,
        uchar*const d_V_median,
        const int py,
        const unsigned width,
        const unsigned limit
    ) {
        extern __shared__ int s_mem[];
        const int *const s_cdf = &s_mem[256];
        const int px = blockIdx.x;
        // after scan, the histo is a CDF (cumulative distribution function)
        // then only only thread will succeed the following test ;-)
        // TODO
        if (((threadIdx.x == 0) || (s_cdf[threadIdx.x - 1] <= limit)) && (s_cdf[threadIdx.x] > limit)){ 
            d_V_median[py * width + px] = threadIdx.x;
        }      
    }


#define CHECK
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
        uchar*const d_V_median, 
        const unsigned width, 
        const unsigned height, 
        const unsigned filter_size
    ) {
        ::fill_shared_memory(d_V, 0, width, height, filter_size);
        // first pixel is specific (no maj): just scan and then apply filter
        ::scan(0); 
#ifdef CHECK
        ::check_scan();
#endif
        ::apply_filter(d_V, d_V_median, 0, width, filter_size*filter_size/2);
        // others came after the first one, only updating the histo
        for(int py=1; py<height; ++py) 
        {
            // maj histo
            ::update_histo(d_V, py, width, height, filter_size);
            // scan
            ::scan(py); 
#ifdef CHECK
            ::check_scan();
#endif
            // apply
            ::apply_filter(d_V, d_V_median, py, width, filter_size*filter_size/2);
        }
    }
}

bool StudentWork1::isImplemented() const {
    return true;
}


void StudentWork1::rgb2h(
    const thrust::device_vector<uchar3>&rgb,
    thrust::device_vector<uchar>&V
)
{
    thrust::transform(
        rgb.begin(),
        rgb.end(),
        V.begin(),
        ::RGB2VFunctor()
    );
}

void StudentWork1::median(
    const thrust::device_vector<uchar> &d_V,
    thrust::device_vector<uchar> &d_V_median,
    const unsigned width,
    const unsigned height,
    const unsigned filter_size
) {    
    dim3 threads(256);    
    if( d_V.size() != width * height ) 
        std::cout << "Problem with the size of d_V" << std::endl;
    if( d_V_median.size() != width * height ) 
        std::cout << "Problem with the size of d_V_median" << std::endl;
    uchar const*const V = d_V.begin().base().get();
    uchar *const F = d_V_median.begin().base().get();
    dim3 blocks(width); 
    ::filter<<<blocks, threads, sizeof(int)*512>>>(V, F, width, height, filter_size);
    std::cout << "do the copy" << std::endl;
}


void StudentWork1::apply_filter(
    const thrust::device_vector<uchar3>&RGB_old,
    const thrust::device_vector<uchar>&V_new,
    thrust::device_vector<uchar3>&RGB_new
) 
{
    thrust::transform(
        RGB_old.begin(), RGB_old.end(),
        V_new.begin(),
        RGB_new.begin(),
        ::FilterFunctor()
    );
}