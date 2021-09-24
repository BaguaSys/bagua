#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <nccl.h>
#include <cub/cub.cuh>
#include "bagua_utils.h"

const float eps = 1e-7;

// Reference: https://github.com/NVIDIA/nccl-tests/blob/2f9bba9f20002e7b7818e7fdeae6e35734260aff/src/common.h#L207
size_t word_size(ncclDataType_t type) {
    switch (type) {
        case ncclChar:
#if NCCL_MAJOR >= 2
            //case ncclInt8:
    case ncclUint8:
#endif
            return 1;
        case ncclHalf:
            //case ncclFloat16:
            return 2;
        case ncclInt:
        case ncclFloat:
#if NCCL_MAJOR >= 2
            //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclDouble:
            //case ncclFloat64:
            return 8;
        default:
            return 0;
    }
}

// Reference: https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/common.cuh#L67
__device__ inline half __hdiv_with_fallback(const half a, const half b) {
#if __CUDA_ARCH__ >= 530
    return __hdiv(a, b);
#else
    float out;
    out = __half2float(a) / __half2float(b);
    return __float2half_rn(out);
#endif
}

__device__ inline half __havg_with_fallback(const half a, const half b) {
#if __CUDA_ARCH__ >= 530
    return __hadd(a, b) / __float2half(2.0);
#else
    float out;
    out = (__half2float(a) + __half2float(b)) / 2.0;
    return __float2half_rn(out);
#endif
}

__device__ inline half __hsub_with_fallback(const half a, const half b) {
#if __CUDA_ARCH__ >= 530
    return __hsub(a, b);
#else
    float out;
    out = __half2float(a) - __half2float(b);
    return __float2half_rn(out);
#endif
}

__device__ inline half __hadd_with_fallback(const half a, const half b) {
#if __CUDA_ARCH__ >= 530
    return __hadd(a, b);
#else
    float out;
    out = __half2float(a) + __half2float(b);
    return __float2half_rn(out);
#endif
}

__device__ inline half __haddmul_with_fallback(const half a, const half b, const half factor) {
#if __CUDA_ARCH__ >= 530
    return __hadd(a, __hmul(b, factor));
#else
    float out;
    out = __half2float(a) + __half2float(b) * __half2float(factor);
    return __float2half_rn(out);
#endif
}

// Reference: https://github.com/dmlc/cub/blob/master/cub/thread/thread_operators.cuh
struct Sum {
    /// Boolean sum operator, returns <tt>a + b</tt>
    template<typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct Max
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return ((b > a) ? b : a);
    }

};

template <>
__device__ __forceinline__ half Max::operator()<half>(const half &a, const half &b) const {
#if __CUDA_ARCH__ >= 530
	return __hgt(b, a) ? b: a;
#else
	return (__half2float(b) > __half2float(a) ? b : a);
#endif
}


struct Min
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return ((b < a) ? b : a);
    }

};

template <>
__device__ __forceinline__ half Min::operator()<half>(const half &a, const half &b) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(b, a) ? b: a;
#else
    return (__half2float(b) < __half2float(a) ? b : a);
#endif
}


template<typename T>
__device__ inline float __accum_to_float(float a, T b) {
    return a + b;
}

template<>
__device__ inline float __accum_to_float<half>(float a, half b) {
    return a + __half2float(b);
}

template<typename T, bool average>
__device__ inline T __from_float(float a, int n, T placeholder) {
   if (average) {
       return a / n;
   } else {
       return a;
   }
}

template<>
__device__ inline half __from_float<half, true>(float a, int n, half placeholder) {
   return  __float2half(a / n);
}

template<>
__device__ inline half __from_float<half, false>(float a, int n, half placeholder) {
   return  __float2half(a);
}

template<typename ReductionOpT, unsigned int block_dim_y>
__device__ void
block_y_reduce(float sdata[][block_dim_y], unsigned int tidx, unsigned int tidy, ReductionOpT reduction_op) {
    if (block_dim_y >= 32) {
        if (tidy < 16) { sdata[tidx][tidy] = reduction_op(sdata[tidx][tidy], sdata[tidx][tidy + 16]); }
        __syncthreads();
    }
    if (block_dim_y >= 16) {
        if (tidy < 8) { sdata[tidx][tidy] = reduction_op(sdata[tidx][tidy], sdata[tidx][tidy + 8]); }
        __syncthreads();
    }
    if (block_dim_y >= 8) {
        if (tidy < 4) { sdata[tidx][tidy] = reduction_op(sdata[tidx][tidy], sdata[tidx][tidy + 4]); }
        __syncthreads();
    }
    if (block_dim_y >= 4) {
        if (tidy < 2) { sdata[tidx][tidy] = reduction_op(sdata[tidx][tidy], sdata[tidx][tidy + 2]); }
        __syncthreads();
    }
    if (block_dim_y >= 2) {
        if (tidy < 1) { sdata[tidx][tidy] = reduction_op(sdata[tidx][tidy], sdata[tidx][tidy + 1]); }
        __syncthreads();
    }
}

__global__ void average_inplace_f32(float *x, float *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = (x[i] + y[i]) / 2.0;
    }
}

__global__ void average_inplace_f16(__half *x, __half *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __havg_with_fallback(x[i], y[i]);
    }
}

__global__ void substract_inplace_f32(float *x, float *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] -= y[i];
    }
}

__global__ void substract_inplace_f16(__half *x, __half *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __hsub_with_fallback(x[i], y[i]);
    }
}

__global__ void add_inplace_f32(float *x, float *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] += y[i];
    }
}

__global__ void add_inplace_f16(__half *x, __half *y, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __hadd_with_fallback(x[i], y[i]);
    }
}

__global__ void addmul_inplace_f32(float *x, float *y, int N, float factor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] += y[i] * factor;
    }
}

__global__ void addmul_inplace_f16(__half *x, __half *y, int N, __half factor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __haddmul_with_fallback(x[i], y[i], factor);
    }
}

__global__ void divide_inplace_f32(float *x, float D_, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = x[i] / D_;
    }
}

__global__ void divide_inplace_f16(__half *x, float D_, int N) {
    __half D__ = __float2half(D_);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        x[i] = __hdiv_with_fallback(x[i], D__);
    }
}

__global__ void async_model_average(float *tensor, const float *reduced_tensor_copy, 
		const float *tensor_copy, const float nranks, const int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
	
	tensor[i] += reduced_tensor_copy[i] / nranks - tensor_copy[i];
        /*if (tensor[i] != tensor[i]) {
            printf("nan encountered!");
        }*/
//        atomicAdd(&tensor[i], reduced_tensor_copy[i] / nranks - tensor_copy[i]);
    }
}

template<typename T>
size_t array_min_max_size(
        const T *input_array,
        int num_items,
        T *output_array,
        cudaStream_t stream) {

     void *dev_buffer = NULL;
     size_t dev_buffer_bytes = 0;

    CUDACHECK(cub::DeviceReduce::Min(
                dev_buffer,
                dev_buffer_bytes,
                input_array,
                output_array,
                num_items,
                stream));

    return dev_buffer_bytes;
}

template<>
size_t array_min_max_size<half>(
        const half *input_array,
        int num_items,
        half *output_array,
        cudaStream_t stream) {

    void *dev_buffer = NULL;
    size_t dev_buffer_bytes = 0;

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array,
            num_items,
            Min(),
            __float2half(65504),  // FIXME
            stream);

    return dev_buffer_bytes;
}

template<typename T>
void array_min_max(
        const T *input_array,
        int num_items,
        void *dev_buffer,
        size_t dev_buffer_bytes,
        T *output_array,
        cudaStream_t stream) {

    CUDACHECK(cub::DeviceReduce::Min(
                dev_buffer,
                dev_buffer_bytes,
                input_array,
                output_array,
                num_items,
                stream));
    
    CUDACHECK(cub::DeviceReduce::Max(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array + 1,
            num_items,
            stream));


//    float *val = (float *) malloc(sizeof(float));
//    cudaMemcpy(val, output_max, sizeof(float), cudaMemcpyDeviceToHost);
//    std::cout << "max " << *val << std::endl;
}

template<>
void array_min_max<half>(
        const half *input_array,
        int num_items,
        void *dev_buffer,
        size_t dev_buffer_bytes,
        half *output_array,
        cudaStream_t stream) {

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array,
            num_items,
            Min(),
            __float2half(65504),  // FIXME
            stream);

    cub::DeviceReduce::Reduce(
            dev_buffer,
            dev_buffer_bytes,
            input_array,
            output_array + 1,
            num_items,
            Max(),
            __float2half(-65504),  // FIXME
            stream);
}

template<unsigned int block_dim_x, unsigned int block_dim_y, typename T, bool average>
__global__ void reduce_chunk_inplace(T *input, int chunk_size, int num_chunks, int target_chunk) {

    __shared__ float sdata[block_dim_x][block_dim_y];

    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // load to shared memory
    float sum = 0.0;
    for (int i = idy; i < num_chunks && idx < chunk_size; i += blockDim.y) {
        sum = __accum_to_float(sum, input[chunk_size * i + idx]);
    }

    sdata[tidx][tidy] = sum;
    __syncthreads();

    block_y_reduce<Sum, block_dim_y>(sdata, tidx, tidy, Sum());

    // write to global memory
    T *output = input + target_chunk * chunk_size;
    if (tidy == 0 && idx < chunk_size) {
        output[idx] = __from_float<T, average>(sdata[tidx][tidy], num_chunks, output[idx]);
    }
}

template<typename T>
__device__ inline uint8_t __minmax_uint8_compress(T f, float scale, float lower_bound, float upper_bound) {
    float level = f * scale;
    level = min(level, upper_bound);
    return level - lower_bound;

}

template<>
__device__ inline uint8_t __minmax_uint8_compress<float>(float f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(f * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<>
__device__ inline uint8_t __minmax_uint8_compress<half>(half f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(__half2float(f) * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<typename T>
__device__ inline T __minmax_uint8_decompress(uint8_t i, float scale, float lower_bound, float upper_bound, T placeholder) {
    return (i + lower_bound) / scale;
}

template<>
__device__ inline half __minmax_uint8_decompress<half>(uint8_t i, float scale, float lower_bound, float upper_bound, half placeholder) {
    return __float2half((i + lower_bound) / scale);
}

template<typename T>
__device__ inline float __load_as_float(T * array) {
    return array[0];
}

template<>
__device__ inline float __load_as_float<half>(half * array) {
    return __half2float(array[0]);
}

template<typename T>
__device__ inline void __store_float(T * array, float data) {
    array[0] = data;
}

template<>
__device__ inline void __store_float<half>(half * array, float data) {
    array[0] = __float2half(data);
}


template<typename T>
__global__ void
compress_float_to_uint8(T *input, int chunk_size, int chunk_offset, int num_chunks, uint8_t *output,
                      size_t output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float min_ = __load_as_float(reinterpret_cast<T *>(output + idy * chunk_offset));
    float max_ = __load_as_float(reinterpret_cast<T *>(output + idy * chunk_offset + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;
    for (int i = idx; i < chunk_size; i += blockDim.x * gridDim.x) {
        int k = idy * chunk_size + i;
        int o = idy * chunk_offset + 32 + i;
        output[o] = __minmax_uint8_compress(input[k], scale, lower_bound, upper_bound);
    }

    if (idx == 0) {
        // write max min to output buffer
        __store_float(reinterpret_cast<T *>(output + idy * chunk_offset), min_);
        __store_float(reinterpret_cast<T *>(output + idy * chunk_offset + sizeof(T)), max_);
    }
}

template<typename T>
__global__ void
decompress_uint8_to_float(uint8_t *input, size_t input_size, int chunk_size, int chunk_offset, int num_chunks,
                          T *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float min_ = __load_as_float(reinterpret_cast<T *>(input + idy * chunk_offset));
    const float max_ = __load_as_float(reinterpret_cast<T *>(input + idy * chunk_offset + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upper_bound = rintf(max_ * scale);
    float lower_bound = upper_bound - 255.0;

    for (int i = idx; i < chunk_size; i += blockDim.x * gridDim.x) {
        int k = idy * chunk_size + i;
        int o = idy * chunk_offset + 32 + i;
        output[k] = __minmax_uint8_decompress(input[o], scale, lower_bound, upper_bound, output[k]);
    }
}

template<typename T, bool average>
void reduce_chunk_inplace_host(T *input, int chunk_size, int num_chunks, int target_chunk, cudaStream_t stream) {
    if (num_chunks <= 4) {
        dim3 num_blocks(DIVUP(chunk_size, 512), 1);
        dim3 num_threads(512, 2);
        reduce_chunk_inplace<512, 2, T, average><<<num_blocks, num_threads, 0, stream>>>(input, chunk_size, num_chunks,
                                                                               target_chunk);
    } else if (num_chunks <= 8) {
        dim3 num_blocks(DIVUP(chunk_size, 256), 1);
        dim3 num_threads(256, 4);
        reduce_chunk_inplace<256, 4, T, average><<<num_blocks, num_threads, 0, stream>>>(input, chunk_size, num_chunks,
                                                                               target_chunk);
    } else if (num_chunks <= 16) {
        dim3 num_blocks(DIVUP(chunk_size, 128), 1);
        dim3 num_threads(128, 8);
        reduce_chunk_inplace<128, 8, T, average><<<num_blocks, num_threads, 0, stream>>>(input, chunk_size, num_chunks,
                                                                               target_chunk);
    } else if (num_chunks <= 32) {
        dim3 num_blocks(DIVUP(chunk_size, 64), 1);
        dim3 num_threads(64, 16);
        reduce_chunk_inplace<64, 16, T, average><<<num_blocks, num_threads, 0, stream>>>(input, chunk_size, num_chunks,
                                                                               target_chunk);
    } else {
        dim3 num_blocks(DIVUP(chunk_size, 32), 1);
        dim3 num_threads(32, 32);
        reduce_chunk_inplace<32, 32, T, average><<<num_blocks, num_threads, 0, stream>>>(input, chunk_size, num_chunks,
                                                                               target_chunk);
    }
    CUDACHECK(cudaGetLastError());
}

template<typename T>
void compress_float_to_uint8_host(T *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    int chunk_offset = output_size / num_chunks;
    int remaining_elem = input_num_element;
    for (int i = 0; i < num_chunks; i++) {
        if ((target_chunk == -1) || (i == target_chunk)) {
            array_min_max(input + i * chunk_size, std::min(remaining_elem, chunk_size), dev_buffer, dev_size,
                          reinterpret_cast<T *>(output + i * chunk_offset), stream);
        }
        remaining_elem -= chunk_size;
    }

    if (target_chunk == -1) {
        dim3 num_blocks(DIVUP(chunk_size, 1024), num_chunks);
        compress_float_to_uint8<<<num_blocks, 1024, 0, stream>>>(input, chunk_size, chunk_offset, num_chunks, output,
                                                               output_size);
    } else {
        dim3 num_blocks(DIVUP(chunk_size, 1024), 1);
        T *chunk_input = input + target_chunk * chunk_size;
        uint8_t *chunk_output = output + target_chunk * chunk_offset;

        compress_float_to_uint8<<<num_blocks, 1024, 0, stream>>>(chunk_input, chunk_size, chunk_offset, 1, chunk_output,
                                                               chunk_offset);
    }
    CUDACHECK(cudaGetLastError());
}

template<typename T>
void decompress_uint8_to_float_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, T *output,
                                   cudaStream_t stream) {

    int chunk_offset = input_size / num_chunks;
    dim3 num_blocks(DIVUP(chunk_size, 1024), num_chunks);
    decompress_uint8_to_float<<<num_blocks, 1024, 0, stream>>>(input, input_size,
                                                             chunk_size, chunk_offset, num_chunks, output);
    CUDACHECK(cudaGetLastError());
}

extern "C" {
void divide_inplace_f32_host(float *x, float D_, int N, cudaStream_t stream) {
    divide_inplace_f32<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, D_, N);
    CUDACHECK(cudaGetLastError());
}

void divide_inplace_f16_host(__half *x, float D_, int N, cudaStream_t stream) {
    divide_inplace_f16<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, D_, N);
    CUDACHECK(cudaGetLastError());
}

void add_inplace_f32_host(float *x, float *y, int N, cudaStream_t stream) {
    add_inplace_f32<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void add_inplace_f16_host(__half *x, __half *y, int N, cudaStream_t stream) {
    add_inplace_f16<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void addmul_inplace_f32_host(float *x, float *y, int N, const float factor, cudaStream_t stream) {
    addmul_inplace_f32<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N, factor);
    CUDACHECK(cudaGetLastError());
}

void addmul_inplace_f16_host(__half *x, __half *y, int N, const float factor, cudaStream_t stream) {
    addmul_inplace_f16<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N, __float2half(factor));
    CUDACHECK(cudaGetLastError());
}

void substract_inplace_f32_host(float *x, float *y, int N, cudaStream_t stream) {
    substract_inplace_f32<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void substract_inplace_f16_host(__half *x, __half *y, int N, cudaStream_t stream) {
    substract_inplace_f16<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void average_inplace_f32_host(float *x, float *y, int N, cudaStream_t stream) {
    average_inplace_f32<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void average_inplace_f16_host(__half *x, __half *y, int N, cudaStream_t stream) {
    average_inplace_f16<<<DIVUP(N, 1024), 1024, 0, stream>>>(x, y, N);
    CUDACHECK(cudaGetLastError());
}

void async_model_average_host(float *tensor, const float *reduced_tensor_copy, 
		const float *tensor_copy, const float nranks, const int N, cudaStream_t stream) {
    async_model_average<<<DIVUP(N, 1024), 1024, 0, stream>>>(tensor, reduced_tensor_copy, tensor_copy, nranks, N);
    CUDACHECK(cudaGetLastError());
}

//// decentralize, recvbuf should get the average of sendbuf and peer's sendbuf
//ncclResult_t ncclPeerAverage(void *sendbuf, void *recvbuf, size_t sendcount,
//                             int peer_rank, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
//    NCCLCHECK(ncclGroupStart());
//    NCCLCHECK(ncclSend(sendbuf, sendcount, datatype, peer_rank, comm, stream));
//    NCCLCHECK(ncclRecv(recvbuf, sendcount, datatype, peer_rank, comm, stream));
//    NCCLCHECK(ncclGroupEnd());
//
//    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sendcount; i += blockDim.x * gridDim.x) {
//        // FIXME: this is not always float
////        float f = sendbuf[i];
////        sendbuf[i] = (sendbuf[i] + recvbuf[i]) / 2;
//    }
//    return ncclSuccess;
//}

void reduce_mean_f32_inplace_host(float *input, int chunk_size, int num_chunks, int target_chunk, cudaStream_t stream) {
    reduce_chunk_inplace_host<float, true>(input, chunk_size, num_chunks, target_chunk, stream);
}

void reduce_mean_f16_inplace_host(half *input, int chunk_size, int num_chunks, int target_chunk, cudaStream_t stream) {
    reduce_chunk_inplace_host<half, true>(input, chunk_size, num_chunks, target_chunk, stream);
}

void reduce_sum_f32_inplace_host(float *input, int chunk_size, int num_chunks, int target_chunk, cudaStream_t stream) {
    reduce_chunk_inplace_host<float, false>(input, chunk_size, num_chunks, target_chunk, stream);
}

void reduce_sum_f16_inplace_host(half *input, int chunk_size, int num_chunks, int target_chunk, cudaStream_t stream) {
    reduce_chunk_inplace_host<half, false>(input, chunk_size, num_chunks, target_chunk, stream);
}
void compress_f32_to_uint8_host(float *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    compress_float_to_uint8_host(input, input_num_element, chunk_size, num_chunks, output, output_size, dev_buffer, dev_size, target_chunk, stream);
}

void decompress_uint8_to_f32_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, float *output,
                                  cudaStream_t stream) {
    decompress_uint8_to_float_host(input, input_size, chunk_size, num_chunks, output, stream);
}

void compress_f16_to_uint8_host(half *input, int input_num_element, int chunk_size, int num_chunks, uint8_t *output,
                                size_t output_size, void *dev_buffer, size_t dev_size, int target_chunk,
                                cudaStream_t stream) {
    compress_float_to_uint8_host(input, input_num_element, chunk_size, num_chunks, output, output_size, dev_buffer, dev_size, target_chunk, stream);
}

void decompress_uint8_to_f16_host(uint8_t *input, size_t input_size, int chunk_size, int num_chunks, half *output, 
		                  cudaStream_t stream) {
    decompress_uint8_to_float_host(input, input_size, chunk_size, num_chunks, output, stream);
}

size_t array_min_max_size_f32_host(float *input, int input_num_element, float *output, cudaStream_t stream) {
    return array_min_max_size(input, input_num_element, output, stream);
}

size_t array_min_max_size_f16_host(half *input, int input_num_element, half *output, cudaStream_t stream) {
    return array_min_max_size(input, input_num_element, output, stream);
}

}

ncclResult_t ncclAllToAll(void *sendbuf,
                          void *recvbuf,
                          size_t count,
                          ncclDataType_t datatype,
                          ncclComm_t comm,
                          int nranks,
                          int rank,
                          cudaStream_t stream) {
    if (sendbuf == recvbuf) {
        return ncclInvalidUsage;
    }

    // awkward workaround for nvcc bug
    intptr_t sendbuff = reinterpret_cast<intptr_t>(sendbuf);
    intptr_t recvbuff = reinterpret_cast<intptr_t>(recvbuf);
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nranks; ++r) {
        intptr_t r_sendbuf = sendbuff + r * count * word_size(datatype);
        intptr_t r_recvbuf = recvbuff + r * count * word_size(datatype);
        if (r != rank) {
            // awkward workaround for nvcc bug
            int peer = (int) r;
            NCCLCHECK(ncclSend(reinterpret_cast<const void *>(r_sendbuf), count, datatype, peer, comm, stream));
            NCCLCHECK(ncclRecv(reinterpret_cast<void *>(r_recvbuf), count, datatype, peer, comm, stream));
        } else {
            CUDACHECK(cudaMemcpyAsync(reinterpret_cast<void *>(r_recvbuf), reinterpret_cast<const void *>(r_sendbuf),
                            count * word_size(datatype), cudaMemcpyDeviceToDevice, stream));
        }
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
}
