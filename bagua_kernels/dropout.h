#include <ATen/ATen.h>
#ifdef OLD_GENERATOR
#include <ATen/CUDAGenerator.h>
#else
#include <ATen/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>
#include <stdio.h>

using namespace std; 

const int UNROLL = 4;


template <
        typename scalar_t,
        typename accscalar_t,
        typename IndexType
        >
__global__ void add_dropout_residual_train_kernel(scalar_t const                *inputs,
                                        scalar_t const                *bias,
                                        scalar_t const                *residual,
                                        scalar_t                      *outputs,
                                        uint8_t                       *mask,
                                        IndexType                      totalElements, 
                                        accscalar_t                    p, 
                                        std::pair<uint64_t, uint64_t>  seeds
                                        ) 
{
  accscalar_t pinv = accscalar_t(1.0)/p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  //printf("seeds: first: %f, second: %f", seeds.first, seeds.second);
  curandStatePhilox4_32_10_t state;
  curand_init(
      seeds.first,
      idx,
      seeds.second,
      &state);

  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
       float4 rand = curand_uniform4(&state);
       scalar_t src[UNROLL];
       scalar_t bias_src[UNROLL];
       scalar_t residual_src[UNROLL];
       rand.x = rand.x < p;
       rand.y = rand.y < p;
       rand.z = rand.z < p;
       rand.w = rand.w < p;
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               src[ii]     = inputs[li];
               bias_src[ii] = bias[li];
               residual_src[ii] = residual[li];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               accscalar_t int1 = (src[ii] + bias_src[ii]) * (&rand.x)[ii] * pinv;
               outputs[li] = static_cast<scalar_t>(static_cast<accscalar_t>(residual_src[ii]) + int1);
               mask[li]    = (uint8_t)(&rand.x)[ii];
           }
       }
       __syncthreads();
  }
}


template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
__global__ void add_dropout_residual_test_kernel(scalar_t const                *inputs,
                                scalar_t const                *bias,
                                scalar_t const                *residual,
                                scalar_t                      *outputs,
                                IndexType                      totalElements
                                ) 
{
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
       scalar_t src[UNROLL];
       scalar_t bias_src[UNROLL];
       scalar_t residual_src[UNROLL];
        
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               src[ii]          = inputs[li];
               bias_src[ii]     = bias[li];
               residual_src[ii] = residual[li];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
	           outputs[li] = src[ii] + bias_src[ii] + residual_src[ii];
           }
       }
       __syncthreads();
  }
}


template<typename scalar_t, 
         typename accscalar_t, 
         typename IndexType
        >
__global__ void bagua_masked_scale_kernel(scalar_t const *inputs, 
                                         scalar_t       *add_dropout_inputs_grad_outputs,
                                        //  scalar_t       *add_dropout_bias_grad_outputs,
                                         uint8_t const  *mask, 
                                         IndexType       totalElements,
                                         accscalar_t     scale
                                        )
{
  IndexType idx          = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) * blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) 
  {
       scalar_t add_dropout_inputs_grad_src[UNROLL];
    //    scalar_t add_dropout_bias_grad_src[UNROLL];
       scalar_t msk[UNROLL];
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
               add_dropout_inputs_grad_src[ii] = static_cast<scalar_t>(inputs[li]);
            //    add_dropout_bias_grad_src[ii] = static_cast<scalar_t>(inputs[li]);
               msk[ii] = static_cast<scalar_t>(mask[li]);
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
            //    add_dropout_inputs_grad_outputs[li] = static_cast<accscalar_t>(add_dropout_inputs_grad_src[ii]) * scale * static_cast<accscalar_t>(msk[ii]);
            //    add_dropout_bias_grad_outputs[li] = static_cast<accscalar_t>(add_dropout_bias_grad_src[ii] * scale * static_cast<accscalar_t>(msk[ii]));
               add_dropout_inputs_grad_outputs[li] = add_dropout_inputs_grad_src[ii] * scale * msk[ii];
            //    add_dropout_bias_grad_outputs[li] = add_dropout_bias_grad_src[ii] * scale * msk[ii];
           }
       }
  }
}

template <
        typename scalar_t,
        typename accscalar_t,
        typename IndexType
        >
void add_dropout_residual_train(scalar_t const *inputs,
                           scalar_t const *bias,
                           scalar_t const *residual,
                           scalar_t       *outputs,
                           uint8_t        *mask,
                           IndexType       totalElements, 
                           accscalar_t     p)
{
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();

  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size -1)/block_size);
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  int64_t counter_offset = ((totalElements - 1)/(block_size * grid.x * UNROLL) + 1) * UNROLL;
  
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
#ifdef OLD_GENERATOR
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
#else
    // std::lock_guard<std::mutex> lock(gen->mutex_);
    // rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
    std::lock_guard<std::mutex> lock(gen.mutex());
    rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(counter_offset);
#endif
  }

  add_dropout_residual_train_kernel<scalar_t, accscalar_t, IndexType><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(inputs, bias, residual, outputs, mask, totalElements, p, rng_engine_inputs);
  C10_CUDA_CHECK(cudaGetLastError());
}


template <typename scalar_t,
          typename accscalar_t,
          typename IndexType
         >
void add_dropout_residual_test(scalar_t const *inputs,
                   scalar_t const *bias,
                   scalar_t const *residual,
                   scalar_t       *outputs,
                   IndexType       totalElements
		          )
{
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size -1)/block_size);
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  add_dropout_residual_test_kernel<scalar_t, accscalar_t, IndexType><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(inputs, bias, residual, outputs, totalElements);
  C10_CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t,
         typename accscalar_t,
         typename IndexType
        >
void bagua_masked_scale_cuda(scalar_t const *inputs, 
                            scalar_t       *add_dropout_inputs_grad_outputs,
                            // scalar_t       *add_dropout_bias_grad_outputs,
                            uint8_t const  *mask, 
                            IndexType       totalElements,
                            accscalar_t     scale
                         )
{
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size -1)/block_size);
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  bagua_masked_scale_kernel<scalar_t, accscalar_t, IndexType><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(inputs, add_dropout_inputs_grad_outputs, mask, totalElements, scale);
  C10_CUDA_CHECK(cudaGetLastError());
}