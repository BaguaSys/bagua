#include <vector>
#include <math.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "strided_batched_gemm.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

namespace multihead_attn {
namespace cross {
namespace matmul1 {

std::vector<torch::Tensor> fwd_cuda(
                               int                  heads,
                               torch::Tensor const& inputs_q,
                               torch::Tensor const& inputs_k
                             )
{
  const int   embed_dim      = inputs_q.size(2);
  const int   sequences      = inputs_q.size(1);
  const int   q_seq_len      = inputs_q.size(0);
  const int   k_seq_len      = inputs_k.size(0);
  const int   head_dim       = embed_dim / heads;

  const int   attn_batches      = heads * sequences;
  const int   lead_dim_q        = attn_batches * head_dim;
  const int   lead_dim_k        = attn_batches * head_dim;
  const int   batch_stride_q    = head_dim;
  const int   batch_stride_k    = head_dim;

  const float beta           = 0.0;
  const float scale          = 1.0 / sqrt(static_cast<float>(head_dim));

  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  auto act_options  = inputs_q.options().requires_grad(false);
  torch::Tensor outputs   = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);

  void* outputs_ptr = static_cast<void*>(outputs.data_ptr());

  char a_layout_t{'t'};
  char b_layout_n{'n'};

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // MatMul1 of Dot-Product Attention Plus scaling by 1/Sqrt(head size)
  gemm_switch_fp32accum(     state,
                             a_layout_t,
                             b_layout_n,
                             k_seq_len,
                             q_seq_len,
                             head_dim,
                             scale,
                             static_cast<const half*>(inputs_k.data_ptr()),
                             lead_dim_k,
                             batch_stride_k,
                             static_cast<const half*>(inputs_q.data_ptr()),
                             lead_dim_q,
                             batch_stride_q,
                             beta,
                             static_cast<half*>(outputs_ptr),
                             k_seq_len,
                             k_seq_len * q_seq_len,
                             attn_batches);

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {
      outputs
  };
}

std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& inputs_q,
                               torch::Tensor const& inputs_k
                                   )
{
  const int   embed_dim         = inputs_q.size(2);
  const int   sequences         = inputs_q.size(1);
  const int   q_seq_len         = inputs_q.size(0);
  const int   k_seq_len         = inputs_k.size(0);
  const int   head_dim          = embed_dim / heads;

  const int   attn_batches      = heads * sequences;
  const int   lead_dim_q        = attn_batches * head_dim;
  const int   lead_dim_k        = attn_batches * head_dim;
  const int   batch_stride_q    = head_dim;
  const int   batch_stride_k    = head_dim;

  const float beta           = 0.0;
  const float scale          = 1.0 / sqrt(static_cast<float>(head_dim));

  // TODO: Streams can be used in Backprop but I haven't added more than one
  // in my first attempt to create the code
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Output Tensor Allocations
  torch::Tensor inputs_q_grads   = torch::zeros_like(inputs_q);
  torch::Tensor inputs_k_grads   = torch::zeros_like(inputs_k);

  auto inputs_q_ptr = static_cast<half*>(inputs_q.data_ptr());
  auto inputs_k_ptr = static_cast<half*>(inputs_k.data_ptr());

  auto inputs_q_grads_ptr = static_cast<half*>(inputs_q_grads.data_ptr());
  auto inputs_k_grads_ptr = static_cast<half*>(inputs_k_grads.data_ptr());

  char a_layout_n{'n'};
  char b_layout_n{'n'};
  char b_layout_t{'t'};

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // Matmul1 Dgrad1
  gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_n,
                             head_dim,
                             q_seq_len,
                             k_seq_len,
                             scale,
                             inputs_k_ptr,
                             lead_dim_k,
                             batch_stride_k,
                             static_cast<half*>(output_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             beta,
                             inputs_q_grads_ptr,
                             lead_dim_q,
                             batch_stride_q,
                             attn_batches);

  // Matmul1 Dgrad2
  gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_t,
                             head_dim,
                             k_seq_len,
                             q_seq_len,
                             scale,
                             inputs_q_ptr,
                             lead_dim_q,
                             batch_stride_q,
                             static_cast<half*>(output_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             beta,
                             inputs_k_grads_ptr,
                             lead_dim_k,
                             batch_stride_k,
                             attn_batches);

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {
           inputs_q_grads,
           inputs_k_grads
         };
}

} // end namespace matmul1
} // end cross
} // end namespace multihead_attn
