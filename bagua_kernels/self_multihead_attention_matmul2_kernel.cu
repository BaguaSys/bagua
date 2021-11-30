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
#include "softmax.h"
#include "dropout.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

namespace multihead_attn {
namespace self {
namespace matmul2 {

std::vector<torch::Tensor> fwd_cuda(
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& attention_probs
                             )
{

  // Embedding of Q, K and V
  const int   embed_dim      = inputs.size(2) / 3;
  const int   sequences      = inputs.size(1);
  const int   q_seq_len      = inputs.size(0);
  const int   k_seq_len      = q_seq_len;
  const int   head_dim       = embed_dim / heads;

  const int   attn_batches   = heads * sequences;
  const int   lead_dim       = attn_batches * 3 * head_dim;
  const int   batch_stride   = 3 * head_dim;

  const float alpha          = 1.0;
  const float beta           = 0.0;

  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  auto act_options  = inputs.options().requires_grad(false);

  torch::Tensor outputs           = torch::empty({q_seq_len, attn_batches, head_dim},    act_options);

  // Input Linear Results Pointers to Q, K, and V of interviewed activations
  void* inputs_q_ptr   = static_cast<void*>(inputs.data_ptr());
  void* inputs_k_ptr   = static_cast<void*>(static_cast<half*>(inputs.data_ptr()) + head_dim);
  void* inputs_v_ptr   = static_cast<void*>(static_cast<half*>(inputs.data_ptr()) + 2 * head_dim);

  char a_layout_t{'t'};
  char a_layout_n{'n'};
  char b_layout_n{'n'};

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // Matmul2
  // attention_probs: {attn_batches, q_seq_len, k_seq_len}
  // V: {k_seq_len, attn_batches, head_dim}
  gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_n,
                             head_dim,
                             q_seq_len,
                             k_seq_len,
                             alpha,
                             static_cast<const half*>(inputs_v_ptr),
                             lead_dim,
                             batch_stride,
                             static_cast<const half*>(attention_probs.data_ptr()),
                             k_seq_len,
                             k_seq_len * q_seq_len,
                             beta,
                             static_cast<half*>(outputs.data_ptr()),
                             head_dim * attn_batches,
                             head_dim,
                             attn_batches);

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {
      outputs
  };
}

std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& attention_probs
                                   )
{
  const int   embed_dim      = inputs.size(2) / 3;
  const int   sequences      = inputs.size(1);
  const int   q_seq_len      = inputs.size(0);
  const int   k_seq_len      = q_seq_len;
  const int   head_dim       = embed_dim / heads;

  const int   attn_batches   = heads * sequences;
  const int   lead_dim       = attn_batches * 3 * head_dim;
  const int   batch_stride   = 3 * head_dim;

  const float alpha          = 1.0;
  const float beta           = 0.0;

  // TODO: Streams can be used in Backprop but I haven't added more than one
  // in my first attempt to create the code
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Output Tensor Allocations
  torch::Tensor inputs_grads   = torch::empty_like(inputs);
  torch::Tensor attention_probs_grads   = torch::empty_like(attention_probs);

  auto inputs_q_ptr = static_cast<half*>(inputs.data_ptr());
  auto inputs_k_ptr = static_cast<half*>(inputs.data_ptr()) + head_dim;
  auto inputs_v_ptr = static_cast<half*>(inputs.data_ptr()) + 2 * head_dim;

  auto inputs_q_grads_ptr = static_cast<half*>(inputs_grads.data_ptr());
  auto inputs_k_grads_ptr = static_cast<half*>(inputs_grads.data_ptr()) + head_dim;
  auto inputs_v_grads_ptr = static_cast<half*>(inputs_grads.data_ptr()) + 2 * head_dim;

  char a_layout_n{'n'};
  char a_layout_t{'t'};
  char b_layout_n{'n'};
  char b_layout_t{'t'};

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // MatMul2 Dgrad1
  gemm_switch_fp32accum(     state,
                             a_layout_t,
                             b_layout_n,
                             k_seq_len,
                             q_seq_len,
                             head_dim,
                             alpha,
                             static_cast<const half*>(inputs_v_ptr),
                             lead_dim,
                             batch_stride,
                             static_cast<const half*>(output_grads.data_ptr()),
                             head_dim * attn_batches,
                             head_dim,
                             beta,
                             static_cast<half*>(attention_probs_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len * q_seq_len,
                             attn_batches);

  // Matmul2 Dgrad2
  gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_t,
                             head_dim,
                             k_seq_len,
                             q_seq_len,
                             alpha,
                             static_cast<const half*>(output_grads.data_ptr()),
                             head_dim * attn_batches,
                             head_dim,
                             static_cast<const half*>(attention_probs.data_ptr()),
                             k_seq_len,
                             k_seq_len * q_seq_len,
                             beta,
                             inputs_v_grads_ptr,
                             lead_dim,
                             batch_stride,
                             attn_batches);

  BAGUA_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {
      inputs_grads,
      attention_probs_grads
  };
}

} // end namespace matmul2
} // end namespace self
} // end namespace multihead_attn
