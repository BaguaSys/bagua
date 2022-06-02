#include <vector>
#include <math.h>
#include <iostream>
using namespace std; 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "dropout.h"

namespace add_dropout_residual {

std::vector<torch::Tensor> fwd_cuda(torch::Tensor const& inputs,
                      torch::Tensor const& bias,
                      torch::Tensor const& residual,
                      float                dropout_prob,
                      bool                 is_training
                      )
{
  auto act_options                = inputs.options().requires_grad(false);
  auto mask_options               = act_options.dtype(torch::kUInt8);
  torch::Tensor outputs           = torch::empty_like(inputs, act_options);
  torch::Tensor dropout_add_mask  = torch::empty_like(inputs, mask_options);
  const int total_tokens   = inputs.size(0) * inputs.size(1) * inputs.size(2);
  
  if (is_training) {
    add_dropout_residual_train<at::Half,float,uint32_t>(
                          static_cast<at::Half const*>(inputs.data_ptr()), 
                          static_cast<at::Half const*>(bias.data_ptr()),
                          static_cast<at::Half const*>(residual.data_ptr()), 
                          static_cast<at::Half*>(outputs.data_ptr()), 
                          static_cast<uint8_t*>(dropout_add_mask.data_ptr()),
                          total_tokens,
                          (1.0f - dropout_prob));
    }
    else{
    add_dropout_residual_test<at::Half,float,uint32_t>(
                              static_cast<at::Half const*>(inputs.data_ptr()), 
                              static_cast<at::Half const*>(bias.data_ptr()), 
                              static_cast<at::Half const*>(residual.data_ptr()), 
                              static_cast<at::Half*>(outputs.data_ptr()), 
                              total_tokens);
  }
  return {outputs,
          dropout_add_mask
  };
}



std::vector<torch::Tensor> bwd_cuda(torch::Tensor const& output_grads,
                      torch::Tensor const& dropout_add_mask,
                      float                dropout_prob
                      )
{
    const int total_tokens   = output_grads.size(0) * output_grads.size(1) * output_grads.size(2);
    torch::Tensor add_dropout_inputs_grads = torch::empty_like(output_grads);
    // torch::Tensor add_dropout_bias_grads = torch::empty_like(output_grads);


    bagua_masked_scale_cuda<at::Half,float,uint32_t>(
                                static_cast<at::Half const*>(output_grads.data_ptr()),
                                static_cast<at::Half*>(add_dropout_inputs_grads.data_ptr()),
                                // static_cast<at::Half*>(add_dropout_bias_grads.data_ptr()),
                                static_cast<uint8_t const*>(dropout_add_mask.data_ptr()),
                                total_tokens,
                                (1.0 / (1.0 - dropout_prob)));
    return {add_dropout_inputs_grads,
            add_dropout_inputs_grads.clone()};
}

}