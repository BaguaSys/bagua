#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace std; 

namespace add_dropout_residual {

std::vector<torch::Tensor> fwd_cuda(torch::Tensor const& inputs,
                      torch::Tensor const& bias,
                      torch::Tensor const& residual,
                      float                dropout_prob,
                      bool                 is_training
                      );

std::vector<torch::Tensor> bwd_cuda(torch::Tensor const& output_grads, 
                                    torch::Tensor const& dropout_add_mask,
                                    float                dropout_prob
                                    );

std::vector<torch::Tensor> fwd(torch::Tensor const& inputs,
                  torch::Tensor const& bias,
                  torch::Tensor const& residual,
                  float                dropout_prob,
                  bool                 is_training
                  )
{ 
  AT_ASSERTM(inputs.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(bias.dim()  == inputs.dim(), "bias and inputs must have the same dimensions");
  AT_ASSERTM(residual.dim() == 3, "expected 3D tensor");

  AT_ASSERTM(inputs.type().scalarType()                == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(bias.type().scalarType()                  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(residual.type().scalarType()              == at::ScalarType::Half, "Only HALF is supported");

  return fwd_cuda(inputs,
                  bias,
                  residual,
                  dropout_prob,
                  is_training
                  );
}


std::vector<torch::Tensor> bwd(torch::Tensor const& output_grads, 
                              torch::Tensor const& dropout_add_mask,
                              float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()          == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_add_mask.dim()      == 3, "expected 3D tensor");
  
  AT_ASSERTM(output_grads.type().scalarType()          == at::ScalarType::Half,  "Only HALF is supported");
  AT_ASSERTM(dropout_add_mask.type().scalarType()      == at::ScalarType::Byte,  "Only BYTE is supported");
  
  return bwd_cuda(output_grads, 
                  dropout_add_mask,
                  dropout_prob
                  );
}

} // end namespace add_dropout_residual

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_dropout_residual::fwd, "Dropout with inputs adding bias and Residual add Forward.");
  m.def("backward", &add_dropout_residual::bwd, "Dropout with inputs adding bias and Residual add Backward.");
}