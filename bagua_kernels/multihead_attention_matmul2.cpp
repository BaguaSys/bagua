/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace cross {
namespace matmul2 {

std::vector<torch::Tensor> fwd_cuda(
    int                  heads,
    torch::Tensor const& inputs,
    torch::Tensor const& attention_probs
    );

std::vector<torch::Tensor> bwd_cuda(
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& inputs,
    torch::Tensor const& attention_probs
    );

std::vector<torch::Tensor> fwd(
    int                  heads,
    torch::Tensor const& inputs,
    torch::Tensor const& attention_probs
    ) {

  AT_ASSERTM(inputs.dim()          == 3, "expected 3D tensor");
  AT_ASSERTM(attention_probs.dim()  == 3, "expected 3D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(attention_probs.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");

  return fwd_cuda(heads,
          inputs,
          attention_probs);
}

std::vector<torch::Tensor> bwd(
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& inputs,
    torch::Tensor const& attention_probs
    ) {

  AT_ASSERTM(output_grads.dim()    == 3, "expected 3D tensor");
  AT_ASSERTM(inputs.dim()          == 3, "expected 3D tensor");
  AT_ASSERTM(attention_probs.dim()  == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.scalar_type()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(attention_probs.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");

  return bwd_cuda(heads,
          output_grads,
          inputs,
          attention_probs);
}

} // end namespace matmul2
} // end namespace cross
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", 
        &multihead_attn::cross::matmul2::fwd,
	"Multihead Attention Matmul2 (Softmax(·) * V) Forward");
  m.def("backward", 
        &multihead_attn::cross::matmul2::bwd,
	"Multihead Attention Matmul2 (Softmax(·) * V) Backward");
}
