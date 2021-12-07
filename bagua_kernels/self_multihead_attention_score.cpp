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
namespace self {
namespace attention_score {

std::vector<torch::Tensor> fwd_cuda(
    bool                 is_training,
    int                  heads,
    torch::Tensor const& inputs,
    const uint8_t*       attention_mask,
    float                coeff,
    float                dropout_prob
    );

std::vector<torch::Tensor> bwd_cuda(
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& dropout_results,
    torch::Tensor const& softmax_results,
    torch::Tensor const& inputs,
    float                coeff,
    torch::Tensor const& dropout_mask,
    float                dropout_prob
    );

std::vector<torch::Tensor> fwd(
    bool                 use_mask,
    bool                 is_training,
    int                  heads,
    torch::Tensor const& inputs,
    torch::Tensor const& attention_mask,
    float                coeff,
    float                dropout_prob
    ) {

  AT_ASSERTM(inputs.dim()          == 3, "expected 3D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    AT_ASSERTM(attention_mask.dim()                     == 2,                    "expected 2D tensor");
    AT_ASSERTM(attention_mask.type().scalarType()       == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(
	  is_training,
          heads,
          inputs,
          use_mask ? static_cast<const uint8_t*>(attention_mask.data_ptr()) : nullptr,
          coeff,
          dropout_prob);
}

std::vector<torch::Tensor> bwd(
    bool                 use_mask,
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& dropout_results,
    torch::Tensor const& softmax_results,
    torch::Tensor const& inputs,
    float                coeff,
    torch::Tensor const& dropout_mask,
    float                dropout_prob
    ) {

  AT_ASSERTM(output_grads.dim()    == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs.dim()          == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_mask.dim()    == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.scalar_type()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads,
          output_grads,
          dropout_results,
          softmax_results,
          inputs,
          coeff,
          dropout_mask,
          dropout_prob);
}

} // end namespace attention_score
} // end namespace self
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", 
        &multihead_attn::self::attention_score::fwd,
	"Self Multihead Attention Scaled Dot-Product Attention Score Forward");
  m.def("backward", 
        &multihead_attn::self::attention_score::bwd,
	"Self Multihead Attention Scaled Dot-Product Attention Score Backward");
}
