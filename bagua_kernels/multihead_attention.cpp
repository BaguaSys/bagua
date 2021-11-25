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
namespace multihead_attention {

torch::Tensor fwd_cuda(
    int                  heads,
    torch::Tensor const& inputs_q,
    torch::Tensor const& inputs_kv
    );

torch::Tensor bwd_cuda(
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& inputs_q,
    torch::Tensor const& inputs_kv
    );

torch::Tensor fwd(
    int                  heads,
    torch::Tensor const& inputs_q,
    torch::Tensor const& inputs_kv) {

  AT_ASSERTM(inputs_q.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()        == 3, "expected 3D tensor");

  AT_ASSERTM(inputs_q.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");

  return fwd_cuda(heads, inputs_q, inputs_kv);
}

torch::Tensor bwd(
    int                  heads,
    torch::Tensor const& output_grads,
    torch::Tensor const& inputs_q,
    torch::Tensor const& inputs_kv) {

  AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_q.dim()     == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()    == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.scalar_type() == at::ScalarType::Half, 
      "Only HALF is supported");
  AT_ASSERTM(inputs_q.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");

  return bwd_cuda(heads, output_grads, inputs_q, inputs_kv);
}

} // end namespace multihead_attention
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", 
        &multihead_attn::multihead_attention::fwd,
	"Multihead Attention -- Forward.");
  m.def("backward", 
        &multihead_attn::multihead_attention::bwd,
	"Multihead Attention -- Backward.");
}
