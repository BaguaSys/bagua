import torch
import bagua_multihead_attn_cuda
import bagua_self_multihead_attn_cuda


class FusedMultiheadAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs_q, inputs_kv):
        heads_t = torch.tensor([heads])

        (outputs,) = bagua_multihead_attn_cuda.forward(heads, inputs_q, inputs_kv)
        ctx.save_for_backward(heads_t, inputs_q, inputs_kv)
        return outputs

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs_q, inputs_kv = ctx.saved_tensors

        inputs_q_grads, inputs_kv_grads = bagua_multihead_attn_cuda.backward(
            heads_t[0], output_grads, inputs_q, inputs_kv
        )

        return None, inputs_q_grads, inputs_kv_grads


class FusedSelfMultiheadAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs):
        heads_t = torch.tensor([heads])

        (outputs,) = bagua_self_multihead_attn_cuda.forward(heads, inputs)
        ctx.save_for_backward(heads_t, inputs)
        return outputs

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs = ctx.saved_tensors

        (inputs_grads,) = bagua_self_multihead_attn_cuda.backward(
            heads_t[0], output_grads, inputs
        )

        return None, inputs_grads
