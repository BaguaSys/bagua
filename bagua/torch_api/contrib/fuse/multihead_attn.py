import torch
import bagua_multihead_attn_raw_cuda
import bagua_self_multihead_attn_raw_cuda


class MultiheadAttnRawScoreFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs_q, inputs_kv):
        heads_t = torch.tensor([heads])

        (outputs,) = bagua_multihead_attn_raw_cuda.forward(heads, inputs_q, inputs_kv)
        ctx.save_for_backward(heads_t, inputs_q, inputs_kv)
        return outputs

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs_q, inputs_kv = ctx.saved_tensors

        inputs_q_grads, inputs_kv_grads = bagua_multihead_attn_raw_cuda.backward(
            heads_t[0], output_grads, inputs_q, inputs_kv
        )

        return None, inputs_q_grads, inputs_kv_grads


class SelfMultiheadAttnRawScoreFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs, scale):
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])

        (outputs,) = bagua_self_multihead_attn_raw_cuda.forward(heads, inputs, scale)
        ctx.save_for_backward(heads_t, inputs, scale_t)
        return outputs

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs, scale_t = ctx.saved_tensors

        (inputs_grads,) = bagua_self_multihead_attn_raw_cuda.backward(
            heads_t[0], output_grads, inputs, scale_t[0]
        )

        return None, inputs_grads
