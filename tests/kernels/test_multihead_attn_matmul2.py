import torch
import unittest
import math
from bagua.torch_api.contrib.fuse.multihead_attn import MultiheadAttnMatmul2Func


class NaiveAttnMatMul2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs, attention_probs):

        embedding_dim = inputs.size(2) // 2
        head_dim = embedding_dim // heads

        # inputs: [seql_q, batches=seqs*heads, 2, head_dim]
        # attention_probs: [batches=seqs*heads, seql_q, seql_k]
        inputs_view = inputs.view(inputs.size(0), inputs.size(1) * heads, 2, head_dim)
        keys = inputs_view[:, :, 0, :]
        values = inputs_view[:, :, 1, :]

        # allocate memory: [seql_q, batches=seqs*heads, head_dim]
        matmul2_results = torch.empty(
            (attention_probs.size(1), attention_probs.size(0), values.size(2)),
            dtype=attention_probs.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)

        # Matmul2
        # Input1: (activation)  [seqs*heads, seql_q, seql_k]
        # Input2: (values)      [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim]
        matmul2_results = torch.bmm(
            attention_probs, values.transpose(0, 1), out=matmul2_results
        )

        matmul2_results = matmul2_results.transpose(0, 1).contiguous()
        # view: [seql_q, seqs, heads* head_dim]
        matmul2_results = matmul2_results.view(inputs.size(0), inputs.size(1), -1)

        heads_t = torch.tensor([heads])
        ctx.save_for_backward(heads_t, inputs, attention_probs)

        return matmul2_results

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs, attention_probs = ctx.saved_tensors

        embedding_dim = inputs.size(2) // 2
        head_dim = embedding_dim // heads_t[0]

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # inputs: [seql_q, batches=seqs*heads, 2, head_dim]
        inputs_view = inputs.view(
            inputs.size(0), inputs.size(1) * heads_t[0], 2, head_dim
        )
        keys = inputs_view[:, :, 0, :]
        values = inputs_view[:, :, 1, :]

        # Slice out q,k,v from one big set of gradients entering the input linear's bprop  (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        inputs_grads = torch.empty_like(inputs)
        inputs_grads_view = inputs_grads.view(
            inputs.size(0), inputs.size(1) * heads_t[0], 2, head_dim
        )
        keys_grads = inputs_grads_view[:, :, 0, :]
        values_grads = inputs_grads_view[:, :, 1, :]

        # attention_probs: [batches=seqs*heads, seql_q, seql_k]
        attention_probs_grads = torch.empty_like(attention_probs)

        # output_grads: [seql_q, seqs, heads* head_dim] -> [seql_q, batches=seqs*heads, head_dim]
        output_grads = output_grads.view(
            output_grads.size(0),
            output_grads.size(1) * heads_t[0],
            -1,
        ).transpose(0, 1)

        # Matmul2 - DGRAD1
        # Input1: (data grads)  [seqs*heads, seql_q, head_dim]
        # Input2: (values)      [seql_k, seqs*heads, head_dim] transpose(0,1).transpose(1,2)
        # Output:               [seqs*heads, seql_q, seql_k]
        attention_probs_grads = torch.bmm(
            output_grads,
            values.transpose(0, 1).transpose(1, 2),
            out=attention_probs_grads,
        )
        # Matmul2 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, head_dim]
        # Input2: (activations) [batches=seqs*heads, seql_q, seql_k] transpose(1,2)
        # Output:               [seqs*heads, seql_k, head_dim]
        values_grads = torch.bmm(
            attention_probs.transpose(1, 2),
            output_grads,
            out=values_grads.transpose(0, 1),
        )

        return None, inputs_grads, attention_probs_grads


def construct_inputs(seed=47):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    seq_length = 80
    sequences = 10
    hidden_dim = 1024
    heads = 16

    inputs = torch.randn(
        seq_length,
        sequences,
        hidden_dim * 2,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)

    attention_probs = torch.randn(
        sequences * heads,
        seq_length,
        seq_length,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    return inputs, attention_probs


class TestSelfMultiheadAttn(unittest.TestCase):
    def test_self_multihead_attn(self):

        ref_inputs, ref_attention_probs = construct_inputs()
        tst_inputs, tst_attention_probs = construct_inputs()

        grads = torch.randn_like(tst_inputs)

        ref_outputs = NaiveAttnMatMul2Func.apply(
            16, ref_inputs, ref_attention_probs
        )
        tst_outputs = MultiheadAttnMatmul2Func.apply(
            16, tst_inputs, tst_attention_probs
        )

        ref_inputs.backward(grads)
        tst_inputs.backward(grads)

        self.assertTrue(torch.equal(ref_inputs, tst_inputs))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(
            torch.allclose(ref_inputs.grad, tst_inputs.grad, atol=1e-3, rtol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
