import torch
import unittest
import math
from bagua.torch_api.contrib.fuse.multihead_attn_func import (
    SelfMultiheadAttnMatmul1Func,
    SelfMultiheadAttnMatmul2Func,
)


class NaiveSelfAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs):

        embedding_dim = inputs.size(2) // 3
        head_dim = embedding_dim // heads
        scale = 1.0 / math.sqrt(head_dim)

        # inputs: [seql_q, batches=seqs*heads, 3, head_dim]
        inputs_view = inputs.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = inputs_view[:, :, 0, :]
        keys = inputs_view[:, :, 1, :]
        values = inputs_view[:, :, 2, :]

        # Matmul1
        # The output tensor is specified prior to the Batch GEMM because baddbmm requires its specification
        # baddbmm is used to apply the scale parameter via the Batched GEMM's alpha parameter instead of
        # a separate elementwise operation.
        # Input1: (Queries) [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Input2: (Keys)    [seql_k, seqs*heads, head_dim] transpose(0,1) transpose(1,2)
        # output:           [seqs*heads, seql_q, seql_k]
        matmul1_results = torch.empty(
            (queries.size(1), queries.size(0), keys.size(0)),
            dtype=queries.dtype,
            device=torch.device("cuda"),
        )
        matmul1_results = torch.baddbmm(
            matmul1_results,
            queries.transpose(0, 1),
            keys.transpose(0, 1).transpose(1, 2),
            out=matmul1_results,
            beta=0.0,
            alpha=scale,
        )

        # Matmul2
        # Input1: (activation)  [seqs*heads, seql_q, seql_k]
        # Input2: (values)      [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim]
        # allocate memory: [seql_q, batches=seqs*heads, head_dim]
        matmul2_results = torch.empty(
            (matmul1_results.size(1), matmul1_results.size(0), values.size(2)),
            dtype=matmul1_results.dtype,
            device=torch.device("cuda"),
        ).transpose(1, 0)
        matmul2_results = torch.bmm(
            matmul1_results, values.transpose(0, 1), out=matmul2_results
        )

        matmul2_results = matmul2_results.transpose(0, 1).contiguous()
        # view: [seql_q, seqs, heads* head_dim]
        matmul2_results = matmul2_results.view(inputs.size(0), inputs.size(1), -1)

        heads_t = torch.tensor([heads])
        ctx.save_for_backward(heads_t, inputs, matmul1_results)

        return matmul2_results

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs, matmul1_results = ctx.saved_tensors

        embedding_dim = inputs.size(2) // 3
        head_dim = embedding_dim // heads_t[0]
        scale = 1.0 / math.sqrt(head_dim)

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # inputs: [seql_q, batches=seqs*heads, 3, head_dim]
        inputs_view = inputs.view(
            inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim
        )
        queries = inputs_view[:, :, 0, :]
        keys = inputs_view[:, :, 1, :]
        values = inputs_view[:, :, 2, :]

        # Slice out q,k,v from one big set of gradients entering the input linear's bprop  (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        inputs_grads = torch.empty_like(inputs)
        inputs_grads_view = inputs_grads.view(
            inputs_grads.size(0), inputs_grads.size(1) * heads_t[0], 3, -1
        )
        queries_grads = inputs_grads_view[:, :, 0, :]
        keys_grads = inputs_grads_view[:, :, 1, :]
        values_grads = inputs_grads_view[:, :, 2, :]

        # matmul1_results: [batches=seqs*heads, seql_q, seql_k]
        matmul1_results_grads = torch.empty_like(matmul1_results)

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
        matmul1_results_grads = torch.bmm(
            output_grads,
            values.transpose(0, 1).transpose(1, 2),
            out=matmul1_results_grads,
        )
        # Matmul2 - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, head_dim]
        # Input2: (activations) [batches=seqs*heads, seql_q, seql_k] transpose(1,2)
        # Output:               [seqs*heads, seql_k, head_dim]
        values_grads = torch.bmm(
            matmul1_results.transpose(1, 2),
            output_grads,
            out=values_grads.transpose(0, 1),
        )

        # Matmul - DGRAD1
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k]
        # Input2: (keys)        [seql_k, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_q, head_dim] transpose(0,1)
        queries_grads = torch.baddbmm(
            queries_grads.transpose(0, 1),
            matmul1_results_grads,
            keys.transpose(0, 1),
            out=queries_grads.transpose(0, 1),
            beta=0.0,
            alpha=scale,
        )
        # Matmul - DGRAD2
        # Input1: (data grads)  [seqs*heads, seql_q, seql_k] transpose(1,2)
        # Input2: (queries)     [seql_q, seqs*heads, head_dim] transpose(0,1)
        # Output:               [seqs*heads, seql_k, head_dim] transpose(0,1)
        keys_grads = torch.baddbmm(
            keys_grads.transpose(0, 1),
            matmul1_results_grads.transpose(1, 2),
            queries.transpose(0, 1),
            out=keys_grads.transpose(0, 1),
            beta=0.0,
            alpha=scale,
        )

        return None, inputs_grads


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
        hidden_dim * 3,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    return inputs


class TestSelfMultiheadAttn(unittest.TestCase):
    def test_self_multihead_attn(self):

        ref_inputs = construct_inputs()
        tst_inputs = construct_inputs()

        ref_outputs = NaiveSelfAttnFunc.apply(16, ref_inputs)

        tst_matmul1 = SelfMultiheadAttnMatmul1Func.apply(16, tst_inputs, 1.0)
        tst_outputs = SelfMultiheadAttnMatmul2Func.apply(16, tst_inputs, tst_matmul1)

        ref_outputs.sum().backward()
        tst_outputs.sum().backward()

        self.assertTrue(torch.equal(ref_inputs, tst_inputs))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(
            torch.allclose(ref_inputs.grad, tst_inputs.grad, atol=1e-3, rtol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
