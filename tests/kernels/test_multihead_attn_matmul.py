import torch
import unittest
import math
from bagua.torch_api.contrib.fuse.multihead_attn import (
    MultiheadAttnMatmul1Func,
    MultiheadAttnMatmul2Func,
)


class NaiveAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heads, inputs_q, inputs_k, inputs_v):

        embedding_dim = inputs_q.size(2)
        head_dim = embedding_dim // heads
        scale = 1.0 / math.sqrt(head_dim)

        # inputs: [seql_q, batches=seqs*heads, head_dim]
        queries = inputs_q.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)
        keys = inputs_k.view(inputs_k.size(0), inputs_k.size(1) * heads, head_dim)
        values = inputs_v.view(inputs_v.size(0), inputs_v.size(1) * heads, head_dim)

        # Matmul1 Batched GEMMs
        # The output tensor is specified prior to the Batch GEMM because baddbmm requires its specification
        # baddbmm is used to apply the scale parameter via the Batched GEMM's alpha parameter instead of
        # a separate elementwise operation.
        # Input1: (Queries) [seql_q, seqs*heads, head_dim] tranpose(0,1)
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
        matmul2_results = matmul2_results.view(inputs_q.size(0), inputs_q.size(1), -1)

        heads_t = torch.tensor([heads])
        ctx.save_for_backward(heads_t, inputs_q, inputs_k, inputs_v, matmul1_results)

        return matmul2_results

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, inputs_q, inputs_k, inputs_v, matmul1_results = ctx.saved_tensors

        embedding_dim = inputs_q.size(2)
        head_dim = embedding_dim // heads_t[0]
        scale = 1.0 / math.sqrt(head_dim)

        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        # Sequences and heads are combined to make the batch of the Batched GEMM
        # input_lin_results: [seql_q, seqs, heads(16), 3, head_dim(64)]
        # input_lin_results: [seql_q, batches=seqs*heads, 3, head_dim]
        queries = inputs_q.view(
            inputs_q.size(0), inputs_q.size(1) * heads_t[0], head_dim
        )
        keys = inputs_k.view(inputs_k.size(0), inputs_k.size(1) * heads_t[0], head_dim)
        values = inputs_v.view(
            inputs_v.size(0), inputs_v.size(1) * heads_t[0], head_dim
        )

        # Slice out q,k,v from one big set of gradients entering the input linear's bprop  (should only impact meta data, no copies!)
        # The gradients are identical in size to the Input Linear outputs.
        # The tensor is declared before hand to properly slice out query, key, and value grads.
        inputs_q_grads = torch.empty_like(inputs_q)
        inputs_k_grads = torch.empty_like(inputs_k)
        inputs_v_grads = torch.empty_like(inputs_v)

        queries_grads = inputs_q_grads.view(
            inputs_q_grads.size(0), inputs_q_grads.size(1) * heads_t[0], -1
        )
        keys_grads = inputs_k_grads.view(
            inputs_k_grads.size(0), inputs_k_grads.size(1) * heads_t[0], -1
        )
        values_grads = inputs_v_grads.view(
            inputs_v_grads.size(0), inputs_v_grads.size(1) * heads_t[0], -1
        )

        # attention_probs: [batches=seqs*heads, seql_q, seql_k]
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

        return None, inputs_q_grads, inputs_k_grads, inputs_v_grads


def construct_inputs(seed=47):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    seq_length_q = 80
    seq_length_k = 100
    seq_length_v = 100
    sequences = 10
    hidden_dim = 1024
    heads = 16

    inputs_q = torch.randn(
        seq_length_q,
        sequences,
        hidden_dim,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    inputs_k = torch.randn(
        seq_length_k,
        sequences,
        hidden_dim,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    inputs_v = torch.randn(
        seq_length_v,
        sequences,
        hidden_dim,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ).requires_grad_(True)
    return inputs_q, inputs_k, inputs_v


class TestSelfMultiheadAttn(unittest.TestCase):
    def test_self_multihead_attn(self):

        ref_inputs_q, ref_inputs_k, ref_inputs_v = construct_inputs()
        tst_inputs_q, tst_inputs_k, tst_inputs_v = construct_inputs()

        ref_outputs = NaiveAttnFunc.apply(16, ref_inputs_q, ref_inputs_k, ref_inputs_v)

        tst_matmul1 = MultiheadAttnMatmul1Func.apply(16, tst_inputs_q, tst_inputs_k)
        tst_outputs = MultiheadAttnMatmul2Func.apply(16, tst_inputs_v, tst_matmul1)

        ref_outputs.sum().backward()
        tst_outputs.sum().backward()

        self.assertTrue(torch.equal(ref_inputs_q, tst_inputs_q))
        self.assertTrue(torch.equal(ref_inputs_k, tst_inputs_k))
        self.assertTrue(torch.equal(ref_inputs_v, tst_inputs_v))

        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(
            torch.allclose(ref_inputs_q.grad, tst_inputs_q.grad, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(ref_inputs_k.grad, tst_inputs_k.grad, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(ref_inputs_v.grad, tst_inputs_v.grad, atol=1e-3, rtol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
