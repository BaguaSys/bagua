import unittest
import torch
import torch.nn as nn
from bagua.torch_api.fast_modules import FusedLayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_layernorm(self):
        torch.manual_seed(0)
        N, C = 16, 8
        inp_ = torch.randn(N, C).cuda()
        inp_.requires_grad_(True)
        dy = 0.1 * torch.randn_like(inp_)

        layer_norm = nn.LayerNorm(8, device="cuda")
        fused_layernorm = FusedLayerNorm(8)

        output_torch = layer_norm(inp_)
        output_triton = fused_layernorm(inp_)

        output_triton.backward(dy, retain_graph=True)
        dx_tri = inp_.grad.clone()

        inp_.grad = None
        output_torch.backward(dy, retain_graph=True)
        dx_tor = inp_.grad.clone()

        self.assertTrue(torch.allclose(output_torch, output_triton))
        self.assertTrue(torch.allclose(dx_tri, dx_tor))


if __name__ == "__main__":
    unittest.main()
