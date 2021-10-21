import torch
import bagua.torch_api as bagua


class MegatronBaseMLP(torch.nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, activation=torch.nn.GELU()):
        super(MegatronBaseMLP, self).__init__()

        self.dense_h_to_4h = torch.nn.Linear(hidden_size, ffn_hidden_size)
        self.activation = activation
        self.dense_4h_to_h = torch.nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, input):
        x = self.dense_h_to_4h(input)
        x = self.activation(x)
        x = self.dense_4h_to_h(x)
        return x


class MegatronMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        num_local_experts,
        top_k,
        activation=torch.nn.GELU(),
    ):
        super(MegatronMLP, self).__init__()

        self.hidden_size = hidden_size
        self.megatron_mlp = bagua.moe.MoE(
            hidden_size,
            MegatronBaseMLP(hidden_size, ffn_hidden_size, activation),
            num_local_experts,
            top_k,
        )

    def forward(self, input):
        output, _, _ = self.megatron_mlp(input)
        return (
            output,
            torch.zeros(self.hidden_size, dtype=input.dtype, device=input.device),
        )
