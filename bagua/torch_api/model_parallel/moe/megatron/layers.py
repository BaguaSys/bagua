import torch


class MoeBaseMLP(torch.nn.Module):
    def __init__(self):
        super(MoeBaseMLP, self).__init__()
        from megatron import get_args
        args = get_args()

        self.dense_h_to_4h = torch.nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.dense_4h_to_h = torch.nn.Linear(args.ffn_hidden_size, args.hidden_size)

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        return  self.dense_4h_to_h(intermediate)
