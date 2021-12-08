from .multihead_attn_func import SelfMultiheadAttnScoreFunc
from torch import nn


class SelfMultiheadAttnScore(nn.Module):
    def __init__(self, num_heads, dropout=0.0, coefficient=1.0):
        super(SelfMultiheadAttnScore, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.coefficient = coefficient

    def forward(self, inputs, attention_mask):
        return SelfMultiheadAttnScoreFunc.apply(
            self.training,
            self.num_heads,
            inputs,
            attention_mask,
            self.coefficient,
            self.dropout,
        )
