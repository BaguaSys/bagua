# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab
#
# All rights reserved.
#
# The file has been adapted from DeepSpeed:
#   https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py
# Git commit hash: bff6126f0ddbd1a03da66867571ac87b11c21ac1
# We retain the following license from the original files:

# Copyright 2020 The Microsoft DeepSpeed Team

import bagua.torch_api as bagua
import logging
import torch
import torch.distributed as dist

from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import typing


class MoE(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        expert,
        num_local_experts=1,
        k=1,
        output_dropout_prob=0.0,
        capacity_factor=1.0,
        eval_capacity_factor=1.0,
        min_capacity=4,
        noisy_gate_policy: typing.Optional[str] = None,
    ):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_local_experts (int, optional): default=1, number of local experts per gpu.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            output_dropout_prob (float, optional): default=0.0, output dropout probability.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        """

        super(MoE, self).__init__()

        assert noisy_gate_policy is None or noisy_gate_policy in [
            "None",
            "Jitter",
            "RSample",
        ], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )

        self.num_experts = num_local_experts * bagua.get_world_size()
        logging.info(
            f"num_experts: {self.num_experts} | num_local_experts: {num_local_experts} | world_size: {bagua.get_world_size()}"
        )

        experts = Experts(expert, num_local_experts)
        self.deepspeed_moe = MOELayer(
            TopKGate(
                hidden_size,
                self.num_experts,
                k,
                capacity_factor,
                eval_capacity_factor,
                min_capacity,
                noisy_gate_policy,
            ),
            experts,
            num_local_experts,
            group=dist.group.WORLD,
        )

        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        output = self.dropout(output)
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
