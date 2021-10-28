# Copyright (c) 2021 Kuaishou AI Platform & DS3 Lab
#
# All rights reserved.
#
# The file has been adapted from DeepSpeed:
#   https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
# Git commit hash: bff6126f0ddbd1a03da66867571ac87b11c21ac1
# We retain the following license from the original files:

# Copyright 2020 The Microsoft DeepSpeed Team

import torch
import copy


class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()

        self.bagua_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)]
        )
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.bagua_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.expert = True

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.bagua_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output
