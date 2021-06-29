#!/usr/bin/env python3

from bagua.torch_api.dev.bucket import BaguaBucket
from bagua.torch_api.dev.distributed_dev import BaguaModule
from bagua.torch_api.dev.algorithms import Algorithm
from torch.optim.optimizer import Optimizer

class OnebitAdamAlgorithm(Algorithm):
    def __init__(self):

        pass

    def init_tensors(self, bagua_module: BaguaModule):
        optimizers = bagua_module.bagua_optimizers

        parameters = bagua_module._bagua_build_params()
        for name, param in parameters:
           param._one_bit_name = name

        print("size of optimizer.params_in_group:{}".format(len(optimizer.params_in_group[0])))
        tensor_groups = []
        for optimizer in optimizers:
            for param_group in optimizer.params_in_group:
                for param in param_group:
                    tensor_groups.append(param.bagua_ensure_grad().to_bagua_tensor(param._one_bit_name))

        return [tensor_groups]

    def init_operations(
            self,
            bagua_module: BaguaModule,
            bucket: BaguaBucket,
    ):
        bucket.backend_bucket.clear_ops()
        bucket.backend_bucket.append_centralized_synchronous_op(
            bagua_module.bagua_inter_node_communicator,
            bagua_module.bagua_intra_node_communicator,
            hierarchical=True,
            average=True,
            scattergather=True,
            compression="MinMaxUInt8",
        )



class OnebitAdamOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        compression_start=100,
        is_bert=False,
        freeze_test_step=-1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, compression_start=compression_start, is_bert=is_bert, freeze_test_step=freeze_test_step, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(OnebitAdamOptimizer, self).__init__(params, defaults)

        self.params_in_group = []
        self.exp_avgs_in_group = []

        ###
        for group_id, group in enumerate(self.param_groups):

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            self.params_in_group.append(params_with_grad)
            self.exp_avgs_in_group.append(exp_avgs)

    def __setstate__(self, state):
        super(OnebitAdam, self).__setstate__(state)

    def step(self, closure=None):
        pass

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        # for group_id, group in enumerate(self.param_groups):

        #     state = self.state[group_id]
        #     state["step"] += 1

        #     lr = group["lr"]
        #     compression_start = group["compression_start"]
        #     is_bert = group["is_bert"]
        #     freeze_test_step = group["freeze_test_step"]

        #     beta1, beta2 = group["betas"]
        #     eps = group["eps"]
        #     weight_decay = group["weight_decay"]

        #     for param in group["params"]:

        #         param_weights = param.data
        #         param_grads = param.grad

        #         # no compression
        #         if (
        #             state["step"] < compression_start
        #             or state["group_numel"] < 8 * self.size
        #             or freeze_test_step > 0
        #         ):
        #             # allreduce grads
        #             communicator.allreduce_tensor(param_grads, param_grads, cupy_stream)
        #             param_grads.div_(self.size)

        #             if not is_bert:
        #                 if weight_decay != 0:
        #                     param_grads.add_(param_weights, alpha=weight_decay)

        #             state["exp_avg"].mul_(beta1).add_(param_grads, alpha=1 - beta1)

        #             # if freeze exp_avg_sq
        #             if (
        #                 freeze_test_step > 0
        #                 and state["step"] >= freeze_test_step
        #             ):
        #                 # freeze exp_avg_sq for testing
        #                 if state["step"] == freeze_test_step:
        #                     print(
        #                         "Rank{} 1bit-adam freeze test starts from the step {}".format(
        #                             self.rank, freeze_test_step
        #                         )
        #                     )
        #                 pass
        #             else:
        #                 # update exp_avg_sq as normal
        #                 state["exp_avg_sq"].mul_(beta2).addcmul_(param_grads, param_grads, value=1 - beta2)

        #         # with compression
        #         else:

        #             if state["step"] == compression_start:
        #                 print(
        #                     "Rank{} 1bit-adam quantization starts from the step {}".format(
        #                         self.rank,
        #                         compression_start,
        #                     )
        #                 )

        #             if not is_bert:
        #                 if weight_decay != 0:
        #                     param_grads.add_(param_weights, alpha=weight_decay)

        #             state["exp_avg"].mul_(beta1).add_(param_grads, alpha=1 - beta1)

        #             aggregated_exp_avg = cen_lowprec_sync(
        #                 state["exp_avg"],
        #                 onebit_quantize,
        #                 onebit_dequantize,
        #                 communicator,
        #                 cupy_stream,
        #                 state["worker_error"],
        #                 state["server_error"],
        #             )

        #             state["exp_avg"].copy_(aggregated_exp_avg)

                
        #         ## communication is finished.
        #         ## update param.
        #         if is_bert:
        #             update = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps)
        #             if weight_decay != 0:
        #                 update += weight_decay * param_weights
        #             param_weights.add_(-lr * update)
        #         else:
        #             bias_correction1 = 1 - beta1 ** state["step"]
        #             bias_correction2 = 1 - beta2 ** state["step"]

        #             denom = (
        #                 state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
        #             ).add_(eps)
        #             step_size = lr / bias_correction1
        #             update = state["exp_avg"] / denom

        #             param_weights.add_(-step_size * update)

        # return loss
