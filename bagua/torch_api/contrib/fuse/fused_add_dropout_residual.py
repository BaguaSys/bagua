import torch
import add_dropout_residual_impl

class FusedAddDropOutResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, bias, residual, dropout_prob, is_training):
        
        dropout_prob_tensor = torch.tensor([dropout_prob])
        dropout_add_result, dropout_add_mask = add_dropout_residual_impl.forward(inputs,
                                                bias,
                                                residual,
                                                dropout_prob,
                                                is_training
                                                )
        ctx.save_for_backward(dropout_add_mask, dropout_prob_tensor)
        return dropout_add_result

    @staticmethod
    def backward(ctx, output_grads):
        """
        Since the residual grad same with `output_grads`, we directly return the `output_grads` as redisual_grad here.
        """
        # if not output_grads.is_contiguous():
        # output_grads = output_grads.clone()
        dropout_add_mask, dropout_prob_tensor = ctx.saved_tensors
        inputs_grad, bias_grad = add_dropout_residual_impl.backward(output_grads.contiguous(),
                                                            dropout_add_mask,
                                                            dropout_prob_tensor[0]
                                                            )
        return inputs_grad, bias_grad, output_grads, None, None
        # return output_grads, output_grads, output_grads, None, None

fused_add_dropout_residual_func = FusedAddDropOutResidual.apply
