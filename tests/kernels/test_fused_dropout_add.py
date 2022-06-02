import torch
from bagua.torch_api.contrib.fuse.fused_add_dropout_residual import fused_add_dropout_residual_func

prob = 0.0
prof = 5
is_training = True


def construct_inputs(seed=47):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic=True

    n = 512
    h = 4
    w = 1024
    inputs = torch.randn(n, h, w).cuda().half()
    residual = torch.randn(n, h, w).cuda().half()
    bias = torch.randn(n, h, w).cuda().half()
    d_loss_tensor = torch.randn((n, h, w)).cuda().half()
    inputs.requires_grad_()
    residual.requires_grad_()
    bias.requires_grad_()

    return inputs, bias, residual, d_loss_tensor


def do_bench(fn, prob, is_training, backward=False):
    
    inputs, bias, residual, d_loss_tensor = construct_inputs()
    grad_to_none = [inputs, bias, residual]
    n_warmup = 10
    n_repeat = 20
    
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event   = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    for _ in range(n_warmup):
        fn(inputs, bias, residual, prob, is_training)
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        
        if not backward:
            # record time of `fn`
            start_event[i].record()
            result = fn(inputs, bias, residual, prob, is_training)
            end_event[i].record()
        else:
            # record time of `fn`
            result = fn(inputs, bias, residual, prob, is_training)
            start_event[i].record()
            # result.backward(d_loss_tensor)
            result.sum().backward()
            end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()

    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    # print((times))
    return torch.mean(times[1:]).item(), result, inputs.grad, bias.grad, residual.grad


def torch_func(inputs, bias, residual, prob, is_training):
    return torch.nn.functional.dropout(inputs + bias, p=prob, training = is_training) + residual


def bagua_func(inputs, bias, residual, prob, is_training):
    return fused_add_dropout_residual_func(inputs, bias, residual, prob, is_training)


#test for training mode forward.
bagua_time, bagua_res, bagua_inputs_grad, bagua_bias_grad, bagua_residual_grad = do_bench(bagua_func, prob, is_training=True)
print("bagua_time for training mode: {}".format(bagua_time))

torch_time,  torch_res, torch_inputs_grad, torch_bias_grad, torch_residual_grad = do_bench(torch_func, prob, is_training=True)
print("torch_time for training mode: {}".format(torch_time))
print("result equal {}".format(torch.equal(torch_res, bagua_res)))


#test for training mode backward.
bagua_time, bagua_res, bagua_inputs_grad, bagua_bias_grad, bagua_residual_grad = do_bench(bagua_func, prob, is_training=True, backward=True)
print("bagua_time for inference mode: {}".format(bagua_time))

torch_time,  torch_res, torch_inputs_grad, torch_bias_grad, torch_residual_grad = do_bench(torch_func, prob, is_training=True, backward=True)
print("torch_time for inference mode: {}".format(torch_time))
print("inputs grad equal {}".format(torch.equal(torch_inputs_grad, bagua_inputs_grad)))
print("bias grad equal {}".format(torch.equal(torch_bias_grad, bagua_bias_grad)))
print("inputs residual equal {}".format(torch.equal(torch_residual_grad, bagua_residual_grad)))


#test for testing mode forward.
bagua_time, bagua_res, bagua_inputs_grad, bagua_bias_grad, bagua_residual_grad = do_bench(bagua_func, prob, is_training=False)
print("bagua_time for inference mode: {}".format(bagua_time))

torch_time,  torch_res, torch_inputs_grad, torch_bias_grad, torch_residual_grad = do_bench(torch_func, prob, is_training=False)
print("torch_time for inference mode: {}".format(torch_time))
print("result equal {}".format(torch.equal(torch_res, bagua_res)))






########### below is an initial implementation for profiling ###########
'''
inputs_1, bias_1, residual_1, d_loss_tensor_1 = construct_inputs()
inputs, bias, residual, d_loss_tensor = construct_inputs()
cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')


if prof > 0:
    print("Profiling begun...")
    torch.cuda.cudart().cudaProfilerStart()


#test for training mode forward.
start_1 = torch.cuda.Event(enable_timing=True)
end_1 = torch.cuda.Event(enable_timing=True)

start_2 = torch.cuda.Event(enable_timing=True)
end_2 = torch.cuda.Event(enable_timing=True)
for _ in range(5):
    ret = fused_add_dropout_residual_func(inputs, bias, residual, prob, is_training)
    # cache.zero_()
torch.cuda.synchronize()
# cache.zero_()

if prof>0:
    torch.cuda.nvtx.range_push("fused forward")
start_1.record()
ret = fused_add_dropout_residual_func(inputs, bias, residual, prob, is_training)
end_1.record()
if prof>0:
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("fused time: {}".format(start_1.elapsed_time(end_1)))

for _ in range(5):
    tmp = torch.nn.functional.dropout(inputs_1 + bias_1, p=prob, training = is_training)
    torch_res = tmp + residual_1
    # cache.zero_()
torch.cuda.synchronize()
# cache.zero_()

if prof>0:
    torch.cuda.nvtx.range_push("unfused forward")
start_2.record()
tmp = torch.nn.functional.dropout(inputs_1 + bias_1, p=prob, training = is_training)
torch_res = tmp + residual_1
end_2.record()
if prof>0:
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("no fused time: {}".format(start_2.elapsed_time(end_2)))
print("training forward equal: {}".format(torch.equal(ret, torch_res)))


#test for training mode backward.
start_3 = torch.cuda.Event(enable_timing=True)
end_3 = torch.cuda.Event(enable_timing=True)

start_4 = torch.cuda.Event(enable_timing=True)
end_4 = torch.cuda.Event(enable_timing=True)

for _ in range(5):
    cache.zero_()
    inputs.grad= None
    bias.grad= None
    residual.grad= None
    ret.backward(d_loss_tensor, retain_graph=True)
    # ret.sum().backward(retain_graph=True)
torch.cuda.synchronize()

cache.zero_()
inputs.grad= None
bias.grad= None
residual.grad= None

if prof>0:
    torch.cuda.nvtx.range_push("fused backward")
start_3.record()
ret.backward(d_loss_tensor)
# ret.sum().backward()
end_3.record()
if prof>0:
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
#print(bias.grad)
print("fused backward time: {}".format(start_3.elapsed_time(end_3)))
print(torch.equal(inputs.grad, bias.grad))
#print(inputs.grad)
for _ in range(5):
    inputs_1.grad= None
    bias_1.grad= None
    residual_1.grad= None
    torch_res.backward(d_loss_tensor_1, retain_graph=True)
    cache.zero_()
    # torch_res.sum().backward(retain_graph=True)
torch.cuda.synchronize()
inputs_1.grad= None
bias_1.grad= None
residual_1.grad= None
cache.zero_()
if prof>0:
    torch.cuda.nvtx.range_push("unfused backward")
start_4.record()
torch_res.backward(d_loss_tensor_1)
# torch_res.sum().backward()
end_4.record()
if prof>0:
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("not fused backward time: {}".format(start_4.elapsed_time(end_4)))
print(torch.equal(inputs.grad, bias.grad))
#print(inputs.grad)



#test for testing mode forward.
is_training =False

start_1 = torch.cuda.Event(enable_timing=True)
end_1 = torch.cuda.Event(enable_timing=True)

start_2 = torch.cuda.Event(enable_timing=True)
end_2 = torch.cuda.Event(enable_timing=True)

for _ in range(5):
    ret = fused_add_dropout_residual_func(inputs, bias, residual, prob, is_training)
    cache.zero_()
torch.cuda.synchronize()
cache.zero_()

if prof>0:
    torch.cuda.nvtx.range_push("fused test forward")
start_1.record()
ret = fused_add_dropout_residual_func(inputs, bias, residual, prob, is_training)
end_1.record()
if prof>0:
    torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()
print("fused time: {}".format(start_1.elapsed_time(end_1)))

for _ in range(5):
    tmp = torch.nn.functional.dropout(inputs_1 + bias_1, p=prob, training = is_training)
    torch_res = tmp + residual_1
    cache.zero_()
torch.cuda.synchronize()
cache.zero_()

if prof>0:
    torch.cuda.nvtx.range_push("unfused test forward")
start_2.record()
tmp = torch.nn.functional.dropout(inputs_1 + bias_1, p=prob, training = is_training)
torch_res = tmp + residual_1
end_2.record()
if prof>0:
    torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()
print(torch.equal(ret, torch_res))
print("no fused time: {}".format(start_2.elapsed_time(end_2)))
'''
