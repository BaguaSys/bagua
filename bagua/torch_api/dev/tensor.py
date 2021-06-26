#!/usr/bin/env python3
import torch
import bagua_core as B
from bagua.torch_api.utils import to_bagua_datatype


class BaguaTensor(torch.Tensor):
    def __init__(self, original_tensor: torch.Tensor) -> None:
        super().__init__()
        self.inner = original_tensor
        ## here only register the param.grad
        self.bagua_tensor = B.BaguaTensorPy(
            ptr=original_tensor.grad.data_ptr(),
            num_elem=original_tensor.numel(),
            num_elem_allocated=original_tensor.__dict__.get("allocated_size", original_tensor.numel()),
            dtype=to_bagua_datatype(original_tensor.dtype),
            device_id=original_tensor.grad.device.index,
        )
    def mark_communication_ready(self, bagua_backend, cuda_event):
        bagua_backend.mark_communication_ready(
            self.bagua_tensor,
            cuda_event,
        )
    def set_storage(self, storage: torch.Storage, storage_offset: int = 0):
        with torch.no_grad():
            self.inner.set_(storage, storage_offset, self.inner.shape)

if __name__ == "__main__":
    import math

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

    loss_fn = torch.nn.MSELoss(reduction="sum")

    for param in model.parameters():
        print(param.shape)
        tensor2 = torch.zeros_like(param.data)
        print(tensor2.shape)
        with torch.no_grad():
            param.set_(tensor2.storage(), 0, param.shape)
            print(param.shape)
        # param.data = BaguaTensor(param.data)

    learning_rate = 1e-6
    for t in range(2000):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
        model.zero_grad()
        loss.backward()
        continue
