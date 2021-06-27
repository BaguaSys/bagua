#!/usr/bin/env python3
import torch
import bagua_core as B
from bagua.torch_api.utils import to_bagua_datatype
import gorilla

@gorilla.patches(torch.Tensor)
class BaguaTensor(object):
    def to_bagua_tensor(self, name: str):
        self.bagua_tensor_name = name
        self.bagua_tensor = B.BaguaTensorPy(
            ptr=self.data_ptr(),
            num_elem=self.numel(),
            num_elem_allocated=self.__dict__.get("allocated_size", self.numel()),
            dtype=to_bagua_datatype(self.dtype),
            device_id=self.device.index,
        )
        return self

    @property
    def grad(self):
        if self.grad is None:
            with torch.no_grad():
                t = torch.zeros_like(self.data)
                self.grad = t

    def mark_communication_ready(self, bagua_backend, cuda_event):
        bagua_backend.mark_communication_ready(
            self.bagua_tensor,
            cuda_event,
        )

    # def _set(self):
    #     pass

    def set_storage(self, storage: torch.Storage, storage_offset: int = 0):
        with torch.no_grad():
            self.set_(storage, storage_offset, self.shape)
        if self.bagua_tensor is not None:
            self.bagua_tensor.reset_ptr(self.data_ptr())


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
