#!/usr/bin/env python3
import torch


class BaguaTensor(torch.Tensor):
    def __init__(self, original_tensor: torch.Tensor) -> None:
        super().__init__()
        self.inner = original_tensor

    def mark_communication_ready(self):
        ...

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
