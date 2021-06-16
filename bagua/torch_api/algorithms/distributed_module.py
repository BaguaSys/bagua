import torch


class DistributedModule(torch.nn.Module):
    """
    A base class for distributed module.
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super(DistributedModule, self).__init__()
        self.module = module
        if hasattr(module, "_bagua_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = [
                ("module." + k) for k in module._bagua_params_and_buffers_to_ignore  # type: ignore
            ]
        else:
            self.parameters_to_ignore = []

    def unwrap(self):
        return self.module

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        return result
