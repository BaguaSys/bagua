import torch


class MinMaxUInt8:
    def __init__(self):
        self.eps = 1e-7
        self.quantization_level = 255

    def compress(self, tensor):
        _max = torch.max(tensor)
        _min = torch.min(tensor)

        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level

        level = (tensor * scale).int()
        level = torch.clamp(level, min=lower_bound)
        return _min, _max, level - lower_bound

    def decompress(self, _min, _max, compressed):
        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level
        return (compressed + lower_bound) / scale
