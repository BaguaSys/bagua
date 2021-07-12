import torch


class MinMaxUInt8:
    def __init__(self):
        self.eps = 1e-7
        self.quantization_level = 255.0

    def compress(self, tensor: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        _min = torch.min(tensor)
        _max = torch.max(tensor)
        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level

        level = torch.round(tensor * scale)
        level = torch.clamp(level, max=upper_bound)

        _minmax = torch.zeros(2, dtype=tensor.dtype, device=tensor.device)
        _minmax[0] = _min
        _minmax[1] = _max
        return _minmax, (level - lower_bound).to(torch.uint8)

    def decompress(
        self, _minmax: torch.Tensor, compressed: torch.Tensor
    ) -> torch.Tensor:
        _min = _minmax[0]
        _max = _minmax[1]

        scale = self.quantization_level / (_max - _min + self.eps)
        upper_bound = torch.round(_max * scale)
        lower_bound = upper_bound - self.quantization_level
        return (compressed.float() + lower_bound) / scale


if __name__ == "__main__":
    x = torch.rand(100).cuda()
    _minmax, compressed = MinMaxUInt8().compress(x)
    decompressed = MinMaxUInt8().decompress(_minmax, compressed)

    diff = x - decompressed
    print(f"{diff}, {torch.norm(diff)}")
