import torch
import cupy
from utils import qsgd_compress, qsgd_decompress
import argparse

# TODO: change to formal benchmark tasks https://github.com/ionelmc/pytest-benchmark
def main():
    parser = argparse.ArgumentParser(description="Test QSGD Compression Error")
    parser.add_argument(
        "--dimension", type=int, default=10000, metavar="N", help="length of tensor"
    )
    parser.add_argument(
        "--norms",
        type=str,
        default="2,4,8,16,inf",
        metavar="N",
        help="pnorm used for qsgd quantization",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default="2,4,8",
        metavar="N",
        help="bits used for qsgd quantization",
    )
    parser.add_argument(
        "--random_test_num",
        type=int,
        default=100,
        metavar="N",
        help="number of test time",
    )
    args = parser.parse_args()

    norms = [float(i) for i in args.norms.split(",")]
    bits = [int(i) for i in args.bits.split(",")]
    print("norms: ", norms)
    print("bits: ", bits)
    errors = torch.zeros(len(norms), len(bits))
    random_test_num = args.random_test_num
    dimension = args.dimension

    torch.manual_seed(42)
    tensor = torch.rand([args.random_test_num, args.dimension]).cuda()
    cupy_stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    cupy_stream.use()
    for t in range(args.random_test_num):
        input_t = tensor[t, :].detach().clone()
        for i, porm in enumerate(norms):
            for j, bit in enumerate(bits):
                norm, packed_sign, compressed_ints = qsgd_compress(
                    input_t, pnorm=porm, quan_bits=bit
                )
                quan_t = qsgd_decompress(
                    norm, packed_sign, compressed_ints, quan_bits=bit
                )
                errors[i, j] += (
                    (quan_t - input_t).norm() / input_t.norm() / random_test_num
                )
    print("error: ", errors)


if __name__ == "__main__":
    main()
