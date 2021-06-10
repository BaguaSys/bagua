from enum import Enum


class Compressor(Enum):
    NoneCompressor = None
    Uint8Compressor = "MinMaxUInt8"
