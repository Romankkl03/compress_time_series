import numpy as np
from pathlib import Path
import sys
import pandas as pd
from sz.SZ3.tools.pysz.pysz import SZ


lib_extension = {
        "darwin": "libSZ3c.dylib",  # macOS
        "win32": "SZ3c.dll",  # Windows
    }.get(sys.platform, "libSZ3c.so")  # Linux (по умолчанию)
sz = SZ(f"/usr/local/lib/{lib_extension}")


def compress_sz3_df(df):
    data = df.values.transpose()
    enc_data = []
    for d in data:
        data_cmpr, cmpr_ratio = sz.compress(d, eb_mode=0, eb_pwr=0, eb_rel=0, eb_abs=0.03)
        enc_data.append(data_cmpr)
    return enc_data


def decompress_sz3_df(enc_data, shape, type='float64'):
    data_dec = []
    for enc, sh in zip(enc_data, shape):
        data_dec.append(sz.decompress(enc, sh, type))
    return data_dec


def get_sz3_size(enc_data):
    mem = 0
    for enc in enc_data:
        mem += enc.nbytes
    return mem
