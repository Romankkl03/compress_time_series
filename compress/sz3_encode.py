import numpy as np
from pathlib import Path
import sys
import pandas as pd
from sz.SZ3.tools.pysz.pysz import SZ
from compress.general_functions import get_float_bytes


lib_extension = {
        "darwin": "libSZ3c.dylib",  # macOS
        "win32": "SZ3c.dll",  # Windows
    }.get(sys.platform, "libSZ3c.so")  # Linux (по умолчанию)
sz = SZ(f"/usr/local/lib/{lib_extension}")


def compress_sz3_df(df):
    data = df.values.transpose()
    enc_data = []
    for d in data:
        data_cmpr, _ = sz.compress(d, eb_mode=0, eb_pwr=0, eb_rel=0, eb_abs=0.03)
        enc_data.append(data_cmpr)
    return enc_data


def decompress_sz3(enc_data, shape, type='float64'):
    if isinstance(enc_data, np.ndarray):
        data_dec = sz.decompress(enc_data, shape, type)
    else:
        data_dec = []
        for enc, sh in zip(enc_data, shape):
            data_dec.append(sz.decompress(enc, sh, type))
    dec_df = pd.DataFrame(data_dec).transpose()
    dec_df.columns = [f"sensor_{i}" for i in range(dec_df.shape[1])]  
    return dec_df


def compress_sz3_all(df):
    data = df.values.transpose()
    data_cmpr, _ = sz.compress(data, eb_mode=0, eb_pwr=0, eb_rel=0, eb_abs=0.03)
    return data_cmpr


def get_compress_info_sz3(df, enc_data):
    init_mem = get_float_bytes(df)
    total = 0
    if isinstance(enc_data, np.ndarray):
        total = enc_data.nbytes
    else:
        for enc in enc_data:
            total += enc.nbytes
    print(f'Размер сжатых данных: {total} байт', '\n')
    print(f'Коэффициент сжатия: {np.round(init_mem/total, 3)}')
