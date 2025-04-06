import lz4.frame
import pandas as pd
import struct
import numpy as np

def LZ4_compress_df(df):
    compressed_df = {}
    for col in df.columns:
        arr = df[col].values
        arr_bytes = arr.tobytes()
        compressed_df[col] = lz4.frame.compress(arr_bytes)
    return compressed_df


def LZ4_decompress_df(compressed_df):
    decompressed_df = {}
    for col, compressed_data in compressed_df.items():
        decompressed_data = lz4.frame.decompress(compressed_data)
        arr = np.frombuffer(decompressed_data, dtype=np.float64)
        decompressed_df[col] = arr
    df = pd.DataFrame(decompressed_df)
    return df


def lz4_one(arr):
    arr_bytes = arr.tobytes()
    compressed = lz4.frame.compress(arr_bytes)
    return [compressed, '']


def get_lz4_memory(compressed_df):
    memory = 0
    for col in compressed_df:
        memory += len(compressed_df[col])
    print(f'Размер сжатых данных: {memory} байт', '\n')
