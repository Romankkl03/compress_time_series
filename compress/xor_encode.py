import pandas as pd
import numpy as np
import struct
import sys
from compress.general_functions import get_float_bytes


def fxor(a, b):
    rtrn = []
    a = struct.pack('d', a)
    b = struct.pack('d', b)
    for ba, bb in zip(a, b):
        rtrn.append(ba ^ bb)
    return bin(int.from_bytes(bytes(rtrn), sys.byteorder))[2:].zfill(64)


def float_to_binary(num):
    num = bytes(struct.pack('d', num))
    return bin(int.from_bytes(num, sys.byteorder))[2:].zfill(64)


def xor_compress(ts):
    result = []
    s = float_to_binary(ts[0])
    try:
        off_pred = s.index('1')
        len_pred = len(s)-s[::-1].index('1')-1-off_pred
        result.append(s)
    except:
        off_pred = 0
        len_pred = 0
        result.append('0')
    for i in range(1, len(ts)):
        xor = fxor(ts[i-1], ts[i])
        try:
            off = xor.index('1')#начало ненулевого кода xor
        except:
            s = '0'
            result.append(s)
            continue
        length = len(xor)-xor[::-1].index('1')-1-off# длина последовательности - 1
        if (off<off_pred) or (off+length>off_pred+len_pred) or (i==1):
            offset_b = bin(off)[2:]
            len_seq_b = bin(length)[2:]
            s = '11'+('0'*(5-len(offset_b))+offset_b+'0'*(6-len(len_seq_b))
                      +len_seq_b+xor[off:off+length+1])
            result.append(s)
            off_pred = off
            len_pred = length
        else:
            s = '10'+'0'*(off-off_pred)+xor[off:off+length+1]+'0'*(off_pred+len_pred-length-off)
            result.append(s)
    return result


def xor_compress_df(df):
    res = []
    for col in df.columns:
        time_series = df[col].values
        res.append(xor_compress(time_series))
    return res


def binary_to_float(binary_str):
    num_bytes = int(binary_str, 2).to_bytes(8, sys.byteorder)
    return struct.unpack('d', num_bytes)[0]


def reverse_fxor(xor_result, a):
    a_bytes = struct.pack('d', a)
    xor_bytes = int(xor_result, 2).to_bytes(8, sys.byteorder)
    b_bytes = bytes(ba ^ xb for ba, xb in zip(a_bytes, xor_bytes))
    return struct.unpack('d', b_bytes)[0]


def decompress_xor(compressed_list):
    result = []
    if compressed_list[0] == '0':
        result.append(0.0)
    else:
        result.append(binary_to_float(compressed_list[0]))
    prev_res = result[0]
    for i in range(1, len(compressed_list)):
        if compressed_list[i] == '0':
            result.append(prev_res)
        elif compressed_list[i][:2] == '11':
            offset = int(compressed_list[i][2:7], 2)
            length = int(compressed_list[i][7:13], 2) + 1
            xor = compressed_list[i][13:]
            xor = '0' * offset + xor + '0' * (64 - offset - length)
            prev_res = reverse_fxor(xor, prev_res)
            result.append(prev_res)
        else:
            xor = compressed_list[i][2:]
            xor = '0' * offset + xor + '0' * (64 - len(xor) - offset)
            prev_res = reverse_fxor(xor, prev_res)
            result.append(prev_res)
    return result


def decompress_xor_df(compressed_df: list):
    result = []
    for compressed_list in compressed_df:
        result.append(decompress_xor(compressed_list))
    result_df = pd.DataFrame(result).transpose()
    result_df.columns = [f'sensor_{i}' for i in range(len(result))]
    return result_df


def get_xor_memory(compressed_list):
    infb = 0
    for r in compressed_list:
        infb += len(''.join(r))
    print(f'Память сжатых XOR данных: {infb} бит')
    print(f'Память сжатых XOR данных: {infb/8} байт')
    
    
def get_xor_memory_df(init_df, compressed_df):
    init_bytes = get_float_bytes(init_df)
    infb = 0
    for r in compressed_df:
        for s in r:
            infb += len(s)
    comp_bytes = infb//8
    print(f'Размер сжатых XOR данных: {comp_bytes} байт')
    print(f'Коэффициент сжатия: {np.round(init_bytes/comp_bytes, 3)}')
