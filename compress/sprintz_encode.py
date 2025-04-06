import pandas as pd
import numpy as np


def delta_encoding(df):
    delta = []
    for i in range(1, df.shape[0]):
        delta.append((df.iloc[i]-df.iloc[i-1]).to_list())
    return pd.DataFrame(delta)


def zigzag_encoding(df):
    res = []
    for i in range(df.shape[0]):
        raw = []
        for j in (df.columns):
            val = df.iloc[i][j]
            if val<0:
                raw.append(abs(val)*2-1)
            else:
                raw.append(val*2)
        res.append(raw)
    return res


# def int_to_bits(n):
#     return bin(n)[2:]
def int_to_bits(n):
    """
    Кодирует целое число (включая отрицательные) в двоичное представление.
    """
    if n >= 0:
        return bin(n << 1)[2:]  # Сдвиг влево для положительных чисел
    else:
        return bin((n << 1) ^ -1)[2:]  # Сдвиг влево и XOR для отрицательных чисел


def sprintz_encode(chunk_df):
    align = 16
    #ToDo: add len for header format
    enc = [[int_to_bits(value) for value in chunk_df.iloc[0].to_list()]]
    delta = delta_encoding(chunk_df)
    zigzag = zigzag_encoding(delta)
    max_val = list(map(max, zip(*zigzag)))
    max_val = [format(m, '08b') for m in max_val]
    max_len = [8-(m.find('1')) if '1' in m else 0 for m in max_val]
    header = ''.join([format(m, '03b') for m in max_len])
    zigzag = [list(row) for row in zip(*zigzag)] #transpose
    bin = ''
    for i in range(len(zigzag)):
        bits = max_len[i]
        if bits==0:
            continue
        for v in zigzag[i]:
            bin+=format(v, f'0{bits}b')
    enc.append(header+'0'*(align-len(header))+bin)
    return enc


def compress_sprintz(df, chunk_size = 8):
    res = []
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size].copy()
        #ToDo: find max appropriate chunk_size
        res.append(sprintz_encode(chunk_df))
    return res


def delta_decoding(zigzag, row_1):
    res_delta = [row_1]
    for i in range(zigzag.shape[0]):
        res_delta.append(zigzag.iloc[i]+res_delta[-1])
    return pd.DataFrame(res_delta)


def zigzag_decoding(zz_list):
    res = []
    for l in zz_list:
        raw = []
        for i in range(len(l)):
            if l[i]%2==0:
                raw.append(l[i]//2)
            else:
                raw.append(-(l[i]+1)//2)
        res.append(raw)
    return pd.DataFrame(res)


def decode_header(header):
    max_len = [int(header[i:i+3], 2) for i in range(0, len(header), 3)]
    return max_len


def get_row_num(length, chunk_num, bits):
    if length/chunk_num==sum(bits):
        return chunk_num
    else:
        return int(length/sum(bits))

def bits_to_int(bits):
    """
    Декодирует двоичное представление обратно в целое число.
    """
    n = int(bits, 2)  # Преобразование строки битов в целое число
    if n & 1 == 0:  # Если последний бит равен 0 (положительное число)
        return n >> 1
    else:  # Если последний бит равен 1 (отрицательное число)
        return -(n >> 1) - 1
    
def decode_sprintz(enc_str, first_row, cols, chunk_size = 8):
    first_row = [bits_to_int(bin_str) for bin_str in first_row]
    len_header = 3*cols
    bits = decode_header(enc_str[:len_header])
    enc_str = enc_str[len_header+(16-len_header):]
    dec_str = []
    len_str = len(enc_str)
    row_num = get_row_num(len_str, chunk_size, bits)
    for bit in bits:
        if bit==0:
            dec_str.append([0]*row_num)
        else:
            s = enc_str[:bit*row_num]
            dec_str.append([int(s[i*bit:(i+1)*bit], 2) for i in range(row_num)])
            enc_str = enc_str[bit*row_num:]
    zigzag = [list(row) for row in zip(*dec_str)]
    delta = zigzag_decoding(zigzag)
    res = delta_decoding(delta, first_row)
    res.columns = [f'sensor_{i+1}' for i in range(cols)]
    return res


def decompress_sprintz(enc_str, num_cols = 5, chunk_size = 8):
    res = []
    for enc in enc_str:
        res.append(decode_sprintz(enc[1], enc[0], num_cols, chunk_size))
    df_res = pd.concat(res)
    df_res.index = range(df_res.shape[0])
    return df_res


def get_sprintz_memory(enc_res):
    infb = 0
    for r in enc_res:
        infb+=len(''.join(r[0]))
        infb+=len(r[1])
    print(f'Память сжатых Sprintz данных: {infb} бит')
    print(f'Память сжатых Sprintz данных: {infb//8} байт', '\n')