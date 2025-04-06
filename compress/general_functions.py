import pandas as pd


def get_memory_init(df):
    memory = 0
    for i in range(df.shape[0]):
        for j in df.columns:
            memory+=len(bin(df.iloc[i][j])[2:])
    print(f'Размер исходных данных: {memory} бит')
    print(f'Размер исходных данных: {memory//8} байт', '\n')


def get_float_bytes(df):
    original_bytes = df.values.nbytes
    print(f'Размер исходных данных: {original_bytes*8} бит')
    print(f'Размер исходных данных: {original_bytes} байт', '\n')