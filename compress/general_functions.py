import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random as rn


def get_memory_init(df):
    memory = 0
    for i in range(df.shape[0]):
        for j in df.columns:
            memory+=len(bin(df.iloc[i][j])[2:])
    memory = memory//8
    print(f'Размер исходных данных: {memory} байт', '\n')
    return memory


def get_float_bytes(df):
    original_bytes = df.values.nbytes
    print(f'Размер исходных данных: {original_bytes} байт', '\n')
    return original_bytes


def get_geo_dict(df):
    # make geo data
    sen_num = df.shape[1]
    x_y = []
    for i in range(sen_num):
        x_y.append([rn.uniform(.0, 6.0), rn.uniform(.0, 6.0)])
        print(f'sensor_{i}:', x_y[i])
    # plt.grid(True)
    # for i in range(len(x_y)):
    #     plt.plot(x_y[i][0], x_y[i][1], 'o', label = i)
    #     plt.legend(loc = 'best', fancybox = True, shadow = True)
    # plt.plot(3*np.ones(300), np.arange(0, 6, 0.02))
    # plt.plot(np.arange(0, 6, 0.02), 3*np.ones(300))
    geo_dict = dict(zip(df.columns, x_y))
    return dict(zip(df.columns, x_y))


def create_geo_plot(d: dict):
    x_coords = [coord[0] for coord in d.values()]  # Координаты X
    y_coords = [coord[1] for coord in d.values()]  # Координаты Y
    labels = list(d.keys())  # Метки (названия сенсоров)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, color='blue', s=100)  # Точки на плоскости
    for i, label in enumerate(labels):
        plt.text(x_coords[i] + 0.1, y_coords[i] + 0.1, label, fontsize=10)
    plt.title('Координаты сенсоров на плоскости')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)  # Линия X=0
    plt.axvline(0, color='black', linewidth=0.5)  # Линия Y=0
    plt.show()
