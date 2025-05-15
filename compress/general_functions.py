import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random as rn
from sklearn.metrics import mean_squared_error
import tracemalloc
import psutil
import os
import gc
import threading
import time


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
    sen_num = df.shape[1]
    x_y = []
    for i in range(sen_num):
        x_y.append([rn.uniform(.0, 6.0), rn.uniform(.0, 6.0)])
        print(f'sensor_{i}:', x_y[i])
    geo_dict = dict(zip(df.columns, x_y))
    return dict(zip(df.columns, x_y))


def create_geo_plot(d: dict):
    x_coords = [coord[0] for coord in d.values()]
    y_coords = [coord[1] for coord in d.values()]
    labels = list(d.keys())
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_coords_norm = [(x - x_min) / (x_max - x_min) for x in x_coords]
    y_coords_norm = [(y - y_min) / (y_max - y_min) for y in y_coords]
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords_norm, y_coords_norm, color='blue', s=100)
    for i, label in enumerate(labels):
        plt.text(x_coords_norm[i] + 0.01, y_coords_norm[i] + 0.01, label, fontsize=10)
    plt.title('Нормализованные координаты сенсоров на плоскости')
    plt.xlabel('X (нормализованная)')
    plt.ylabel('Y (нормализованная)')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()


def get_errors(df, dec_df):
    mse = np.round(mean_squared_error(df, dec_df), 6)
    print(f"MSE: {mse} \n")
    true_all = df.values.flatten()
    pred_all = dec_df.values.flatten()
    mask = true_all != 0
    true_all = true_all[mask]
    pred_all = pred_all[mask]
    overall_mape = np.nanmean(np.abs((true_all - pred_all)
                                     / true_all)) * 100
    print(f"MAPE: {np.round(overall_mape, 2)} %", "\n")
    return mse, np.round(overall_mape, 2)


def get_peak_resource(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, np.round(peak / 1024, 2)


def cnn_resource_usage(func, *args, **kwargs):
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    peak_mem = [mem_before]
    def monitor():
        while not stop_flag[0]:
            mem_now = process.memory_info().rss
            if mem_now > peak_mem[0]:
                peak_mem[0] = mem_now
            time.sleep(0.01)
    stop_flag = [False]
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    result = func(*args, **kwargs)
    stop_flag[0] = True
    monitor_thread.join()
    peak_kib = (peak_mem[0] / 1024) - (mem_before/ 1024)
    return result, round(peak_kib, 2)
