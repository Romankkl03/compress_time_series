from compress.bypass import geo_sort, get_corr_lists
from compress.sz3_encode import compress_sz3_all, decompress_sz3, compress_sz3_df
from sz.SZ3.tools.pysz.pysz import SZ
from compress.general_functions import get_float_bytes
import sys
import os
import pandas as pd


# Подавляем C++ логи TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Только ошибки
os.environ['ABSL_CPP_MIN_LOG_LEVEL'] = '3'  # Только ошибки absl

# Установим параметры логирования absl (через стандартный логгер TensorFlow не работает)
os.environ['FLAGS_logtostderr'] = '0'
os.environ['FLAGS_minloglevel'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_v'] = '0'

import logging

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from tensorflow.keras.layers import Input, Dense, Flatten, SeparableConv1D, Concatenate
from tensorflow.keras.models import Model
import lz4.frame
import numpy as np
import pywt
import matplotlib.pyplot as plt
import time
import zstandard as zstd

import contextlib


@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_fd = os.dup(2)
        sys.stdout, sys.stderr = devnull, devnull
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            os.dup2(old_fd, 2)
            os.close(old_fd)


def create_training_plot(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


lib_extension = {
        "darwin": "libSZ3c.dylib",  # macOS
        "win32": "SZ3c.dll",  # Windows
    }.get(sys.platform, "libSZ3c.so")  # Linux (по умолчанию)
sz = SZ(f"/usr/local/lib/{lib_extension}")

def compress_model(model, model_compress="zstd"):
    with suppress_all_output():
        model.export("saved_model")
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        converter.experimental_sparsify_model = True
        tflite_model = converter.convert()
        if model_compress == "zstd":
            compressed_model = zstd.ZstdCompressor(level=10).compress(tflite_model)
        else:
            compressed_model = lz4.frame.compress(tflite_model)
    print('Size of compressed model (bytes):', len(compressed_model))
    return compressed_model
# def compress_model(model, model_compress="zstd"):
#     model.export("saved_model")
#     converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
#     converter.experimental_sparsify_model = True
#     tflite_model = converter.convert()
#     if model_compress == "zstd":
#         compressed_model = zstd.ZstdCompressor(level=10).compress(tflite_model)
#     else:
#         compressed_model = lz4.frame.compress(tflite_model)
#     print('Size of compressed model (bytes):', len(compressed_model))
#     return compressed_model


def compress_sz3_one(d, er_abs_sz3=0.0001):
    data_cmpr, _ = sz.compress(d, eb_mode=0, eb_pwr=0, eb_rel=0, eb_abs=er_abs_sz3)
    return data_cmpr


def decompress_sz3_one(v, shape, v_type=np.float64):
    data_dec = sz.decompress(v, shape, v_type)
    return data_dec


def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return 0.7 * mse + 0.3 * mae


def get_learned_model(df, window_size=64, num_epochs=50, 
                      extra_layer=False, conv_filter=2,
                      plot_flag=False):
    main_sensor = df.iloc[:, 0].values.astype(np.float32)
    dependent_sensors = df.iloc[:, 1:].values.astype(np.float32)
    X = main_sensor.reshape(-1, 1)
    Y = dependent_sensors
    X_windows = []
    Y_targets = []
    output_dim = Y.shape[1]
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i+window_size])
        Y_targets.append(Y[i+window_size]) 
    X_windows = np.array(X_windows).reshape(-1, window_size, 1)
    Y_targets = np.array(Y_targets).reshape(-1, output_dim)
    
    inputs = Input(shape=(window_size, 1))
    x = SeparableConv1D(conv_filter, kernel_size=3,
                        activation="swish",
                        padding='same')(inputs)
    x = Flatten()(x)
    if extra_layer:
        x = Dense(8, activation="swish")(x)
    x = Dense(4, activation="swish")(x)
    outputs = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss)
    start_time = time.time()
    history = model.fit(X_windows, Y_targets, epochs=num_epochs, batch_size=8, verbose=0, validation_split=0)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Время обучения: {training_time:.2f} секунд")
    if plot_flag == True:
        create_training_plot(history)
    return model


def get_learned_model_dwt(df, window_size=64, num_epochs=50, conv_filter=2, plot_flag=False):
    main_sensor = df.iloc[:, 0].values.astype(np.float32)
    dependent_sensors = df.iloc[:, 1:].values.astype(np.float32)
    output_dim = dependent_sensors.shape[1]

    X_windows = []
    X_dwt_feats = []
    Y_targets = []

    wavelet = 'db4'
    dwt_level = 2

    for i in range(len(main_sensor) - window_size):
        window = main_sensor[i:i+window_size]
        X_windows.append(window.reshape(-1, 1))
        dwt_coeffs = pywt.wavedec(window, wavelet=wavelet, level=dwt_level)
        dwt_feat = np.concatenate(dwt_coeffs)
        X_dwt_feats.append(dwt_feat)
        Y_targets.append(dependent_sensors[i + window_size])

    X_windows = np.array(X_windows)
    X_dwt_feats = np.array(X_dwt_feats)
    Y_targets = np.array(Y_targets)

    input_seq = Input(shape=(window_size, 1), name="seq_input")
    x_seq = SeparableConv1D(conv_filter, kernel_size=3, activation="swish", padding='same')(input_seq)
    x_seq = Flatten()(x_seq)
    input_dwt = Input(shape=(X_dwt_feats.shape[1],), name="dwt_input")
    x = Concatenate()([x_seq, input_dwt])
    x = Dense(8, activation="swish")(x)
    x = Dense(4, activation="swish")(x)
    output = Dense(output_dim, activation='sigmoid')(x)

    model = Model(inputs=[input_seq, input_dwt], outputs=output)
    model.compile(optimizer='adam', loss=custom_loss)
    history = model.fit([X_windows, X_dwt_feats], Y_targets, epochs=num_epochs, batch_size=8, verbose=0)
    if plot_flag == True:
        create_training_plot(history)
    return model


def compress_cnn_cluster(df,
                         use_dwt=False,
                         window_size=64, num_epochs=50, extra_layer=False,
                         conv_filter=2, plot_flag=False, er_abs_sz3=0.0001,
                         model_compress="zstd"):
    main_sensor = compress_sz3_one(df.iloc[:, 0].values, er_abs_sz3)
    main_sen = decompress_sz3_one(main_sensor, (df.shape[0],))
    df.iloc[:,0] = np.abs(main_sen)
    if use_dwt:
        model = get_learned_model_dwt(df, window_size, num_epochs, conv_filter, plot_flag) #, plot_flag
    else:
        model = get_learned_model(df, window_size, num_epochs, extra_layer, conv_filter, plot_flag)
    compressed_model = compress_model(model, model_compress)
    remainder = compress_sz3_df(df.iloc[:window_size, 1:], 0.03)
    #print("Размер кластера в байтах:", main_sensor.nbytes+remainder+len(compressed_model))
    res = [main_sensor, compressed_model, remainder]
    return res


def compress_cnn_sz3(df_init,
                     x_y_dict,
                     cor_lvl = 0.8,
                     use_dwt=False,
                     window_size=64,
                     num_epochs=50,
                     extra_layer=False,
                     conv_filter=2,
                     plot_flag=False,
                     er_abs_sz3=0.0001,
                     model_compress="zstd"):
    cl_sp = {}
    sensors = list(df_init.columns)
    iter_sen = sensors[0]
    iter_clust = []
    while len(sensors)!=0:
        sensors.remove(iter_sen)
        iter_clust.append(iter_sen)
        sen_queue = geo_sort(x_y_dict, sensors, iter_sen)
        if sen_queue:
            for sen in sen_queue:
                cor = get_corr_lists(df_init[iter_sen], df_init[sen])
                if cor>cor_lvl and len(iter_clust)<4:
                    iter_clust.append(sen)
                    sensors.remove(sen)
                elif len(iter_clust)>1:
                    cl_sp[tuple(iter_clust)] = compress_cnn_cluster(df_init[iter_clust], use_dwt, 
                                                                    window_size, num_epochs, extra_layer, conv_filter,
                                                                    plot_flag, er_abs_sz3, model_compress)
                    iter_sen = sen
                    iter_clust = []
                    break
                else:
                    dec_one = [compress_sz3_one(df_init[iter_clust[0]].values, 0.03)]
                    cl_sp[tuple(iter_clust)] = dec_one
                    iter_sen = sen
                    iter_clust = []
                    break
                if not sensors and len(iter_clust)>1:
                    cl_sp[tuple(iter_clust)] = compress_cnn_cluster(df_init[iter_clust], use_dwt, 
                                                                    window_size, num_epochs, extra_layer, conv_filter,
                                                                    plot_flag, er_abs_sz3, model_compress)
        else:
            cl_sp[tuple(iter_clust)] = [compress_sz3_one(df_init[iter_sen].values, 0.03)]
            break
    return cl_sp


def get_model(compressed_model, model_compress="zstd"):
    with suppress_all_output():
        if model_compress == "zstd":
            decompressor = zstd.ZstdDecompressor()
            decompressed_tflite = decompressor.decompress(compressed_model)
        else:
            decompressed_tflite = lz4.frame.decompress(compressed_model)
        interpreter = tf.lite.Interpreter(model_content=decompressed_tflite)
        interpreter.allocate_tensors()
    return interpreter


def get_predict(interpreter, main_sensor, window_size=64):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X = main_sensor.reshape(-1, 1)
    X_windows = []
    for i in range(len(X) - window_size):
        X_windows.append(X[i:i+window_size])
    X_windows = np.array(X_windows).reshape(-1, window_size, 1)
    restored_Y = []
    for i in range(len(X_windows)):
        input_tensor = X_windows[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        restored_Y.append(output[0])
    restored_Y = np.array(restored_Y)
    return restored_Y


def get_predict_dwt(interpreter, main_sensor, window_size=64):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_windows = []
    X_dwt_feats = []
    wavelet = 'db4'
    dwt_level = 2
    for i in range(len(main_sensor) - window_size):
        window = main_sensor[i:i+window_size]
        X_windows.append(window.reshape(-1, 1))

        dwt_coeffs = pywt.wavedec(window, wavelet=wavelet, level=dwt_level)
        dwt_feat = np.concatenate(dwt_coeffs)
        X_dwt_feats.append(dwt_feat)

    X_windows = np.array(X_windows).astype(np.float32)
    X_dwt_feats = np.array(X_dwt_feats).astype(np.float32)
    restored_Y = []
    for i in range(len(X_windows)):
        interpreter.set_tensor(input_details[0]['index'], X_windows[i:i+1])
        interpreter.set_tensor(input_details[1]['index'], X_dwt_feats[i:i+1])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        restored_Y.append(output[0])
    restored_Y = np.array(restored_Y)
    return restored_Y


def decomress_cnn_sz3(res, shape, use_dwt=False, window_size=64, model_compress="zstd"):
    sens = res.keys()
    dec_res = {}
    for s in sens:
        main_sen = np.abs(decompress_sz3_one(res[s][0], shape))
        dec_res[s[0]] = main_sen
        if len(s) > 1:
            shape_r = [(window_size,) for _ in range(len(s)-1)]
            remainder = decompress_sz3(res[s][2],
                    shape_r).values
            interpeter = get_model(res[s][1], model_compress)
            if use_dwt:
                preds = get_predict_dwt(interpeter, main_sen)
            else:
                preds = get_predict(interpeter, main_sen)
            preds = np.concatenate((remainder, preds)).transpose()
            for dep_sen, val in zip(s[1:], preds):
                dec_res[dep_sen] = val
    res_df = pd.DataFrame(dec_res)
    return res_df


def get_compress_info_cnn_sz3(df, enc_data):
    init_mem = get_float_bytes(df)
    total = 0
    for k in enc_data.keys():
        total += enc_data[k][0].nbytes
        if len(k) > 1:
            total += len(enc_data[k][1])
            # total += enc_data[k][2].nbytes
            for e in enc_data[k][2]:
                total += e.nbytes
    print(f'Размер сжатых данных: {total} байт', '\n')
    print(f'Коэффициент сжатия: {np.round(init_mem/total, 3)}')
    return np.round(init_mem/total, 3)
