from compress.bypass import geo_sort, get_corr_lists
from compress.sz3_encode import compress_sz3_all, decompress_sz3
from sz.SZ3.tools.pysz.pysz import SZ
from compress.general_functions import get_float_bytes
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, SeparableConv1D
from tensorflow.keras.models import Model
import lz4.frame
import numpy as np


lib_extension = {
        "darwin": "libSZ3c.dylib",  # macOS
        "win32": "SZ3c.dll",  # Windows
    }.get(sys.platform, "libSZ3c.so")  # Linux (по умолчанию)
sz = SZ(f"/usr/local/lib/{lib_extension}")


def compress_model(model):
    model.export("saved_model")  
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
    tflite_model = converter.convert()
    compressed_model = lz4.frame.compress(tflite_model)
    #TODO: add path
    return compressed_model


def compress_sz3_one(d):
    data_cmpr, _ = sz.compress(d, eb_mode=0, eb_pwr=0, eb_rel=0, eb_abs=0.03)
    return data_cmpr


def decompress_sz3_one(v, shape, v_type=np.float64):
    data_dec = sz.decompress(v, shape, v_type)
    return data_dec


def get_learned_model(df, window_size=64):
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
    x = SeparableConv1D(2, kernel_size=3,
                        activation='selu',
                        padding='same')(inputs)
    x = Flatten()(x)
    x = Dense(4, activation='selu')(x)  
    outputs = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_windows, Y_targets, epochs=50, batch_size=64, verbose=1)
    return model


def compress_cnn_cluster(df):
    main_sensor = compress_sz3_one(df.iloc[:, 0].values)
    main_sen = decompress_sz3_one(main_sensor, (df.shape[0],))
    df.iloc[:,0] = main_sen
    model = get_learned_model(df)
    compressed_model = compress_model(model)
    main_sensor = compress_sz3_one(df.iloc[:, 0].values)
    remainder = compress_sz3_all(df.iloc[:64, 1:])
    res = [main_sensor, compressed_model, remainder]
    return res


def compress_cnn_sz3(df_init, x_y_dict, cor_lvl = 0.8):
    cl_sp = {}
    sensors = list(df_init.columns)
    iter_sen = sensors[0]
    while len(sensors)!=0:
        iter_clust = []
        sensors.remove(iter_sen)
        iter_clust.append(iter_sen)
        sen_queue = geo_sort(x_y_dict, sensors, iter_sen)
        if sen_queue:
            for sen in sen_queue:
                cor = get_corr_lists(df_init[iter_sen], df_init[sen])
                if cor>cor_lvl:
                    iter_clust.append(sen)
                    sensors.remove(sen)
                elif len(iter_clust)>1: 
                    cl_sp[tuple(iter_clust)] = compress_cnn_cluster(df_init[iter_clust])
                    iter_sen = sen
                    break
                else:
                    dec_one = [compress_sz3_one(df_init[iter_clust[0]].values)]
                    cl_sp[tuple(iter_clust)] = dec_one
                    iter_sen = sen
                    break
        else:
            cl_sp[tuple(iter_clust)] = [compress_sz3_one(df_init[iter_sen].values)]
            break
    return cl_sp


def get_model(compressed_model):
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


def decomress_cnn_sz3(res, shape):
    sens = res.keys()
    dec_res = {}
    for s in sens:
        main_sen = decompress_sz3_one(res[s][0], shape)
        dec_res[s[0]] = main_sen
        if len(s) > 1:
            remander = decompress_sz3(res[s][2], (64,len(s)-1)).values.transpose()
            interpeter = get_model(res[s][1])
            preds = get_predict(interpeter, main_sen)
            preds = np.concatenate((remander, preds)).transpose()
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
            total += enc_data[k][2].nbytes
    print(f'Размер сжатых данных: {total} байт', '\n')
    print(f'Коэффициент сжатия: {np.round(init_mem/total, 3)}')
