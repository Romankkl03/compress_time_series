import numpy as np
import lz4.frame
import pandas as pd
from sklearn.decomposition import PCA
from compress.bypass import geo_sort, get_corr_lists
from compress.general_functions import get_float_bytes
import pickle
import sys


def compress_pca(pca: PCA):
    pca_data = {
        "components": pca.components_.astype(np.float32),  # float32 вместо float64
        "mean": pca.mean_.astype(np.float32),
        "variance": pca.explained_variance_ratio_.astype(np.float32)
    }
    pca_bytes = pickle.dumps(pca_data)
    compressed_pca = lz4.frame.compress(pca_bytes)
    return compressed_pca


def LZ4_pca_compress(df_comp: pd.DataFrame):
    pca_n = max(2, int(0.5 * len(df_comp.columns)))
    pca = PCA(n_components=pca_n)  # Уменьшаем до 2 компонент
    transformed = pca.fit_transform(df_comp)
    compressed = lz4.frame.compress(transformed.tobytes())
    pca_compress = compress_pca(pca)
    return [compressed, pca_compress]


def lz4_one(arr):
    arr_bytes = arr.tobytes()
    compressed = lz4.frame.compress(arr_bytes)
    return compressed


def spatial_clustering_PCA_LZ4(df_init, x_y_dict, cor_lvl = 0.8):
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
                    cl_sp[iter_sen] = LZ4_pca_compress(df_init[iter_clust])
                    iter_sen = sen
                    break
                else:
                    dec_one = [lz4_one(df_init[iter_clust[0]].values), '']
                    cl_sp[iter_sen] = dec_one
                    iter_sen = sen
                    break
                if not sensors and len(iter_clust)>1:
                    cl_sp[iter_sen] = LZ4_pca_compress(df_init[iter_clust])
                    iter_sen = sen
                    break
        else:
            arr = df_init[iter_sen].values
            arr_bytes = arr.tobytes()
            cl_sp[iter_sen] = [lz4.frame.compress(arr_bytes), '']
            break
    return cl_sp


def decompress_pca(compressed_pca):
    pca_bytes = lz4.frame.decompress(compressed_pca)
    pca_data = pickle.loads(pca_bytes)
    pca = PCA()
    pca.components_ = pca_data["components"]
    pca.mean_ = pca_data["mean"]
    pca.explained_variance_ratio_ = pca_data["variance"]
    return pca


def spatial_lz4_decompress(enc_dict, clust_dict):
    res = []
    for key in enc_dict.keys():
        if enc_dict[key][1] == '':
            decompressed_data = lz4.frame.decompress(enc_dict[key][0])
            arr = np.frombuffer(decompressed_data, 
                                dtype=np.float64)
            res.append(arr)
        else:
            pca = decompress_pca(enc_dict[key][1])
            decompressed_data = lz4.frame.decompress(enc_dict[key][0])
            arr = np.frombuffer(decompressed_data, 
                                dtype=np.float64).reshape(-1, 
                                                    pca.components_.shape[0]) 
            res_dec = pca.inverse_transform(arr).transpose()
            res.extend(res_dec)
    df = pd.DataFrame(res).transpose()
    cols = []
    for _, l in clust_dict.items():
        if len(l)==1:
            cols.append(l[0])
        else:
            cols.extend(l)
    df.columns = cols
    df = df.sort_index(axis=1)
    return df


def get_compress_info_spatial_PCA_LZ4(df, res):
    init_mem = get_float_bytes(df)
    total = sum(sys.getsizeof(v) for k in res.keys() for v in res[k])
    print(f'Размер сжатых данных: {total} байт', '\n')
    print(f'Коэффициент сжатия: {np.round(init_mem/total, 3)}')
    return np.round(init_mem/total, 3)