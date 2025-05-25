import pandas as pd
import numpy as np
from compress.xor_encode import xor_compress, decompress_xor
from compress.bypass import geo_sort, get_corr_lists
from compress.general_functions import get_float_bytes


def xor_cluster_compress(df: pd.DataFrame) -> list:
    res = {}
    cols = df.columns
    res[cols[0]] = xor_compress(list(df[cols[0]].values))
    for num_ts in range(1,len(cols)):
        res[cols[num_ts]] = xor_compress(list((df[cols[0]]-df[cols[num_ts]]).values))
    return res


def spatial_clustering_xor(df, x_y_dict, cor_lvl=0.85):
    cl_sp = {}
    sensors = list(df.columns)
    iter_sen = sensors[0]
    while len(sensors)!=0:
        iter_clust = []
        sensors.remove(iter_sen)
        iter_clust.append(iter_sen)
        sen_queue = geo_sort(x_y_dict, sensors, iter_sen)
        if sen_queue:
            for sen in sen_queue:
                cor = get_corr_lists(df[iter_sen], df[sen])
                if cor>cor_lvl:
                    iter_clust.append(sen)
                    sensors.remove(sen)
                elif len(iter_clust)>1:
                    cl_sp[iter_sen] = xor_cluster_compress(df[iter_clust])
                    iter_sen = sen
                    break
                else:   
                    cl_sp[iter_sen] = {iter_sen: xor_compress(list(df[iter_clust].values))}
                    iter_sen = sen
                    break
        else:
            cl_sp[iter_sen] = {iter_sen: xor_compress(list(df[iter_clust].values))}
            break
    return cl_sp


def spatial_XOR_decompress(enc_dict):
    res = []
    cols = []
    for key in enc_dict.keys():
        clust_res = []
        clust_keys = list(enc_dict[key].keys())
        cols.append(key)
        if len(clust_keys)==1:
            res.append(decompress_xor(enc_dict[key][key]))
        else:
            clust_res.append(decompress_xor(enc_dict[key][key]))
            clust_keys.remove(key)
            for cl_key in clust_keys:
                dec_dif = decompress_xor(enc_dict[key][cl_key])
                dec_dif = list(np.array(clust_res[0]) - np.array(dec_dif))
                rounded_numbers = [round(num, 15) for num in dec_dif]
                
                clust_res.append(rounded_numbers)
                cols.append(cl_key)
            res.extend(clust_res)
    df = pd.DataFrame(res).transpose()
    df.columns = cols
    df = df.sort_index(axis=1)
    return df


def get_compress_info_spatial_xor(init_df, res: dict):
    init_bytes = get_float_bytes(init_df)
    total = 0
    keys_res = res.keys()
    for key in keys_res:
        keys_clust = res[key].keys()
        for k in keys_clust:
            for l in res[key][k]:
                total+=len(l)
    total = total/8
    print(f'Размер сжатых данных: {total} байт', '\n')
    print(f'Коэффициент сжатия: {np.round(init_bytes/total, 3)}')
    return np.round(init_bytes/total, 3)
