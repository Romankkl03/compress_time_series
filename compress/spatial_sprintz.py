import numpy as np
import lz4.frame
import pandas as pd
from compress.bypass import geo_sort, get_corr_lists
from compress.sprintz_encode import compress_sprintz, decompress_sprintz
from compress.spatial_lz4 import lz4_one


def sprintz_cluster_compress(df_comp: pd.DataFrame) -> list:
    cols = df_comp.columns
    for num_ts in range(1,len(cols)):
        df_comp[cols[num_ts]] = df_comp[cols[0]]-df_comp[cols[num_ts]]
    res = compress_sprintz(df_comp)
    return res


def spatial_clustering_sprintz(df_init, x_y_dict, cor_lvl = 0.7):
    cl_sp = {}
    sensors = list(df_init.columns)
    while len(sensors)!=0:
        iter_sen = sensors[0]
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
                elif len(iter_clust) != 1:
                    cl_sp[iter_sen] = sprintz_cluster_compress(df_init[iter_clust])
                    iter_sen = sen
                    break
                else:
                    #take compress for one
                    cl_sp[iter_sen] = [lz4_one(df_init[iter_clust[0]].values/(10**10))]
        else:
            cl_sp[iter_sen] = [lz4_one(df_init[iter_clust[0]].values/(10**10))]
            break
    return cl_sp


def lz4_decode(enc):
    decompressed_data = lz4.frame.decompress(enc)
    arr = np.frombuffer(decompressed_data, dtype=np.float64)
    lst = list(arr*(10**10))
    lst = [int(x) for x in lst]
    return lst


def spatial_sprintz_decompress(enc_dict, clust_dict):
    res_lz = []
    cols_lz = []
    res_clust = []
    res = pd.DataFrame()
    for key in enc_dict.keys():
        if len(clust_dict[key]) == 1:
            res_lz.append(lz4_decode(enc_dict[key][0]))
            cols_lz.append(key)
        else:
            res_dec = decompress_sprintz(enc_dict[key], len(clust_dict[key]))
            first_col = res_dec.iloc[:, 0]  
            res_dec.iloc[:, 1:] = (first_col.values.reshape(-1, 1) -
                                   res_dec.iloc[:, 1:])
            res_dec.columns = clust_dict[key]
            res_clust.append(res_dec)
    if len(res_lz) != 0:
        res = pd.DataFrame(res_lz).transpose()
        res.columns = cols_lz
    if len(res_clust) > 0:
        res_clust_df = pd.concat(res_clust, axis=1)
        res = (pd.concat([res, res_clust_df], axis=1) 
               if not res.empty else res_clust_df)
    res = res.sort_index(axis=1)
    return res