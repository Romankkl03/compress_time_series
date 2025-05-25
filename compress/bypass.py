import math
import numpy as np


def geo_sort(x_y_dict, sensors, init_sen)->np.array:
    """
    Function to sort sensors based on their distance from the initial sensor
    """
    if len(sensors) == 0:
        return None
    xy_dist = {}
    for sen in sensors:
        if sen != init_sen:
            distance = math.dist(x_y_dict[init_sen], x_y_dict[sen])
            xy_dist[distance] = sen
    dist_keys = sorted(xy_dist.keys())
    res = []
    for dist in dist_keys:
        res.append(xy_dist[dist])
    #print(f'Cенсоры в зависимости от их удаленности от сенсора {init_sen}:', res)
    return res


def get_corr_lists(list1, list2):
    correlation = np.corrcoef(list1, list2)[0, 1]
    return correlation


def spatial_clustering(df, x_y_dict, cor_lvl = 0.8):
    cl_sp = {}
    sensors = list(df.columns)
    iter_sen = sensors[0]
    iter_clust = []
    while len(sensors)!=0:
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
                    cl_sp[iter_sen] = iter_clust
                    iter_sen = sen
                    iter_clust = []
                    break
                else:
                    cl_sp[iter_sen] = iter_clust
                    iter_sen = sen
                    iter_clust = []
                    break
                if not sensors and len(iter_clust)>1:
                    cl_sp[iter_sen] = iter_clust
        else:
            cl_sp[iter_sen] = iter_clust
            break
    return cl_sp
