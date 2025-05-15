import numpy as np
import pandas as pd
from settings import SPEED_DATA_PATH


def create_nab_data():
    speed_init = pd.read_csv(SPEED_DATA_PATH, parse_dates=[0], index_col=[0])
    speed_init['index'] = np.arange(speed_init.shape[0])
    sensor_data = {}
    for i in range(5):
        sensor_data[f'sensor_{i+1}'] = speed_init.loc[speed_init['index'] % 5 == i,
                                                ['value']]['value'].to_list()
    df_speed = pd.DataFrame(sensor_data)
    return df_speed