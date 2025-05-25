import os
import yaml
import zipfile
import requests
import pandas as pd
from settings import (
    DATA_PATH,
    GEO_PATH,
    COUNTRY_DF_URL
)


def get_country_df():
    zip_path = DATA_PATH / "EMHIRES_PV_COUNTRY.zip"
    excel_path = DATA_PATH / "EMHIRESPV_TSh_CF_Country_19862015.xlsx"
    if not os.path.exists(DATA_PATH / "country_df.csv"):
        if not os.path.exists(zip_path):
            print("Скачиваем архив")
            response = requests.get(COUNTRY_DF_URL, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        if not os.path.exists(excel_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_PATH)
        df_wind = pd.read_excel(excel_path)
        df_wind = df_wind[['AT', 'BE', 'BG', 'CH', 'CZ', 'DE']]
        df_wind = df_wind.head(10000)
        df_wind = df_wind.loc[(
            df_wind[list(df_wind.columns)] != 0).any(axis=1)].reset_index(drop=True)
        df_wind.columns = [f'sensor_{i}' for i in range(df_wind.shape[1])]
        df_wind.to_csv(DATA_PATH / "country_df.csv", index=False)
    else:
        df_wind = pd.read_csv(DATA_PATH / "country_df.csv")
    with open(GEO_PATH / 'country_geo_dict.yaml', 'r') as file:
            wind_geo_dict = yaml.safe_load(file)
    return df_wind, wind_geo_dict
