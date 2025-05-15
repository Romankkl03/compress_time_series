import os
import yaml
import zipfile
import requests
import geopandas as gpd
import pandas as pd
from settings import (DATA_PATH, NUTS_PATH, NUTS_ZIP_PATH, GEO_URL,
                      GEO_JSON_PATH, GEO_PATH,
                      EMHIRES_URL, EMHIRES_ZIP_PATH,
                      EMHIRES_EXCEL_PATH)


def create_geo_wind_dict(nuts_codes):
    if not os.path.exists(NUTS_PATH):
        os.makedirs(NUTS_PATH, exist_ok=True)
    if not os.path.exists(NUTS_ZIP_PATH):
        print("Скачивание архива геоданных")
        r = requests.get(GEO_URL)
        with open(NUTS_ZIP_PATH, 'wb') as f:
            f.write(r.content)
    if not os.path.exists(GEO_JSON_PATH):
        print("Распаковка geojson.zip")
        with zipfile.ZipFile(NUTS_ZIP_PATH, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith("NUTS_RG_01M_2021_4326_LEVL_2.geojson"):
                    zip_ref.extract(file, NUTS_PATH)
                    os.rename(os.path.join(NUTS_PATH, file), GEO_JSON_PATH)
                    break
    gdf = gpd.read_file(GEO_JSON_PATH)
    gdf_3035 = gdf.to_crs(epsg=3035)
    gdf_3035['centroid'] = gdf_3035.geometry.centroid
    gdf_3035['x'] = gdf_3035.centroid.x
    gdf_3035['y'] = gdf_3035.centroid.y
    filtered = gdf_3035[gdf_3035['NUTS_ID'].isin(nuts_codes)].copy()
    geo_dict = {
        key: [x, y]
        for key, x, y in zip(filtered['NUTS_ID'], filtered['x'], filtered['y'])
    }
    filtered = filtered['NUTS_ID'].tolist()
    return filtered, geo_dict


def create_nuts_data(country="AT"):
    if (not os.path.exists(DATA_PATH / f"{country}_wind.csv") or
        not os.path.exists(GEO_PATH / f"geodict_{country}.yaml")
    ):
        if not os.path.exists(EMHIRES_EXCEL_PATH):
            print("Загрузка датасета")
            r = requests.get(EMHIRES_URL)
            with open(EMHIRES_ZIP_PATH, 'wb') as f:
                f.write(r.content)
            print("Распаковка датасета")
            with zipfile.ZipFile(EMHIRES_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_PATH)
        print("Чтение датасета")
        df_wind = pd.read_excel(EMHIRES_EXCEL_PATH, nrows=40000)
        df_wind = df_wind.filter(like=country)
        df_wind = df_wind.loc[(df_wind != 0).any(axis=1)].reset_index(drop=True)
        df_wind = df_wind.iloc[:5371]
        df_wind = df_wind.sort_index(axis=1)
        nuts_codes = list(df_wind.columns)
        filtered_codes, geo_dict = create_geo_wind_dict(nuts_codes=nuts_codes) 
        df_wind = df_wind[filtered_codes]
        column_mapping = {old_name: f'sensor_{i}' for i, old_name in enumerate(df_wind.columns)}
        df_wind.rename(columns=column_mapping, inplace=True)
        df_wind.to_csv(DATA_PATH / f"{country}_wind.csv", index=False)
        geo_dict = {column_mapping[key]: value for key, value in geo_dict.items()}
        if not os.path.exists(GEO_PATH):
            os.makedirs(GEO_PATH, exist_ok=True)
        with open(GEO_PATH / f"geodict_{country}.yaml", "w") as file:
            yaml.dump(geo_dict, file)
        return df_wind, geo_dict
    else:
        df_wind = pd.read_csv(DATA_PATH / f"{country}_wind.csv")
        with open(GEO_PATH / f"geodict_{country}.yaml", 'r') as file:
            geo_dict = yaml.safe_load(file)
        return df_wind, geo_dict
