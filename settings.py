from pathlib import Path

ROOT_PATH = Path(__file__).parents[0]
SPEED_DATA_PATH = ROOT_PATH / "data" / "speed_6005.csv"
WIND_DATA_PATH_INIT = ROOT_PATH / "data" / "EMHIRESPV_TSh_CF_Country_19862015.csv"
WIND_DATA_PATH = ROOT_PATH / "data" / "wind_df.csv"
DATA_PATH = ROOT_PATH / "data"
NUTS_PATH = ROOT_PATH / "nuts_data"
NUTS_ZIP_PATH = ROOT_PATH / "nuts_data" / "ref-nuts-2021-01m.geojson.zip"
GEO_JSON_PATH = ROOT_PATH / "nuts_data" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
GEO_PATH = ROOT_PATH / "nuts_data"/ "geo_dicts"
GEO_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-01m.geojson.zip"
EMHIRES_URL = "https://zenodo.org/records/8340501/files/EMHIRES_PV_NUTS2.zip?download=1"
EMHIRES_ZIP_PATH = ROOT_PATH / "data" / "EMHIRES_WIND_ONSHORE_NUTS2.zip"
EMHIRES_EXCEL_PATH = ROOT_PATH / "data" / "EMHIRES_PVGIS_TSh_CF_n2_19862015_reformatt.xlsx"
RESULTS_PATH = ROOT_PATH / "results"
MODEL_PARAMETERS_PATH = ROOT_PATH / "parameters" / "model_parameters.json"
