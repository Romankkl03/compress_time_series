from pathlib import Path


ROOT_PATH = Path(__file__).parents[0]
DATA_PATH = ROOT_PATH / "data"
SPEED_DATA_PATH = DATA_PATH / "speed_6005.csv"
WIND_DATA_PATH_INIT = DATA_PATH / "EMHIRESPV_TSh_CF_Country_19862015.csv"
COUNTRY_DF_URL = "https://zenodo.org/records/8340501/files/EMHIRES_PV_COUNTRY.zip?download=1"
WIND_DATA_PATH = DATA_PATH / "wind_df.csv"
NUTS_PATH = DATA_PATH / "nuts_data"
NUTS_ZIP_PATH = NUTS_PATH / "ref-nuts-2021-01m.geojson.zip"
GEO_JSON_PATH = NUTS_PATH / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
GEO_PATH = NUTS_PATH / "geo_dicts"
GEO_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-01m.geojson.zip"
EMHIRES_URL = "https://zenodo.org/records/8340501/files/EMHIRES_PV_NUTS2.zip?download=1"
EMHIRES_ZIP_PATH = ROOT_PATH / "data" / "EMHIRES_WIND_ONSHORE_NUTS2.zip"
EMHIRES_EXCEL_PATH = ROOT_PATH / "data" / "EMHIRES_PVGIS_TSh_CF_n2_19862015_reformatt.xlsx"
RESULTS_PATH = ROOT_PATH / "results"
MODEL_PARAMETERS_PATH = ROOT_PATH / "parameters" / "model_parameters.json"
