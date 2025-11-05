import joblib
import pandas as pd
import requests

# Carregar o modelo salvo
modelo = joblib.load("modelo_inundacoes.pkl")

# (Opcional) Carregar o LabelEncoder, se você salvou
try:
    le = joblib.load("label_encoder_provincia.pkl")
except:
    le = None


def get_nasa_data(lat, lon, year=2025, month=5):
    """
    Busca dados meteorológicos médios do mês/ano informados.
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters=T2M,PRECTOT,WS10M,RH2M,ALLSKY_SFC_SW_DWN"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={year}&end={year}&format=JSON"
    )

    response = requests.get(url).json()
    dados = response["properties"]["parameter"]

    # Extrair os valores do mês
    key = f"{year}{month:02d}"
    features = {
        "precipitacao_mm": dados["PRECTOTCORR"][key],
        "temperatura_C": dados["T2M"][key],
        "humidade_percent": dados["RH2M"][key],
        "vento_kmh": dados["WS10M"][key],
        "rad_solar_Wm2": dados["ALLSKY_SFC_SW_DWN"][key],
    }
    return features

# Exemplo: dados para Luanda (maio de 2025)
luanda_data = get_nasa_data(-8.839, 13.289, year=2025, month=1)
print(luanda_data)

# Montar um DataFrame para previsão
df_pred = pd.DataFrame([{
    'provincia': le.transform(['Luanda'])[0] if le else 0,  # usa o LabelEncoder se existir
    'latitude': -8.839,
    'longitude': 13.289,
    'ano': 2025,
    'mes': 5,
    'precipitacao_mm': luanda_data['precipitacao_mm'],
    'temperatura_C': luanda_data['temperatura_C'],
    'humidade_percent': luanda_data['humidade_percent'],
    'vento_kmh': luanda_data['vento_kmh'],
    'rad_solar_Wm2': luanda_data['rad_solar_Wm2']
}])

# Fazer a previsão
pred = modelo.predict(df_pred)[0]

# Interpretar o resultado
if pred == 1:
    print("Risco de inundação detectado!")
else:
    print("Sem risco de inundação.")
