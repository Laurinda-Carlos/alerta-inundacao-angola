import requests
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# Função para buscar dados meteorológicos da NASA POWER API
# ============================================================
def get_weather_from_nasa(lat: float, lon: float, date: str = None):
    """
    Obtém dados meteorológicos (precipitação, temperatura e humidade) 
    da NASA POWER API para a localização e data especificadas.

    Parâmetros:
        lat (float): Latitude da localização.
        lon (float): Longitude da localização.
        date (str): Data no formato 'YYYY-MM-DD'. Se None, usa o dia atual.

    Retorna:
        pandas.DataFrame com colunas:
        ['latitude', 'longitude', 'mes', 'precipitacao_mm', 'temperatura_C', 'humidade_percent']
    """

    # Data atual se não especificada
    if not date:
        date = datetime(2024, 1, 1).strftime("%Y-%m-%d")

    # Endpoint NASA POWER
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR,T2M,RH2M"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={date.replace('-', '')}&end={date.replace('-', '')}"
        "&format=JSON"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        # Extrair parâmetros meteorológicos
        params = data.get("properties", {}).get("parameter", {})
        precip = list(params.get("PRECTOTCORR", {}).values())[0]
        temp = list(params.get("T2M", {}).values())[0]
        humid = list(params.get("RH2M", {}).values())[0]

        # Criar DataFrame compatível com o modelo
        df = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "mes": datetime.strptime(date, "%Y-%m-%d").month,
            "precipitacao_mm": precip,
            "temperatura_C": temp,
            "humidade_percent": humid
        }])

        print(f"[✔] Dados da NASA obtidos com sucesso para {date}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"[✖] Erro ao consultar NASA POWER API: {e}")
        return pd.DataFrame()

# ============================================================
# Exemplo de uso direto (para testes)
# ============================================================
if __name__ == "__main__":
    # Exemplo: Luanda, Angola
    lat, lon = -8.8383, 13.2344
    df_weather = get_weather_from_nasa(lat, lon)
    print(df_weather)
