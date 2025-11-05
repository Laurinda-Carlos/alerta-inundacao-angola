import streamlit as st
import pandas as pd
import joblib
import requests

# ================================
# 1. Carregar modelo e encoder
# ================================
modelo = joblib.load("modelo_inundacoes.pkl")

try:
    le = joblib.load("labelencoder_provincia.pkl")
except:
    le = None
    st.warning("⚠️ LabelEncoder não encontrado. As províncias serão numéricas.")

# ================================
# 2. Função para buscar dados da NASA
# ================================
def get_nasa_data(lat, lon, year=2025, month=5):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters=T2M,PRECTOTCORR,WS10M,RH2M,ALLSKY_SFC_SW_DWN"
        f"&community=RE&longitude={lon}&latitude={lat}"
        f"&start={year}&end={year}&format=JSON"
    )

    try:
        response = requests.get(url).json()
        dados = response["properties"]["parameter"]
        key = f"{year}{month:02d}"
        return {
            "precipitacao_mm": dados["PRECTOTCORR"][key],
            "temperatura_C": dados["T2M"][key],
            "humidade_percent": dados["RH2M"][key],
            "vento_kmh": dados["WS10M"][key],
            "rad_solar_Wm2": dados["ALLSKY_SFC_SW_DWN"][key],
        }
    except Exception as e:
        st.error(f"Erro ao buscar dados da NASA: {e}")
        return None

# ================================
# 3. Interface Streamlit
# ================================
# Configuração da página
st.set_page_config(
    page_title="Nvula",
    page_icon="🌊",
    layout="centered"
)

st.title("Previsão de Risco de Inundação (Angola)")

st.sidebar.header("Dados de Entrada")

# Entradas do usuário
provincia = st.sidebar.selectbox("Província", ["Luanda", "Benguela", "Huíla"])
latitude = st.sidebar.number_input("Latitude", value=-8.839 if provincia == "Luanda" else -12.576)
longitude = st.sidebar.number_input("Longitude", value=13.289 if provincia == "Luanda" else 13.406)
ano = st.sidebar.number_input("Ano", value=2025, min_value=2000, max_value=2100)
mes = st.sidebar.slider("Mês", 1, 12, 5)

usar_api = st.sidebar.checkbox("Usar dados automáticos da NASA", value=True)

# ================================
# 4. Obter dados meteorológicos
# ================================
if usar_api:
    with st.spinner("A buscar dados da NASA..."):
        nasa_data = get_nasa_data(latitude, longitude, ano, mes)

    if nasa_data:
        st.success("Dados obtidos com sucesso!")
    else:
        st.error("Erro ao obter dados da NASA. Verifique a conexão ou coordenadas.")
else:
    st.write("Insere manualmente os dados meteorológicos:")
    nasa_data = {
        "precipitacao_mm": st.number_input("Precipitação (mm)", value=100.0),
        "temperatura_C": st.number_input("Temperatura (°C)", value=27.0),
        "humidade_percent": st.number_input("Humidade (%)", value=75.0),
        "vento_kmh": st.number_input("Vento (km/h)", value=12.0),
        "rad_solar_Wm2": st.number_input("Radiação Solar (W/m²)", value=200.0)
    }

if st.button("🔍 Prever Risco de Inundação"):
    if nasa_data is None:
        st.error("Não foi possível obter os dados meteorológicos.")
    else:
        # Montar dataframe
        df_pred = pd.DataFrame([{
            'provincia': le.transform([provincia])[0] if le else 0,
            'latitude': latitude,
            'longitude': longitude,
            'ano': ano,
            'mes': mes,
            **nasa_data
        }])

        # Fazer previsão
        pred = modelo.predict(df_pred)[0]
        prob = modelo.predict_proba(df_pred)[0][1] if hasattr(modelo, "predict_proba") else None

        # Mostrar resultados
        if pred == 1:
            st.error(f"Risco de Inundação Detectado! (Probabilidade: {prob:.2%})" if prob else "🚨 Risco de Inundação Detectado!")
        else:
            st.success(f"Sem Risco de Inundação. (Probabilidade: {prob:.2%})" if prob else "✅ Sem Risco de Inundação.")

        st.write("Dados usados para previsão:")
        st.dataframe(df_pred)
