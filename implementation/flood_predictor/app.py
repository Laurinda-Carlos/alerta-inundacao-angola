from flask import Flask, request, jsonify, render_template
from markupsafe import Markup

import pandas as pd
import joblib
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import folium

# Dicionário com imagens das províncias
img_info = {
    "Huambo": "https://media-cdn.tripadvisor.com/media/photo-c/1280x250/07/8c/b4/9a/huambo.jpg",
    "Cabinda": "https://2.bp.blogspot.com/-dy8aDGWahKE/WrEvnXYHPbI/AAAAAAAAUZY/nLHK-vJzq_A5d-xrqIq4wnRkbdvt2jgZQCEwYBhgL/s1600/cabinda1.png",
    "Uige": "https://static.africa-press.net/angola/sites/65/2023/06/sm_1687433351.90146.jpg",
    "Huila": "https://dilemma-x.net/wp-content/uploads/2017/02/luanda-skyline-17.jpg",
    "Benguela": "https://cdn-images.rtp.pt/EPG/imagens/43085_70372_73197.jpg?amp;w=270",
    "Cunene": "https://welcometoangola.co.ao/wp-content/uploads/2021/01/foz-rio-cunene-pedro-carreno-1240x827-1.jpg",
    "Moxico": "https://welcometoangola.co.ao/wp-content/uploads/2021/01/JGPM.jpg",
    "Luanda": "https://dilemma-x.net/wp-content/uploads/2017/02/luanda-skyline-17.jpg"
}
# ===============================
# Inicializar Flask
# ===============================
app = Flask(__name__)

# ===============================
# Carregar modelos e artefatos
# ===============================
ARTIFACTS_DIR = "models"

# Melhor modelo
MODEL_FILE = max(
    [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("melhor_modelo_inundacoes")],
    key=lambda x: os.path.getctime(os.path.join(ARTIFACTS_DIR, x))
)
model = joblib.load(os.path.join(ARTIFACTS_DIR, MODEL_FILE))

# Features esperadas
FEATURES_FILE = max(
    [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("feature_columns")],
    key=lambda x: os.path.getctime(os.path.join(ARTIFACTS_DIR, x))
)
feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, FEATURES_FILE))

# Label encoders
ENCODERS_FILE = max(
    [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("label_encoders")],
    key=lambda x: os.path.getctime(os.path.join(ARTIFACTS_DIR, x))
)
label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, ENCODERS_FILE))

# Estatísticas por província
PROV_STATS_FILE = max(
    [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("provincia_stats")],
    key=lambda x: os.path.getctime(os.path.join(ARTIFACTS_DIR, x))
)
provincia_stats = joblib.load(os.path.join(ARTIFACTS_DIR, PROV_STATS_FILE))

# ===== Função auxiliar para previsão de risco =====
def prever_risco(df_row, provincia):
    df_model = pd.DataFrame([{
        "precipitacao": df_row["precipitacao_mm"],
        "temperatura": df_row["temperatura_C"],
        "umidade": df_row["humidade_percent"],
        "provincia": provincia
    }])
    # Preencher colunas faltantes
    for col in feature_columns:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[feature_columns]
    prob = model.predict_proba(df_model)[:,1][0]
    risco = "Baixo" if prob < 0.33 else "Médio" if prob < 0.66 else "Alto"
    return prob, risco
# ===============================
# Rotas do Flask
# ===============================

# Página principal
@app.route("/")
def index():
    return render_template("index.html")

# ====================================================
# Dados simulados (podes substituir pela tua base real)
# ====================================================
provincia_stats = [
    {"nome": "Luanda", "latitude": -8.8383, "longitude": 13.2344, "risco": "Alto"},
    {"nome": "Benguela", "latitude": -12.5783, "longitude": 13.4072, "risco": "Médio"},
    {"nome": "Huambo", "latitude": -12.7761, "longitude": 15.7392, "risco": "Baixo"},
    {"nome": "Cabinda", "latitude": -5.5500, "longitude": 12.2000, "risco": "Médio"},
    {"nome": "Uíge", "latitude": -7.6167, "longitude": 15.0500, "risco": "Médio"},
    {"nome": "Huíla", "latitude": -14.9167, "longitude": 13.5000, "risco": "Baixo"},
    {"nome": "Cunene", "latitude": -17.0667, "longitude": 15.7333, "risco": "Alto"},
    {"nome": "Moxico", "latitude": -11.7833, "longitude": 19.9167, "risco": "Médio"}
]


# ====================================================
# Função auxiliar para escolher a cor por risco
# ====================================================
def cor_por_risco(risco):
    cores = {
        "Alto": "red",
        "Médio": "orange",
        "Baixo": "green"
    }
    return cores.get(risco, "gray")

# ====================================================
# Rota para exibir o mapa
# ====================================================
@app.route("/mapa")
def mapa():
    # Define o ponto central de Angola
    mapa_angola = folium.Map(location=[-12.5, 17.5], zoom_start=6, tiles="CartoDB positron")

    # Adiciona marcadores das províncias
    for p in provincia_stats:
        folium.Marker(
            location=[p["latitude"], p["longitude"]],
            popup=f"<b>{p['nome']}</b><br>Risco: {p['risco']}",
            icon=folium.Icon(color=cor_por_risco(p["risco"]))
        ).add_to(mapa_angola)

    # Renderiza o HTML do mapa
    mapa_html = mapa_angola._repr_html_()

    # Passa o HTML para o template
    return render_template("mapa.html", mapa_html=Markup(mapa_html))

# Página de informações (estática)
@app.route("/informacoes")
def informacoes():
    return render_template("informacoes.html")

# Página de documentação
@app.route("/documentacao")
def documentacao():
    return render_template("documetacao.html")

# ===============================
# API Endpoints
# ===============================

# POST /api/predict
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    # Checar campos obrigatórios
    required_fields = ["precipitacao", "temperatura", "umidade", "provincia"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Campo '{field}' é obrigatório"}), 400

    # Criar DataFrame com uma linha
    df = pd.DataFrame([data])

    # Codificar província
    if "provincia" in df.columns and df["provincia"].iloc[0] in label_encoders["provincia"].classes_:
        df["provincia"] = label_encoders["provincia"].transform(df["provincia"])
    else:
        return jsonify({"error": f"Província '{data['provincia']}' inválida"}), 400

    # Preencher valores ausentes com zero (ou medianas futuras se quiser)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Garantir a ordem correta das features
    df = df[feature_columns]

    # Prever
    probabilidade = model.predict_proba(df)[:, 1][0]
    risco = "Baixo" if probabilidade < 0.33 else "Médio" if probabilidade < 0.66 else "Alto"

    return jsonify({
        "provincia": data["provincia"],
        "probabilidade_inundacao": round(probabilidade, 2),
        "risco": risco
    })

# GET /api/provincias
@app.route("/api/provincias", methods=["GET"])
def api_provincias():
    provincias_risco = []
    for prov, stats in provincia_stats.iterrows():
        flood_rate = stats["flood_rate_prov"]
        risco = "Baixo" if flood_rate < 0.33 else "Médio" if flood_rate < 0.66 else "Alto"
        provincias_risco.append({
            "provincia": prov,
            "risco": risco
        })
    return jsonify(provincias_risco)

from datetime import datetime, timedelta

@app.route("/api/hero/<provincia>", methods=["GET"])
def api_hero(provincia):
    # Carregar dados do CSV da província
    df = pd.read_csv(f"datasets_provincias/{provincia}_nasa_power.csv")

    # Últimos valores de vento e humidade
    ultima_linha = df.iloc[-1]
    vento = ultima_linha["vento_kmh"]
    humidade = ultima_linha["humidade_percent"]

    # Previsão de risco atual
    data_atual = {
        "precipitacao": ultima_linha["precipitacao_mm"],
        "temperatura": ultima_linha["temperatura_C"],
        "umidade": ultima_linha["humidade_percent"],
        "provincia": provincia
    }
    df_model = pd.DataFrame([data_atual])
    df_model["provincia"] = label_encoders["provincia"].transform(df_model["provincia"])
    df_model = df_model[feature_columns]
    probabilidade = model.predict_proba(df_model)[:, 1][0]
    risco_atual = "Baixo" if probabilidade < 0.33 else "Médio" if probabilidade < 0.66 else "Alto"

    # Previsão dos próximos 4 meses
    meses = []
    hoje = datetime.today()
    for i in range(4):
        mes_data = hoje + timedelta(days=30*i)
        # Exemplo simples usando média histórica de flood_rate (se existir)
        flood_rate = df["flood_rate"].mean() if "flood_rate" in df.columns else 0
        risco_mes = "Baixo" if flood_rate < 0.33 else "Médio" if flood_rate < 0.66 else "Alto"
        meses.append({"mes": mes_data.strftime("%B"), "risco": risco_mes})

    return jsonify({
        "provincia": provincia,
        "vento": vento,
        "humidade": humidade,
        "risco_atual": risco_atual,
        "previsao_meses": meses
    })

# ===============================
# Executar Flask
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
