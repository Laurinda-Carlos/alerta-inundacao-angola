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
"""
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
"""
modelo = joblib.load("models/melhor_modelo_inundacoes_20251113_052939.pkl")
feature_columns = joblib.load("models/feature_columns_20251113_052939.pkl")
label_encoders = joblib.load("models/label_encoders_20251113_052939.pkl")
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

# Página de documentação
@app.route("/prever")
def prever():
    return render_template("predict.html")
# ===============================
# API Endpoints
# ===============================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extrair variáveis
        provincia = data.get("provincia", "Luanda")
        mes = int(data.get("mes", 11))
        precipitacao = float(data.get("precipitacao", 0))
        temperatura = float(data.get("temperatura", 0))
        humidade = float(data.get("humidade", 0))
        vento = float(data.get("vento", 0))
        radiacao = float(data.get("radiacao_solar", 0))

        # Criar DataFrame
        entrada = pd.DataFrame([{
            "provincia": provincia,
            "mes": mes,
            "precipitacao": precipitacao,
            "temperatura": temperatura,
            "humidade": humidade,
            "vento": vento,
            "radiacao_solar": radiacao
        }])

        # Aplicar Label Encoders, se houver colunas categóricas
        for col, encoder in label_encoders.items():
            if col in entrada.columns:
                entrada[col] = encoder.transform(entrada[col])

        # Garantir alinhamento de colunas com o treino
        entrada = entrada.reindex(columns=feature_columns, fill_value=0)

        # Fazer previsão
        prob = modelo.predict_proba(entrada)[0][1]  # probabilidade da classe "inundação"

        # Classificar o risco
        if prob >= 0.7:
            risco = "Alto"
        elif prob >= 0.4:
            risco = "Médio"
        else:
            risco = "Baixo"

        return jsonify({
            "provincia": provincia,
            "mes": mes,
            "probabilidade_flood": round(float(prob), 3),
            "risco": risco
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

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

# ===============================
# Executar Flask
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
