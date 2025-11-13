import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Carregar artefatos
# -----------------------------
modelo_filename = "melhor_modelo_inundacoes_20251112_021908.pkl"
feature_columns_filename = "feature_columns_20251112_021908.pkl"
label_encoders_filename = "label_encoders_20251112_021908.pkl"
provincia_stats_filename = "provincia_stats_20251112_021908.pkl"

# Carregar pipeline treinado
modelo = joblib.load(modelo_filename)

# Carregar lista de features esperadas
feature_columns = joblib.load(feature_columns_filename)

# Carregar LabelEncoders
label_encoders = joblib.load(label_encoders_filename)

# Carregar estatísticas por província
provincia_stats = joblib.load(provincia_stats_filename)

print("Artefatos carregados com sucesso!")

# -----------------------------
# 2. Função para engenharia de features
# -----------------------------
def criar_features(df):
    # Features de interação
    df['precip_temp_interaction'] = df['precipitacao_mm'] * df['temperatura_C']
    df['precip_humidity_interaction'] = df['precipitacao_mm'] * df['humidade_percent']
    df['temp_humidity_interaction'] = df['temperatura_C'] * df['humidade_percent']
    
    # Sazonalidade
    df['estacao_chuvosa'] = df['mes'].apply(lambda x: 1 if x in [11, 12, 1, 2, 3] else 0)
    df['trimestre'] = (df['mes'] - 1) // 3 + 1
    
    # Risco composto
    df['risco_precipitacao'] = df['precipitacao_mm'].apply(lambda x: 0 if x < 100 else (1 if x < 180 else 2))
    df['risco_humidade'] = df['humidade_percent'].apply(lambda x: 0 if x < 70 else (1 if x < 85 else 2))
    df['score_risco_total'] = df['risco_precipitacao'] + df['risco_humidade'] + df['estacao_chuvosa']
    
    # Agregação por província
    df = df.merge(provincia_stats, left_on='provincia', right_index=True, how='left')
    df['precip_vs_prov_mean'] = df['precipitacao_mm'] - df['precip_mean_prov']
    df['temp_vs_prov_mean'] = df['temperatura_C'] - df['temp_mean_prov']
    
    # Polinomiais
    df['precipitacao_quadratica'] = df['precipitacao_mm'] ** 2
    df['temperatura_quadratica'] = df['temperatura_C'] ** 2
    
    # Codificação de variáveis categóricas
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Garantir a mesma ordem das features
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

# -----------------------------
# 3. Receber dados de entrada
# Exemplo manual
# -----------------------------
entrada = pd.DataFrame([{
    'provincia': 'Luanda',
    'mes': 11,
    'precipitacao_mm': 1,
    'temperatura_C': 28,
    'humidade_percent': 80,
    'latitude': -8.839987,
    'longitude': 13.289437
}])

# Criar features
X_input = criar_features(entrada)

# -----------------------------
# 4. Previsão de probabilidade
# -----------------------------
probabilidade_flood = modelo.predict_proba(X_input)[:, 1]  # probabilidade de flood = 1

# -----------------------------
# 5. Exibir resultados
# -----------------------------
entrada['probabilidade_flood'] = probabilidade_flood
entrada['risco'] = entrada['probabilidade_flood'].apply(
    lambda x: 'Baixo' if x < 0.33 else ('Médio' if x < 0.66 else 'Alto')
)

print("\nProbabilidade de risco de inundação:")
print(entrada[['provincia', 'mes', 'probabilidade_flood', 'risco']])
