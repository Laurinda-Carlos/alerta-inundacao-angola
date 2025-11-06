
# 1. INSTALAR BIBLIOTECAS (se necessário)
# !pip install xgboost -q
# !pip install plotly -q

# 2. IMPORTAR BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print(" Bibliotecas importadas com sucesso!")

# 3. CARREGAMENTO DOS DADOS
# Carregar dataset
df = pd.read_csv('dataset_realista.csv')
print(f" Dataset 'dataset_realista.csv' carregado com sucesso!")
print(f"Dimensões do dataset: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head())

# 4. EXPLORAÇÃO E ANÁLISE DE DADOS (EAD)
print(" EXPLORAÇÃO COMPLETA DO DATASET")
print(f" Dimensões do dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Análise da variável target
print(f"\n ANÁLISE DA VARIÁVEL TARGET 'flood':")
target_analysis = df['flood'].value_counts()
target_percentage = df['flood'].value_counts(normalize=True) * 100
print(f"   Flood = 0: {target_analysis[0]} amostras ({target_percentage[0]:.1f}%)")
print(f"   Flood = 1: {target_analysis[1]} amostras ({target_percentage[1]:.1f}%)")

# Estatísticas descritivas detalhadas
print(f"\n ESTATÍSTICAS DESCRITIVAS DETALHADAS:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# Análise de valores ausentes
print(f"\n ANÁLISE DE VALORES AUSENTES:")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("  Nenhum valor ausente encontrado")
else:
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"   {col}: {missing} valores ausentes ({missing/len(df)*100:.1f}%)")

# Análise de outliers usando IQR
print(f"\n DETECÇÃO DE OUTLIERS (Método IQR):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Análise por província
print(f"\n ANÁLISE POR PROVÍNCIA:")
print(f"   Número de províncias: {df['provincia'].nunique()}")
provincia_stats = df.groupby('provincia').agg({
    'flood': ['count', 'mean'],
    'precipitacao_mm': 'mean',
    'temperatura_C': 'mean'
}).round(3)

for prov in df['provincia'].unique():
    prov_data = df[df['provincia'] == prov]
    flood_rate = prov_data['flood'].mean() * 100
    avg_precip = prov_data['precipitacao_mm'].mean()
    print(f"   {prov}: {len(prov_data)} amostras, {flood_rate:.1f}% flood, {avg_precip:.1f}mm precip")

# Visualizações
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gráfico 1: Distribuição do Target
fig1 = px.pie(df, names='flood', title='Distribuição da Variável Target (Flood)',
              color='flood', color_discrete_map={0:'lightblue', 1:'salmon'})
fig1.update_traces(textinfo='percent+label')
fig1.show()

# Gráfico 2: Distribuição por Província
fig2 = px.sunburst(df, path=['provincia', 'flood'],
                   title='Distribuição de Flood por Província',
                   color='flood', color_continuous_scale='Blues')
fig2.show()

# Gráfico 3: Precipitação vs Flood
fig3 = px.box(df, x='flood', y='precipitacao_mm',
              title='Precipitação vs Ocorrência de Flood',
              color='flood', color_discrete_map={0:'lightblue', 1:'salmon'})
fig3.update_layout(xaxis_title='Flood', yaxis_title='Precipitação (mm)')
fig3.show()

# Gráfico 4: Mapa Interativo de Floods
fig4 = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                         color="flood", size="precipitacao_mm",
                         hover_name="provincia",
                         hover_data=["temperatura_C", "humidade_percent"],
                         title="Mapa de Ocorrências de Flood em Angola",
                         color_continuous_scale=px.colors.diverging.RdYlBu_r,
                         zoom=5, height=500)
fig4.update_layout(mapbox_style="open-street-map")
fig4.show()

# 5. PRÉ-PROCESSAMENTO DE DADOS
print("\n=== PRÉ-PROCESSAMENTO DE DADOS ===")

# Separar features e target
X = df.drop('flood', axis=1)
y = df['flood']

# Definir colunas numéricas e categóricas
numeric_features = ['latitude', 'longitude', 'ano', 'mes', 'precipitacao_mm', 
                   'temperatura_C', 'humidade_percent', 'vento_kmh', 'rad_solar_Wm2']
categorical_features = ['provincia']

# Criar transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print(" Pré-processador criado com sucesso!")

# 6. DIVISÃO DOS DADOS
print("\n=== DIVISÃO DOS DADOS ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Dados de treino: {X_train.shape[0]} amostras")
print(f" Dados de teste: {X_test.shape[0]} amostras")
print(f" Proporção de flood no treino: {y_train.mean():.3f}")
print(f" Proporção de flood no teste: {y_test.mean():.3f}")

# 7. MODELAGEM
print("\n=== MODELAGEM ===")

# Definir modelos
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# Criar pipelines
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Treinar e avaliar modelos
results = {}

for name, pipeline in pipelines.items():
    print(f"\n--- Treinando {name} ---")
    
    # Treinar modelo
    pipeline.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Armazenar resultados
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'model': pipeline
    }
    
    print(f" Acurácia: {accuracy:.4f}")
    print(f" Precisão: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1-Score: {f1:.4f}")
    print(f" AUC-ROC: {auc:.4f}")

# 8. COMPARAÇÃO DE MODELOS
print("\n=== COMPARAÇÃO DE MODELOS ===")

# Criar DataFrame com resultados
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('f1', ascending=False)
print("\n Comparação de Modelos (ordenado por F1-Score):")
print(results_df[['accuracy', 'precision', 'recall', 'f1', 'auc']].round(4))

# Gráfico de comparação
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x = np.arange(len(metrics))
width = 0.25

for i, (model, result) in enumerate(results.items()):
    values = [result[metric] for metric in metrics]
    ax.bar(x + i*width, values, width, label=model)

ax.set_xlabel('Métricas')
ax.set_ylabel('Score')
ax.set_title('Comparação de Modelos')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()

# 9. MODELO FINAL
print("\n=== MODELO FINAL ===")
best_model_name = results_df.index[0]
best_model = results[best_model_name]['model']
print(f" Melhor modelo: {best_model_name}")

# Relatório de classificação detalhado
y_pred_final = best_model.predict(X_test)
print("\n Relatório de Classificação:")
print(classification_report(y_test, y_pred_final))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Modelo Final')
plt.ylabel('Valor Real')
plt.xlabel('Previsão')
plt.show()

# 10. CURVA ROC
print("\n=== CURVA ROC ===")
plt.figure(figsize=(10, 8))

for name, result in results.items():
    pipeline = result['model']
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC - Comparação de Modelos')
plt.legend()
plt.grid(True)
plt.show()

# 11. VALIDAÇÃO CRUZADA
print("\n=== VALIDAÇÃO CRUZADA ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, pipeline in pipelines.items():
    print(f"\n Validando {name} com Validação Cruzada...")
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    cv_results[name] = {
        'mean_f1': cv_scores.mean(),
        'std_f1': cv_scores.std(),
        'scores': cv_scores
    }
    print(f" F1-Score médio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 12. OTIMIZAÇÃO DE HIPERPARÂMETROS
print("\n=== OTIMIZAÇÃO DE HIPERPARÂMETROS ===")

# Otimizar Random Forest
print(" Otimizando Random Forest...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    rf_pipeline, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=1
)
grid_search_rf.fit(X_train, y_train)

print(f" Melhores parâmetros: {grid_search_rf.best_params_}")
print(f" Melhor score: {grid_search_rf.best_score_:.4f}")

# 13. FEATURE IMPORTANCE
print("\n=== FEATURE IMPORTANCE ===")

# Obter feature names após pré-processamento
feature_names = numeric_features.copy()
categorical_features_encoded = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names.extend(categorical_features_encoded)

# Obter importâncias do melhor modelo
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Criar DataFrame com importâncias
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n Top 10 Features Mais Importantes:")
    print(feature_importance_df.head(10))
    
    # Gráfico de importância
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Features Mais Importantes')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()

# 14. SALVAR MODELO
print("\n=== SALVANDO MODELO ===")
import joblib

# Salvar o melhor modelo
joblib.dump(best_model, 'melhor_modelo_flood.pkl')
print(" Modelo salvo como 'melhor_modelo_flood.pkl'")

# Salvar o pré-processador separadamente
joblib.dump(preprocessor, 'preprocessor.pkl')
print(" Pré-processador salvo como 'preprocessor.pkl'")

# 15. EXEMPLO DE PREDIÇÃO
print("\n=== EXEMPLO DE PREDIÇÃO ===")
sample_data = X_test.iloc[:3].copy()
print(" Dados de exemplo:")
print(sample_data)

predictions = best_model.predict(sample_data)
probabilities = best_model.predict_proba(sample_data)

print("\n Previsões:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f" Amostra {i+1}: Flood = {pred} (probabilidade: {prob[1]:.4f})")

print("\n=== TREINAMENTO CONCLUÍDO ===")
print(f" Melhor modelo: {best_model_name}")
print(f" F1-Score no teste: {results[best_model_name]['f1']:.4f}")
print(f" AUC-ROC no teste: {results[best_model_name]['auc']:.4f}")