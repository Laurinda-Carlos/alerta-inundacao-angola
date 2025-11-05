# 1. Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Carregar dataset
df = pd.read_csv("dataset.csv")

# 3. Explorar rapidamente os dados
print(df.head())
print(df.info())
print(df['flood'].value_counts())

# 4. Pré-processamento
# Converter 'provincia' para numérico (LabelEncoder)
le = LabelEncoder()
df['provincia'] = le.fit_transform(df['provincia'])

# Definir features (X) e target (y)
X = df.drop('flood', axis=1)
y = df['flood']

# 5. Dividir em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Inicializar modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000)
}

# 7. Treinar e avaliar modelos
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append([name, acc, prec, rec, f1])
    print(f"\nModelo: {name}")
    print(classification_report(y_test, y_pred))

# 8. Comparar desempenho
results_df = pd.DataFrame(results, columns=['Modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score'])
print("\nResultados comparativos:")
print(results_df)
#------------------------
import joblib

# Suponha que este foi o melhor modelo
melhor_modelo = models["Random Forest"]

# Salvar o modelo em arquivo
joblib.dump(melhor_modelo, "modelo_inundacoes.pkl")
import joblib

# salvar o label encoder
joblib.dump(le, "labelencoder_provincia.pkl")


print("✅ Modelo salvo como 'modelo_inundacoes.pkl'")

# 9. Visualizar comparativamente
results_df.set_index('Modelo')[['Acurácia','Precisão','Recall','F1-Score']].plot(kind='bar', figsize=(8,5))
plt.title("Comparação de Modelos - Previsão de Inundações")
plt.ylabel("Pontuação")
plt.show()
