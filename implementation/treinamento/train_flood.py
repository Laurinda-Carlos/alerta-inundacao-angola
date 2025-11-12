# sistema_prevencao_inundacoes_angola.py
"""
SISTEMA DE PREVENÇÃO DE INUNDAÇÕES PARA ANGOLA
Descrição: Sistema de machine learning para predição e prevenção de inundações
no contexto climático de Angola.
"""

# CONFIGURAÇÕES INICIAIS E IMPORTAÇÕES

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Tratamento de desbalanceamento
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Modelos avançados
from xgboost import XGBClassifier

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

logger.info("Bibliotecas carregadas com sucesso!")

# FUNÇÕES DE CARREGAMENTO DE DADOS

def carregar_dados(caminho_arquivo):
    """
    Carrega dados de arquivo CSV ou Excel
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo de dados
        
    Returns:
        pandas.DataFrame: DataFrame com os dados carregados
    """
    logger.info(f"Carregando dados de: {caminho_arquivo}")
    
    try:
        if caminho_arquivo.endswith('.csv'):
            df = pd.read_csv(caminho_arquivo)
        elif caminho_arquivo.endswith('.xlsx'):
            df = pd.read_excel(caminho_arquivo)
        else:
            raise ValueError("Formato não suportado. Use CSV ou Excel.")
        
        logger.info(f"Dataset carregado! Shape: {df.shape}")
        
        # Mostrar preview
        print("\nPRÉ-VISUALIZAÇÃO DOS DADOS:")
        print(df.head())
        
        print(f"\nESTATÍSTICAS BÁSICAS:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

# ANÁLISE EXPLORATÓRIA

def analise_exploratoria(df):
    """
    Realiza análise exploratória dos dados
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados
    """
    print("INICIANDO ANÁLISE EXPLORATÓRIA...")
    
    # 1. Informações básicas
    print(f"\nSHAPE DO DATASET: {df.shape}")
    print(f"\nTIPOS DE DADOS:")
    print(df.dtypes)
    
    # 2. Valores faltantes
    print(f"\nVALORES FALTANTES:")
    missing_data = df.isnull().sum()
    for col, missing in missing_data[missing_data > 0].items():
        print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # 3. Distribuição do target
    print(f"\nDISTRIBUIÇÃO DO TARGET (flood):")
    target_dist = df['flood'].value_counts()
    print(target_dist)
    
    # 4. Correlações
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5)
    plt.title('Mapa de Correlação - Variáveis Numéricas')
    plt.tight_layout()
    plt.show()
    
    # 5. Distribuição de features numéricas
    numeric_features = ['precipitacao_mm', 'temperatura_C', 'humidade_percent']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(numeric_features):
        if i < len(axes):
            df[feature].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'Distribuição de {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequência')
    
    # Remover eixo extra se necessário
    if len(numeric_features) < len(axes):
        fig.delaxes(axes[len(numeric_features)])
    
    plt.tight_layout()
    plt.show()
    
    # 6. Boxplots por status de inundação
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='flood', y='precipitacao_mm')
    plt.title('Precipitação vs Inundação')
    plt.show()

# ENGENHARIA DE FEATURES PARA ANGOLA

def criar_features_angola(df):
    """
    Cria features específicas para o contexto de Angola
    
    Args:
        df (pandas.DataFrame): DataFrame original
        
    Returns:
        pandas.DataFrame: DataFrame com features adicionais
    """
    logger.info("Criando features climáticas para Angola...")
    
    # Mapeamento de regiões climáticas
    REGIOES_CLIMATICAS = {
        'Litoral': ['Luanda', 'Benguela', 'Cabinda', 'Namibe', 'Zaire'],
        'Semi-Árido': ['Cunene', 'Namibe', 'Cuando Cubango'],
        'Planalto Central': ['Huambo', 'Bié', 'Huíla'],
        'Norte Úmido': ['Uíge', 'Malanje', 'Kwanza Norte']
    }
    
    def classificar_regiao(provincia):
        for regiao, provincias in REGIOES_CLIMATICAS.items():
            if provincia in provincias:
                return regiao
        return 'Outra'
    
    # Features baseadas em conhecimento de domínio
    if 'provincia' in df.columns:
        df['regiao_climatica'] = df['provincia'].apply(classificar_regiao)
    
    # Estações climáticas de Angola
    def estacao_angola(mes):
        # Chuva: Out-Abr | Seca: Mai-Set
        if mes in [10, 11, 12, 1, 2, 3, 4]:
            return 'Chuvosa'
        else:
            return 'Seca'
    
    if 'mes' in df.columns:
        df['estacao'] = df['mes'].apply(estacao_angola)
        df['estacao_chuvosa'] = (df['mes'].isin([10, 11, 12, 1, 2, 3, 4])).astype(int)
    
    # Features de risco
    df['precipitacao_extrema'] = (df['precipitacao_mm'] > 150).astype(int)
    df['risco_composto'] = (df['precipitacao_mm'] * df['humidade_percent']) / 100
    
    # Interações climáticas
    df['temp_umidade_interaction'] = df['temperatura_C'] * df['humidade_percent']
    df['precipitacao_quadrado'] = df['precipitacao_mm'] ** 2
    
    # Indicadores de tendência
    if 'dia' in df.columns:
        df['final_de_semana'] = ((df['dia'] % 7) >= 5).astype(int)
    
    logger.info(f"Features criadas! Colunas: {list(df.columns)}")
    return df

# SISTEMA DE MODELAGEM

class SistemaPrevencaoInundacoes:
    """Sistema completo de prevenção de inundações"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = None
        self.feature_names = []
        
    def preparar_dados(self, df):
        """Prepara dados para modelagem"""
        
        # Definir features base
        features_base = [
            'precipitacao_mm', 'temperatura_C', 'humidade_percent', 
            'vento_kmh', 'rad_solar_Wm2'
        ]
        
        # Features de engenharia
        features_engenharia = [
            'estacao_chuvosa', 'precipitacao_extrema', 'risco_composto',
            'temp_umidade_interaction', 'precipitacao_quadrado'
        ]
        
        # Combinar todas as features
        self.feature_names = features_base + features_engenharia
        
        # Garantir que todas as features existem
        features_finais = [f for f in self.feature_names if f in df.columns]
        
        X = df[features_finais]
        y = df['flood']
        
        logger.info(f"Features selecionadas: {features_finais}")
        logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")
        
        return X, y, features_finais
    
    def criar_preprocessador(self, features):
        """Cria pipeline de pré-processamento"""
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features)
            ]
        )
        
        return self.preprocessor
    
    def treinar_modelos(self, X_train, y_train, features):
        """Treina múltiplos modelos com otimização"""
        
        # Calcular pesos para balanceamento
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Configuração dos modelos
        modelos_config = {
            'XGBoost': {
                'model': XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    scale_pos_weight=weight_dict[1]/weight_dict[0]
                ),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 6, 9],
                    'classifier__learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__class_weight': ['balanced', None]
                }
            }
        }
        
        # Treinar cada modelo
        for nome, config in modelos_config.items():
            logger.info(f"Treinando {nome}...")
            
            try:
                # Pipeline com SMOTE
                pipeline = ImbPipeline([
                    ('preprocessor', self.preprocessor),
                    ('smote', SMOTE(random_state=RANDOM_STATE)),
                    ('classifier', config['model'])
                ])
                
                # Busca de hiperparâmetros
                busca = RandomizedSearchCV(
                    pipeline, config['params'], 
                    n_iter=10, cv=3, scoring='f1',
                    random_state=RANDOM_STATE, n_jobs=-1
                )
                
                busca.fit(X_train, y_train)
                self.models[nome] = busca.best_estimator_
                
                logger.info(f"{nome} treinado! Melhor F1: {busca.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Erro em {nome}: {e}")
    
    def avaliar_modelos(self, X_test, y_test):
        """Avalia todos os modelos treinados"""
        
        resultados = {}
        
        for nome, modelo in self.models.items():
            logger.info(f"Avaliando {nome}...")
            
            # Previsões
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
            
            # Métricas
            metrics = {
                'acuracia': accuracy_score(y_test, y_pred),
                'precisao': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_proba),
                'modelo': modelo,
                'previsoes': y_pred,
                'probabilidades': y_proba
            }
            
            # Matriz de confusão
            metrics['matriz_confusao'] = confusion_matrix(y_test, y_pred)
            
            resultados[nome] = metrics
            
            print(f"\n{nome}:")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  Precisão: {metrics['precisao']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
        
        self.results = resultados
        return resultados
    
    def visualizar_resultados(self, df, X_test, y_test):
        """Cria visualizações completas dos resultados"""
        
        if not self.results:
            logger.warning("Nenhum resultado para visualizar")
            return
        
        # 1. Comparação de modelos
        modelos = list(self.results.keys())
        f1_scores = [self.results[m]['f1'] for m in modelos]
        auc_scores = [self.results[m]['auc_roc'] for m in modelos]
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(modelos))
        width = 0.35
        
        plt.bar(x_pos - width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        plt.bar(x_pos + width/2, auc_scores, width, label='AUC-ROC', alpha=0.8)
        
        plt.xlabel('Modelos')
        plt.ylabel('Score')
        plt.title('Comparação de Desempenho dos Modelos')
        plt.xticks(x_pos, modelos, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Curvas ROC
        plt.figure(figsize=(10, 8))
        
        # Linha de base
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Linha Base')
        
        for nome, resultado in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, resultado['probabilidades'])
            auc_score = resultado['auc_roc']
            
            plt.plot(fpr, tpr, label=f'{nome} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC - Comparação de Modelos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 3. Mapa de calor de correlações
        plt.figure(figsize=(12, 10))
        correlacoes = df[self.feature_names + ['flood']].corr()
        sns.heatmap(correlacoes, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlação entre Features e Inundações')
        plt.tight_layout()
        plt.show()
        
        # 4. Importância de features (apenas para modelos que suportam)
        for nome, resultado in self.results.items():
            modelo = resultado['modelo']
            if hasattr(modelo.named_steps['classifier'], 'feature_importances_'):
                importancias = modelo.named_steps['classifier'].feature_importances_
                indices = np.argsort(importancias)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Importância de Features - {nome}')
                plt.bar(range(len(importancias)), importancias[indices])
                plt.xticks(range(len(importancias)), [self.feature_names[i] for i in indices], rotation=45)
                plt.tight_layout()
                plt.show()

# FUNÇÃO PRINCIPAL

def main():
    """Função principal do sistema"""
    
    print("INICIANDO SISTEMA DE PREVENÇÃO DE INUNDAÇÕES ANGOLA")
    print("=" * 50)
    
    try:
        # 1. Carregar dados
        caminho_arquivo = input("Digite o caminho para o arquivo de dados: ").strip()
        df = carregar_dados(caminho_arquivo)
        
        # 2. Análise exploratória
        analise_exploratoria(df)
        
        # 3. Engenharia de features
        df = criar_features_angola(df)
        
        # 4. Sistema de modelagem
        sistema = SistemaPrevencaoInundacoes()
        X, y, features = sistema.preparar_dados(df)
        
        # 5. Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\nDIVISÃO DOS DADOS:")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        print(f"  Features: {len(features)} variáveis")
        
        # 6. Pré-processamento
        preprocessor = sistema.criar_preprocessador(features)
        
        # 7. Treinamento
        sistema.treinar_modelos(X_train, y_train, features)
        
        if not sistema.models:
            logger.error("Nenhum modelo foi treinado com sucesso")
            return
        
        # 8. Avaliação
        resultados = sistema.avaliar_modelos(X_test, y_test)
        
        # 9. Visualizações
        sistema.visualizar_resultados(df, X_test, y_test)
        
        # 10. Selecionar melhor modelo
        melhor_modelo_nome = max(resultados, key=lambda x: resultados[x]['f1'])
        melhor_modelo = resultados[melhor_modelo_nome]['modelo']
        
        print(f"\n{'='*50}")
        print(f"MELHOR MODELO: {melhor_modelo_nome}")
        print(f"F1-Score: {resultados[melhor_modelo_nome]['f1']:.4f}")
        print(f"AUC-ROC: {resultados[melhor_modelo_nome]['auc_roc']:.4f}")
        print(f"{'='*50}")
        
        # 11. Salvar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"modelo_inundacoes_angola_{timestamp}.pkl"
        
        with open(nome_arquivo, 'wb') as f:
            pickle.dump({
                'modelo': melhor_modelo,
                'features': features,
                'preprocessor': preprocessor,
                'metricas': resultados[melhor_modelo_nome],
                'timestamp': timestamp
            }, f)
        
        print(f"Modelo salvo como: {nome_arquivo}")
        
        # 12. Exemplo de predição
        print(f"\nEXEMPLO DE PREDIÇÃO:")
        exemplo = X_test.iloc[0:1]
        predicao = melhor_modelo.predict(exemplo)
        probabilidade = melhor_modelo.predict_proba(exemplo)[0, 1]
        
        print(f"  Dados: {exemplo.values[0]}")
        print(f"  Predição: {'INUNDAÇÃO' if predicao[0] == 1 else 'NORMAL'}")
        print(f"  Probabilidade: {probabilidade:.1%}")
        
        print("\nSISTEMA CONCLUÍDO COM SUCESSO!")
        
    except Exception as e:
        logger.error(f"Erro no sistema: {e}")
        raise

# EXECUÇÃO

if __name__ == "__main__":
    main()