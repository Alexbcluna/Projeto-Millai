# ANALISE E MODELAGEM 
# Basico de entender dados e prever notas

import os
import pickle
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_dados():
    """
    Carrega os dados do arquivo CSV e retorna o DataFrame.
    """
    diretorio = os.path.dirname(os.path.abspath(__file__))
    arquivo = os.path.join(diretorio, 'klike_challenge_dataset.csv')
    df = pd.read_csv(arquivo)
    print(f"  OK! Encontrados: {len(df)} campanhas, {len(df.columns)} colunas")
    return df


def explorar_dados(df):
    """
    Faz exploração básica dos dados: estatísticas do klike_score e contagem por plataforma.
    """
    melhor_score = df['klike_score'].max()
    pior_score = df['klike_score'].min()
    media_score = df['klike_score'].mean()

    print("  Klike Score (nota das campanhas):")
    print(f"    - Melhor: {melhor_score:.1f}")
    print(f"    - Pior: {pior_score:.1f}")
    print(f"    - Media: {media_score:.1f}")

    print("\n  Campanhas por plataforma:")
    contagem_plat = df['platform'].value_counts()
    for plat, qtd in contagem_plat.items():
        print(f"    - {plat}: {qtd} campanhas")


def criar_graficos(df):
    """
    Cria e salva gráficos de correlação, impacto do hook e importância de features.
    """
    # Gráfico 1: Correlação com klike_score
    plt.figure(figsize=(12, 6))
    colunas_numeros = df.select_dtypes(include=['float64', 'int64'])
    correlacoes = colunas_numeros.corr()['klike_score'].sort_values(ascending=False)
    top_10 = correlacoes[1:11]

    plt.barh(range(len(top_10)), top_10.values)
    plt.yticks(range(len(top_10)), top_10.index)
    plt.xlabel('Correlação (quanto influencia)')
    plt.title('O que mais influencia o Klike Score?')
    plt.tight_layout()
    plt.savefig('01_correlacoes_klike_score.png', dpi=300)
    print("  OK - Gráfico 1 salvo: 01_correlacoes_klike_score.png")

    # Gráfico 2: Impacto do Hook
    plt.figure(figsize=(10, 6))
    media_com_hook = df[df['has_hook'] == 1]['klike_score'].mean()
    media_sem_hook = df[df['has_hook'] == 0]['klike_score'].mean()

    plt.bar(['Sem Hook', 'Com Hook'], [media_sem_hook, media_com_hook], color=['red', 'green'])
    plt.ylabel('Klike Score Médio')
    plt.title('Ter Hook no início do vídeo AJUDA?')
    plt.tight_layout()
    plt.savefig('02_hook_impacto.png', dpi=300)
    print("  OK - Gráfico 2 salvo: 02_hook_impacto.png")


def preparar_dados(df):
    """
    Prepara os dados para modelagem: engenharia de features, limpeza e normalização.
    Retorna X, y, normalizador e colunas usadas.
    """
    tabela = df.copy()

    # Feature engineering
    print("  Criando novas variáveis...")
    tabela['hook_e_cta'] = (tabela['has_hook'] == 1) & (tabela['has_cta'] == 1)
    tabela['tem_rosto'] = (tabela['has_face'] == 1)
    tabela['engajamento_por_segundo'] = tabela['engagement_rate'] / (tabela['video_duration_s'] + 1)
    print("    OK - 3 novas variáveis criadas")

    # Selecionar colunas para modelo
    colunas_modelo = [col for col in tabela.columns
                      if col not in ['campaign_id', 'date', 'klike_score', 'platform',
                                     'campaign_type', 'target_audience']]
    X = tabela[colunas_modelo].select_dtypes(include=['number', 'bool'])
    y = tabela['klike_score']

    # Converter booleanos para int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Preencher NaNs com mediana
    X = X.astype(float)
    for col in X.columns:
        media_col = X[col].median()
        X[col] = X[col].fillna(media_col)

    print("  Dados preparados:")
    print(f"    - {len(X.columns)} variáveis de entrada")
    print(f"    - {len(X)} campanhas para aprender")

    # Dividir treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    print("  Dados divididos:")
    print(f"    - {len(X_treino)} campanhas para TREINO")
    print(f"    - {len(X_teste)} campanhas para TESTE")

    # Normalizar
    normalizador = StandardScaler()
    X_treino_norm = normalizador.fit_transform(X_treino)
    X_teste_norm = normalizador.transform(X_teste)
    print("    - Dados normalizados OK")

    return X_treino_norm, X_teste_norm, y_treino, y_teste, normalizador, X.columns.tolist()


def treinar_modelos(X_treino_norm, X_teste_norm, y_treino, y_teste):
    """
    Treina Regressão Linear e Random Forest, compara e retorna o melhor modelo.
    """
    # Modelo 1: Regressão Linear
    print("\n  Treinando Modelo 1: Regressão Linear...")
    modelo_simples = LinearRegression()
    modelo_simples.fit(X_treino_norm, y_treino)
    y_pred_simples = modelo_simples.predict(X_teste_norm)
    r2_simples = r2_score(y_teste, y_pred_simples)
    mae_simples = mean_absolute_error(y_teste, y_pred_simples)
    print(f"    Resultado: {r2_simples:.4f} (quanto melhor, mais perto de 1.0)")
    print(f"    Erro médio: {mae_simples:.2f} pontos de nota")

    # Modelo 2: Random Forest
    print("\n  Treinando Modelo 2: Random Forest...")
    modelo_inteligente = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_inteligente.fit(X_treino_norm, y_treino)
    y_pred_inteligente = modelo_inteligente.predict(X_teste_norm)
    r2_inteligente = r2_score(y_teste, y_pred_inteligente)
    mae_inteligente = mean_absolute_error(y_teste, y_pred_inteligente)
    print(f"    Resultado: {r2_inteligente:.4f} (quanto melhor, mais perto de 1.0)")
    print(f"    Erro médio: {mae_inteligente:.2f} pontos de nota")

    # Escolher melhor
    print("\n  Comparação:")
    if r2_simples > r2_inteligente:
        melhor_modelo = modelo_simples
        melhor_tipo = "Regressão Linear"
        melhor_r2 = r2_simples
    else:
        melhor_modelo = modelo_inteligente
        melhor_tipo = "Random Forest"
        melhor_r2 = r2_inteligente

    print(f"    MELHOR: {melhor_tipo} com score {melhor_r2:.4f}")
    return melhor_modelo, melhor_tipo


def salvar_modelo(melhor_modelo, normalizador, colunas):
    """
    Salva o modelo, normalizador e colunas em um arquivo pickle.
    """
    caixa = {
        'modelo': melhor_modelo,
        'normalizador': normalizador,
        'colunas': colunas
    }
    diretorio = os.path.dirname(os.path.abspath(__file__))
    arquivo_modelo = os.path.join(diretorio, 'best_model.joblib')
    joblib.dump(caixa, arquivo_modelo)
    print("  OK - Modelo salvo em: best_model.joblib")


def grafico_importancia(modelo, colunas, tipo_modelo):
    """
    Cria gráfico de importância das features dependendo do tipo de modelo.
    """
    plt.figure(figsize=(10, 6))
    
    if 'Random Forest' in tipo_modelo:
        importancia = modelo.feature_importances_
        titulo = 'Top 10 Variáveis mais Importantes (Random Forest)'
    elif 'Regressão Linear' in tipo_modelo:
        importancia = np.abs(modelo.coef_)
        titulo = 'Top 10 Coeficientes Absolutos (Regressão Linear)'
    else:
        print("  Modelo não suporta gráfico de importância")
        return
    
    indices = np.argsort(importancia)[::-1][:10]
    top_features = np.array(colunas)[indices]
    top_imp = importancia[indices]

    plt.barh(range(len(top_features)), top_imp)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importância (quanto influencia)')
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig('03_feature_importance.png', dpi=300)
    print("  OK - Gráfico 3 salvo: 03_feature_importance.png")


def resumo_final(df, X_treino, X_teste, melhor_tipo):
    """
    Imprime resumo final do que foi feito.
    """
    print("\n" + "="*70)
    print("ANÁLISE COMPLETA!")
    print("="*70)
    print("\nResumo do que foi feito:")
    print(f"  1. Carregamos {len(df)} campanhas")
    print(f"  2. Analisamos que {len(X_treino[0])} colunas influenciam a nota")
    print(f"  3. Treinamos 2 modelos com {len(X_treino)} campanhas")
    print(f"  4. Testamos com {len(X_teste)} campanhas novas")
    print(f"  5. Melhor modelo foi: {melhor_tipo}")
    print("  6. Salvamos o modelo para usar depois")
    print("\nArquivos criados:")
    print("  - 01_correlacoes_klike_score.png (o que influencia a nota)")
    print("  - 02_hook_impacto.png (impacto de ter hook)")
    print("  - 03_feature_importance.png (variáveis mais importantes)")
    print("  - best_model.joblib (o modelo treinado)")
    print("\n" + "="*70)


# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ANÁLISE DE CAMPANHAS DE VÍDEO - PASSO A PASSO")
    print("="*70)

    # Carregar dados
    print("\n[PASSO 1] Carregando dados do arquivo CSV...")
    df_campanhas = carregar_dados()

    # Explorar dados
    print("\n[PASSO 2] Explorando os dados...")
    explorar_dados(df_campanhas)

    # Criar gráficos
    print("\n[PASSO 3] Criando gráficos...")
    criar_graficos(df_campanhas)

    # Preparar dados
    print("\n[PASSO 4] Preparando dados para treinar...")
    X_treino_norm, X_teste_norm, y_treino, y_teste, normalizador, colunas = preparar_dados(df_campanhas)

    # Treinar modelos
    print("\n[PASSO 5] Treinando modelos (algoritmos de predição)...")
    melhor_modelo, melhor_tipo = treinar_modelos(X_treino_norm, X_teste_norm, y_treino, y_teste)

    # Salvar modelo
    print("\n[PASSO 6] Salvando o modelo...")
    salvar_modelo(melhor_modelo, normalizador, colunas)

    # Gráfico de importância
    print("\n[PASSO 7] Criando gráfico de importância...")
    grafico_importancia(melhor_modelo, colunas, melhor_tipo)

    # Resumo final
    resumo_final(df_campanhas, X_treino_norm, X_teste_norm, melhor_tipo)
