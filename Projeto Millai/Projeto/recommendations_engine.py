# MOTOR DE RECOMENDACOES - VERSAO SIMPLES
# Este script carrega o modelo e gera dicas para melhorar videos

import pandas as pd  # trabalhar com tabelas
import numpy as np   # contas
import pickle  # carregar arquivo do modelo
import joblib  # carregar modelo ML
import os  # trabalhar com arquivos

print("\n" + "="*70)
print("MOTOR DE RECOMENDACOES - GERAR DICAS PARA MELHORAR VIDEOS")
print("="*70)

# CARREGA O MODELO TREINADO

print("\n[PASSO 1] Carregando o modelo treinado...")

diretorio = os.path.dirname(os.path.abspath(__file__))
arquivo_modelo = os.path.join(diretorio, 'best_model.joblib')

# Abrir a "caixa" que salvamos com o modelo
caixa = joblib.load(arquivo_modelo)

modelo = caixa['modelo']
normalizador = caixa['normalizador']
colunas_modelo = caixa['colunas']

print("  OK - Modelo carregado!")

# CARREGA OS DADOS PARA CALCULAR BENCHMARKS

print("\n[PASSO 2] Carregando dados de referencia...")

arquivo_csv = os.path.join(diretorio, 'klike_challenge_dataset.csv')
dados = pd.read_csv(arquivo_csv)

print("  OK - {} campanhas carregadas".format(len(dados)))

# PASSO 3: CALCULAR BENCHMARKS (REFERENCIAS) POR PLATAFORMA

print("\n[PASSO 3] Calculando referencias (benchmarks) de cada plataforma...\n")

# Para cada plataforma, vamos descobrir:
# - Qual a media de nota?
# - Qual o comprimento ideal do video?
# - Quanto melhora a nota se tiver hook? E CTA? E rosto?

benchmarks = {}

for plataforma in dados['platform'].unique():
    print("  Analisando {}...".format(plataforma))
    
    # Pegar apenas as campanhas dessa plataforma
    dados_plat = dados[dados['platform'] == plataforma]
    
    # Media geral de nota
    media_geral = dados_plat['klike_score'].mean()
    
    # Duracao ideal = aquela em que mais campanhas tiveram bom resultado
    duracao_ideal = dados_plat['video_duration_s'].median()
    
    # Impacto do Hook: quanto melhora se tiver hook?
    com_hook = dados_plat[dados_plat['has_hook'] == 1]['klike_score'].mean()
    sem_hook = dados_plat[dados_plat['has_hook'] == 0]['klike_score'].mean()
    impacto_hook = ((com_hook - sem_hook) / sem_hook * 100) if sem_hook > 0 else 0
    
    # Impacto do CTA (call-to-action = "clique aqui")
    com_cta = dados_plat[dados_plat['has_cta'] == 1]['klike_score'].mean()
    sem_cta = dados_plat[dados_plat['has_cta'] == 0]['klike_score'].mean()
    impacto_cta = ((com_cta - sem_cta) / sem_cta * 100) if sem_cta > 0 else 0
    
    # Impacto de ter rosto no video
    com_face = dados_plat[dados_plat['has_face'] == 1]['klike_score'].mean()
    sem_face = dados_plat[dados_plat['has_face'] == 0]['klike_score'].mean()
    impacto_face = ((com_face - sem_face) / sem_face * 100) if sem_face > 0 else 0
    
    # Guardar tudo numa "pasta" de referencia
    benchmarks[plataforma] = {
        'media': media_geral,
        'duracao_ideal': duracao_ideal,
        'impacto_hook': impacto_hook,
        'impacto_cta': impacto_cta,
        'impacto_face': impacto_face,
        'com_hook': com_hook,
        'sem_hook': sem_hook
    }
    
    print("    - Media de nota: {:.1f}".format(media_geral))
    print("    - Duracao ideal: {:.0f}s".format(duracao_ideal))
    print("    - Melhora com hook: +{:.1f}%".format(impacto_hook))
    print("    - Melhora com CTA: +{:.1f}%".format(impacto_cta))
    print("    - Melhora com rosto: +{:.1f}%".format(impacto_face))

# FUNCAO PARA GERAR RECOMENDACOES

print("\n[PASSO 4] Definindo como gerar recomendacoes...")

def gerar_recomendacoes(campanha_info, plataforma, benchmarks):
    """
    Recebe os dados de uma campanha e retorna dicas de melhoria
    
    campanha_info = dicionario com os dados da campanha
    plataforma = string com nome da plataforma (Meta, TikTok, LinkedIn)
    benchmarks = dicionario com referencias
    """
    
    recomendacoes = []
    bench = benchmarks[plataforma]
    
    # Verificar se tem HOOK (primeiro segundo)
    if campanha_info.get('has_hook', 0) != 1:
        impacto = bench['impacto_hook']
        if impacto > 10:  # So recomenda se melhora mais de 10%
            prioridade = 90 if impacto > 20 else 80
            rec = {
                'acao': 'Adicione um HOOK no primeiro segundo do video',
                'explicacao': 'Videos COM hook em {} ficam com nota {:.0f}, sem hook ficam {:.0f}'.format(
                    plataforma, bench['com_hook'], bench['sem_hook']),
                'impacto': '+{:.1f}% de melhora esperada'.format(impacto),
                'prioridade': prioridade
            }
            recomendacoes.append(rec)
    
    # Verificar DURACAO
    duracao = campanha_info.get('video_duration_s', 0)
    duracao_maxima = bench['duracao_ideal'] * 2
    if duracao > duracao_maxima:
        rec = {
            'acao': 'Reduza o tamanho do video',
            'explicacao': 'Video tem {:.0f}s, mas videos longos perdem espectador. Ideal: ate {:.0f}s'.format(
                duracao, duracao_maxima),
            'impacto': 'Menos pessoas saem no meio',
            'prioridade': 85
        }
        recomendacoes.append(rec)
    
    # Verificar se tem CTA
    if campanha_info.get('has_cta', 0) != 1:
        impacto = bench['impacto_cta']
        if impacto > 5:
            rec = {
                'acao': 'Adicione um CTA (Call-To-Action = "clique", "veja mais", etc)',
                'explicacao': 'Videos COM CTA melhoram +{:.1f}% em {}'.format(impacto, plataforma),
                'impacto': '+{:.1f}%'.format(impacto),
                'prioridade': 70
            }
            recomendacoes.append(rec)
    
    # Verificar se tem ROSTO
    if campanha_info.get('has_face', 0) != 1:
        impacto = bench['impacto_face']
        if impacto > 10:
            rec = {
                'acao': 'Inclua um rosto/pessoa no video',
                'explicacao': 'Pessoas se conectam melhor com rostos. Melhora +{:.1f}% em {}'.format(
                    impacto, plataforma),
                'impacto': '+{:.1f}%'.format(impacto),
                'prioridade': 80
            }
            recomendacoes.append(rec)
    
    # Verificar TEXTO NA TELA
    densidade_texto = campanha_info.get('text_density', 0)
    if densidade_texto > 0.15:  # Mais de 15% de texto eh muito
        rec = {
            'acao': 'Reduza a quantidade de TEXTO na tela',
            'explicacao': 'Texto cobre {:.1f}% da tela (maximo recomendado: 15%)'.format(densidade_texto * 100),
            'impacto': 'Video fica mais limpo e profissional',
            'prioridade': 60
        }
        recomendacoes.append(rec)
    
    return recomendacoes

print("  OK - Funcao criada!")

# APLICAR A FUNCAO EM CAMPANHAS REAIS

print("\n[PASSO 5] Gerando recomendacoes para exemplo...")

# Campanha TikTok SaaS (sem hook)
print("\n" + "-"*70)
print("EXEMPLO 1: CAMPANHA TIKTOK - PRODUTO SAAS")
print("-"*70)

camp1 = {
    'has_hook': 0,  # NAO TEM hook
    'has_cta': 1,   # TEM CTA
    'has_face': 1,  # TEM rosto
    'video_duration_s': 45,
    'text_density': 0.08
}

recs1 = gerar_recomendacoes(camp1, 'TikTok', benchmarks)
print("\nRecomendacoes geradas: {}".format(len(recs1)))
for i, rec in enumerate(recs1, 1):
    print("\n  [Prioridade {}] DICA {}: {}".format(rec['prioridade'], i, rec['acao']))
    print("    Por que: {}".format(rec['explicacao']))
    print("    Impacto: {}".format(rec['impacto']))

# Campanha LinkedIn B2B (sem CTA)
print("\n" + "-"*70)
print("EXEMPLO 2: CAMPANHA LINKEDIN - B2B")
print("-"*70)

camp2 = {
    'has_hook': 1,   # TEM hook
    'has_cta': 0,    # NAO TEM CTA
    'has_face': 1,   # TEM rosto
    'video_duration_s': 25,
    'text_density': 0.12
}

recs2 = gerar_recomendacoes(camp2, 'LinkedIn', benchmarks)
print("\nRecomendacoes geradas: {}".format(len(recs2)))
for i, rec in enumerate(recs2, 1):
    print("\n  [Prioridade {}] DICA {}: {}".format(rec['prioridade'], i, rec['acao']))
    print("    Por que: {}".format(rec['explicacao']))
    print("    Impacto: {}".format(rec['impacto']))

# Campanha Meta E-commerce (muito ruim)
print("\n" + "-"*70)
print("EXEMPLO 3: CAMPANHA META - E-COMMERCE")
print("-"*70)

camp3 = {
    'has_hook': 0,   # NAO TEM hook
    'has_cta': 1,    # TEM CTA
    'has_face': 0,   # NAO TEM rosto
    'video_duration_s': 120,  # Muito longo!
    'text_density': 0.20  # Muito texto!
}

recs3 = gerar_recomendacoes(camp3, 'Meta', benchmarks)
print("\nRecomendacoes geradas: {}".format(len(recs3)))
for i, rec in enumerate(recs3, 1):
    print("\n  [Prioridade {}] DICA {}: {}".format(rec['prioridade'], i, rec['acao']))
    print("    Por que: {}".format(rec['explicacao']))
    print("    Impacto: {}".format(rec['impacto']))

# RESUMO FINAL

print("\n" + "="*70)
print("MOTOR DE RECOMENDACOES FUNCIONANDO!")
print("="*70)
print("\nResumo:")
print("  - Calculamos referencias para {} plataformas".format(len(benchmarks)))
print("  - Testamos com 3 campanhas reais")
print("  - Campanha 1 recebeu {} recomendacoes".format(len(recs1)))
print("  - Campanha 2 recebeu {} recomendacoes".format(len(recs2)))
print("  - Campanha 3 recebeu {} recomendacoes".format(len(recs3)))
print("\nEste motor pode ser usado para:")
print("  1. Analisar QUALQUER campanha nova")
print("  2. Gerar dicas automaticas")
print("  3. Priorizar o que melhorar PRIMEIRO")
print("\n" + "="*70)
