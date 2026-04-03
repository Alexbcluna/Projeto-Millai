# Klike Challenge – Análise e Recomendação de Anúncios

## Sobre o projeto

Esse projeto foi desenvolvido com o objetivo de analisar campanhas de anúncios em vídeo e entender quais fatores mais influenciam o desempenho.

Além da análise, também foi criado um modelo de machine learning capaz de prever o desempenho de novas campanhas e um sistema simples de recomendação para melhorar anúncios.

---

## Objetivos

* Explorar e entender os dados das campanhas
* Identificar padrões de sucesso (insights)
* Treinar um modelo para prever o *Klike Score*
* Criar recomendações automáticas para melhorar anúncios

---

## Estrutura do projeto

* `analise_modelo.py` → faz toda a análise e treinamento do modelo
* `recommendations_engine.py` → gera recomendações com base nos dados/modelo
* `dataset.csv` → base de dados utilizada

---

## Etapas do projeto

### 1. Análise Exploratória (EDA)

Foi feita uma análise inicial para entender o comportamento dos dados:

* Distribuição das campanhas por plataforma
* Identificação de campanhas com melhor desempenho
* Avaliação de variáveis como:

  * presença de *hook*
  * uso de *CTA*
  * presença de *rosto humano*
  * duração do vídeo

Também foram utilizadas correlações para identificar quais variáveis mais impactam o *Klike Score*.

---

### Análise de Correlação

Foi realizada uma análise de correlação entre as variáveis numéricas e o *Klike Score*, com o objetivo de identificar quais fatores mais influenciam o desempenho das campanhas.

Principais observações:

* Variáveis como *CTR e conversões* apresentaram forte correlação positiva com o score
* Elementos como *hook e CTA* também mostraram impacto relevante
* A *duração do vídeo* apresentou tendência de correlação negativa, indicando que vídeos mais curtos tendem a performar melhor

Essa análise foi importante para orientar tanto a criação de novas features quanto o sistema de recomendação.

---

### 2. Features

Foram criadas novas variáveis para melhorar a análise e o modelo:

* `hook_e_cta` → combinação de hook + CTA
* `engajamento_por_segundo` → eficiência do vídeo

Essas features ajudam o modelo a entender melhor o comportamento dos anúncios.

---

### 3. Preparação dos dados

* Remoção de colunas irrelevantes (ID, datas, etc)
* Conversão de valores booleanos para numéricos
* Tratamento de valores ausentes (preenchimento com mediana)
* Normalização dos dados (StandardScaler)
* Divisão em treino e teste

---

### 4. Modelagem

Foram utilizados dois modelos:

* Regressão Linear (modelo mais simples)
* Random Forest (modelo mais robusto)

Os modelos foram avaliados com:

* *R² (coeficiente de determinação)*
* *MAE (erro absoluto médio)*

O melhor modelo é escolhido automaticamente com base no desempenho.

---

### 5. Salvamento do modelo

O modelo final, junto com o scaler e as colunas utilizadas, é salvo para uso futuro.

---

### 6. Sistema de Recomendação

O arquivo `recommendations_engine.py` foi criado para sugerir melhorias nos anúncios com base nos dados.

Exemplos de recomendações:

* Adicionar *hook* nos primeiros segundos
* Incluir *CTA (call to action)*
* Utilizar *legendas*
* Reduzir duração do vídeo
* Inserir *rosto humano*

Essas recomendações são baseadas nos padrões encontrados durante a análise.

---

## Principais insights encontrados

* Anúncios com *hook* tendem a ter melhor desempenho
* A presença de *CTA* impacta positivamente conversões
* Vídeos mais curtos tendem a ter maior retenção
* Elementos visuais (como rosto humano) podem aumentar engajamento

---

## Como executar

1. Instale as dependências:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

2. Execute o script principal:

```bash
python analise_modelo.py
```

3. Para gerar recomendações:

```bash
python recommendations_engine.py
```

---

## Uso de ferramentas de IA

Ferramentas de inteligência artificial foram utilizadas como apoio para esclarecimento de conceitos, revisão de código e melhoria da documentação, sempre com validação do conteúdo durante o desenvolvimento do projeto.

---

## Considerações finais

Esse projeto foi importante para praticar:

* Análise de dados
* Manipulação com Pandas
* Criação de features
* Machine Learning básico
* Geração de insights de negócio

Além disso, ajudou a entender como dados podem ser usados para melhorar decisões em campanhas de marketing.

## Projeto Desenvolvido para Estágio