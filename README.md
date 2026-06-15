# Steam Main Game Churn Prediction

## Descrição

Este projeto tem como objetivo prever o abandono do jogo principal de um usuário da Steam utilizando técnicas de Machine Learning.

A proposta consiste em identificar jogadores com risco de deixar de jogar o título ao qual dedicam a maior parte de seu tempo na plataforma, permitindo analisar padrões de engajamento e comportamento associados ao churn.

Para isso, foi desenvolvido um modelo de classificação baseado em Random Forest, utilizando métricas extraídas do histórico de atividade dos usuários.

O projeto também incorpora ferramentas amplamente utilizadas em ambientes profissionais de Ciência de Dados:

* MLflow para rastreamento de experimentos;
* FastAPI para disponibilização do modelo através de uma API REST;
* Docker para conteinerização e portabilidade da aplicação.

---

## Objetivo

O objetivo do projeto é prever se um jogador está:

* Ativo em seu jogo principal;
* Em risco de abandono (churn).

Inicialmente o dataset foi construído com três categorias:

* Ativo;
* Possível churn;
* Churn.

Posteriormente foram realizados experimentos binários agrupando a classe "possível churn" às demais categorias para avaliar diferentes estratégias de classificação.

---

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-Learn
* MLflow
* FastAPI
* Uvicorn
* Docker
* Joblib

---

## Dataset

O dataset foi construído a partir de informações públicas da Steam.

As principais variáveis utilizadas foram:

* playtime_forever
* game_count
* recent_active_games
* avg_playtime
* top_game_ratio

### Descrição das Features

**playtime_forever**

Tempo total acumulado de jogo na conta.

**game_count**

Quantidade total de jogos possuídos pelo usuário.

**recent_active_games**

Quantidade de jogos utilizados recentemente.

**avg_playtime**

Tempo médio de jogo por título.

**top_game_ratio**

Proporção do tempo total dedicada ao jogo mais jogado da biblioteca.

Essa variável é importante para representar a relação do usuário com seu jogo principal.

---

## Construção do Churn

O status de churn foi definido utilizando métricas de engajamento recente em relação ao histórico de atividade do jogador.

Durante os experimentos foram utilizados dois conjuntos de variáveis:

### Sem Data Leakage

Utilizadas apenas informações comportamentais gerais:

* playtime_forever
* game_count
* recent_active_games
* avg_playtime
* top_game_ratio

### Com Data Leakage

Foram adicionadas variáveis utilizadas diretamente na construção do rótulo:

* playtime_2weeks
* engagement_ratio

Esse experimento foi realizado apenas para demonstrar o impacto do data leakage em modelos de Machine Learning.

Como esperado, o modelo apresentou métricas artificialmente elevadas devido ao acesso direto às informações usadas para definir o status de churn.

---

## Experimentos Realizados

Os experimentos foram registrados utilizando MLflow.

Entre os testes realizados estão:

1. Random Forest sem data leakage;
2. Random Forest com data leakage;
3. Random Forest sem balanceamento de classes;
4. Random Forest com 500 árvores;
5. Random Forest utilizando conjunto mínimo de features;
6. Random Forest otimizado com Grid Search;
7. Classificação binária agrupando "possível churn" com "churn";
8. Classificação binária agrupando "possível churn" com "ativo".

---

## Modelo Final

Após a comparação dos experimentos, foi selecionado o modelo binário que agrupa a classe "possível churn" com "churn".

Essa abordagem se mostrou mais adequada para identificar usuários com risco de abandonar seu jogo principal.

---

## Estrutura do Projeto

projeto/

├── data/

│ └── raw/

│ ├── steam_games_dataset.csv

│ └── binaryC_churn_model.pkl

│

├── src/

│ ├── modelo.py

│ └── api.py

│

├── mlflow.db

├── requirements.txt

├── Dockerfile

├── docker-compose.yml

└── README.md

---

## Execução com Docker

### Construir a imagem

```bash
docker build -t steam-churn .
```

### Executar o container

```bash
docker run -p 8000:8000 steam-churn
```

---

## Acessando a API

Após iniciar o container:

```text
http://localhost:8000/docs
```

A documentação interativa da API será exibida automaticamente.

---

## Endpoint de Previsão

### POST /predict

Exemplo de entrada:

```json
{
  "playtime_forever": 250000,
  "game_count": 20,
  "recent_active_games": 1,
  "avg_playtime": 12000,
  "top_game_ratio": 0.80
}
```

Exemplo de resposta:

```json
{
  "prediction": "churn"
}
```

---

## Rastreamento de Experimentos

Para visualizar os experimentos registrados:

```bash
python -m mlflow ui
```

Acesse:

```text
http://127.0.0.1:5000
```

---

## Autores

Gabriel Cavalcante
Vitor Félix
Tharso Oliveira
João Paulo Costa

Projeto desenvolvido para a disciplina de Ciência de Dados.
