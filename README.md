# Classificador de Canal de Vendas - K-Nearest Neighbors (k-NN)

Este projeto implementa um classificador para prever o canal de vendas (HoReCa ou Retail) de clientes de um conjunto de dados de vendas. A solução utiliza o algoritmo K-Nearest Neighbors (k-NN) com ajuste de hiperparâmetros, balanceamento de classes e avaliação de desempenho.

## Arquivo de dados

O arquivo de dados, `Wholesale customers data.csv`, tem informações sobre clientes de atacado, incluindo as colunas `Region`, `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicatessen` e `Channel`.

## Estrutura do Código

### 1. Importação de Bibliotecas

O código utiliza as seguintes bibliotecas principais:
- **Pandas** para manipulação de dados;
- **Scikit-Learn** para modelagem, separação de dados, ajuste de hiperparâmetros e avaliação do modelo.

### 2. Pré-processamento dos Dados

1. **Carregar Dados**: O arquivo CSV `Wholesale customers data.csv` é carregado.
2. **Codificação de Dados Categóricos**: As colunas `Channel` e `Region` são codificadas para valores numéricos.
   - `Channel`: 0 para HoReCa, 1 para Retail.
   - `Region`: 0 para Lisbon, 1 para Oporto, 2 para Other.
3. **Reordenação das Colunas**: As colunas são reordenadas para facilitar a leitura e o processamento.

### 3. Balanceamento das Classes

Para lidar com classes desbalanceadas, a classe minoritária (Retail) é aumentada por meio do processo de **oversampling**, garantindo que o classificador tenha dados equilibrados para treinamento.

### 4. Separação em Conjuntos de Treinamento e Teste

Os dados balanceados são divididos em conjunto de treinamento (80%) e conjunto de teste (20%) usando a função `train_test_split`.

### 5. Ajuste de Hiperparâmetros

O código utiliza o `GridSearchCV` para otimizar os hiperparâmetros do classificador k-NN, explorando diferentes valores para:
   - Número de vizinhos (n_neighbors)
   - Peso dos pontos (`weights`)
   - Métrica de distância (`metric`)

### 6. Avaliação do Modelo

O modelo treinado é avaliado com o conjunto de teste usando:
   - Acurácia (métrica geral de precisão)
   - Relatório de classificação detalhado com métricas como `precision`, `recall` e `f1-score`.

### 7. Classificação de Novos Dados

Uma função `classify_input()` permite que o usuário insira dados manualmente para classificação, determinando o canal de vendas (HoReCa ou Retail) com base no modelo treinado.

## Uso

1. **Executar o Script**: Após o carregamento dos dados, o script treina o modelo e exibe as métricas de avaliação.
2. **Classificação de Novos Clientes**: O usuário pode inserir dados para prever o canal de vendas. Os valores permitidos para cada campo são:
   - `Region`: 0 (Lisbon), 1 (Oporto), 2 (Other)
   - `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicatessen`: valores numéricos que representam o consumo anual em cada categoria.

## Dependências

- `pandas`
- `scikit-learn`

Para instalar as dependências utilize o seguinte comando:
```bash
    pip install pandas scikit-learn
```

### Colaborador
- Leonardo Matias

