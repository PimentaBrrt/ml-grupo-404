# Modelo de Machine Learning - K-means, Random Forest e SVM

Para esse projeto, foi utilizado um dataset obtido no Projeto Integrado, na raspagem do site da [**Growth**](https://www.gsuplementos.com.br/).

## Objetivo

O dataset possui informações sobre diversos produtos da Growth. O objetivo da análise será prever o quão recomendado será um produto de acordo com as outras informações da base.

## Workflow

Os pontos *"etapas"* são o passo-a-passo da realização do projeto.

### Etapa 1 - Exploração de Dados

Primeiramente, para entender melhor a base de dados, vamos descobrir quantas linhas e colunas o dataset possui.

=== "Saída"

    ``` python exec="1" 
    --8<-- "docs/projeto3/exploring-ds.py"
    ```

=== "Código"

    ``` python exec="0" 
    --8<-- "docs/projeto3/exploring-ds.py"
    ```

Como foi possível observar no código acima, o dataset possui **182 linhas** e **14 colunas**, com cada linha possuindo os dados de um produto.

#### Colunas do Dataset

Em seguida, é necessário descobrir a natureza dos dados. Isso será feito rodando as linhas de código abaixo:

``` python exec="0" 
import pandas as pd

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

print(df.info())
```

As informações obtidas foram as seguintes:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `nome` | String | Nome do produto |
| `preco` | String | Preço do produto |
| `estrelas_media` | String | Média de estrelas do produto |
| `avaliacoes` | Inteiro | Total de avaliações realizadas por usuários do produto |
| `recomendacoes` | Inteiro | Porcentagem de recomendações do produto |
| `formato` | String | Formato do produto (Ex: Pó, Líquido, Alimento, etc.) |
| `aminoacidos` | Inteiro | Variável binária que indica se o produto possui a categoria aminoacidos |
| `carboidratos` | Inteiro | Variável binária que indica se o produto possui a categoria carboidratos |
| `clinical` | Inteiro | Variável binária que indica se o produto possui a categoria clinical |
| `proteinas` | Inteiro | Variável binária que indica se o produto possui a categoria proteinas |
| `termogenicos` | Inteiro | Variável binária que indica se o produto possui a categoria termogenicos |
| `veganos` | Inteiro | Variável binária que indica se o produto possui a categoria veganos |
| `vegetarianos` | Inteiro | Variável binária que indica se o produto possui a categoria vegetarianos |
| `vitaminas` | Inteiro | Variável binária que indica se o produto possui a categoria vitaminas |

#### Visualização das variáveis

Em seguida, é essencial realizar gráficos para visualizar como cada uma das variáveis se comportam, com o objetivo de entender melhor a base da dados. Todas variáveis da base são quantitativas, sendo onze contínuas e duas discretas.

##### Variáveis Quantitativas Contínuas

=== "preco"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto3/visualizations/preco.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto3/visualizations/preco.py"
        ```

=== "estrelas_media"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto3/visualizations/e_media.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto3/visualizations/e_media.py"
        ```

=== "avaliacoes"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto3/visualizations/avals.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto3/visualizations/avals.py"
        ```

##### Variável Quantitativa Discreta `recomendacoes`

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/projeto3/visualizations/recom.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/visualizations/recom.py"
    ```

##### Variáveis Qualitativas Nominais

=== "formato"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto3/visualizations/formato.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto3/visualizations/formato.py"
        ```

=== "categorias (todas as 8 variáveis)"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto3/visualizations/cats.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto3/visualizations/cats.py"
        ```

Aqui obtemos informações valiosas. Principalmente, que a coluna `avaliacoes` possui *outliers* e que a variável-alvo `recomendacoes` possui valores zerados.

### Etapa 2 - Pré-processamento

#### 1° Passo: Identificação de valores nulos

Através da linha de código abaixo, pode-se identificar que não há valores nulos na base. Portanto, pularemos o passo de tratamento de valores nulos.

``` python exec="0"
print(df.isnull().sum())
```

Não há valores nulos no dataset, contanto, a variável-alvo `recomendacoes`, como pudemos observar nas visualizações, possui valores zerados. Rodando esse código, obtemos **14 registros**, que representa menos de 10% da base. Portanto, vamos remover esses valores da base através do código abaixo:

``` python exec="0"
droppar = []

for i in range(len(df)):
    if df.loc[i, "recomendacoes"] == 0:
        droppar.append(i)

df = df.drop(index=droppar)
```

#### 2° Passo: Remoção de colunas desimportantes

Vamos remover a variável `nome` da base, já que ela não contribuirá para as predições.

``` python
df = df.drop("nome", axis=1)
```

#### 3° Passo: Correção dos tipos de dados

As colunas `preco` e `estrelas_media` deveriam ser *float*, contudo, estão como *object* devido à separação dos decimais por ",". Vamos resolver isso:

``` python
df["preco"] = df["preco"].str.replace(",", ".").astype(float)
df["estrelas_media"] = df["estrelas_media"].str.replace(",", ".").astype(float)
```

#### 4° Passo: Padronização das features numéricas

Agora, é necessário padronizar as features numéricas da base. Para as variáveis quantitativas não desbalancearem os cálculos dos modelos.
Para a padronização, foi utilizado o *StandardScaler()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_num = ["preco", "estrelas_media", "avaliacoes", "recomendacoes"]

df_scaled = scaler.fit_transform(df[features_num])

```

#### 5° Passo: Codificação de variáveis categóricas

Por fim, vamos codificar as variáveis categóricas.
Utilizaremos a técnica de One-Hot Encoding para codificar essas variáveis, utilizando o *OneHotEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
categorical_cols = ["formato", "aminoacidos", "carboidratos", "clinical", "proteinas", "termogenicos", "veganos", "vegetarianos", "vitaminas"]

df_encoded = encoder.fit_transform(df[categorical_cols])

```

### Etapa 3 - Divisão de dados

Em seguida, vamos realizar a divisão dos dados em conjuntos de *treino* e *teste*.

- **Conjunto de Treino:** Utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** Utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, foi utilizada a função *train_test_split()* do `scikit-learn`. Os parâmetros utilizados são:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

- **stratify=y:** Esse atributo definido como *y* é essencial devido à natureza da coluna `Wine_Type`. Com essa definição, será mantida a mesma proporção das categorias em ambos os conjuntos, reduzindo o viés.

=== "Saída"

    ```python exec="1"
    --8<-- "docs/projeto3/division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/projeto3/division.py"
    ```

Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting*.

### Etapa 6 - Treinamento do Modelo Random Forest



### Etapa 7 - Avaliação do Modelo Random Forest



### Etapa 8 - Treinamento do Modelo SVM



### Etapa 9 - Avaliação do Modelo SVM



### Etapa 10 - Conclusão Final
