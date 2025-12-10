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
features_num = ["preco", "estrelas_media", "avaliacoes"]

df_scaled = pd.DataFrame(scaler.fit_transform(df[features_num]), columns=features_num, index=df.index)

```

#### 5° Passo: Codificação de variáveis categóricas

Por fim, vamos codificar as variáveis categóricas.
Utilizaremos a técnica de One-Hot Encoding para codificar essas variáveis, utilizando o *OneHotEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

df_encoded = encoder.fit_transform(df[["formato"]]).toarray()
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(["formato"]), index=df.index)

```

#### Código Final do Pré-processamento

``` python exec="0"
--8<-- "docs/projeto3/preprocessing.py"
```

### Etapa 3 - Divisão de dados

Em seguida, vamos realizar a divisão dos dados em conjuntos de *treino* e *teste*.

- **Conjunto de Treino:** Utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** Utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, foi utilizada a função *train_test_split()* do `scikit-learn`. Os parâmetros utilizados são:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

=== "Saída"

    ```python exec="1"
    --8<-- "docs/projeto3/division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/projeto3/division.py"
    ```

Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting*.

### Etapa 4 - Regressão Linear Múltipla

Agora, vamos fazer um modelo de *Regressão Linear Múltipla* com a base.

#### Multicolinearidade

Primeiramente, vamos checar por **Multicolinearidade** no modelo para garantir que as variáveis independentes não tenham seus coeficientes inflados por correlações entre si:

``` python exec="1"
--8<-- "docs/projeto3/rlm/mcl.py"
```

Como todos valores estão abaixo de **5**, inclusive, *muito próximos do ideal de 1*, não há multicolinearidade no modelo, então não precisamos retirar variáveis.

#### Stepwise 

Iremos realizar o **Método de Seleção de Variáveis** para deixar no modelo apenas as variáveis relevantes para a predição (e que não tenham correlação espúria).

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/rlm/step.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/rlm/step.py"
    ```

#### Modelo Final

Agora, finalmente, vamos rodar o modelo final, checar os coeficientes e o *R²*:

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/rlm/training.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/rlm/training.py"
    ```

O *R²* do modelo foi ótimo, de **92%**, significando que a regressão linear realizada possui uma forte capacidade de explicação da variação da variável `recomendacoes` através das variáveis selecionadas no stepwise. 

#### Validação Cruzada

Contanto, devido ao alto valor de *R²*, surge uma suspeita de *overfitting* no modelo. Vamos fazer uma validação cruzada para validar essa hipótese:

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/rlm/cross-val.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/rlm/cross-val.py"
    ```

A variação do *R²* entre o conjunto de treino e de teste foi de apenas **1,63%**, indicando que não há *overfitting*. Contanto, podemos observar que um dos folds obteve *R²* de **63,88%**, enquanto o resto variou entre bons valores, de **84%** à **91%**. Isso pode indicar que há instabilidade nos dados da base, com uma quantidade considerável de ruído e dataset com poucos registros.

### Etapa 5 - Treinamento do Modelo Random Forest

Agora, vamos realizar o treinamento do modelo Random Forest.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/rf/training.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/rf/training.py"
    ```

### Etapa 6 - Avaliação do Modelo Random Forest

Obtivemos um bom *R²*, de **91,39%**, o que significa que o modelo de *Random Forest* consegue explicar **91,39%** da variação de `recomendacoes`. Além disso, o *RMSE* foi de **1.27**, um bom valor para uma variável que varia de 0 a 100. Ele significa que, em média, a predição erra em 1.27 pontos percentuais de recomendações.

Pudemos observar que, por uma imensa margem, a variável mais importante do modelo é `estrelas_media`. O restante possui pouco poder explicativo, mas devem ser mantidas para fazer ajuste fino dos valores preditos.

#### Validação Cruzada

Antes de seguirmos, vamos fazer uma rápida validação cruzada para garantir que não há *overfitting*.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/rf/cross-val.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/rf/cross-val.py"
    ```

Podemos observar que sim, o modelo sofre de *overfitting*. A *R²* no treino é de **97,78%**, porém, no teste, cai significativamente, para **89,07%**. Além disso, a média de *R²* dos folds foi de **79,08%**, chegando a ter um fold com apenas **61,65%** de *R²*. 

Mesmo após alguns testes, não consegui reduzir o *overfitting*. Por isso, podemos concluir que o Random Forest não é o melhor modelo para esse problema. Provavelmente, isso acontece devido à combinação dos seguintes fatores:

- **Dataset pequeno:** Random Forest costuma precisar de muitos dados para não oscilar entre folds;

- **Target com variação baixa:** O modelo se confude ao tentar fazer splits certeiros;

- **Ruído no dataset:** Árvores podem amplificar possível ruído na base.

### Etapa 7 - Treinamento do Modelo SVM

Considerando que a base é pequena, e relembrando do aprendizado do *Projeto Extra*, vamos diretamente utilizar o *kernel linear* para fazer o modelo.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/svm/training.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/svm/training.py"
    ```

### Etapa 8 - Avaliação do Modelo SVM

Obtivemos um ótimo *R²*, de **92,16%**, o que significa que o modelo de *SVM* consegue explicar **92,16%** da variação de `recomendacoes`. Além disso, o *RMSE* foi de **1.21**, um bom valor para uma variável que varia de 0 a 100. Ele significa que, em média, a predição erra em 1.21 pontos percentuais de recomendações.

#### Validação cruzada

Vamos testar novamente *overfitting* no modelo:

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/svm/cross-val.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/svm/cross-val.py"
    ```

Dessa vez, parece que não há *overfitting* no modelo, já que obtivemos *R²* de **90,34%** no conjunto treino e **92,16%** no conjunto teste. Novamente, o mesmo sinal de instabilidade na base observado na *Regressão Linear Múltipla* aparece aqui, com variação alta entre os folds mesmo sem o modelo sofrer de *overfitting*.

### Etapa 9 - Treinamento do Modelo KNN

Agora, vamos fazer o treinamento do modelo *KNN*.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/knn/training.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/knn/training.py"
    ```

### Etapa 10 - Avaliação do Modelo KNN

Obtivemos um bom *R²*, de **91,28%**, o que significa que o modelo de *KNN* consegue explicar **91,28%** da variação de `recomendacoes`. Além disso, o *RMSE* foi de **1.28**, um bom valor para uma variável que varia de 0 a 100. Ele significa que, em média, a predição erra em 1.28 pontos percentuais de recomendações.

#### Validação cruzada

Agora, vamos para o momento crítico novamente. Apesar do *SVM* não ter tido *overfitting*, podemos ver se o *KNN* não sofre com o ruído presente na base.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto3/knn/cross-val.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto3/knn/cross-val.py"
    ```

Novamente, houve *overfitting* no modelo, com uma variação de *R²* entre o conjunto treino e teste de **aproximadamente 8,5%**. Além disso, os folds ainda possuem uma enorme variação no valor de *R²*, indicando que a instabilidade dos dados também afeta o *KNN*.

### Etapa 11 - Conclusão Final

O objetivo deste projeto foi prever o percentual de recomendações dos produtos da Growth a partir de variáveis numéricas, categóricas e binárias obtidas por raspagem. Trata-se de um **dataset pequeno**, com algumas variáveis com *outliers* e certo nível de ruído, o que impactou diretamente o desempenho e estabilidade de alguns modelos.

Após o pré-processamento, incluindo padronização, codificação categórica e remoção de registros inconsistentes, quatro abordagens principais foram testadas: **Regressão Linear Múltipla**, **Random Forest**, **KNN** e **SVM (kernel linear)**.

- **Random Forest:** apresentou bom desempenho inicial, mas sofreu com *overfitting* severo e grande oscilação entre folds. Com poucos dados e um alvo de baixa variabilidade, as árvores não conseguiram generalizar bem.

- **KNN:** apesar de ter obtido resultados sólidos, mostrou instabilidade na validação cruzada e maior sensibilidade ao ruído. Comportamento esperado para métodos baseados em vizinhança.

- **Regressão Linear Múltipla:** apresentou bom desempenho geral, com alto *R²* de **92%** e sem evidência de *overfitting*. Contudo, sofre com a instabilidade dos dados da base, como é possível observar pela variação do *R²* entre os folds.

- **SVM Linear:** entregou o melhor equilíbrio geral. Apresentou alto *R²* de **92,16%** (0,16% maior do que da regressão), baixo RMSE, ótima generalização e quase nenhuma evidência de *overfitting*, sendo robusto mesmo com poucas observações. Apesar disso, também sofre com a instabilidade dos dados da base, mesmo que bem menos do que os outros modelos.

Portanto, **o modelo escolhido como solução final foi o SVM Linear**, pois forneceu o melhor desempenho consistente, maior estabilidade e se mostrou o método mais adequado para a estrutura e tamanho reduzido do dataset. Apesar disso, a **Regressão Linear Múltipla** possui desempenho quase idêntico, então, também pode ser utilizada nesse caso.