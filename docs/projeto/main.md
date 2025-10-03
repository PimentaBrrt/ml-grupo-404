# Modelo de Machine Learning - Árvore de Decisões, KNN e K-Means

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering){:target='_blank'}, e foram adaptados de outra base de vinhos, 

## Objetivo

O dataset possui informações diversas sobre resultados de análises químicas de vinhos produzidos na mesma região da Itália, mas derivados de três diferentes cultivares. A análise determinou as quantidade de 13 constituintes encontrados em cada um dos três tipos de vinho. O objetivo da análise é clusterizar a base através do k-means e, com os modelos supervisionados, prever o tipo de vinho com base nos dados fornecidos.

## Workflow

Os pontos *"etapas"* são o passo-a-passo da realização do projeto.

### Etapa 1 - Exploração de Dados

Primeiramente, para entender melhor a base de dados, vamos descobrir quantas linhas e colunas o dataset possui.

=== "Saída"

    ``` python exec="1" 
    --8<-- "docs/projeto/exploring-ds.py"
    ```

=== "Código"

    ``` python exec="0" 
    --8<-- "docs/projeto/exploring-ds.py"
    ```

Como foi possível observar no código acima, o dataset possui **178 linhas** e **13 colunas**, com cada linha possuindo os dados de um vinho.

#### Colunas do Dataset

Em seguida, é necessário descobrir a natureza dos dados. Isso será feito rodando as linhas de código abaixo:

``` python exec="0" 
import pandas as pd

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

print(df.info())
```

As informações obtidas foram as seguintes:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Alcohol` | Float | Teor alcoólico do vinho |
| `Malic_Acid` | Float | Ácido málico |
| `Ash` | Float | Cinzas |
| `Ash_Alcanity` | Float | Alcalinidade das cinzas |
| `Magnesium` | Inteiro | Magnésio |
| `Total_Phenols` | Float | Total de fenóis |
| `Flavanoids` | Float | Flavanóides |
| `Nonflavanoid_Phenols` | Float | Fenóis não-flavanóides |
| `Proanthocyanins` | Float | Proantocianidinas |
| `Color_Intensity` | Float | Intensidade da cor do vinho |
| `Hue` | Float | Saturação do vinho |
| `OD280` | Float | Relação OD280/OD315 de vinhos diluídos  |
| `Proline` | Inteiro | Prolina |

#### Visualização das variáveis

Em seguida, é essencial realizar gráficos para visualizar como cada uma das variáveis se comportam, com o objetivo de entender melhor a base da dados. Todas variáveis da base são quantitativas, sendo onze contínuas e duas discretas.

##### Variáveis Quantitativas Contínuas

Para cada uma das variáveis numéricas contínuas, será feito um histograma com o objetivo de visualizar a frequência de valores.

=== "Alcohol"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/alcohol.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/alcohol.py"
        ```

=== "Malic_Acid"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/malic_acid.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/malic_acid.py"
        ```

=== "Ash"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/ash.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/ash.py"
        ```

=== "Ash_Alcanity"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/ash-alc.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/ash-alc.py"
        ```

=== "Total_Phenols"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/total-phenols.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/total-phenols.py"
        ```

=== "Flavanoids"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/flavanoids.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/flavanoids.py"
        ```

=== "Nonflavanoid_Phenols"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/flavanoids-n.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/flavanoids-n.py"
        ```

=== "Proanthocyanins"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/proa.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/proa.py"
        ```

=== "Color_Intensity"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/color.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/color.py"
        ```

=== "Hue"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/hue.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/hue.py"
        ```

=== "OD280"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/od280.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/od280.py"
        ```

##### Variáveis Quantitativas Discretas

Para ambas variáveis numéricas discretas, também faremos histogramas.

=== "Magnesium"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/magnesium.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/magnesium.py"
        ```

=== "Proline"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/projeto/visualizations/proline.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/projeto/visualizations/proline.py"
        ```

Através das análises, foi possível alcançar uma compreensão mais aprofundada do funcionamento de cada uma das variáveis no dataset, além de haver insights valiosos nesses gráficos.

### Etapa 2 - Pré-processamento

#### 1° Passo: Identificação de valores nulos

Através da linha de código abaixo, pode-se identificar que não há valores nulos na base. Portanto, pularemos o passo de tratamento de valores nulos.

``` python exec="0"
print(df.isnull().sum())
```

#### 2° Passo: Remoção de colunas desimportantes

Não há colunas desimportantes para a análise no dataset. Um exemplo de coluna seria um identificador único do vinho. Todas são viáveis para o modelo de predição.

#### 3° Passo: Padronização das features numéricas

Por fim, é necessário padronizar as features numéricas da base. Ao invés da normalização, será utilizada a técnica de padronização devido aos outliers nas features numéricas.
Para a padronização, foi utilkizado o *StandardScaler()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(df)

```

### Etapa 3 - K-Means

Nessa etapa, realizaremos um modelo K-Means para clusterizar a base e obter categorias que serão a variável-alvo da previsão dos modelos supervisionados. 

#### Elbow Method

Antes de treinar o modelo, é necessário descobrir o número de clusters que será utilizado. Para isso, aplicaremos o Elbow Method.

=== "Elbow"

    ![Elbow Method](../images/elbow.svg)

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/k-means/elbow.py"
    ```

Podemos observar que o cotovelo está em $k = 3$, logo, esse será o número de clusters utilizado para o K-Means.

#### Treinamento do K-Means

Para a formação dos clusters do K-Means, foi utilizado a técnica do PCA (Principal Component Analysis). 

=== "K-Means PCA"

    <figure markdown="span">
        ![K-Means](../images/k-means.svg)
        <figcaption>Silhouette Score: 0.2849</figcaption>
    </figure>

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/k-means/training1.py"
    ```

### Etapa 4 - Avaliação do K-Means

#### Silhouette Score

O modelo alcançou um Silhouette Score de **0.2849**, indicando uma estrutura de clusters potencialmente artificial. Na escala de -1 a +1, este valor se enquadra na categoria **Fraca**, porém ainda acima do limiar de 0.25 que indicaria ausência de clusters naturais.

#### Variância Explicada

O PCA aplicado para visualização explica **55.41%** da variância total dos dados. Embora seja uma representação simplificada, é suficiente para identificar padrões gerais, porém pode não capturar estruturas mais complexas não-lineares.

#### Conclusão da avaliação

O **silhouette score** ficou baixo, por isso, vamos tentar outra técnica que não o PCA. Vamos explorar o t-SNE (t-Distributed Stochastic Neighbor Embedding), uma técnica não-linear que pode revelar melhor estruturas locais e agrupamentos não capturados pelo PCA.

#### Re-treinamento do k-means

=== "K-Means t-SNE"

    <figure markdown="span">
        ![K-Means](../images/k-means-tsne.svg)
        <figcaption>Silhouette Score: 0.5928</figcaption>
    </figure>

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/k-means/training2.py"
    ```

Foi possível observar uma melhora significativa no **silhouette score**, de 0.2849 com PCA para 0.5928 com t-SNE, indicando uma estrutura de clusters muito mais definida.

Os centróides na visualização t-SNE podem parecer "estranhos" porque esta técnica prioriza a preservação de estruturas locais em detrimento de relações globais e densidades. O t-SNE distorce intencionalmente o espaço para destacar agrupamentos próximos, o que explica a posição não convencional dos centróides na visualização.

Em resumo, o trade-off vale a pena: mesmo com a perda de informações sobre densidades e estruturas globais, a qualidade da clusterização melhorou drasticamente, revelando padrões que não eram aparentes com PCA.

#### Criando a coluna da variável-alvo

Agora, vamos criar a coluna que conterá a variável categórica `Wine_Type`, criada a partir da clusterização do K-Means com t-SNE. Essa coluna será a variável-alvo das análises preditivas feitas pelos modelos supervisionados adiante.

=== "Saída"

    ``` python exec="1"
    --8<-- "docs/projeto/k-means/create-type.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/k-means/create-type.py"
    ```

Com isso, podemos partir para as próximas etapas.

### Etapa 5 - Divisão de dados

Em seguida, vamos realizar a divisão dos dados em conjuntos de *treino* e *teste*.

- **Conjunto de Treino:** Utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** Utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, foi utilizada a função *train_test_split()* do `scikit-learn`. Os parâmetros utilizados são:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

- **stratify=y:** Esse atributo definido como *y* é essencial devido à natureza da coluna `Wine_Type`. Com essa definição, será mantida a mesma proporção das categorias em ambos os conjuntos, reduzindo o viés.

=== "Saída"

    ```python exec="1"
    --8<-- "docs/projeto/division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/projeto/division.py"
    ```

Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting*.

### Etapa 6 - Treinamento do modelo Decision Tree

Agora, vamos treinar um modelo de árvore de decisões (Decision Tree) para prever a variável alvo `Wine_Type` para os dados do conjunto *teste*. Nosso objetivo aqui é treinar e avaliar o modelo, para depois compará-lo ao KNN e decidir o melhor para este caso.

=== "Decision Tree"
    
    <figure markdown="span">
        ![Decision-Tree](../images/d-tree.svg)
    </figure>

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/decision-tree/training.py"
    ```

### Etapa 7 - Avaliação do modelo Decision Tree

Agora, vamos realizar a avaliação do modelo treinado. Primeiramente, vamos ver a acurácia do modelo e a importância de cada uma das features utilizadas para a predição.

=== "Saída"

    ```python exec="1" html="1"
    --8<-- "docs/projeto/decision-tree/aval-decision.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/projeto/decision-tree/aval-decision.py"
    ```

#### Acurácia

O modelo atingiu uma boa acurácia, de **88,89%**, bem próximo do ideal de **95%**. Isso significa que, em 88,89% das previsões feitas, o tipo de vinho predito está correto.

#### Importância das features

- Na tabela de importância das features, podemos notar que a variável mais importante para a previsão é a `Proline`, com **42,74%** de importância na previsão. 

- Diversas variáveis tiveram uma importância nula, sendo elas: `Magnesium`, `Ash`, `Ash_Alcanity`, `Proanthocyanins`, `Nonflavanoid_Phenols`, `Total_Phenols` e `Hue`

- As variáveis `Malic_Acid` e `Alcohol` tiveram uma importância quase irrelevante na predição, de **1,82%** e **0,49%** respectivamente.

#### Matriz de Confusão

Agora, vamos visualizar a matriz de confusão do modelo.

=== "Saída"

    Matriz de confusão

    ![CM-Decision-Tree](../images/cm-d-tree.svg)

    Métricas de qualidade

    ``` python exec="1"
    --8<-- "docs/projeto/decision-tree/cm-decision.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/projeto/decision-tree/cm-decision.py"
    ```

#### Avaliação das métricas

**Pontos Positivos**

- Boa acurácia geral: 89% - modelo consegue classificar corretamente a maioria das instâncias

- Excelente precisão para Wine Type 3: 100% - quando o modelo classifica como tipo 3, está sempre correto

- Recall alto para Wine Type 1: 92% - consegue identificar quase todos os vinhos do tipo 1

- Balanceamento razoável: Métricas similares entre as classes

**Pontos de Melhoria**

*Problema com Wine Type 3:*

- Recall de 85% - o modelo falha em identificar 15% dos vinhos do tipo 3

Isso significa que 15% dos vinhos tipo 3 estão sendo classificados erroneamente como outros tipos

*Precisão do Wine Type 1:*

- 80% - quando o modelo diz "é tipo 1", em 20% dos casos está errado

### Etapa 8 - Treinamento do Modelo KNN



### Etapa 9 - Avaliação do modelo KNN



### Etapa 10 - Relatório Final


