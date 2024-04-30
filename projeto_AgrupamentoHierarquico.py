import streamlit             as st
import io

import numpy                 as np
import pandas                as pd
import matplotlib.pyplot     as plt
import seaborn               as sns

from gower                   import gower_matrix

from scipy.spatial.distance  import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster



@st.cache_data(show_spinner=False)
def calcularGowerMatrix(data_x, cat_features):
    return gower_matrix(data_x=data_x, cat_features=cat_features)


@st.cache_data(show_spinner=False)
# Definir a função para criar um dendrograma
def dn(color_threshold: float, num_groups: int, Z: list) -> None:
    """
    Cria e exibe um dendrograma.

    Parameters:
        color_threshold (float): Valor de threshold de cor para a coloração do dendrograma.
        num_groups (int): Número de grupos para o título do dendrograma.
        Z (list): Matriz de ligação Z.

    Returns:
        None
    """
    plt.figure(figsize=(24, 6))
    plt.ylabel(ylabel='Distância')
    
    # Adicionar o número de grupos como título
    plt.title(f'Dendrograma Hierárquico - {num_groups} Grupos')

    # Criar o dendrograma com base na matriz de ligação Z
    dn = dendrogram(Z=Z, 
                    p=6, 
                    truncate_mode='level', 
                    color_threshold=color_threshold, 
                    show_leaf_counts=True, 
                    leaf_font_size=8, 
                    leaf_rotation=45, 
                    show_contracted=True)
    plt.yticks(np.linspace(0, .6, num=31))
    plt.xticks([])

    # Exibir o dendrograma criado
    st.pyplot(plt)



# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(
        page_title="Projeto de Agrupamento hierárquico",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    st.sidebar.markdown('''
                      
                        # **Profissão: Cientista de Dados**
                        ### **Projeto de Agrupamento Hierárquico**

                        ---
                        ''', unsafe_allow_html=True)
    



    with st.sidebar.expander(label="Bibliotecas/Pacotes", expanded=False):
        st.code('''
                import streamlit             as st
                import io

                import numpy                 as np
                import pandas                as pd
                import matplotlib.pyplot     as plt
                import seaborn               as sns

                from gower                   import gower_matrix

                from scipy.spatial.distance  import squareform
                from scipy.cluster.hierarchy import linkage
                from scipy.cluster.hierarchy import dendrogram
                from scipy.cluster.hierarchy import fcluster
                ''', language='python')
        



    st.markdown('''

                <!-- # **Profissão: Cientista de Dados** -->
                ### **Módulo 31** | Streamlit V (Exercício 2)

                **Aluna:** Rayssa Athayde<br>

                ---
                ''', unsafe_allow_html=True)


    st.markdown('''
                <a name="intro"></a> 

                # Agrupamento hierárquico

                Neste projeto foi utilizada a base [online shoppers purchase intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). [Web Link](https://doi.org/10.1007/s00521-018-3523-0).

                A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente relacionar o design da página e o perfil do cliente.
                
                ***"Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?"***

                O objetivo é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

                |Variável                |Descrição                                                                                                                      |Atributo   | 
                | :--------------------- |:----------------------------------------------------------------------------------------------------------------------------  | --------: | 
                |Administrative          | Quantidade de acessos em páginas administrativas                                                                              |Numérico   | 
                |Administrative_Duration | Tempo de acesso em páginas administrativas                                                                                    |Numérico   | 
                |Informational           | Quantidade de acessos em páginas informativas                                                                                 |Numérico   | 
                |Informational_Duration  | Tempo de acesso em páginas informativas                                                                                       |Numérico   | 
                |ProductRelated          | Quantidade de acessos em páginas de produtos                                                                                  |Numérico   | 
                |ProductRelated_Duration | Tempo de acesso em páginas de produtos                                                                                        |Numérico   | 
                |BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão                            |Numérico   | 
                |ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações                      |Numérico   | 
                |PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico |Numérico   | 
                |SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc)                                                                    |Numérico   | 
                |Month                   | Mês                                                                                                                           |Categórico | 
                |OperatingSystems        | Sistema operacional do visitante                                                                                              |Categórico | 
                |Browser                 | Browser do visitante                                                                                                          |Categórico | 
                |Region                  | Região                                                                                                                        |Categórico | 
                |TrafficType             | Tipo de tráfego                                                                                                               |Categórico | 
                |VisitorType             | Tipo de visitante: novo ou recorrente                                                                                         |Categórico | 
                |Weekend                 | Indica final de semana                                                                                                        |Categórico | 
                |Revenue                 | Indica se houve compra ou não                                                                                                 |Categórico |

                *Variáveis calculadas pelo Google Analytics*

                ''', unsafe_allow_html=True)


    st.markdown(''' 
                ## Visualização dos Dados
                <a name="visualizacao"></a> 
                ''', unsafe_allow_html=True)
    

    st.markdown(''' 
                ### Carregar e ler dados de arquivo .csv
                <a name="read_csv"></a> 
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # Ler o arquivo CSV 'online_shoppers_intention.csv' e armazenar os dados em um DataFrame chamado df
        df = pd.read_csv('online_shoppers_intention.csv')

        # Exibir o DataFrame df, mostrando os dados carregados do arquivo CSV
        st.dataframe(df)


    st.markdown(''' 
                ### Visualização da contagem de valores na coluna 'Revenue'
                <a name="value_counts"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # Exibir a contagem de valores na coluna 'Revenue'
        st.text(df.Revenue.value_counts())


    st.markdown(''' 
                ## Análise Descritiva
                <a name="descritiva"></a>
                ''', unsafe_allow_html=True)
    

    st.markdown(''' 
                ### Informações sobre a estrutura do DataFrame
                <a name="info"></a>
                ''', unsafe_allow_html=True)
    # Imprimir informações sobre a estrutura do DataFrame
    st.info(f''' 
            Quantidade de linhas: {df.shape[0]}

            Quantidade de colunas: {df.shape[1]}

            Quantidade de valores missing: {df.isna().sum().sum()} 
            ''')
    with st.echo():
        ""
        # Exibir informações detalhadas sobre o DataFrame, incluindo os tipos de dados de cada coluna e a contagem de valores não nulos
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())


    st.markdown(''' 
                ### Resumo estatístico para variáveis numéricas
                <a name="describe"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # Exibir estatísticas descritivas para colunas numéricas do DataFrame
        st.dataframe(df.describe())


    st.markdown(''' 
                ### Representação gráfica da correlação entre variáveis
                <a name="corr"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # Criar um mapa de calor (heatmap) para visualizar a correlação entre as colunas do DataFrame
        sns.heatmap(df.corr(numeric_only=True), cmap='viridis')

        # Exibir o mapa de calor
        st.pyplot(plt)


    st.markdown('''
                ## Seleção de variáveis
                <a name="feature_selection"></a>
                ''', unsafe_allow_html=True)
    

    st.markdown(''' 
                ### Variáveis que descrevem o padrão de navegação na sessão
                <a name="padrao_navegacao"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # variáveis que descrevem o padrão de navegação na sessão
        padrao_navegacao = ['Administrative', 
                            'Informational', 
                            'ProductRelated', 
                            'BounceRates', 
                            'ExitRates', 
                            'VisitorType'] 

        # tipos de dados das variáveis relacionadas ao padrão de navegação na sessão, 
        # criar um DataFrame e renomear as colunas
        st.dataframe((df[padrao_navegacao]
                    .dtypes
                    .reset_index()
                    .rename(columns={'index': 'Variável (padrao_navegacao)', 
                                      0: 'Tipo'})
                     ))

    st.markdown(''' 
                ### Variáveis que indicam a característica da data
                <a name="indicadores_temporais"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # variáveis que indicam a característica da data
        indicadores_temporais = ['SpecialDay', 'Month', 'Weekend']

        # tipos de dados das variáveis relacionadas à característica da data 
        # criar um DataFrame e renomear as colunas
        st.dataframe(df[indicadores_temporais].dtypes.reset_index().rename(columns={'index': 'Variável (indicadores_temporais)', 
                                                                                  0: 'Tipo'}), hide_index=True)


    st.markdown(''' 
                ### Seleção das variáveis numéricas e categóricas
                <a name="cat_selection"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # variáveis numéricas
        var_num = ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay']

        # selecionar as variáveis relacionadas ao padrão de navegação e à característica da data
        df_ = df[padrao_navegacao + indicadores_temporais]

        # selecionar as variáveis categóricas removendo as variáveis numéricas
        df_cat = df_.drop(columns=var_num)


        #categorizar algumas variáveis que são contínuas 
        df_['BounceRates_cat'] = pd.qcut(df_['BounceRates'], 
                                 q = 4, 
                                 duplicates = 'drop') 

        df_['ExitRates_cat'] = pd.qcut(df_['ExitRates'], 
                               q = 4, 
                               duplicates = 'drop') 

        # atualizar variáveis
        padrao_navegacao = ['Administrative', 
                            'Informational', 
                            'ProductRelated', 
                            'BounceRates_cat', 
                            'ExitRates_cat', 
                            'VisitorType']
        
        st.dataframe(df_)


    st.markdown(''' 
                ## Processamento de Variáveis Dummy
                <a name="dummy"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        var = padrao_navegacao + indicadores_temporais #variáveis que serão usadas
        var_cat = ['Month', 'Weekend', 'VisitorType', 'BounceRates_cat', 'ExitRates_cat',] #variáveis categóricas

        df2 = pd.get_dummies(df_[var].dropna()) #criação de dummies para variáveis categóricas
        st.dataframe(df2)

        #colunas que representam as variáveis categóricas
        var_catg = df2.drop(columns=var_num).columns.values

        #lista de valores booleanos indicando se cada coluna é categórica
        vars_cat = [True if column in var_catg else False for column in df2.columns]

    st.markdown(''' 
                ## Agrupamentos Hierárquicos com 3 e 4 grupos 
                <a name="agrupamento"></a>
                ''', unsafe_allow_html=True)


    st.markdown(''' 
                ### Cálculo da Matriz de Distância Gower
                <a name="gower"></a>
                ''', unsafe_allow_html=True)
    
    with st.echo():
            ""
            # Calcular a matriz de distância Gower
            dist_gower = gower_matrix(df2, cat_features=vars_cat)


    st.success('Matriz de distância Gower calculada!')
    with st.echo():
        ""
        # Criar um DataFrame com a matriz de distância Gower
        st.dataframe(pd.DataFrame(dist_gower).head())


    st.markdown(''' 
                ### Cálculo da matriz de ligação a partir da vetorização da distância Gower
                <a name="linkage"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        # Converter a matriz de distância Gower em um vetor
        gdv = squareform(X=dist_gower, force='tovector')

        # Calcular a matriz de ligação usando o método 'complete'
        Z = linkage(y=gdv, method='complete')

        # Criar um DataFrame com a matriz de ligação
        st.dataframe(pd.DataFrame(data=Z, columns=['id1', 'id2', 'dist', 'n']), hide_index=True)


    st.markdown(''' 
                ### Visualização dos agrupamentos: Dendrogramas para diferentes números de grupos
                <a name="dendrogram"></a>
                ''', unsafe_allow_html=True)
    # Para cada quantidade desejada de grupos e valor de threshold de cor, criar e exibir o dendrograma com título
    for qtd, color_threshold in [(3, .46), (4, .435)]:
        print(f'\n{qtd} grupos:')
    
        # Exibir os dendrogramas criados
        dn(color_threshold=color_threshold, num_groups=qtd, Z=Z)



    st.markdown(''' 
                ### Agrupamentos: 3 e 4 grupos
                <a name="grupo_3"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        #Adicionar coluna 'grupo_3' ao DataFrame com base no agrupamento hierárquico
        df2['grupo_3'] = fcluster(Z, 3, criterion='maxclust')
        df2.grupo_3.value_counts()

        #Adicionar coluna 'grupo_4' ao DataFrame com base no agrupamento hierárquico
        df2['grupo_4'] = fcluster(Z, 4, criterion='maxclust')
        df2.grupo_4.value_counts()

        #juntar df inicial com df que apresenta classificações
        df3 = df.reset_index().merge(df2.reset_index(), how='left')
        st.dataframe(df3.head())

    st.markdown(''' 
                ### Análises por VisitorType
                <a name="tipo_visitante"></a>
                ''', unsafe_allow_html=True)
 
    with st.echo():
        ""
        # Criar e exibir uma tabela cruzada normalizada por linha para as variáveis 'VisitorType', 'grupo_3' e 'Revenue'
        vt = df3.groupby(['VisitorType', 'grupo_3'])['index'].count().unstack()

        vtr = df3.groupby(['VisitorType', 'Revenue', 'grupo_3'])['index'].count().unstack()

        st.dataframe(vt)
        st.dataframe(vtr)

    
    st.text('''
            - Tipo de visitante parece ser importante para a divisão dos 3 grupos, já que o grupo 1 apresenta apenas novos visitantes, porém o 2 e 3 se misturam entre visitantes recorrentes.
            - Também parece que novos visitantes estão menos propensos à compras. Grupos 2 e 3 (visitantes recorrentes) também se misturam.
            ''')


    st.markdown(''' 
                ### Análises por SpecialDay
                <a name="data_festiva"></a>
                ''', unsafe_allow_html=True)
 
    with st.echo():
        ""
        sd = df3.groupby(['SpecialDay', 'grupo_3'])['index'].count().unstack()
        st.dataframe(sd)

    
    st.text('''
            - A proximidade com datas festivas parece influenciar o grupo 2, mas outros testes devem ser feitos para saber se essa influência é relevante.
            ''')



    st.markdown(''' 
                <br>

                ### Pair Plot 
                <a name="pairplot"></a>
                ''', unsafe_allow_html=True)
    with st.echo():
        ""
        #análise da relação entre mais variáveis e a propensão à compra - Revenue
        sns.pairplot(data=df3[['BounceRates', 'Revenue',  'grupo_3', 'grupo_4']], hue='Revenue')

        # Exibir o pair plot
        st.pyplot(plt)


    st.markdown('''
                ## Conclusão

                Na divisão em 3 grupos, relacionado à propensão a compras, o grupo 1 (de visitantes novos) parece menos propenso. Os grupos 2 e 3 se misturam em relação ao tipo de visitante (nos dois a maioria são visitantes recorrentes) e estão mais propensos à compras quando comparados ao grupo 1.

                ---
                <a name="final"></a>
                ''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
