#%%[1] Instalando os pacotes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
#%%[2] Criando configurações para os arquivos

anos = [2017,2018,2019,2020,2021,2022,2023,2024]
df_acidentes = []
caminho_arquivos = "Dados/"

#%%[3] Lendo e combinando os arquivos em um único dataset
for ano in anos:
    print(caminho_arquivos+'datatran'+str(ano)+'.csv')
    df = pd.read_csv(caminho_arquivos+'datatran'+str(ano)+'.csv',delimiter=";",encoding='latin-1')
    df_acidentes.append(df)

df_acidentes = pd.concat(df_acidentes, ignore_index=True)
print(df_acidentes.info())

#%%[4] Ajustando dataset
df_acidentes_filtrado = df_acidentes[df_acidentes['uf'].isin(['RS','SC','PR'])]
   

#%%[5] Criando conjunto para identificação de relação

df1_3d = df_acidentes_filtrado.groupby('causa_acidente').agg(
    qtd_acidentes=('id', 'count'),
    qtd_mortos=('mortos', 'sum'),
    qtd_ilesos=('ilesos', 'sum'),
    qtd_feridos=('feridos', 'sum') 
).reset_index()

fig = px.scatter(df1_3d, 
                 x='qtd_acidentes', 
                 y='qtd_mortos', 
                 hover_data=['causa_acidente']) 
fig.update_traces(marker=dict(size=15)) 
fig.show()
#%%[6] Fazendo top 30 para reduzir volume de categorias da variavel
top_30 = df1_3d.nlargest(30, 'qtd_acidentes')
#%%[7] dispersão simples
fig = px.scatter(top_30, 
                 x='qtd_acidentes', 
                 y='qtd_mortos', 
                 hover_data=['causa_acidente']) 
fig.update_traces(marker=dict(size=15)) 
fig.show()
#%%[8] kmeans com top3
ac = top_30.drop(columns=['causa_acidente'])
kmeans = KMeans(n_clusters=3, init='random', random_state=100).fit(ac)
est_pad = ac.apply(zscore, ddof=1)


#%%[9] aplicando kmeans
kmeans = KMeans(n_clusters=3, init='random', random_state=100).fit(ac)

#%%[10]
kmeans_clusters = kmeans.labels_
top_30['cluster_kmeans'] = kmeans_clusters
top_30['cluster_kmeans'] = top_30['cluster_kmeans'].astype('category')
#%%[11] Identificando centroides
cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = ac.columns
cent_finais.index.name = 'cluster'
cent_finais


#%% [12] plotando centroides iedntifiquei que não ficou bem distribuido por causa do "outlier"
plt.figure(figsize=(8,8))
sns.scatterplot(data=top_30, x='qtd_acidentes', y='qtd_mortos', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='qtd_acidentes', y='qtd_mortos', color = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides', fontsize=16)
plt.xlabel('qtd_acidentes', fontsize=16)
plt.ylabel('qtd_mortos', fontsize=16)
plt.legend()
plt.show()
#%% [13] Apliquei elbow para entender um pouco melhor da distribuição
elbow = []
K = range(1,10) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(ac)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,5))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% [14]Plotando Clusters
#neste plot identifiquei q não é uma boa opção fazer total, visto que o agrupamento por 2 clusters nao faz sentido e deixei com 3, mesmo assim nao ficou bom
fig = px.scatter(top_30, 
                    y='qtd_mortos', 
                    x='qtd_acidentes',
                    color='cluster_kmeans',                                        
                    hover_data= ['causa_acidente'])
fig.update_traces(marker=dict(size=15)) 
fig.show()
#%% [15] Cria um novo dataset
# Aqui decidi separar o cluster pela variavel tipo_pista para ver melhor, já q no BI mostrou que as variaveis estao distribuidas diferentemente
df_novo = df_acidentes[df_acidentes['tipo_pista'].isin(['Simples','Múltipla','Dupla'])]
#%% [16] Separando por tipo de pista
df_simples = df_novo[df_novo['tipo_pista'] == 'Simples']
df_dupla = df_novo[df_novo['tipo_pista'] == 'Dupla']
df_multipla = df_novo[df_novo['tipo_pista'] == 'Múltipla']
#%% [17] exibindo para ver se nao ficou sujeira
print(df_simples['tipo_pista'].drop_duplicates())
print(df_dupla['tipo_pista'].drop_duplicates())
print(df_multipla['tipo_pista'].drop_duplicates())

#%% [18] agregando valores
df_s_agg = df_simples.groupby(['tipo_pista', 'causa_acidente']).agg(
    qtd_acidentes=('id', 'count'),
    qtd_mortos=('mortos', 'sum'),
    qtd_ilesos=('ilesos', 'sum'),
    qtd_feridos=('feridos', 'sum') 
).reset_index()

#%% [20] Separando dataset
df_s = df_s_agg.drop(columns=['tipo_pista','causa_acidente'])
#%% [21] Criando Kmeans
kmeans_s = KMeans(n_clusters=2, init='random', random_state=100).fit(df_s)
est_pad = df_s.apply(zscore, ddof=1)
#%% [22] Aplicando ao dataset agregado da pista simples
kmeans_clusters_s = kmeans_s.labels_
df_s_agg['cluster_kmeans'] = kmeans_clusters_s
df_s_agg['cluster_kmeans'] = df_s_agg['cluster_kmeans'].astype('category')

#%% [23] Identificando centroides
cent_finais_s = pd.DataFrame(kmeans_s.cluster_centers_)
cent_finais_s.columns = df_s.columns
cent_finais_s.index.name = 'cluster'
cent_finais_s


#%% [24]plotando centroides
plt.figure(figsize=(8,8))
sns.scatterplot(data=df_s_agg, x='qtd_acidentes', y='qtd_mortos', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='qtd_acidentes', y='qtd_mortos', color = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides', fontsize=16)
plt.xlabel('qtd_acidentes', fontsize=16)
plt.ylabel('qtd_mortos', fontsize=16)
plt.legend()
plt.show()
#%% [25]aplicando elbow
elbow_s = []
K_s = range(1,10) 
for k in K_s:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_s)
    elbow_s.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K_s, elbow_s, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,10))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow - Pista Simples', fontsize=16)
plt.show()

#%% [26] Vi que o numero 3 é o melhor numero de cluster, visto que é o "cotovelo"
kmeans_s = KMeans(n_clusters=3, init='random', random_state=100).fit(df_s)
est_pad = df_s.apply(zscore, ddof=1)
kmeans_clusters_s = kmeans_s.labels_

df_s_agg['cluster_kmeans'] = kmeans_clusters_s
df_s_agg['cluster_kmeans'] = df_s_agg['cluster_kmeans'].astype('category')
cent_finais_s = pd.DataFrame(kmeans_s.cluster_centers_)
cent_finais_s.columns = df_s.columns
cent_finais_s.index.name = 'cluster'
cent_finais_s

#%% [27] centroides pista simples:
plt.figure(figsize=(8,8))
sns.scatterplot(data=df_s_agg, x='qtd_acidentes', y='qtd_mortos', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='qtd_acidentes', y='qtd_mortos', color = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides - Simples', fontsize=16)
plt.xlabel('qtd_acidentes', fontsize=16)
plt.ylabel('qtd_mortos', fontsize=16)
plt.legend()
plt.show()

#%% [28] agregando valores pista dupla
df_d_agg = df_dupla.groupby(['tipo_pista', 'causa_acidente']).agg(
    qtd_acidentes=('id', 'count'),
    qtd_mortos=('mortos', 'sum'),
    qtd_ilesos=('ilesos', 'sum'),
    qtd_feridos=('feridos', 'sum') 
).reset_index()

#%% [29] Separando dataset
df_d = df_d_agg.drop(columns=['tipo_pista','causa_acidente'])

#%% [30]aplicando elbow
elbow_d = []
K_d = range(1,10) 
for k in K_d:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_d)
    elbow_d.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K_d, elbow_d, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,10))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow - Pista Dupla', fontsize=16)
plt.show()

#%%[28] Aplicando para pista dupla
kmeans_d = KMeans(n_clusters=3, init='random', random_state=100).fit(df_d)
est_pad_d = df_d.apply(zscore, ddof=1)
kmeans_clusters_d = kmeans_d.labels_
df_d_agg['cluster_kmeans'] = kmeans_clusters_d
df_d_agg['cluster_kmeans'] = df_d_agg['cluster_kmeans'].astype('category')
cent_finais_d = pd.DataFrame(kmeans_d.cluster_centers_)
cent_finais_d.columns = df_d.columns
cent_finais_d.index.name = 'cluster'
cent_finais_d
#%% [31] Plot do cluster e centroides
plt.figure(figsize=(8,8))
sns.scatterplot(data=df_d_agg, x='qtd_acidentes', y='qtd_mortos', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='qtd_acidentes', y='qtd_mortos', color = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides - Dupla', fontsize=16)
plt.xlabel('qtd_acidentes', fontsize=16)
plt.ylabel('qtd_mortos', fontsize=16)
plt.legend()
plt.show()
#%% [32] agregando valores pista multipla
df_m_agg = df_multipla.groupby(['tipo_pista', 'causa_acidente']).agg(
    qtd_acidentes=('id', 'count'),
    qtd_mortos=('mortos', 'sum'),
    qtd_ilesos=('ilesos', 'sum'),
    qtd_feridos=('feridos', 'sum') 
).reset_index()

#%% [33] Separando dataset
df_m = df_m_agg.drop(columns=['tipo_pista','causa_acidente'])

#%% [34]aplicando elbow
elbow_m = []
K_m = range(1,10) # ponto de parada pode ser parametrizado manualmente
for k in K_m:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_m)
    elbow_m.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K_m, elbow_m, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,10))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow - Pista Multipla', fontsize=16)
plt.show()


#%%[28] Aplicando para pista dupla
kmeans_m = KMeans(n_clusters=3, init='random', random_state=100).fit(df_m)
est_pad_m = df_m.apply(zscore, ddof=1)
kmeans_clusters_m = kmeans_m.labels_
df_m_agg['cluster_kmeans'] = kmeans_clusters_m
df_m_agg['cluster_kmeans'] = df_m_agg['cluster_kmeans'].astype('category')
cent_finais_m = pd.DataFrame(kmeans_m.cluster_centers_)
cent_finais_m.columns = df_m.columns
cent_finais_m.index.name = 'cluster'
cent_finais_m
#%% [31] Plot do cluster e centroides
plt.figure(figsize=(8,8))
sns.scatterplot(data=df_m_agg, x='qtd_acidentes', y='qtd_mortos', hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(data=cent_finais, x='qtd_acidentes', y='qtd_mortos', color = 'red', label = 'Centróides', marker="X", s = 40)
plt.title('Clusters e Centroides - Múltipla', fontsize=16)
plt.xlabel('qtd_acidentes', fontsize=16)
plt.ylabel('qtd_mortos', fontsize=16)
plt.legend()
plt.show()
#%%
fig = px.scatter(df_s_agg, 
                    y='qtd_mortos', 
                    x='qtd_acidentes',
                    color='cluster_kmeans',  
                    title='Clusters e Centroides - Simples',         
                    size = 'qtd_mortos',
                    hover_data=['causa_acidente'])
fig.show()

#%%
fig = px.scatter(df_d_agg, 
                    y='qtd_mortos', 
                    x='qtd_acidentes',
                    color='cluster_kmeans',  
                    title='Clusters e Centroides - Dupla',         
                    size = 'qtd_mortos',
                    hover_data=['causa_acidente'])
fig.show()



#%%
fig = px.scatter(df_m_agg, 
                    y='qtd_mortos', 
                    x='qtd_acidentes',
                    color='cluster_kmeans',  
                    title='Clusters e Centroides - Múltipla',         
                    size = 'qtd_mortos',
                    hover_data=['causa_acidente'])
fig.show()


