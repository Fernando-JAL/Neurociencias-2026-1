import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

def lista_dfs(ensayo_dict):
    ensayos_dfs_ = []
    for sujeto in ensayo_dict.keys():
        ensayos_dfs_.append(ensayo_dict[sujeto]) 
    return ensayos_dfs_
    
def estats(ensayos_dfs):
    th25,th50,th75, promedio, varianza= [],[],[],[],[]
    mu_sigma=[]
    for sujeto in ensayos_dfs:
        #Cálculo de estadísticos
        data=sujeto.stack().values
        th25.append(np.percentile(data, 25))
        th50.append(np.percentile(data, 50))
        th75.append(np.percentile(data, 75))
        promedio.append(np.mean(data))
        varianza.append(np.var(data))
        mu_sigma.append(np.mean(data)+np.std(data))
    return th25, th50, th75, promedio, varianza, mu_sigma


def  capsula(lista_dfs,th,porcentaje):
    #Conjuntamos las matrices
    stack=np.stack([df.values for df in lista_dfs])
    above= stack > th
    #Aplicamos filtro a cada matriz
    N= len(lista_dfs)
    count_above= above.sum(axis=0) #Contando cuantos valores superan el umbral en celda
    
    min_requerido= int(np.ceil(porcentaje*N)) 
    # con esto bastara con quedarnos con las celdas de 'above' que cumplan con miin_requerido
    # para así tener el número de sujetos que superan el umbral
    result_bool=count_above>= min_requerido
    result_df=pd.DataFrame(result_bool,
                           index=lista_dfs[0].index,
                          columns=lista_dfs[0].columns).astype(int)
    return result_df

def grafo2D(df,pos):
    ensayo_grafo=nx.from_pandas_adjacency(df)
    nx.draw_circular(ensayo_grafo, with_labels=True,ax=pos)

    return ensayo_grafo

def metricas_grafo(G):
    #Metricas:
    # Cluster promedio
    clust_coeff= nx.average_clustering(G)
    # Longitud de camino promedio(camino más corto)
    # (b) Longitud de camino promedio (camino más corto)
    try:
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan  # red no conexa
    # Mundo pequeño
    # Comparar con grafo aleaatorio de igual N,K
    G_rand=nx.gnm_random_graph(n=G.number_of_nodes(), m=G.number_of_edges())
    clust_rand= nx.average_clustering(G_rand)
    path_rand= nx.average_shortest_path_length(G_rand)
    small_world_sigma= (clust_coeff/clust_rand)/(path_length/path_rand)
    #Modularidad
    # (d) Modularidad — usando método de comunidades
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, communities)
    
    # (e) Hubs — grado, centralidad de intermediación
    degree_dict = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    betwenness= sorted(betweenness.items(),key=lambda x: x[1],reverse=True)
    
    # (f) Eficiencia global y local
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)

    return (clust_coeff, path_length,small_world_sigma,communities,modularity,betwenness,global_eff,local_eff,degree_dict)


def grafo2D(df,pos):
    ensayo_grafo=nx.from_pandas_adjacency(df)
    nx.draw_circular(ensayo_grafo, with_labels=True,ax=pos)

    return ensayo_grafo
    
def grafo3D(coords,Hub,pos):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=10)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)


