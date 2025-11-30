import numpy as np
import mne
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms import community

def cargar_brainvision(ruta_pre, ruta_post):
     with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        raw_pre = mne.io.read_raw_brainvision(ruta_pre, preload=True)
        raw_post = mne.io.read_raw_brainvision(ruta_post, preload=True)
        return raw_pre, raw_post

def detectar_canales_malos_por_varianza(raw, factor=3.0):
    
    ##Detecta canales malos usando varianza (outliers).
    ##factor: número de desviaciones estándar para marcar un outlier.
    
    data = raw.get_data()
    var = np.var(data, axis=1)
    
    mean = np.mean(var)
    std  = np.std(var)

    umbral_alto = mean + factor * std
    umbral_bajo = mean - factor * std

    idx_malos = np.where((var > umbral_alto) | (var < umbral_bajo))[0]
    nombres_malos = [raw.ch_names[i] for i in idx_malos]

    return nombres_malos, var

def obtener_cerrados(raw_pre, raw_post, dur_bloque=135):
   
    # PRE
    pre_C1 = raw_pre.copy().crop(tmin=dur_bloque, tmax=2*dur_bloque)
    pre_C2 = raw_pre.copy().crop(tmin=3*dur_bloque, tmax=4*dur_bloque)

    # POST
    post_C1 = raw_post.copy().crop(tmin=dur_bloque, tmax=2*dur_bloque)
    post_C2 = raw_post.copy().crop(tmin=3*dur_bloque, tmax=4*dur_bloque)

    # ojos cerrado
    pre_cerrado = mne.concatenate_raws([pre_C1, pre_C2])
    post_cerrado = mne.concatenate_raws([post_C1, post_C2])

    # matrices
    data_pre = pre_cerrado.get_data()
    data_post = post_cerrado.get_data()

    return data_pre,data_post

def matriz_corr(data, ch_names):
    corr = np.corrcoef(data)
    df_corr = pd.DataFrame(corr, index=ch_names, columns=ch_names)
    return df_corr

def lista_dfs(ensayo_dict):
    ensayos_dfs_ = []
    for sujeto in ensayo_dict.keys():
        ensayos_dfs_.append(ensayo_dict[sujeto])
    return ensayos_dfs_
    

# Filtrado de matrices
def capsula(lista_dfs_, th, porcentaje_):
    # Conjuntamos las matrices
    stack = np.stack([df.values for df in lista_dfs_])
    # Aplicamos filtro a cada matriz
    above = stack > th
    
    N = len(lista_dfs_) # = 109
    count_above = above.sum(axis=0) # Contando cuantos valores superan el umbral en celda
    
    min_requerido = int(np.ceil(porcentaje_*N)) # 0.6*109 = 65.4
    # con esto bastara con quedarnos con las celdas de 'above' que cumplan con min_requerido
    # para así tener el número de sujetos que superan el umbral
    result_bool = count_above >= min_requerido
    
    result_df = pd.DataFrame(result_bool, 
                             index=lista_dfs_[0].index, 
                             columns=lista_dfs_[0].columns).astype(int)
# Quitar self-loops
    np.fill_diagonal(result_df.values, 0)



    return result_df
    
def metricas_grafo(G):
    # Métricas:
    #Clusterin promedio
    clust_coeff = nx.average_clustering(G)
    # Longitud de camino promedio (camino más corto)
    try:
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan # red no conexa
    # Coeficiente de mundo pequeño
    # Comparar con grafo aleatorio de igual N, K
    G_rand = nx.gnm_random_graph(n=G.number_of_nodes(), m=G.number_of_edges())
    clust_rand = nx.average_clustering(G_rand)
    path_rand = nx.average_shortest_path_length(G_rand)
    small_world_sigma = (clust_coeff / clust_rand) / (path_length / path_rand)
    # Modularidad
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, communities)
    # Hubs
    degree_dict = dict(G.degree())
    betwenness = nx.betweenness_centrality(G)
    betwenness = sorted(betwenness.items(), key=lambda x: x[1], reverse=True)
    # Eficiencia global y local
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)

    return (clust_coeff, path_length, small_world_sigma, communities, 
            modularity, betwenness, global_eff, local_eff, degree_dict)
            

# Grafo 2D
def grafo2D(df, pos):
    ensayo_grafo = nx.from_pandas_adjacency(df)
    nx.draw_circular(ensayo_grafo, with_labels=True, font_size=7, ax=pos)

    return ensayo_grafo
    

def grafo3D(coords, Hub, pos):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=5)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)

def grafo_comunidades(comunidades, Hub, coords, pos):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=5)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)
    
    colores = ['red', 'green', 'blue', 'black', 'orange']
    for n_comunidad, comunidad in enumerate(comunidades):
        for idx in range(len(comunidad)-1):
            n1, n2 = list(comunidad)[idx], list(comunidad)[idx+1]
            x_ = [coords.loc[n1, 'x'], coords.loc[n2, 'x']]
            y_ = [coords.loc[n1, 'y'], coords.loc[n2, 'y']]
            z_ = [coords.loc[n1, 'z'], coords.loc[n2, 'z']]
            pos.plot(x_, y_, z_, linewidth=3, alpha=0.4, color=colores[n_comunidad])
    