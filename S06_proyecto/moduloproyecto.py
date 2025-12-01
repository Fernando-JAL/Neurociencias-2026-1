import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from scipy import stats
import mne
from scipy.signal import butter, filtfilt
from networkx.algorithms import community
import re

def lista_dfs(ensayo_dict):

    return [ensayo_dict[suj] for suj in ensayo_dict]


# para solo qeudarme con los canales en comun, ya que en algunos se elimiaron 8 canales porque tenian mucho ruido
def canales_comunes(lista_dfs_):
    sets = [set(df.columns) for df in lista_dfs_]
    return set.intersection(*sets)

def recortar_a_comunes(lista_dfs_):
    comunes = canales_comunes(lista_dfs_)
    comunes = sorted(list(comunes)) 
    nuevas = [df.loc[comunes, comunes] for df in lista_dfs_]
    return nuevas, comunes

    
def capsula(lista_dfs_, th, porcentaje_):

    stack = np.stack([df.values for df in lista_dfs_])  # (sujetos,128,128)
    above = stack > th

    N = len(lista_dfs_)
    count_above = above.sum(axis=0)

    min_requerido = int(np.ceil(porcentaje_ * N))

    result_bool = count_above >= min_requerido

    result_df = pd.DataFrame(
        result_bool.astype(int),
        index=lista_dfs_[0].index,
        columns=lista_dfs_[0].columns
    )

    return result_df

# para modificar los nombres de los electrodos mas bonitos
def limpiar_nombres_canales(df):
    nuevas_cols = []
    for c in df.columns:
        c2 = c.replace("EEG", "").replace("CPz", "")
        nuevas_cols.append(c2)

    # limpiar filas (index)
    nuevas_filas = []
    for f in df.index:
        f2 = f.replace("EEG", "").replace("CPz", "")
        nuevas_filas.append(f2)

    df.columns = nuevas_cols
    df.index = nuevas_filas
    return df

for i in range(len(lista)):
    np.fill_diagonal(lista[i], np.nan)
    
def analisis_estadisticos(ensayos_dfs):

    th25, th50, th75 = [], [], []
    promedio, varianza, mu_sigma = [], [], []


    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for sujeto in ensayos_dfs:
        data = sujeto.stack().values  # PLV a vector

        th25.append(np.percentile(data, 25))
        th50.append(np.percentile(data, 50))
        th75.append(np.percentile(data, 75))
        promedio.append(np.mean(data))
        varianza.append(np.var(data))
        mu_sigma.append(np.mean(data) + np.std(data))

        sns.histplot(data, kde=True, ax=axes[0][0], bins=30)
        sns.ecdfplot(data, ax=axes[0][1])

    axes[1][0].plot(th25, label='Percentil 25%')
    axes[1][0].plot(th50, label='Percentil 50%')
    axes[1][0].plot(th75, label='Percentil 75%')
    axes[1][0].plot(promedio, label='Media')
    axes[1][0].plot(varianza, label='Varianza')
    axes[1][0].plot(mu_sigma, label='Mu + Sigma')
    axes[1][0].legend()

    axes[0][0].set_title("Histogramas por sujeto")
    axes[0][1].set_title("ECDF por sujeto")
    axes[1][0].set_title("Estadísticos por sujeto")
    axes[1][1].axis("off")

    plt.tight_layout()
    plt.show()

# tabla
    estadisticos_df = pd.DataFrame(
        columns=['th25', 'th50', 'th75', 'promedio', 'mu_sigma'],
        index=['min', 'max', 'mediana', 'MAD']
    )

    columnas = [th25, th50, th75, promedio, mu_sigma]
    min_, max_, mediana_, mad_ = [], [], [], []

    for col in columnas:
        min_.append(np.min(col))
        max_.append(np.max(col))
        mediana_.append(np.median(col))
        mad_.append(stats.median_abs_deviation(col))

    estadisticos_df.loc['min'] = min_
    estadisticos_df.loc['max'] = max_
    estadisticos_df.loc['mediana'] = mediana_
    estadisticos_df.loc['MAD'] = mad_

    return estadisticos_df


# estadisticos
def metricas_grafo_resumido(G):

    # Clustering promedio
    clust_coeff = nx.average_clustering(G)

    # Longitud de camino promedio
    try:
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan  # red no conexa

    # Grafo aleatorio comparable
    G_rand = nx.gnm_random_graph(n=G.number_of_nodes(), m=G.number_of_edges())

    if not nx.is_connected(G_rand):
        largest = G_rand.subgraph(max(nx.connected_components(G_rand), key=len))
        G_rand = largest

    clust_rand = nx.average_clustering(G_rand)
    path_rand = nx.average_shortest_path_length(G_rand)

    small_world_sigma = (clust_coeff / clust_rand) / (path_length / path_rand)

    # MODULARIDAD + COMUNIDADES
    from networkx.algorithms import community
    comunidades = community.greedy_modularity_communities(G)
    modularidad = community.modularity(G, comunidades)

    # Betweenness centrality completo
    betweenness = nx.betweenness_centrality(G)
    hubs_ordenados = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top5_hubs = hubs_ordenados[:5]

    # Eficiencia
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)

    return {
        "clustering": clust_coeff,
        "path_length": path_length,
        "small_world_sigma": small_world_sigma,
        "modularidad": modularidad,
        "eficiencia_global": global_eff,
        "eficiencia_local": local_eff,
        "top5_hubs": top5_hubs,
        "comunidades": comunidades,      
        "betweenness_completo": hubs_ordenados  
    }

def generar_coords_esfera_para_canales(channel_names):
    """
    Genera coordenadas 3D en una esfera para la lista de canales dada.
    Distribución uniforme aproximada.
    """

    n = len(channel_names)

    # Distribución esférica tipo 'Fibonacci sphere'
    indices = np.arange(0, n)
    phi = np.pi * (3. - np.sqrt(5.))  # ángulo áureo
    y = 1 - (indices / float(n - 1)) * 2  # y va de 1 a -1
    radius = np.sqrt(1 - y * y)
    theta = phi * indices

    x = radius * np.cos(theta)
    z = radius * np.sin(theta)

    coords = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=channel_names)
    return coords


# grafo 2d
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
    
    # Tamaño de nodos (hub más grande)
    nodes_size = [40 if idx != Hub[0] else 200 for idx in coords.index]

    pos.scatter(x, y, z, s=nodes_size, alpha=0.5)

    # Etiquetas
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=5)

    # Colores
    colores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

    # Dibujar aristas de cada comunidad
    for c_i, comunidad in enumerate(comunidades):
        comunidad = list(comunidad)
        # Recorremos la comunidad
        for i in range(len(comunidad)):
            for j in range(i+1, len(comunidad)):
                
                n1, n2 = comunidad[i], comunidad[j]

                # Verificar que ambos nodos existan
                if n1 in coords.index and n2 in coords.index:
                    x_ = [coords.loc[n1, 'x'], coords.loc[n2, 'x']]
                    y_ = [coords.loc[n1, 'y'], coords.loc[n2, 'y']]
                    z_ = [coords.loc[n1, 'z'], coords.loc[n2, 'z']]
                    pos.plot(x_, y_, z_, color=colores[c_i], alpha=0.5, linewidth=2)

    pos.set_title("Comunidades")

    