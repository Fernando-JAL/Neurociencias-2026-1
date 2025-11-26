import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

def lista_dfs(ensayo_dict):
    ensayos_dfs_ = []
    for sujeto in ensayo_dict.keys():
        ensayos_dfs.append(ensayo_dict[sujeto])
    return ensayos_dfs_

def capsula(lista_dfs, th, porcentaje_):
    # conjuntamos las matrices
    stack = np.stack([df.values for df in lista_dfs])

    # aplicamos el filtro a cada matriz
    above = stack > th

    N = len(lista_dfs)
    count_above = above.sum(axis = 0)  # contar cuántos valores superan el umbral en celda

    min_requerido = int(np.ceil(porcentaje_*N)) # np.ceil redondea para arriba
    # con esto bastará con quedarnos con las celdas de "above" que cumplan con min_requerido, para así tener el número de sujetos que superan el umbral
    
    # Vamos a obtener a la matriz binarizada de todos
    result_bool = count_above >= min_requerido

    result_df = pd.DataFrame(result_bool, index=lista_dfs[0].index, columns=lista_dfs[0].columns)

    return result_df

# Métricas grafo
def metricas_grafo(G):
    # Métricas
    # clústering promedio
    clust_coeff = nx.average_clustering(G)

    # longitud de camino promedio (camino más corto)
    try:                                                  # se activa cuando se genera un error
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan # red no conecta
    
    # coeficiente de mundo pequeño
    # grafo aleatorio para comparar
    G_rand = nx.gnm_random_graph(n=G.number_of_nodes(), m=G.number_of_edges())
    clust_rand = nx.average_clustering(G_rand)
    path_rand = nx.average_shortest_path_length(G_rand)
    small_world_sigma = (clust_coeff / clust_rand) / (path_length / path_rand)

    # modularidad
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, communities)

    # hubs
    degree_dict = dict(G.degree())
    betwenness = nx.betweenness_centrality(G)
    betwenness = sorted(betwenness.items(), key=lambda x: x[1], reverse = True)

    # eficiencia global y local
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)

    return  (clust_coeff, path_length, small_world_sigma, communities, modularity, betwenness, global_eff, local_eff, degree_dict)

# Grafo 2D
def grafo_2d(ensayo_df, pos):
    ensayo_grafo_ = nx.from_pandas_adjacency(ensayo_df)
    nx.draw_circular(ensayo_grafo, with_labels = True, font_size = 7.5, ax = pos)
    return ensayo_grafo_

# Grafo 3D
def grafo_3d(Hub, coords, pos):
    x, y, z =coords["x"].values, coords["y"].values, coords["z"].values
    
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index] # cambiar el tamaño del nodo si es el hub
    pos.scatter(x, y, z, alpha = 0.5)
     
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize = 10)
        if coords.index[idx] == hub[0]:
            pos.text(x_, y_, z_, "HUB", color = "red", fontsize = 13) # cambiar el color del texto del nodo si es el 

# Grafo comunidades
def grafo_comunidades(comunidades, Hub, coords, pos):
    x, y, z = coords["x"].values, coords["y"].values, coords["z"].values

    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index] # cambiar el tamaño del nodo si es el hub
    pos.scatter(x, y, z, alpha = 0.5)
 
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize = 10)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, "HUB", color = "red", fontsize = 13) # cambiar el color del texto del nodo si es el hub

    colores = ["yellow", "blue", "orange", "olive", "red"]
    for n_comunidad, comunidad in enumerate(comunidades):  # recorrer las comunidades, plotearemos así las aristas
        for idx in range(len(comunidad)-1):
            n1, n2 = list(comunidad)[idx], list(comunidad)[idx+1]
            # plotear arista
            x_ = [coords.loc[n1, "x"], coords.loc[n2, "x"]]
            y_ = [coords.loc[n1, "y"], coords.loc[n2, "y"]]
            z_ = [coords.loc[n1, "z"], coords.loc[n2, "z"]]
            pos.plot(x_, y_, z_, linewidth = 3, alpha = 0.4, color = colores[n_comunidad])
    pos.set_title("comunidades del ensayo 3")
