
import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sp
from scipy.io import loadmat
import mne
import glob
import os
import powerlaw

def lista_dfs(ensayo_dict):
    return list(ensayo_dict.values())


def estadisticas(lista_de_dfs, pos):
   
    t25, t50, t75, promedio, varianza = [], [], [], [], []
    
    for sujeto in lista_de_dfs:
        data = sujeto.stack().values
        
        
        t25.append(np.percentile(data, 25))
        t50.append(np.percentile(data, 50))
        t75.append(np.percentile(data, 75))
        promedio.append(np.mean(data))
        varianza.append(np.std(data))
        
        
        sns.histplot(data, ax=pos[0][0], kde=True, element="step", fill=False, alpha=0.3, legend=False)
        sns.ecdfplot(data, ax=pos[0][1], alpha=0.3, legend=False)

   
    pos[1][0].plot(t25, label="P25", linestyle='--')
    pos[1][0].plot(t50, label="Mediana", linewidth=2)
    pos[1][0].plot(t75, label="P75", linestyle='--')
    pos[1][0].plot(promedio, label="Promedio", color='k', alpha=0.5)
    pos[1][0].legend()
    
    pos[1][1].plot(varianza, label="Varianza", color='red')
    pos[1][1].legend()
    return(t25, t50, t75, promedio, varianza)

def crear_tabla_resumen(t25, t50, t75, promedio, varianza):
    listas_datos = [t25, t50, t75, promedio, varianza]
    columnas = ["t25", "t50", "t75", "promedio", "varianza"]
    
    minimos, maximos, medias, perc30, medianas = [], [], [], [], []
    
    for lista in listas_datos:
        minimos.append(np.min(lista))
        maximos.append(np.max(lista))
        medias.append(np.mean(lista))
        perc30.append(np.percentile(lista, 30))
        medianas.append(np.median(lista))
        
    df = pd.DataFrame(
        [minimos, maximos, medias, perc30, medianas],
        columns=columnas,
        index=["min", "max", "media", "perc30", "mediana"]
    )
    return df

def capsula(list_df, th, porcentaje_):

    stack = np.stack([df.values for df in list_df])
    above = stack > th
    
    contar = above.sum(axis=0)
    N = len(list_df)
    min_requerido = int(np.ceil(porcentaje_ * N))
    
    resultado_boleano = contar >= min_requerido
    
    result_df = pd.DataFrame(
    resultado_boleano.astype(int), 
    index=list_df[0].index, 
    columns=list_df[0].columns
    )
    
    return result_df

def dibujar_grafo(grafo_obj, pos):
   
    
   
    nx.draw_circular(
        grafo_obj, 
        with_labels=True, 
        font_size=8, 
        node_color='skyblue', 
        ax=pos
    )
def metricas (G):
#Para evitar caminos infinitos habrá que calcular solo en el componente conectado
#"componente conexa gigante" 
    clust = nx.average_clustering(G)
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
    small_world_sigma = (clust / clust_rand) / (path_length / path_rand)
#Modularidad
    from networkx.algorithms import community
    communities= community.greedy_modularity_communities(G)
    modularity= community.modularity(G, communities)
    #HUBS
    degree_dict= dict(G.degree())
    betwenness=nx.betweenness_centrality(G)
    betwenness=sorted(betwenness.items(), key=lambda x:x[1], reverse=True)
    #Eficiencia globaly local
    global_eff= nx.global_efficiency(G)
    local_eff=nx.local_efficiency(G)
    print("metricas de grafo")

    return(clust, path_length, small_world_sigma, communities, modularity, betwenness, global_eff, local_eff, degree_dict)
def grafo_3d (Hub, coords, pos):
    x,y,z= coords["x"],coords["y"].values, coords["z"].values

    nodes_size=[30 if idx !=Hub[0] else 200 for idx in coords.index]
    pos.scatter(x,y,z, alpha=0.5, s=nodes_size )
    for idx, (x_,y_,z_) in enumerate(zip(x,y,z)):
        pos.text(x_,y_,z_,coords.index[idx], fontsize=10)
        if coords.index[idx]== Hub[0]:
            pos.text(x_,y_,z_,"HUB", color="red", fontweight="bold", fontsize=10)

def grafo_de_comunidades(comunidades, Hub, coords, pos):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=10)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)
    
    colores = ['red','green','blue','black','orange']
    for n_comunidad, comunidad in enumerate (comunidades):
        for idx in range(len(comunidad)-1):
            n1, n2= list(comunidad)[idx], list(comunidad)[idx+1]
            x_=[coords.loc[n1,'x'], coords.loc[n2,'x']]
            y_=[coords.loc[n1,'y'], coords.loc[n2,'y']]
            z_=[coords.loc[n1,'z'], coords.loc[n2,'z']]
            pos.plot(x_,y_,z_, linewidth=3, alpha=0.4, color=colores[n_comunidad])
    pos.set_title('comunidades')

def tabla_de_grados (diccionario_de_grado):
    
    media_grados= sum(diccionario_de_grado.values())/len(diccionario_de_grado)
    grado_máximo= max(diccionario_de_grado.values())
    grado_minimo= min(diccionario_de_grado.values())
    nodos_g_altos= dict(sorted(diccionario_de_grado.items(), key=lambda item: item[1], reverse=True))
    top_5_nodos = dict(list(nodos_g_altos.items())[:5])
    d= pd.DataFrame( {"Media de grados": [media_grados], "Grado_máximo":[grado_máximo],
                      "Grado_mínimo":[grado_minimo], "Nodos con mayor grado":[top_5_nodos]})
    return(d.T)
def medidas_de_centralidad (grafo):
    centralidad_grado=nx.degree_centrality(grafo)
    cercania=nx.closeness_centrality(grafo)
    intermediación=nx.betweenness_centrality(grafo)
    grado= dict(grafo.degree())
    columnas=( "nodo", "grado", "intermediación", "cercanía")
    datos = {
        'Grado': grado,
        'Centralidad_de_Grado': centralidad_grado,
        'Cercania': cercania,
        'Intermediacion': intermediación
    }

    df = pd.DataFrame(datos)  
    return df
def distribución_de_grados (d_g):
    grados = list(d_g.keys())
    valores = list(d_g.values())
    plt.figure(figsize=(8, 5))
    plt.bar(grados, valores, color="pink", edgecolor="black", alpha=0.7)
    plt.xlabel('Nodo')
    plt.ylabel('Grado')
    plt.title('Distribución de grado')
    plt.show()

def ley_potencia (dic_g):
    grados= list(dic_g.values())
    results = powerlaw.Fit(grados, discrete=True, verbose=False)
    alpha = results.power_law.alpha
    xmin = results.power_law.xmin
    
    print(f"\n--- RESULTADOS ESTADÍSTICOS ---")
    print(f"Alpha calculado: {alpha:.4f}")
    print(f"Xmin (corte): {xmin}")

def club_de_ricos (Grafo):
    c = nx.rich_club_coefficient(Grafo, normalized=True)
    grados = list(c.keys())
    valores = list(c.values())
    
    plt.figure(figsize=(8, 5))
    plt.plot(grados, valores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Línea base (red aleatoria)')
    plt.xlabel('Grado mínimo k')
    plt.ylabel('Coeficiente de Rich Club Normalizado')
    plt.title('Análisis de Club de Ricos en la red')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
def grafos_comunidades (grafo, comunidades):
    cantidad = len(comunidades)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, cantidad, figsize=(5 * cantidad, 5))
    for i, comunidad in enumerate(comunidades):
        subgrafo = grafo.subgraph(comunidad)
        ax = axes[i]
        pos = nx.spring_layout(subgrafo, seed=42)  
        color_actual = cmap(i % 20)
        # Dibujar el grafo
        nx.draw_networkx_nodes(
            subgrafo, pos,
            node_color=[color_actual],
            node_size=100,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            subgrafo, pos,
            alpha=0.6,
            edge_color='gray',
            ax=ax
        )
        nx.draw_networkx_labels(subgrafo, pos, font_size=15, ax=ax)
        ax.set_title(f'Comunidad {i+1}\n({len(comunidad)} nodos, {subgrafo.number_of_edges()} conexiones)')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
