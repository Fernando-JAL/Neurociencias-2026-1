import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def lista_dfs(ensayo):
    lista_df = []
    for item in ensayo.keys():
        lista_df.append(ensayo[item])
    return lista_df

def capsula(lista_dfs, umbral, porcentaje):
    stack = np.stack([df.values for df in lista_dfs])
    
    #Filtro a cada matriz
    above = stack>umbral
    
    N = len(lista_dfs)
    count_above = above.sum(axis = 0) #Contar cuantos valores de cada matriz supera el umbral
    minimo = int(np.ceil(porcentaje*N))
    
    result_bool = count_above >= minimo
    result_df = pd.DataFrame(result_bool, index=lista_dfs[0].index,
                            columns = lista_dfs[0].columns).astype(int)
    return result_df

def estadisticos_df(lista_dfs):

    todo, p25,p50,p75,promedio,varianza = [],[],[],[],[],[]
    
    fig, axes = plt.subplots(2,2, figsize=(12,7))
    for sujetos in lista_dfs:
        data = sujetos.stack().values
        #Estadísticos
        todo.append(sujetos.stack().values)
        p25.append(np.percentile(data,25))
        p50.append(np.percentile(data,50))
        p75.append(np.percentile(data,75))
        promedio.append(np.mean(data))
        varianza.append(np.var(data))
        sns.histplot(sujetos.stack().values,ax=axes[0][0], kde=True)
        sns.ecdfplot(sujetos.stack().values,ax=axes[0][1])
    todo_final = np.concatenate(todo)

        # Plots
        
    axes[1][0].plot(p25,label = 'Percentil al 25%')
    axes[1][0].plot(p50,label = 'Percentil al 50%')
    axes[1][0].plot(p75,label = 'Percentil al 75%')
    axes[1][0].plot(promedio,label = 'Promedio')
    axes[1][0].plot(varianza,label = 'Varianza')
    axes[1][0].legend()

    estadisticos = pd.DataFrame(index = ['Valor','Min','Max','Media','mediana','p30','p75','Normalidad'], 
                            columns = ['p25','p50','p75','Media','Varianza'])
    lista = [p25,p50,p75,promedio,varianza]
    lista2 = ['p25','p50','p75','Media','Varianza']
    for estadistico,nombre in zip(lista,lista2):
        minn, maxx, mean, p30, p75_, median,  = min(estadistico), max(estadistico), np.mean(estadistico), np.percentile(estadistico,30), np.percentile(estadistico,75), np.median(estadistico)
        shapiro_statistic, shapiro_pvalue = stats.shapiro(estadistico)
        if nombre == 'p25':
            valor = np.percentile(todo_final,25)
        elif nombre == 'p50':
            valor = np.percentile(todo_final,50)
        elif nombre == 'p75':
            valor = np.percentile(todo_final,75)
        elif nombre == 'Media':
            valor = np.mean(todo_final)
        elif nombre == 'Varianza':
            valor = np.var(todo_final)
        if shapiro_pvalue>=0.05:
            valores = [valor,minn, maxx, mean, median, p30, p75_,'Sí']
        else:
            valores = [valor,minn, maxx, mean, median, p30, p75_,'No']
        estadisticos[nombre] = valores
    estadisticos = estadisticos.round(5)
    estadisticos = estadisticos.applymap(lambda x: f"{x:.5f}" if isinstance(x,(float,int)) else x)
    axes[1][1].axis('off')  # desactiva los ejes
    tabla = axes[1][1].table(
        cellText=estadisticos.values,
        rowLabels=estadisticos.index,
        colLabels=estadisticos.columns,
        cellLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(7)
    tabla.scale(0.9, 0.9)

    return

def metricas(df_final):
    G = nx.from_pandas_adjacency(df_final)
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
    comunidades_multiples = [c for c in communities if len(c) > 1]
    comunidades_unicas = [c for c in communities if len(c) == 1]
    modularity = community.modularity(G, communities)
    # Hubs
    degree_dict = dict(G.degree())
    top3 = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    betwenness = nx.betweenness_centrality(G)
    betwenness = sorted(betwenness.items(), key=lambda x: x[1], reverse=True)
    # Eficiencia global y local
    global_eff = nx.global_efficiency(G)
    local_eff = nx.local_efficiency(G)

    lista_c = ['CC',
                'Camino más corto',
                'Sigma',
                'Comms >1',
               'Comms 1',
                'Mod',
                'Cent',
                'Ef. global',
                'Ef. local',
                'Top 3 hubs',
              ]
    metricas_df = pd.DataFrame(columns = lista_c)
    valores_m = [clust_coeff, path_length, small_world_sigma, len(comunidades_multiples), len(comunidades_unicas),
            modularity, betwenness[0], global_eff, local_eff, top3]
    # for nombre, metrica in zip(lista_c,valores_m):
    #     metricas_df[nombre] = metrica
    metricas_df.loc[0, :] = valores_m
    metricas_df
    return (metricas_df, clust_coeff, path_length, small_world_sigma, communities, 
            modularity, betwenness, global_eff, local_eff, degree_dict)

def  grafcomm(comunidades, Hub, coords, pos):

    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=10)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)
    
    colores = [
    "#FF0000",  # rojo
    "#00AEEF",  # azul vibrante
    "#39FF14",  # verde neón
    "#FF00FF",  # magenta
    "#FFA500",  # naranja brillante
    "#FFFF00",  # amarillo puro
    "#8A2BE2",  # azul violeta
    "#00FFEF",  # turquesa neón
    "#FF1493",  # rosa fuerte
    "#00FF00",  # verde puro
    "#FF4500",  # naranja rojizo brillante
    "#1E90FF",  # azul intenso
    "#FFD700",  # dorado intenso
    "#7FFF00",  # chartreuse
    "#DC143C",  # carmesí
    "#00CED1",  # turquesa oscuro
    "#ADFF2F",  # verde lima brillante
    "#FF69B4",  # rosa vibrante
    "#BA55D3",  # púrpura medio vibrante
    "#40E0D0"   # turquesa claro
]

    for n_comunidad, comunidad in enumerate(comunidades):
        for idx in range(len(comunidad)-1):
            n1, n2 = list(comunidad)[idx], list(comunidad)[idx+1]
            x_ = coords.loc[n1,'x'], coords.loc[n2,'x']
            y_ = coords.loc[n1,'y'], coords.loc[n2,'y']
            z_ = coords.loc[n1,'z'], coords.loc[n2,'z']
            pos.plot(x_,y_,z_, linewidth = 3, alpha = 0.4, color=colores[n_comunidad])
    pos.set_title('Comunidades_ensayo_3')

def graf3d(Hub, coords, pos):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values
    
    nodes_size = [30 if idx != Hub[0] else 200 for idx in coords.index]
    pos.scatter(x, y, z, alpha=0.5, s=nodes_size)
    for idx, (x_, y_, z_) in enumerate(zip(x, y, z)):
        pos.text(x_, y_, z_, coords.index[idx], fontsize=10)
        if coords.index[idx] == Hub[0]:
            pos.text(x_, y_, z_, 'HUB', color='red', fontweight='bold', fontsize=10)
def graf2d(df_final, pos):
    G = nx.from_pandas_adjacency(df_final)
    graph = nx.draw_circular(G, with_labels=True, font_size=4, ax = pos)
    plt.show()