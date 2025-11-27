import numpy as np
import pandas as pd
import networkx as nx
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt

def lista_dfs(ensayo_dict):
    ensayos_dfs = []
    for sujeto in ensayo3_dict.keys():
        ensayos_dfs.append(ensayo3_dict[sujeto])
    return ensayos_dfs

def capsula(lista_dfs, th, porcentaje_):
    # Conjuntamos las matrices
    stack = np.stack([df.values for df in lista_dfs])
    # Aplicamos filtro a cada matriz
    above = stack > th
    
    N = len(lista_dfs) # = 109
    count_above = above.sum(axis=0) # Contando cuantos valores superan el umbral en celda
    
    min_requerido = int(np.ceil(porcentaje_*N)) # 0.6*109 = 65.4
    # con esto bastara con quedarnos con las celdas de 'above' que cumplan con min_requerido
    # para así tener el número de sujetos que superan el umbral
    result_bool = count_above >= min_requerido
    
    result_df = pd.DataFrame(result_bool, 
                             index=lista_dfs[0].index, 
                             columns=lista_dfs[0].columns).astype(int)
    return result_df

#Grafo 2D
def grafo2D (df, pos):
    ensayo_grafo = nx.from_pandas_adjacency (df)
    nx.draw_circular(ensayo_grafo, with_labels= True, font_sizes=7, ax = pos)

    return ensayo_grafo