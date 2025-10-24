import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
#creacion de grafo dirigido
G=nx.DiGraph()

G.add_nodes_from(['A','B','C','D','E'])
G.add_edges_from([('A','B'),('A','D'),('B','C'),('C','E'),('D','E')])
plt.figure(figsize=(8,8))
nx.draw_networkx(G,arrows=True, arrowstyle='->', with_labels=True)
plt.show()

#Crear df del grafo dirigido
import numpy as np

# Crear matriz de adyacencia
nodos = ['A', 'B', 'C', 'D','E']
matriz_adj = np.array([
    [0,2.1,0,7.5,0],
    [8.6,0,10.2,0,0],
    [0,7.3,0,0,1.10],
    [6.6,0,0,0,0.8],
     [0,0,3.5,4.2,0]])

adj_df =pd.DataFrame(matriz_adj, index=nodos, columns=nodos)
print("\nMatriz de adyacencia:")
print(adj_df)

G=nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph())
plt.figure(figsize=(6,6))
pos=nx.spring_layout(G) #crea formas aleatorias para los diferentes nodos
nx.draw_networkx(G,pos,with_labels=True, arrows=True, arrowstyle='->',connectionstyle='arc3,rad=0.15')

pesos=nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,pesos,label_pos=0.3)
plt.show()

# Mapa de calor

import seaborn as sns

np.fill_diagonal(adj_df.values, 0)
ax = sns.heatmap(adj_df.values,
                 annot=True, cmap='hot', fmt=".2f",
                 xticklabels=adj_df.columns, yticklabels=adj_df.index)
ax.set(xlabel="channels", ylabel="channels")
plt.show()

# Cuartiles
from scipy import stats
# Calcular cuartiles de todos los valores
valores = adj_df.values.flatten()
q1 = np.percentile(valores, 25)
q2 = np.percentile(valores, 50)  # Mediana
q3 = np.percentile(valores, 75)

print(f"\nCuartiles de todos los valores:")
print(f"Q1 (25%): {q1}")
print(f"Q2 (50% - Mediana): {q2}")
print(f"Q3 (75%): {q3}")

#Mapa de calor con umbral de 50%

