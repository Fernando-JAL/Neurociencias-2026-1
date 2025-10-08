import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.pyplot import ylabel
from sympy.printing.pretty.pretty_symbology import line_width

# Defino los archivos a usar
excel_conectividad = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S03_datasets\EEG.xlsx"
excel_coordenadas = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S03_datasets\EEG_3D_coordinates.xlsx"

# Leo los archivos de excel, como diccionarios
conectividad_dict = pd.read_excel(excel_conectividad, sheet_name=None)
coordenadas_dict = pd.read_excel(excel_coordenadas, sheet_name=None)

# Especifico cuál hoja de cada excel voy a leer, posición 0
motor_df = conectividad_dict[list(conectividad_dict.keys())[0]]
coordenadas_df = coordenadas_dict[list(coordenadas_dict.keys())[0]]

# Redefino el índice para que sea más fácil convertirlo a grafo
motor_df.set_index("Unnamed: 0", inplace=True, drop= True)
coordenadas_df.set_index("Canal", inplace=True, drop= True)

# Dibujemos el grafo
G = nx.from_pandas_adjacency(motor_df)
# plt.figure(figsize=(5, 5))
# nx.draw_circular(G, with_labels=True)
#
# plt.show()

# Creamos el plot2D del grafo
# # Crear un diccionario de posiciones
# pos2D = {canal: (coordenadas_df.loc[canal, 'x'],
#                  coordenadas_df.loc[canal, 'y']) for canal in coordenadas_df.index}
#
# plt.figure(figsize=(4, 4))
# nx.draw_networkx(G, pos=pos2D)
# plt.show()

# Creamos el plot3D del grafo
pos3D = {canal: (coordenadas_df.loc[canal, 'x'],
                 coordenadas_df.loc[canal, 'y'],
                 coordenadas_df.loc[canal, 'z']) for canal in coordenadas_df.index}

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

# Dibujar los nodos en 3d
for canal, (x, y, z) in pos3D.items():
    ax.scatter(x, y, z, s=100)
    ax.text(x, y, z+0.03, canal, fontsize=10, ha='center')

for i, j, data in G.edges(data=True):
    x = [pos3D[i][0], pos3D[j][0]]
    y = [pos3D[i][1], pos3D[j][1]]
    z = [pos3D[i][2], pos3D[j][2]]
    ax.plot(x, y, z, linewidth=data['weight']*10)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Red de conectividad EEG (3D)')
plt.show()

# Crear mapa de calor
import seaborn as sns

np.fill_diagonal(motor_df.values, 0)
ax = sns.heatmap(motor_df.values,
                 annot=True, cmap='hot', fmt=".2f",
                 xticklabels=motor_df.columns, yticklabels=motor_df.index)
ax.set(xlabel="channels", ylabel="channels")
plt.show()
