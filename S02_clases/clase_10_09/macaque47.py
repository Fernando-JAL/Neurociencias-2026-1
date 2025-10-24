import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


file = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S03_datasets\BCT\macaco_M132_F99_LH.csv"
df = pd.read_csv(file)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], color='orange')
ax.scatter(-df['x'], df['y'], df['z'], color='blue')
plt.show()