import networkx as nx
from matplotlib import pyplot as plt
G= nx.Graph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
G.add_edges_from([('D', 'A', {'weight':1.3}),
                  ('A', 'C',  {'weight':3.9}),
                  ('D', 'E',  {'weight':0.8}),
                  ('B', 'C',  {'weight':0.6}),
                  ('A', 'E',  {'weight':2.2}),
                  ('D', 'B',  {'weight':3.4})])
plt.figure(figsize=(2, 2))
nx.draw_networkx(G)

