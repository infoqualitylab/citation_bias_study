import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import netwulf as nw

attr_list = pd.read_csv("Article_attr.csv", usecols=[0, 1, 2, 3])

G = nx.DiGraph()
G.add_nodes_from(attr_list["ID"].to_list())
#stylized_network, config = nw.visualize(G)

edgelist = pd.read_csv("inclusion_net_edges.csv")
a = edgelist["citing_ID"].to_list()
b = edgelist["cited_ID"].to_list()
edge_list = zip(a, b)
G.add_edges_from(edge_list)
G.edges()

year = attr_list['year'].to_list()

coloring = []
for i in attr_list["Attitude"]:
    if i == 'inconclusive':
        coloring.append('pink')
    elif i == 'for':
        coloring.append('green')
    else:
        coloring.append('blue')

labelss = {}
#labelss[0] = 0
for i, val in enumerate(year):
    labelss[i + 1] = val

print(labelss)

nx.draw(
    G,
    with_labels=True,
    labels=labelss,
    node_size=150,
    width=0.5,
    font_size=9,
    alpha=0.5,
    node_color=coloring
)

plt.show()