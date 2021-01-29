import pandas as pd
from igraph import *
import cairo

# create attribute list
# easy for this file because the first 6 columns are in fact attribute list
attr_list = pd.read_csv('HTTLPR.csv', usecols=[0, 1, 2, 3, 4, 5])
attr_list['PaperID'] = attr_list.index
# vertices ids start from 0 in igraph, so we also label the nodes here from 0
# brown year = 2013 but year online = 2012?

# create edge list
matrix = pd.read_csv('HTTLPR.csv')
matrix = matrix.iloc[:, 6:]
# edge_list = [(matrix[col][matrix[col].eq('X')].index[i], matrix.columns.get_loc(col)) for col in matrix.columns for i in range(len(matrix[col][matrix[col].eq('X')].index))]


# search_for_alias
search_dict = pd.Series(attr_list['PaperID'].values, index=attr_list['Study']).to_dict()
search_dict_reverse = pd.Series(attr_list['Study'].values, index=attr_list['PaperID']).to_dict()
matrix.columns = [search_dict[x] for x in matrix.columns]

g = Graph(directed=True)
g.add_vertices(73)

study = attr_list['Study'].to_list()
outcome = attr_list['Outcome'].to_list()

# add attributes to the graph
g.vs['name'] = study
g.vs["label"] = g.vs["name"]
g.vs['outcome'] = outcome
color_dict = {"Positive": "light blue", "Negative": "pink", "Unclear": "orange"}
g.vs["color"] = [color_dict[outcome] for outcome in g.vs["outcome"]]

# create edges for the real graph
edge_list = []
for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
        if isinstance(matrix.iloc[i, j], str):
            s = matrix.iloc[i, j]
            if s.replace(" ", "") == "X":
                edge_list.append((i, j))
                g.add_edge(source=i, target=j)

#vis = plot(g)
#vis.show()

# create simulated network g_sim
g_sim = Graph(directed=True)
g_sim.add_vertices(73)

g_sim.vs['name'] = study
g_sim.vs["label"] = g_sim.vs["name"]
g_sim.vs['outcome'] = outcome
color_dict = {"Positive": "light blue", "Negative": "pink", "Unclear": "orange"}
g_sim.vs["color"] = [color_dict[outcome] for outcome in g_sim.vs["outcome"]]
g_sim.vs['Year'] = attr_list['YearOnline']

year_gap = 2  # allow connection of two papers if citing_year - cited_year >= year_gap

sim_edge_list = []

# construct graph
for i in g_sim.vs.indices:
    for j in g_sim.vs.indices:
        if g_sim.vs[j]['Year'] - g_sim.vs[i]['Year'] >= year_gap:
            g_sim.add_edge(j, i)

#vis_sim = plot(g_sim)
#vis_sim.show()

# select groups by "outcome" column in the HTTLPR dataset
positive_nodes = g.vs.select(outcome='Positive')
negative_nodes = g.vs.select(outcome='Negative')
neutral_nodes = g.vs.select(outcome='Unclear')

positive_nodes_sim = g_sim.vs.select(outcome='Positive')
negative_nodes_sim = g_sim.vs.select(outcome='Negative')
neutral_nodes_sim = g_sim.vs.select(outcome='Unclear')



import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors

neighbor_overlap = []
for node_1_ix in g.vs.indices:
    for node_2_ix in g.vs.indices:
        if node_2_ix > node_1_ix:
            node_1 = g.vs[node_1_ix]
            node_2 = g.vs[node_2_ix]
            node_1_set = set(g.neighbors(node_1, mode=OUT))
            node_2_set = set(g.neighbors(node_2, mode=OUT))
            node_1_set.intersection(node_2_set)
            if (len(node_1_set) + len(node_2_set)) != 0:
                overlap = len(node_1_set.intersection(node_2_set)) / (
                        len(node_1_set) + len(node_2_set) - len(
                    node_1_set.intersection(node_2_set)))  # full is 1, no overlap is zero
                neighbor_overlap.append(overlap)

neighbor_overlap_sim = []
for node_1_ix in g_sim.vs.indices:
    for node_2_ix in g_sim.vs.indices:
        if node_2_ix > node_1_ix:
            node_1 = g_sim.vs[node_1_ix]
            node_2 = g_sim.vs[node_2_ix]
            node_1_set = set(g_sim.neighbors(node_1, mode=OUT))
            node_2_set = set(g_sim.neighbors(node_2, mode=OUT))
            node_1_set.intersection(node_2_set)
            if (len(node_1_set) + len(node_2_set)) != 0:
                overlap = len(node_1_set.intersection(node_2_set)) / (
                        len(node_1_set) + len(node_2_set) - len(
                    node_1_set.intersection(node_2_set)))  # full is 1, no overlap is zero
                neighbor_overlap_sim.append(overlap)


plt.ylabel('Count')
#plt.plot(x, y, linewidth=2.0)

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.title('Pairwise Neighbor Overlap Ratio')
bins = np.linspace(math.ceil(min(neighbor_overlap_sim)),
                   math.floor(max(neighbor_overlap_sim)),
                   20)  # fixed number of bins
plt.xlabel('Real Network')
plt.ylabel('Count')
plt.hist(neighbor_overlap, bins=bins, histtype='step', alpha=0.5)
plt.subplot(132)
plt.hist(neighbor_overlap_sim, bins=bins, histtype='step', alpha=0.5)
plt.xlabel('Simulated Network')
plt.ylabel('Count')
plt.show()
'''
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

plt.show()

'''
'''
#overlapping one
#take a look at bin size
bins = np.linspace(math.ceil(min(neighbor_overlap_sim)),
                   math.floor(max(neighbor_overlap_sim)),
                   20)  # fixed number of bins

plt.xlim([min(neighbor_overlap) - 0.1, max(neighbor_overlap) + 0.1])
plt.ylim([0, 700])
plt.hist(neighbor_overlap, bins=bins, alpha=0.5)
plt.title('Distribution of Pairwise Neighbor Overlap Ratio')
plt.xlabel('Pairwise neighbor overlap ratio')
plt.ylabel('Count')
plt.xlim([min(neighbor_overlap_sim) - 0.1, max(neighbor_overlap_sim) + 0.1])
plt.hist(neighbor_overlap_sim, bins=bins, alpha=0.5)
plt.legend(['Real Network', 'Simulated Network'])
plt.show()
'''