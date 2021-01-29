import pandas as pd
from igraph import *
import cairo
import numpy as np
import math
from matplotlib import pyplot as plt


# create attribute list
# easy for this file because the first 6 columns are in fact attribute list
attr_list = pd.read_csv('HTTLPR.csv', usecols=[0, 1, 2, 3, 4, 5])
attr_list['PaperID'] = attr_list.index

# create edge list
matrix = pd.read_csv('HTTLPR.csv')
matrix = matrix.iloc[:, 6:]

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
g.vs['year'] = attr_list['YearOnline']

# create edges for the real graph
edge_list = []
for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
        if isinstance(matrix.iloc[i, j], str):
            s = matrix.iloc[i, j]
            if s.replace(" ", "") == "X":
                edge_list.append((i, j))
                g.add_edge(source=i, target=j)

# vis = plot(g)
# vis.show()

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

# vis_sim = plot(g_sim)
# vis_sim.show()

# pairwise neighbor overlap ratio
neighbor_overlap = pd.DataFrame(columns=['node_1','node_2','ratio'])
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
                neighbor_overlap = neighbor_overlap.append({'node_1': node_1_ix,
                                         'node_2': node_2_ix,
                                         'ratio': overlap},ignore_index=True)

medium_overlap = neighbor_overlap[neighbor_overlap['ratio'] > 0.5]
# get all the nodes
medium_overlap_nodes = list(set(medium_overlap['node_1']).union(set(medium_overlap['node_2'])))
# non-medium nodes
non_medium_overlap_nodes = list(set(g.vs.indices).difference(set(medium_overlap_nodes)))

medium_overlap_nodes = g.vs.select(medium_overlap_nodes)
non_medium_overlap_nodes = g.vs.select(non_medium_overlap_nodes)

# visualization [any updates need to be here]
medium_overlap_nodes['color'] = 'light blue'
non_medium_overlap_nodes['color'] = 'pink'

plot(g, margin=50, bbox=(600, 600), layout= g.layout_grid_fruchterman_reingold()).show()

# print nodes names
#for node in non_medium_overlap_nodes:
    #print(node['name'],node['year'],node['outcome'])

# neighbor_overlap_sim = []
# for node_1_ix in g_sim.vs.indices:
#     for node_2_ix in g_sim.vs.indices:
#         if node_2_ix > node_1_ix:
#             node_1 = g_sim.vs[node_1_ix]
#             node_2 = g_sim.vs[node_2_ix]
#             node_1_set = set(g_sim.neighbors(node_1, mode=OUT))
#             node_2_set = set(g_sim.neighbors(node_2, mode=OUT))
#             node_1_set.intersection(node_2_set)
#             if (len(node_1_set) + len(node_2_set)) != 0:
#                 overlap = len(node_1_set.intersection(node_2_set)) / (
#                         len(node_1_set) + len(node_2_set) - len(
#                     node_1_set.intersection(node_2_set)))  # full is 1, no overlap is zero
#                 neighbor_overlap_sim.append(overlap)
#
# bins = np.linspace(math.ceil(min(neighbor_overlap_sim)),
#                    math.floor(max(neighbor_overlap_sim)),
#                    20)  # fixed number of bins
#
# plt.xlim([min(neighbor_overlap) - 0.1, max(neighbor_overlap) + 0.1])
# plt.ylim([0, 700])
# plt.hist(neighbor_overlap, bins=bins, alpha=0.5)
# plt.title('Distribution of Pairwise Neighbor Overlap Ratio')
# plt.xlabel('Pairwise neighbor overlap ratio')
# plt.ylabel('Count')
# plt.xlim([min(neighbor_overlap_sim) - 0.1, max(neighbor_overlap_sim) + 0.1])
# plt.hist(neighbor_overlap_sim, bins=bins, alpha=0.5)
# plt.legend(['Real Network', 'Simulated Network'])
# plt.show()

# PART II
# ratio between actual citation and theoretical citations
g_degree = pd.Series(data=g.degree(mode=IN),index=g.vs.indices)
g_sim_degree = pd.Series(data=g_sim.degree(mode=IN),index=g_sim.vs.indices)

degree_ratio = pd.DataFrame(columns=['paperID','degree_ratio'])

last_generation = []
for idx, degree in enumerate(g_degree):
    if g_sim_degree[idx]!=0:
        degree_ratio = degree_ratio.append({'paperID': idx, 'degree_ratio': round(g_degree[idx] / g_sim_degree[idx],3)},
                            ignore_index=True)
    else:
        last_generation.append(idx)

# articles that are insufficiently cited
low_cutoff = 0.1
high_cutoff = 0.9
idx_low = degree_ratio[degree_ratio['degree_ratio'] < low_cutoff].index.tolist()
idx_low = [int(item) for item in idx_low]

idx_high = degree_ratio[degree_ratio['degree_ratio'] > high_cutoff].index.tolist()
idx_high = [int(item) for item in idx_high]


for node in g.vs.indices:
    if node in idx_low:
        g.vs[node]['citation_ratio'] = 'Low'
    elif node in idx_high:
        g.vs[node]['citation_ratio'] = 'High'
    elif node in last_generation:
        g.vs[node]['citation_ratio'] = 'Last Generation'
    else:
        g.vs[node]['citation_ratio'] = 'Average'

# block of code for visualization [ update here ]
color_dict = {"High": "green", "Low": "pink", "Average": "light blue","Last Generation":"white"}
g.vs["color"] = [color_dict[ratio] for ratio in g.vs["citation_ratio"]]

plot(g, vertex_size=30, nodebbox=(600,600),margin=50, edge_size=40, layout= g.layout_grid_fruchterman_reingold()).show()

#Figure 2
color_dict = {"High": "green", "Low": "pink", "Average": "light blue","Last Generation":"white"}
g.vs["color"] = [color_dict[ratio] for ratio in g.vs["citation_ratio"]]
bins = np.linspace(0,1,10)
plt.xlabel('Ratio between actual citations and theoretical citations')
plt.hist(degree_ratio['degree_ratio'], bins=bins, alpha=0.5, color='mediumorchid', edgecolor='mediumorchid', linewidth=1, hatch='///')
plt.ylabel('Count')
plt.show()

# for node in idx_low:
#     print(g.vs[node]['name'],g.vs[node]['year'],g.vs[node]['outcome'])
