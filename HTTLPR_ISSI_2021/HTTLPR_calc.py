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

vis = plot(g)
vis.show()

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

vis_sim = plot(g_sim)
vis_sim.show()

# select groups by "outcome" column in the HTTLPR dataset
positive_nodes = g.vs.select(outcome='Positive')
negative_nodes = g.vs.select(outcome='Negative')
neutral_nodes = g.vs.select(outcome='Unclear')

positive_nodes_sim = g_sim.vs.select(outcome='Positive')
negative_nodes_sim = g_sim.vs.select(outcome='Negative')
neutral_nodes_sim = g_sim.vs.select(outcome='Unclear')

# total edges: 1799 in the simulated network with 2 years as year-gap
# total edges: 488 in the real network

##############################
# Without membership, global #
##############################
ns = len(g.vs)  # 73
ms = len(g.es)  # 488
ns_sim = len(g_sim.vs)  # 73
ms_sim = len(g_sim.es)  # 1799

# average degree
2 * ms_sim / ns_sim  # 49.3
2 * ms / ns  # 13.4

# edge density
4 * ms_sim / ns / (ns - 1)  # 1.37
4 * ms / ns / (ns - 1)  # 0.371

# global clustering coefficient
g.transitivity_undirected()  # 0.366
g_sim.transitivity_undirected()  # 0.602

# average local clustering coefficient
g.transitivity_avglocal_undirected()  # 0.503
g_sim.transitivity_avglocal_undirected()  # 0.600

# calculate triad participation ratio
g_triads = g.cliques(min=3, max=3)
g_sim_triads = g_sim.cliques(min=3, max=3)

len(set([y for x in g_triads for y in x])) / len(g.vs)  # 0.918
len(set([y for x in g_sim_triads for y in x])) / len(g_sim.vs)  # 1.00

# Fraction over median degree (FOMD)
len(g.vs.select(_degree_gt=median(g.degree()))) / len(g.vs)  # 0.479
len(g_sim.vs.select(_degree_gt=median(g_sim.degree()))) / len(g_sim.vs)  # 0.493

# pair-wise neighbor overlap
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

import numpy as np
import math
from matplotlib import pyplot as plt

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

#############################
# With membership, by group #
#############################

# define subgraphs
g_positive = g.subgraph(vertices=positive_nodes)
g_negative = g.subgraph(vertices=negative_nodes)
g_neutral = g.subgraph(vertices=neutral_nodes)

g_sim_positive = g_sim.subgraph(vertices=positive_nodes_sim)
g_sim_negative = g_sim.subgraph(vertices=negative_nodes_sim)
g_sim_neutral = g_sim.subgraph(vertices=neutral_nodes_sim)

# Average degree within the subgroup
mean(g_positive.degree())  # 6.67
mean(g_negative.degree())  # 6.37
mean(g_neutral.degree())  # 1.64

mean(g_sim_positive.degree())  # 16.7
mean(g_sim_negative.degree())  # 24.3
mean(g_sim_neutral.degree())  # 6.91

# average degree -- overall
mean(g.degree(vertices=positive_nodes))  # 15.9
mean(g_sim.degree(vertices=positive_nodes_sim))  # 50.9

mean(g.degree(vertices=negative_nodes))  # 12.4
mean(g_sim.degree(vertices=negative_nodes_sim))  # 48.7

mean(g.degree(vertices=neutral_nodes))  # 11.1
mean(g_sim.degree(vertices=neutral_nodes_sim))  # 48.0

# internal edge density: 2*ms/(ns(ns-1))
ns_positive = len(positive_nodes)
ns_negative = len(negative_nodes)
ns_neutral = len(neutral_nodes)

len(g.es.select(_between=(positive_nodes, positive_nodes)).indices) * 4 / ns_positive / (ns_positive - 1)  # 0.580
len(g.es.select(_between=(negative_nodes, negative_nodes)).indices) * 4 / ns_negative / (ns_negative - 1)  # 0.344
len(g.es.select(_between=(neutral_nodes, neutral_nodes)).indices) * 4 / ns_neutral / (ns_neutral - 1)  # 0.328

# compare to simulated network
len(g_sim.es.select(_between=(positive_nodes_sim, positive_nodes_sim)).indices) * 4 / ns_positive / (
        ns_positive - 1)  # 1.45
len(g_sim.es.select(_between=(negative_nodes_sim, negative_nodes_sim)).indices) * 4 / ns_negative / (
        ns_negative - 1)  # 1.31
len(g_sim.es.select(_between=(neutral_nodes_sim, neutral_nodes_sim)).indices) * 4 / ns_neutral / (
        ns_neutral - 1)  # 1.38

# calculate transitivity
# global clustering coefficient
g_positive.transitivity_undirected()  # 0.448
g_negative.transitivity_undirected()  # 0.348
g_neutral.transitivity_undirected()  # 0

g_sim_positive.transitivity_undirected()  # 0.671
g_sim_negative.transitivity_undirected()  # 0.554
g_sim_neutral.transitivity_undirected()  # 0.642

# average local clustering coefficient
g_positive.transitivity_avglocal_undirected()  # 0.637
g_negative.transitivity_avglocal_undirected()  # 0.377
g_neutral.transitivity_avglocal_undirected()  # 0.0

g_sim_positive.transitivity_avglocal_undirected()  # 0.671
g_sim_negative.transitivity_avglocal_undirected()  # 0.556
g_sim_neutral.transitivity_avglocal_undirected()  # 0.667

# calculate triad participation ratio
# positive
g_positive_triads = g_positive.cliques(min=3, max=3)
g_sim_positive_triads = g_sim_positive.cliques(min=3, max=3)

# triads participation ratio:
len(set([y for x in g_positive_triads for y in x])) / len(positive_nodes)  # 0.875
len(set([y for x in g_sim_positive_triads for y in x])) / len(positive_nodes_sim)  # 1.00

# negative
g_negative_triads = g_negative.cliques(min=3, max=3)
g_sim_negative_triads = g_sim_negative.cliques(min=3, max=3)

# triads participation ratio:
len(set([y for x in g_negative_triads for y in x])) / len(negative_nodes)  # 0.737
len(set([y for x in g_sim_negative_triads for y in x])) / len(negative_nodes_sim)  # 1.00

# Neutral
g_neutral_triads = g_neutral.cliques(min=3, max=3)
g_sim_neutral_triads = g_sim_neutral.cliques(min=3, max=3)

# triads participation ratio:
len(set([y for x in g_neutral_triads for y in x])) / len(neutral_nodes)  # 0.0
len(set([y for x in g_sim_negative_triads for y in x])) / len(negative_nodes_sim)  # 1.00

# fraction over median degree (FOMD)
# find the median degree

# real network
len(positive_nodes.select(_degree_gt=median(g.degree()))) / len(positive_nodes)  # 0.542
len(negative_nodes.select(_degree_gt=median(g.degree()))) / len(negative_nodes)  # 0.447
len(neutral_nodes.select(_degree_gt=median(g.degree()))) / len(neutral_nodes)  # 0.455

# simulated unbiased network
len(positive_nodes_sim.select(_degree_gt=median(g_sim.degree()))) / len(positive_nodes_sim)  # 0.542
len(negative_nodes_sim.select(_degree_gt=median(g_sim.degree()))) / len(negative_nodes_sim)  # 0.526
len(neutral_nodes_sim.select(_degree_gt=median(g_sim.degree()))) / len(neutral_nodes_sim)  # 0.273

# expansion - directed, only count outgoing nodes
# positive nodes
len(g.es.select(_source_in=positive_nodes.indices, _target_in=g.vs.select(outcome_ne='Positive')).indices) / len(
    positive_nodes)  # 2.70
len(g_sim.es.select(_source_in=positive_nodes_sim.indices,
                    _target_in=g_sim.vs.select(outcome_ne='Positive')).indices) / len(
    positive_nodes_sim)  # 10.8

# negative nodes
len(g.es.select(_source_in=negative_nodes.indices, _target_in=g.vs.select(outcome_ne='Negative')).indices) / len(
    negative_nodes)  # 3.87
len(g_sim.es.select(_source_in=negative_nodes_sim.indices,
                    _target_in=g_sim.vs.select(outcome_ne='Negative')).indices) / len(
    negative_nodes_sim)  # 16.8

# neutral nodes
len(g.es.select(_source_in=neutral_nodes.indices, _target_in=g.vs.select(outcome_ne='Unclear')).indices) / len(
    neutral_nodes)  # 6.0
len(g_sim.es.select(_source_in=neutral_nodes_sim.indices,
                    _target_in=g_sim.vs.select(outcome_ne='Negative')).indices) / len(
    neutral_nodes_sim)  # 13.0

# cut ratio: directed
len(g.es.select(_source_in=positive_nodes.indices, _target_in=g.vs.select(outcome_ne='Positive')).indices) \
/ len(g_sim.es.select(_source_in=positive_nodes_sim.indices, _target_in=g_sim.vs.select(outcome_ne='Positive')).indices)
# 0.252

len(g.es.select(_source_in=negative_nodes.indices, _target_in=g.vs.select(outcome_ne='Negative')).indices) \
/ len(g_sim.es.select(_source_in=negative_nodes_sim.indices, _target_in=g_sim.vs.select(outcome_ne='Negative')).indices)
# 0.231

len(g.es.select(_source_in=neutral_nodes.indices, _target_in=g.vs.select(outcome_ne='Unclear')).indices) \
/ len(g_sim.es.select(_source_in=neutral_nodes_sim.indices, _target_in=g_sim.vs.select(outcome_ne='Unclear')).indices)


# Conductance -- directed
def conductance_calc(g, in_group_nodes, out_group_nodes):
    cs = len(g.es.select(_source_in=in_group_nodes, _target_in=out_group_nodes))
    ms = len(g.es.select(_between=(in_group_nodes, in_group_nodes)))
    return cs / (2 * ms + cs)


# conductance positive nodes
conductance_calc(g, positive_nodes, g.vs.select(outcome_ne='Positive'))  # 0.289
conductance_calc(g_sim, positive_nodes_sim, g_sim.vs.select(outcome_ne='Positive'))

# conductance negative nodes
conductance_calc(g, negative_nodes, g.vs.select(outcome_ne='Negative'))  # 0.378
conductance_calc(g_sim, negative_nodes_sim, g_sim.vs.select(outcome_ne='Negative'))  # 0.408

# conductance neutral nodes
conductance_calc(g, neutral_nodes, g.vs.select(outcome_ne='Neutral'))  # 0.806
conductance_calc(g_sim, neutral_nodes_sim, g_sim.vs.select(outcome_ne='Neutral'))  # 0.761


# odfs - directed
def ODF_calc(input_graph, in_group_nodes, out_group_nodes):
    odf = []
    for idx in in_group_nodes.indices:
        internal_degree = len(input_graph.es.select(_source=input_graph.vs[idx], _target_in=in_group_nodes))
        outer_degree = len(input_graph.es.select(_source=input_graph.vs[idx], _target_in=out_group_nodes))
        if (internal_degree + outer_degree) != 0:
            odf.append(outer_degree / (internal_degree + outer_degree))
    return odf


def flake_odf(odf_list):
    return sum((1 for i in odf_list if i > 0.5)) / len(odf_list)


# positive nodes
odf = ODF_calc(g, positive_nodes, g.vs.select(outcome_ne='Positive'))
max(odf)  # 0.750
mean(odf)  # 0.312
flake_odf(odf)  # 0.273

odf_sim = ODF_calc(g_sim, positive_nodes_sim, g_sim.vs.select(outcome_ne='Positive'))
max(odf_sim)  # 0.618
mean(odf_sim)  # 0.444
flake_odf(odf_sim)  # 0.391

# negative
odf = ODF_calc(g, negative_nodes, g.vs.select(outcome_ne='Negative'))
max(odf)  # 1.0
mean(odf)  # 0.637
flake_odf(odf)  # 0.622

odf_sim = ODF_calc(g_sim, negative_nodes_sim, g_sim.vs.select(outcome_ne='Negative'))
max(odf_sim)  # 1.0
mean(odf_sim)  # 0.633
flake_odf(odf_sim)  # 1.0

# neutral nodes
odf = ODF_calc(g, neutral_nodes, g.vs.select(outcome_ne='Unclear'))
max(odf)  # 1.0
mean(odf)  # 0.880
flake_odf(odf)  # 0.909

odf_sim = ODF_calc(g_sim, neutral_nodes_sim, g_sim.vs.select(outcome_ne='Unclear'))
max(odf_sim)  # 1.0
mean(odf_sim)  # 0.877
flake_odf(odf_sim)  # 1.0
