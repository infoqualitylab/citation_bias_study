# Citation bias study

## ISSI_2021  
This folder contains a jupyter notebook for the calculation contained in the work-in-progress paper submitted to ISSI 2021  

## Jasmine  
This folder contains Jasmine Yuan's work  

## NetworkX_migration  
This folder contains files for the iGraph to NetworkX code migration  
March 18, 2021: 
import pandas as pd
import cairo
import math
import numpy as np
import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt

### create attribute list
### easy for this file because the first 6 columns are in fact attribute list
attr_list = pd.read_csv('HTTLPR.csv', usecols=[0, 1, 2, 3, 4, 5])
attr_list['PaperID'] = attr_list.index

data = pd.read_csv("HTTLPR.csv") #reading the .csv file (excel spreadsheet)
new = pd.DataFrame(np.zeros(shape=(73, 73)), columns=data['Study'], index=data['Study']) #making a data frame

# create edge list
matrix = pd.read_csv('HTTLPR.csv')
matrix = matrix.iloc[:, 6:]

# search_for_alias
search_dict = pd.Series(attr_list['PaperID'].values, index=attr_list['Study']).to_dict()
search_dict_reverse = pd.Series(attr_list['Study'].values, index=attr_list['PaperID']).to_dict()
matrix.columns = [search_dict[x] for x in matrix.columns]

for a in data['Study']:
    for b, v in zip(data['Study'],data[a]):
        if v == 'X':
            new.at[a, b] = 1
print(len(b))

#g = nx.generators.directed.kamada_kawai_layout(75, 3, 0.5)
g = nx.generators.directed.random_k_out_graph(75, 3, 0.5)
pos = nx.layout.spring_layout(g)

#node_sizes = [10 + i for i in range(len(g))]
M = g.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [i for i in range(M)]

study = attr_list['Study'].to_list()
outcome = attr_list['Outcome'].to_list()
year = attr_list['Year'].to_list()
#print(year)

#userInp = input("Enter ratio")

coloring = []
for node in g:
    if node < 0.1:
        coloring.append('green')
    elif node > 0.9:
        coloring.append('blue')
    else:
        coloring.append('green')

#print(study)

#make labels
labelss = {}
for i, val in enumerate(study):
    labelss[i] = val

#print(labelss)


### create edges for the real graph
edge_list = []
for i in range(0, matrix.shape[0]):
    for j in range(0, matrix.shape[1]):
        if isinstance(matrix.iloc[i, j], str):
            s = matrix.iloc[i, j]
            if s.replace(" ", "") == "X":
                edge_list.append((i, j))
                #g.add_edge(i, j)

'''                              
nodes = nx.draw_networkx_nodes(
    g,
    pos,
    node_size=node_sizes,
    node_color="red"
)
'''

nx.draw(
    g,
    labels=labelss,
    with_labels=True,
    node_size=150, #node_sizes,
    width=0.5,
    font_size=9,
    alpha=0.5,
    node_color=coloring
)

'''./;
edges = nx.draw_networkx_edges(
    g,
    pos,
    width=0.3,
    edgelist=edge_list,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=10,
    edge_cmap=plt.cm.Blues,
)
'''

ax = plt.gca()
ax.set_axis_off()
plt.show()

g_sim = nx.generators.directed.random_k_out_graph(75, 2, 0.5)
sim_edge_list = []
year_gap = 2

'''
g_sim_edge = nx.draw_networkx_edges(
    g_sim,
    pos,
    label=study
)
'''

### construct graph
for i in year:
    #print("i = ", i)
    for j in range(len(year)):
        if year[j] - i >= year_gap:
            sim_edge_list.append((year[j], i))

print(len(sim_edge_list))

nx.draw(
    g_sim,
    labels=labelss,
    with_labels=True,
    node_size=150, #node_sizes,
    width=0.5,
    font_size=9,
    alpha=0.5,
    node_color=coloring
)

plt.show()

g_degree = nx.in_degree_centrality(g)
print(g_degree)
g_sim_degree = nx.in_degree_centrality(g_sim)

degree_ratio = pd.DataFrame(columns=['paperID','degree_ratio'])

last_generation = []
for idx, degree in enumerate(g_degree):
    if g_sim_degree[idx]!=0:
        degree_ratio = degree_ratio.append({'paperID': idx, 'degree_ratio': round(g_degree[idx] / g_sim_degree[idx],3)}, ignore_index=True)
    else:
        last_generation.append(idx)

#print(degree_ratio)

### articles that are insufficiently cited
low_cutoff = 0.1
high_cutoff = 0.9
idx_low = degree_ratio[degree_ratio['degree_ratio'] < low_cutoff].index.tolist()
idx_low = [int(item) for item in idx_low]
#print(idx_low)

idx_high = degree_ratio[degree_ratio['degree_ratio'] > high_cutoff].index.tolist()
idx_high = [int(item) for item in idx_high]

'''
for node in g.vs.indices:
    if node in idx_low:
        g.vs[node]['citation_ratio'] = 'Low'
    elif node in idx_high:
        g.vs[node]['citation_ratio'] = 'High'
    elif node in last_generation:
        g.vs[node]['citation_ratio'] = 'Last Generation'
    else:
        g.vs[node]['citation_ratio'] = 'Average'
'''

plt.bar(degree_ratio['paperID'], degree_ratio['degree_ratio'], width=3, color="b", alpha=0.5)

plt.title("Histogram")
plt.ylabel("Count")
plt.xlabel("Ratio")
plt.show()