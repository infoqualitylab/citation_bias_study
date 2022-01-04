# this script simulate a toy-model unbiased citation network
# each generation of papers will have a defined probability to cite papers from the prior generations
# and the defined probabilities are computed from observed networks
# there are two input .csv files
# file one is an attribute list including the following columns: paper_id, paper_name, publication year
# opinion
# file two is an edge list for the observed network
import pandas as pd
import igraph as ig
import numpy as np
import cairo
from matplotlib import pyplot as plt


def main():
    obs_edge_list = pd.read_csv('data/obs_edge_list.csv')  # two columns: source, target
    attr_list = pd.read_csv('data/attr_list.csv')  # four columns: ID,Label,Year,Outcome
    potential_edge_list = pd.read_csv('data/potential_edge_list.csv')  # two columns: source, target
    df_prob = compute_p(attr_list, obs_edge_list, potential_edge_list)  # two columns: Year, p(y)

    print("Generating 5 networks")
    for i in range(5):
        g_sim = toy_network_gen(df_prob, attr_list)
        network_vis(g_sim, 'simulated_network' + str(i) + '.png')


def toy_network_gen(prob, attr_list):
    attr_list.sort_values(by='Year', inplace=True)  # make sure attr list is correctly sorted

    # initialize graph
    df_init = attr_list.loc[attr_list['Year'] == prob['Year'][0]]
    g_sim = ig.Graph.DataFrame(edges=pd.DataFrame(columns=['source', 'target']), vertices=df_init, directed=True)

    # generate edges and nodes one-by-one
    for idx in range(len(df_init), len(attr_list) - len(df_init) + 1):
        g_sim.add_vertex(name=idx)
        g_sim.vs[idx]['Label'] = attr_list['Label'][idx]
        g_sim.vs[idx]['Outcome'] = attr_list['Outcome'][idx]
        g_sim.vs[idx]['Year'] = attr_list['Year'][idx]

        current_vs = g_sim.vs[idx]

        # construct edges
        for vs in g_sim.vs.select(Year_lt=current_vs['Year']):
            # draw a random number
            rng = np.random.default_rng()
            if rng.random() <= float(prob.loc[prob['Year'] == current_vs['Year'], 'p(y)']):
                g_sim.add_edge(source=current_vs, target=vs)

    return g_sim


def compute_p(attr_list, obs_edge_list, potential_edge_list):
    g_obs = ig.Graph.DataFrame(edges=obs_edge_list, vertices=attr_list, directed=True)
    network_vis(g_obs, 'observed_network.png')
    g_potential = ig.Graph.DataFrame(edges=potential_edge_list, vertices=attr_list, directed=True)

    # compute probabilities
    generations = pd.unique(g_obs.vs['Year'])
    performed_citation = [len(g_obs.es.select(_source_in=g_obs.vs.select(Year=year))) for year in generations]
    potential_citation = [len(g_potential.es.select(_source_in=g_potential.vs.select(Year=year))) for year in
                          generations]
    p_y = [i / j for i, j in
           zip(performed_citation[1:], potential_citation[1:])]  # omit the first year in the generations
    p_y = [None] + p_y  # add the probability of the first year in generations as None

    prob = pd.DataFrame({'Year': generations,
                         'p(y)': p_y})

    prob = prob.merge(attr_list.groupby('Year').count()['ID'].reset_index().rename(columns={'ID': 'Publications'}),
                      on='Year')
    return prob


def network_vis(g, name):
    for vs in g.vs:
        if vs['Outcome'] == 'Positive':
            vs['Color'] = 'dodgerblue'
        elif vs['Outcome'] == 'Negative':
            vs['Color'] = 'orangered'
        else:
            vs['Color'] = 'white'

    vis = ig.plot(g,
                  vertex_size=20,
                  vertex_color=g.vs['Color'],
                  vertex_label=g.vs['Label'],
                  layout='kk',
                  target=name)
    vis.show()


if __name__ == '__main__':
    main()
