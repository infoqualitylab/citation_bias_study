from igraph import *
from igraph import Graph


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_g():
    g = Graph()
    g.add_vertices(29)
    g.vs["label"] = ["70", "71", "72", "73", "74", "75",
                     "76", "77", "78", "79", "80", "69",
                     "81", "82", "83", "84", "85", "86",
                     "87", "88", "89", "90", "91", "92",
                     "93", "94", "95", "96", "97"]

# starting from 0 so all of the numbers are subtracted by one
    visual_style = {}
    out_name = "graph.png"

    visual_style["bbox"] = (800, 800)
    visual_style["margin"] = 20  # how wide, 27 pixels

    g.vs["color"] = ["white", "white", "white", "white", "white",
                     "white", "white", "white", "white", "white",
                     "white", "white", "blue", "blue", "blue",
                     "blue", "blue", "blue", "blue", "blue",
                     "blue", "blue", "blue", "blue", "blue",
                     "blue", "blue", "blue", "blue"]
    visual_style["width"] = 50

    visual_style["vertex_size"] = 40

    visual_style["edge_curved"] = False  # making the line curved or straight

    my_layout = g.layout_lgl()
    visual_style["layout"] = my_layout

    plot(g, out_name, **visual_style)

    weights = [8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9,
               8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9, 8, 6, 3, 5, 6, 4, 9]
    g.es['weight'] = weights
    g.es['label'] = weights

    g.add_edges([(12, 4), (12, 5), (12, 19), (12, 17), (12, 20),
                 (12, 24), (12, 25), (12, 28), (13, 12), (13, 19),
                 (13, 20), (13, 24), (13, 26), (13, 27), (13, 28),
                 (14, 4), (14, 5), (14, 16), (14, 17), (14, 19),
                 (14, 5), (15, 10), (15, 24), (15, 25), (15, 26),
                 (15, 27), (15, 28), (16, 4), (16, 27), (16, 28),
                 (19, 4), (19, 5), (19, 10), (19, 24), (19, 25),
                 (19, 27), (19, 28), (20, 4), (20, 5), (20, 10),
                 (20, 24), (20, 25), (21, 4), (21, 9), (21, 10),
                 (21, 28), (21, 0), (22, 9), (22, 10), (22, 27),
                 (22, 28), (23, 27), (23, 28), (23, 4), (23, 10),
                 (23, 27), (24, 28), (25, 4), (25, 6), (25, 10),
                 (25, 4), (25, 5), (26, 10), (26, 27), (26, 28),
                 (27, 4), (27, 5), (28, 4), (28, 5), (28, 10)])


# wtf am i doing wrong g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])


if __name__ == '__main__':
    print_hi("PyCharm Jasmine")
    print_g()
