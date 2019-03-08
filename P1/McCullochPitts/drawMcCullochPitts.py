# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edge("x1", "z11", weight=2)
G.add_edge("x1", "z22", weight=1)
G.add_edge("x1", "z33", weight=1)
G.add_edge("x2", "z12", weight=2)
G.add_edge("x2", "z23", weight=1)
G.add_edge("x2", "z31", weight=1)
G.add_edge("x3", "z13", weight=2)
G.add_edge("x3", "z21", weight=1)
G.add_edge("x3", "z32", weight=1)
G.add_edge("z11", "z21", weight=1)
G.add_edge("z11", "z31", weight=1)
G.add_edge("z12", "z22", weight=1)
G.add_edge("z12", "z32", weight=1)
G.add_edge("z13", "z23", weight=1)
G.add_edge("z13", "z33", weight=1)
G.add_edge("z21", "y1", weight=2)
G.add_edge("z22", "y1", weight=2)
G.add_edge("z23", "y1", weight=2)
G.add_edge("z31", "y2", weight=2)
G.add_edge("z32", "y2", weight=2)
G.add_edge("z33", "y2", weight=2)

pos = {
    "x1": (0,2), "x2": (0,1), "x3": (0,0),
    "z11": (1,2), "z12": (1,1), "z13": (1,0),
    "z21": (2,2), "z22": (2,1), "z23": (2,0),
    "z31": (2,1.5), "z32": (2, 0.5), "z33": (2, -0.5),
    "y1": (3, 1.5), "y2": (3, 0.5)
}

plt.title(r'$\theta=2$')
nx.draw(G, arrows = True, with_labels = True, pos = pos, node_size= 500, node_color = "w", alpha = 0.8)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
ax= plt.gca()
ax.collections[0].set_edgecolor("#000000")
plt.savefig("RedMcCullochPitts.png")
plt.show()
