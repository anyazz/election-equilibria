from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from networkx.drawing.nx_agraph import graphviz_layout

def draw_networks(fig, axes, e, thetas, label_X):
    for ax, theta, label in zip(axes.flatten(), thetas, label_X):
        maps = draw_network(ax, e, theta, label)
    plt.ion()
    titles = [r'Node opinion ($\theta$)', "Edge weight"]
    # for m, title in zip(maps, titles):
    #     cbar = fig.colorbar(mappable=m, ax=axes.ravel().tolist())
    #     cbar.ax.set_ylabel(title, rotation=270, labelpad=10)
    plt.show()
    plt.pause(0.001)
    fig.set_size_inches(25, 10.5)
    fig.savefig('test.png', dpi=300)

    input("Press [enter] to continue.")
    # plt.close()


def draw_network(ax, e, theta, label):
    n = len(theta)
    G = nx.DiGraph()
    for i in range(n):
        if e.A.X[i] > 0 or e.B.X[i] > 0:
            G.add_node(i, theta=theta[i], infl=(round(e.A.X[i], 2), round(e.B.X[i], 2)))
        else:
            G.add_node(i, theta=theta[i])
    for i in range(n):
        for j in range(n):
            if e.P[i][j] > 0:
                G.add_edge(i, j, weight=e.P[i][j]) # weight??
    
    # thetalst = nx.get_node_attributes(G,'theta')
    weight = nx.get_edge_attributes(G,'weight')
    weightlst = weight.values()
    mean_weight = np.mean(list(weightlst))

    infl = nx.get_node_attributes(G, 'infl')

    pos = nx.nx_pydot.graphviz_layout(G)
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 10)


    node_norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    node_map = mpl.cm.ScalarMappable(norm=node_norm, cmap=mpl.cm.seismic.reversed())
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=[node_map.to_rgba(i) for i in theta])
    if label:
        nx.draw_networkx_labels(G, pos_attrs, labels=infl, font_size=6, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    edge_norm = mpl.colors.DivergingNorm(vmin=0, vcenter=mean_weight, vmax=1)
    edge_map = mpl.cm.ScalarMappable(norm=edge_norm, cmap=mpl.cm.binary)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax, 
        node_size=35,
        arrowstyle="->",
        arrowsize=10,
        edge_color=[edge_map.to_rgba(w) for w in weightlst],
        width=1,
        connectionstyle='arc3, rad = 0.1'
    )

    # nx.draw_networkx_edge_labels(
    #     G, pos, 
    #     edge_labels=weight, 
    #     label_pos=0.5, 
    #     font_size=10, 
    #     font_color='k', 
    #     font_family='sans-serif', 
    #     font_weight='normal', 
    #     bbox=dict(alpha=0),
    # )

    return node_map, edge_map

    


