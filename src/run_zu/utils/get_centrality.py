import networkx as nx
import numpy as np

def centralissimo(G):
    centralities = []
    # print(G.nodes())
    centralities.append(nx.pagerank(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc, L))
    for i in range(Nc):
        keys = np.array(list(centralities[i].keys())) - 1  # 修复关键
        cenarray[i][keys] = list(centralities[i].values())
    normcen = (cenarray.astype(float) - np.min(cenarray, axis=1)[:, None]) / \
              (np.max(cenarray, axis=1) - np.min(cenarray, axis=1))[:, None]
    return normcen[0]

# G = nx.Graph()
# G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])
# normcen = centralissimo(G)
# for i in normcen:
#     print(i)
# print(len(normcen))
