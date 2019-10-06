import faiss
import numpy as np
import linecache
import json


def train_search(data):
    # train similarity search model
    nb, d = data.shape
    index = faiss.IndexFlatL2(d)   # build the index
    print("Index trained: {}".format(index.is_trained))
    index.add(data)                  # add vectors to the index
    print("Index total: {}".format(index.ntotal))
    return index


def check(nodes, k, emb, ind, f):
    """
    nodes - ids of the nodes we want to check
    k     - nearest neighbours
    emb   - a 2-d numpy array of embeddings
    ind   - index built with faiss
    f     - file containing urls
    read https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    how to print with different colours

    """
    if len(nodes) == 1:
        dist, ind = ind.search(emb[nodes].reshape(1, -1), k)
    else:
        dist, ind = ind.search(emb[nodes], k)
    for row in ind:
        print('\x1b[0;35;43m' + '{} nearest neighbours of node {}'.format(
            k - 1, row[0]) + '\x1b[0m')
        print('\x1b[0;35;43m' + linecache.getline(file, row[0] + 1) + '\x1b[0m')
        for node in row[1:]:
            print("  node {}, {}".format(
                node, linecache.getline(file, node + 1)))


if __name__ == '__main__':
    # x = np.load('node_embeddings/cnr-2000_d64wL30cS20wPN20p2q1numS40e100.npy')
    with open("/data/models/cnr-2000/entity_names_link_0.json", "rt") as tf:
        entities_list = json.load(tf)

    x = np.load("/data/models/cnr-2000/embeddings_link_0.v50.h5")
    idx = train_search(x)
    # nodes = np.random.randint(0, len(x), size=5)
    nodes = [int(entities_list[i]) for i in range(5)]
    k = 6
    check(nodes, k, x, idx, '/data/graphs/cnr-2000/cnr-2000.urls')
