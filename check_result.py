import faiss
import numpy as np
import linecache
import json
import h5py


def train_search(data):
    # train similarity search model
    nb, d = data.shape
    index = faiss.IndexFlatL2(d)   # build the index
    print("Index trained: {}".format(index.is_trained))
    index.add(data)                  # add vectors to the index
    print("Index total: {}".format(index.ntotal))
    return index


def check(nodes, k, emb, ind, f, ent_list):
    """
    nodes    - ids of the nodes we want to check
    k        - nearest neighbours
    emb      - a 2-d numpy array of embeddings
    ind      - index built with faiss
    f        - file containing urls
    ent_list - list of entities id
    read https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    how to print with different colours

    """
    if len(nodes) == 1:
        dist, ind = ind.search(emb[nodes].reshape(1, -1), k)
    else:
        dist, ind = ind.search(emb[nodes], k)
    for row in ind:
        source = int(ent_list[row[0]])
        print('\x1b[0;35;43m' + '{} nearest neighbours of node {}'.format(
            k - 1, source) + '\x1b[0m')
        print('\x1b[0;35;43m' + linecache.getline(file, source + 1) + '\x1b[0m')
        for node in row[1:]:
            neighbor = int(ent_list[node])
            print("  node {}, {}".format(
                node, linecache.getline(file, neighbor + 1)))


if __name__ == '__main__':
    # x = np.load('node_embeddings/cnr-2000_d64wL30cS20wPN20p2q1numS40e100.npy')
    with open("/data/models/cnr-2000/entity_names_link_0.json", "rt") as tf:
        entities_list = json.load(tf)

    hf = h5py.File("/data/models/cnr-2000/embeddings_link_0.v50.h5", 'r')
    x = hf.get("embeddings").value
    # embedding = hf["embeddings"][offset, :]
    idx = train_search(x)
    nodes = np.random.randint(0, len(x), size=5)
    k = 6
    check(nodes, k, x, idx, '/data/graphs/cnr-2000/cnr-2000.urls')
