import numpy as np
import argparse
from utils.data_utils import *
from utils.faiss_utils import train_search
from tqdm import tqdm


def load_XY(basename):
    """
    Load embeddings (X) and possibly the
    labels (Y) of the graph {basename}.
    """
    model_path = Path("/data/models") / basename
    print("Loading data..")
    X, Y = load_data(model_path)
    classes = len(np.unique(Y))
    print("X shape: {}".format(X.shape))
    print("{} classes".format(classes))
    return X, Y


def precision_score(node_ranking, neighs):
    """
    Compute the precision score as explained in
    https://dawn.cs.stanford.edu/2018/03/19/hyperbolics/
    Input:
        - node_ranking (np.array)
        - neighs (list)
    Output:
        - precision score (float)
    """
    # note: positions starting from 1 --> add 1
    neighs_ranks = np.in1d(node_ranking, neighs).nonzero()[0] + 1
    neighs_card = np.arange(len(neighs_ranks)) + 1
    node_score = neighs_card / neighs_ranks
    return node_score.sum() / len(neighs)


def map_score(X, nodes, ind, neigh_num=50):
    """
    Compute the map score of the given embedding.
    If the number of neighbours of the current node
    is bigger than the one given as input, returns
    the current node as an outlier.
    Input:
        - X (np.array), embeddings
        - nodes (list[list]), neighbours of each node
        - ind (faiss index), index used to compute L2
                            distances for the embeddings
        - neigh_num (int), number of neighbours considered
    Output:
        - score (float), map score
        - outliers (list)
        - singleton, number of singleton nodes
    """
    outliers = []
    score = 0
    singleton = 0
    _, ranking = ind.search(X, neigh_num)
    for node, neighs in enumerate(nodes):
        Na = len(neighs)
        if Na == 0:
            singleton += 1
        elif Na > neigh_num // 2:
            outliers.append(node)
        else:
            # start from index=1 to not consider the node itself
            Ra = ranking[node, 1:]
            score += precision_score(Ra, neighs)
    return score, outliers, singleton


def dataset_map(X, out_nodes, neighs=50):
    """
    Compute the MAP score on all the embeddings
    given as input.
    """
    ind = train_search(X)
    if len(X) > 10000:
        n = 10000
        iters = len(X) // n
        splits = np.array_split(X, iters)
        out_node_split = np.array_split(out_nodes, iters)
    else:
        iters = 1
        splits = X
        out_node_split = out_nodes
    score = 0
    singleton = 0
    for data, nodes in tqdm(zip(splits, out_node_split), total=iters):
        split_score, outliers, sing = map_score(
            data, nodes, ind, neigh_num=neighs)
        singleton += sing
        score += split_score
        while len(outliers) > 0:
            neighs *= 2
            split_score, outliers, _ = map_score(
                data[outliers], nodes[outliers], ind, neigh_num=neighs)
            score += split_score
    return score / (len(X) - singleton)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Measure mean average precision of produced embeddings.')
    parser.add_argument('-b', '--basename', type=str, default='indochina-2004',
                        help='name of the graph to use')
    args = parser.parse_args()
    basename = args.basename
    embeddings = load_XY(basename)
    X = embeddings[0]
    ent_list = get_entities_list(basename)
    ent_list = np.array(ent_list)
    perm = np.argsort(ent_list)
    X = X[perm]
    out_nodes = nodes_from_ascii(basename)
    out_nodes = [out_nodes[i] for i in ent_list[perm]]
    assert len(X) == len(out_nodes)
    ind = train_search(X)
    score = dataset_map(X, out_nodes)
    print("MAP score on {}: {}".format(basename, score))
