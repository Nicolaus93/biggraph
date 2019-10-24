import numpy as np
from utils.data_utils import read_ascii_graph
from pathlib import Path


def precision(Na, Ra_bi, bi):
    """
    Assuming Na and Ra_bi are 2
    arrays of indices.
    """
    set1 = set(Na)
    index = np.where(Ra_bi == bi)[0][0]
    set2 = set(Ra_bi[:index + 1])
    return len(set1.intersection(set2) / len(set2))


def map(graph):

    out_nodes, in_nodes, out_degree, in_degree = read_ascii_graph(graph)
    print("Computing map score")
    V = len(out_nodes)
    score = 0
    for node in graph:
        node_score = 0
        Na = len(out_nodes)
        for neighbor in node.neighbours:
            node_score += precision(node, neighbor)
        score += node_score / Na
    return score / V


if __name__ == "__main__":

    basename = "cnr-2000"
    graph = Path("/data/graphs") / basename / ("ascii.graph-txt")
    map_score = map(basename)
    print(map_score)
