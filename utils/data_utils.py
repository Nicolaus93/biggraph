import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import trange


def iter_partitions(model_path, names=False):
    """
    Returns the path of partitions and number of nodes in them,
    if names is set to false. Otherwise returns nodes "names".
    model_path - Path
    names      - bool
    """
    config_dict = model_path / "config.json"
    with config_dict.open() as tf:
        config = json.load(tf)

    # determine the number of partitions
    num_partitions = config["entities"]["link"]["num_partitions"]
    for i in range(num_partitions):
        # count the nodes in each partition
        count_file = model_path / "entity_count_link_{}.txt".format(i)
        with open(count_file, "rt") as f:
            count = int(f.readline())
        if not names:
            yield model_path / "embeddings_link_{}.v50.h5".format(i), count
        else:
            yield model_path / "entity_names_link_{}.json".format(i), count


def load_data(model_path):
    """
    Load data saved in model_path.
    Input:
        - model_path (Path)
    Output:
        - X (np.array), embeddings
        - Y (np.array), labels (if possible)
                        otherwise array of zeros.
    """
    x_arrays = []
    y_arrays = []
    for partition, count in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f["embeddings"][:]
        x_arrays.append(X)
        try:
            Y = h5f["labels"][:]
            y_arrays.append(Y)
        except KeyError:
            print("Labels not defined")
    if len(y_arrays) > 0:
        X = np.vstack(x_arrays)
        Y = np.hstack(y_arrays)
        return X, Y
    else:
        X = np.vstack(x_arrays)
        Y = np.zeros(len(X))
        return X, Y


def get_entities_list(basename):
    entities_list = []
    model_path = Path("/data/models") / basename
    for json_f, _ in iter_partitions(model_path, names=True):
        with open(json_f, "rt") as f:
            entities = json.load(f)
        entities_list += [int(i) for i in entities]
    return entities_list


def create(n, constructor=list):
    for _ in range(n):
        yield constructor()


def nodes_from_ascii(basename, in_nodes=False):
    """
    Read nodes from ascii file.
    Input:
        - basename (str), name of the graph
        - in_nodes (bool), if True return in_nodes
    Output:
        nodes (list), list of out_nodes
                    (in_nodes) if in_nodes=True
    """
    ascii_path = Path("/data/graphs") / basename / ("ascii.graph-txt")
    assert ascii_path.exists(), "Graph not found!"
    with ascii_path.open() as f:
        line = f.readline()
        V = int(line.split()[0])
        print("{} vertices".format(V))
        print("reading..")
        nodes = list(create(V))
        singleton = 0
        for i in trange(V):
            line = f.readline()
            if line[0] == "\n":
                singleton += 1
            else:
                if in_nodes:
                    for node in line.split():
                        nodes[int(node)].append(i)
                else:
                    nodes[i] = [int(j) for j in line.split()]
        print("Found {} singleton nodes".format(singleton))
    return nodes


def edges_from_ascii(basename, rm_singleton=False):
    """
    Input:
        basename (str)      - name of the graph
        rm_singleton (bool) - whether to remove singleton nodes
    Output:
        out_nodes (list) - list of numpy arrays containing
            the out nodes of every node in the graph
        in_nodes (list)  - list of numpy arrays containing
            the in nodes of every node in the graph
        out_degree       - list of out degrees of nodes
        in_degree        - list of in degrees of nodes
    TODO: remove nodes with no links?
    """
    ascii_path = Path("/data/graphs") / basename / ("ascii.graph-txt")
    assert ascii_path.exists(), "Graph not found!"
    with ascii_path.open() as f:
        line = f.readline()
        line_tot = int(line.split()[0])
        print("{} lines".format(line_tot))
        print("reading..")
        out_nodes = [0] * line_tot
        in_nodes = [0] * line_tot
        out_degree = [0] * line_tot
        for i in trange(line_tot):
            line = f.readline()
            if line[0] == '\n' and not rm_singleton:
                # don't remove singleton
                # assume node is linked to itself
                in_ = np.array([i])
                out_ = np.array([i])
            else:
                in_ = np.fromstring(line, dtype=int, sep=' ')
                out_ = np.ones(len(in_), dtype=int) * i
            out_nodes[i] = out_
            in_nodes[i] = in_
            out_degree[i] = len(out_)
    print("finished reading, now preprocessing..")
    out_nodes = np.hstack(out_nodes)
    in_nodes = np.hstack(in_nodes)
    unique_elements, in_degree_temp = np.unique(in_nodes, return_counts=True)
    # some nodes might have in_degree = 0
    in_degree = np.zeros(len(out_degree))
    in_degree[unique_elements] = in_degree_temp
    return out_nodes, in_nodes, out_degree, in_degree
