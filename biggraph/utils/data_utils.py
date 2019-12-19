import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import trange


def iter_embeddings(model_path, h5=True):
    """
    updated version of iter_partitions
    NOTICE: returns objects NOT ordered
    """
    temp = []
    if h5:
        for h5_file in model_path.glob('embeddings_link*.h5'):
            temp.append(h5_file)
    else:
        for json_file in model_path.glob('entity_names_link_*.json'):
            temp.append(json_file)
    temp = sorted(temp)
    for i in temp:
        yield i


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
    for partition in iter_embeddings(model_path):
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
    for json_f in iter_embeddings(model_path, h5=False):
        with json_f.open() as f:
            entities = json.load(f)
        entities_list += [int(i) for i in entities]
    return entities_list


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
        nodes = [list() for i in range(V)]
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
