import numpy as np
import linecache
import json
import h5py
import argparse
from pathlib import Path
from utils.faiss_utils import train_search


def check(nodes, k, idx, f, ent_list):
    """
    nodes    - 2d array of nodes we want to check
    k        - nearest neighbours
    ind      - index built with faiss
    f        - file containing urls
    ent_list - list of entities id
    read https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    how to print with different colours

    """
    if len(nodes) == 1:
        dist, ind = idx.search(nodes.reshape(1, -1), k)
    else:
        dist, ind = idx.search(nodes, k)
    for row in ind:
        source = int(ent_list[row[0]])
        print('\x1b[0;35;43m' + '{} nearest neighbours of node {}'.format(
            k - 1, source) + '\x1b[0m')
        print('\x1b[0;35;43m' + linecache.getline(f, source + 1) + '\x1b[0m')
        for node in row[1:]:
            neighbor = int(ent_list[node])
            print("  node {}, {}".format(
                node, linecache.getline(f, neighbor + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate embeddings on given graph.')
    parser.add_argument('basename', help='name of the graph to use')
    parser.add_argument('format', help='file format storing original data.')
    args = parser.parse_args()
    basename = args.basename
    f_format = args.format
    assert f_format == 'ids' or f_format == 'urls', "not the right format!"
    model_path = Path("/data/models") / basename
    assert model_path.is_dir(), "model directory not found"
    with (model_path / "entity_names_link_0.json").open() as tf:
        entities_list = json.load(tf)
    hf_path = list(model_path.glob("embeddings_link_0*.h5"))[0]
    hf = h5py.File(hf_path)
    x = hf["embeddings"][:]
    idx = train_search(x)
    nodes_id = np.random.randint(len(x), size=5)
    nodes = x[nodes_id, :]
    k = 6
    ids_file = Path("/data/graphs") / basename / (basename + '.' + f_format)
    try:
        check(nodes, k, idx, str(ids_file), entities_list)
    except Exception as e:
        print("e")
        print("urls file not found!")
