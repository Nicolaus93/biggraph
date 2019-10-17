import json
import h5py
import numpy as np
from pathlib import Path


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
