import json
import h5py
import numpy as np
import faiss


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
        Y = h5f["labels"][:]
        x_arrays.append(X)
        y_arrays.append(Y)
    return np.vstack(x_arrays), np.hstack(y_arrays)


def train_search(data):
    """train similarity search model
        as explained in faiss tutorial.
    """
    nb, d = data.shape
    index = faiss.IndexFlatL2(d)   # build the index
    print("Index trained: {}".format(index.is_trained))
    index.add(data)                  # add vectors to the index
    print("Index total: {}".format(index.ntotal))
    return index
