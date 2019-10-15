import json
import h5py
import numpy as np


def iter_partitions(model_path):
    """
    Returns the path of partitions and number of nodes in them.
    model_path - Path
    """
    config_dict = model_path / "config.json"
    with config_dict.open() as tf:
        config = json.load(tf)

    # determine the number of partitions
    num_partitions = config['entities']['link']['num_partitions']
    for i in range(num_partitions):
        # count the nodes in each partition
        count_file = model_path / "entity_count_link_{}.txt".format(i)
        with open(count_file, "rt") as f:
            count = int(f.readline())
        yield model_path / 'embeddings_link_{}.v50.h5'.format(i), count


def load_data(model_path):

    x_arrays = []
    y_arrays = []
    for partition, count in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f["embeddings"][:]
        Y = h5f["labels"][:]
        x_arrays.append(X)
        y_arrays.append(Y)
    return np.vstack(x_arrays), np.hstack(y_arrays)
