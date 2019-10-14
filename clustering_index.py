import h5py
import json
import numpy as np
from collections import Counter
from utils.helper import iter_partitions
from pathlib import Path


def define_clusters(basename, dim=64):
    """
    Return clusters in a dictionary,
    where the keys are the labels and
    the values is a 2d numpy array containing
    the vectors in the cluster.
    Input:
        model_path - Path
        dim        - dimension of the embeddings
    """
    model_path = Path("/data/models") / basename
    labels_info = model_path / "labels.json"
    with labels_info.open() as tf:
        labels_dict = json.load(tf)

    clusters = dict()
    for domain in labels_dict["count"]:
        rows = labels_dict["count"][domain]
        label = labels_dict["labels"][domain]
        # TODO: dim can be inferred
        clusters[label] = np.zeros((rows, dim))

    for partition, _ in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        x = h5f['embeddings']
        y = h5f['labels']
        assert len(x) == len(y), "Aborting, x and y have different size."
        pos = Counter()
        for i, j in zip(x, y):
            index = pos[j]
            clusters[labels_dict["num_labels"][j]][index] = x
            pos[j] += 1

    for c in clusters:
        # TODO: add overwrite
        h5f = h5py.File(model_path / "cluster{}".format(c), 'w')
        h5f.create_dataset('nodes', data=clusters[c])
        h5f.close()

    return clusters


def dunn_index(data):
    return 0


if __name__ == "__main__":
    basename = 'indochina-2004'
    _ = define_clusters(basename)
