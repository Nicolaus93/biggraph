import h5py
import json
import numpy as np
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
    for partition, _ in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f['embeddings'][:]
        Y = h5f['labels'][:]
        assert len(X) == len(Y), "Aborting, x and y have different size."
        # define clusters
        for class_label in labels_dict["num_labels"]:
            cl = int(class_label)
            try:
                clusters[cl] = np.vstack((clusters[cl], X[Y == cl]))
            except KeyError:
                clusters[cl] = X[Y == cl, :]

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
