import h5py
import json
import numpy as np
from utils.helper import iter_partitions
from pathlib import Path


def define_clusters(basename, override=False, dim=64):
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
    clusters_path = model_path / basename / "clusters"
    if clusters_path.is_dir():
        print("Clusters already existing.")
        if not override:
            return clusters_path
        else:
            print("Overwriting new clusters..")
    else:
        try:
            clusters_path.mkdir()
        except OSError:
            print("Fix your paths!!")
            return clusters_path

    labels_info = model_path / "labels.json"
    with labels_info.open() as tf:
        labels_dict = json.load(tf)

    clusters = dict()
    for partition, _ in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f["embeddings"][:]
        Y = h5f["labels"][:]
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
        h5f = h5py.File(clusters_path / "cluster{}.h5".format(c), 'w')
        h5f.create_dataset("nodes", data=clusters[c])
        h5f.close()

    return clusters_path


def davies_bouldin_index(clusters):

    centroid = [0] * len(clusters)
    avg_distance = [0] * len(clusters)
    for pos, array in enumerate(clusters):
        centroid[pos] = np.mean(array, axis=0)
        avg_distance[pos] = np.mean(array - centroid[pos], axis=0)

    centroid = np.array(centroid)
    avg_distance = np.array(avg_distance)
    best = np.zeros(len(clusters))
    for i, c1 in enumerate(centroid):
        best[i] = 0
        for j, c2 in enumerate(centroid):
            if i == j:
                continue
            index = (avg_distance[i] + avg_distance[j]) \
                / np.linalg.norm(c1, c2)
            if index > best[i]:
                best[i] = index
    return best


def dunn_index(clusters):
    return 0


if __name__ == "__main__":
    basename = 'indochina-2004'
    clusters_path = define_clusters(basename)
    clusters = dict()
    for pos, f in enumerate(clusters_path.iterdir()):
        h5f = h5py.File(f, 'r')
        array = h5f["nodes"][:]
        clusters[pos] = array
    indices = davies_bouldin_index(clusters)
