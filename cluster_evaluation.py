import h5py
import numpy as np
from pathlib import Path
from utils.helper import iter_partitions
from sklearn import metrics
from sklearn.cluster import KMeans


def load_data(model_path):

    x_arrays = []
    y_arrays = []
    for partition, count in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f["embeddings"].value
        Y = h5f["labels"].values
        x_arrays.append(X)
        y_arrays.append(Y)
    return np.vstack(x_arrays), np.hstack(y_arrays)


def bench_k_means(estimator, name, data, labels):
    estimator.fit(data)
    print(name)
    print("homogeneity score: {}".format(
        metrics.homogeneity_score(labels, estimator.labels_)))
    print("completeness score: {}".format(
        metrics.completeness_score(labels, estimator.labels_)))
    print("v measure score: {}".format(
        metrics.v_measure_score(labels, estimator.labels_)))
    print("adjusted rand score: {}".format(
        metrics.adjusted_rand_score(labels, estimator.labels_)))
    print("adjusted mutual info score: {}".format(
        metrics.adjusted_mutual_info_score(labels, estimator.labels_),
        average_method='arithmetic'))
    print("silhouette score: {}".format(
        metrics.silhouette_score(data, estimator.labels_,
                                 metric='euclidean',
                                 sample_size=300)))


if __name__ == "__main__":
    # todo: add argparse
    model_path = Path("/data/models/indochina-2004")
    print("Loading data..")
    X, Y = load_data(model_path)
    classes = len(np.unique(Y))
    print("X shape: {}".format(X.shape))
    print("{} classes.".format(classes))
    score = metrics.silhouette_score(X, Y, metric='euclidean',
                                     sample_size=1000)
    print("silhouette_score: {}".format(score))
    print("")
    algo = KMeans(init='k-means++', n_clusters=classes, n_init=10,
                  max_iter=1000)
    bench_k_means(algo, "k-means++", X, Y)
    print("")
    algo = KMeans(init='random', n_clusters=classes, n_init=10,
                  max_iter=1000)
    bench_k_means(algo, "random", X, Y)
