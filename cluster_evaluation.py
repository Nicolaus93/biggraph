import h5py
import numpy as np
from pathlib import Path
from utils.helper import iter_partitions
from sklearn import metrics
from sklearn.cluster import KMeans


def load_data(model_path):

    for partition, count in iter_partitions(model_path):
        h5f = h5py.File(partition, 'r')
        X = h5f["embeddings"][:]
        Y = h5f["labels"][:]
        try:
            X = np.vstack((X, h5f["embeddings"][:]))
            Y = np.hstack((Y, h5f["labels"][:]))
        except NameError:
            X = h5f["embeddings"][:]
            Y = h5f["labels"][:]
    return X, Y


def bench_k_means(estimator, labels, data):
    estimator.fit(data)
    print('%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=300)))


if __name__ == "__main__":

    model_path = Path("/data/models/indochina-2004")
    X, Y = load_data(model_path)
    classes = len(np.unique(Y))
    bench_k_means(KMeans(init='k-means++', n_clusters=classes, n_init=10),
                  name="k-means++", data=X)

    bench_k_means(KMeans(init='random', n_clusters=classes, n_init=10),
                  name="random", data=X)
