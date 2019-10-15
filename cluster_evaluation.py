import numpy as np
import faiss
import json
from pathlib import Path
from utils.helper import load_data, iter_partitions, train_search
from sklearn import metrics
from sklearn.cluster import KMeans
from check_result import check
from os.path import join


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

    ncentroids = 5
    niter = 100
    verbose = True
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(X)

    # algo = KMeans(init='k-means++', n_clusters=classes, n_init=10,
    #               max_iter=1000)
    # bench_k_means(algo, "k-means++", X, Y)
    # print("")
    # algo = KMeans(init='random', n_clusters=classes, n_init=10,
    #               max_iter=1000)
    # bench_k_means(algo, "random", X, Y)

    entities_list = []
    for json_f, _ in iter_partitions(model_path, names=True):
        with open(json_f, "rt") as f:
            entities = json.load(f)
        entities_list += [i for i in entities]

    basename = "indochina-2004"
    # urls_file = Path('/data/graphs/', basename, (basename + '.urls'))
    urls_file = join('/data/graphs/', basename, (basename + '.urls'))
    k = 50
    idx = train_search(X)
    check(kmeans.centroids, k, X, idx, urls_file, entities_list)
