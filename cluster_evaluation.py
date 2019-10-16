import numpy as np
import faiss
import json
import argparse
from pathlib import Path
from utils.helper import load_data, iter_partitions, train_search
from sklearn import metrics
from sklearn.cluster import KMeans
from check_result import check


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
    parser = argparse.ArgumentParser(
        description='Use Kmeans on given graph embeddings.')
    parser.add_argument('-b', '--basename', type=str, default='indochina-2004',
                        help='name of the graph to use')
    parser.add_argument('-n', type=int, default=5,
                        help='number of centroids')
    parser.add_argument('-it', type=int, default=50,
                        help="number of iterations")
    parser.add_argument('-v', "--verbose", default=True,
                        help="verbosity")
    parser.add_argument("-k" "--k_nearest", default=20,
                        help="k centroids nearest neighbours")

    args = parser.parse_args()
    basename = args.basename
    ncentroids = args.n
    niter = args.it
    verbose = args.verbose
    k = args.k

    model_path = Path("/data/models") / basename
    print("Loading data..")
    X, Y = load_data(model_path)
    classes = len(np.unique(Y))
    print("X shape: {}".format(X.shape))
    print("{} classes.".format(classes))
    score = metrics.silhouette_score(X, Y, metric='euclidean',
                                     sample_size=1000)
    print("silhouette_score: {}".format(score))
    print("")

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

    urls_file = Path('/data/graphs/', basename, (basename + '.urls'))
    idx = train_search(X)
    check(kmeans.centroids, k, X, idx, urls_file.as_posix(), entities_list)
