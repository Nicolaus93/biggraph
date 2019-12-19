import faiss
import linecache
from pathlib import Path
from utils.data_utils import get_entities_list


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


def kmeans(X, n_cent=5, niter=50, verbose=True):
    """
    Use k-means algorithm from faiss.
    """
    d = X.shape[1]
    ncentroids = n_cent
    algo = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    algo.train(X)
    return algo


def PCA(X, out=2):
    """
    Use PCA algorithm from faiss.
    """
    d = X.shape[1]
    mat = faiss.PCAMatrix(d, out)
    mat.train(X)
    assert mat.is_trained
    tr = mat.apply_py(X)
    return tr


def centroid_neigh(basename, k_means, X, entities, n=15):
    """
    Find the n-nearest neighbours to k-means
    cluster centroids.
    """
    d = X.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X)
    D, Ind = index.search(k_means.centroids, n)
    find_neighbours(basename, Ind, entities)


def find_neighbours(basename, idx, ent_list):
    """
    Helper function for centroid_neigh.
    """
    ids_file = Path('/data/graphs/') / basename / (basename + '.urls')
    if not ids_file.exists():
        ids_file = Path('/data/graphs/') / basename / (basename + '.ids')
    assert ids_file.exists(), "File not found!"
    f = ids_file.as_posix()
    for pos, cluster in enumerate(idx):
        print("\x1b[0;35;43m Cluster {} \x1b[0m".format(pos))
        for node in cluster:
            line = ent_list[node]
            print(linecache.getline(f, line + 1))
