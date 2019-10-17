import faiss


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
