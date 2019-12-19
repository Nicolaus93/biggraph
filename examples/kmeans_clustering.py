from biggraph.embedding import Embedding
from biggraph.utils.faiss_utils import kmeans, closest_to


if __name__ == '__main__':

    mod = "../data/models/indochina-2004"
    emb = Embedding(mod)
    emb.load_XY()
    x = emb.get_X()
    y = emb.get_Y()
    print(x.shape)

    emb.set_entities()
    ent = emb.get_entities()
    print(ent[:10])

    itwiki_kmeans = kmeans(x, 50, niter=100)
    clusters = closest_to(itwiki_kmeans.centroids, x)
    for pos, cluster in enumerate(clusters):
        print("Cluster {}".format(pos + 1))
        for node in emb.get_node_id(cluster):
            print(node)
