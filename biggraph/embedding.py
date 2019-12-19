from biggraph.utils.data_utils import *
from pathlib import Path
import linecache


class Embedding(object):

    def __init__(self, model_path=None, basename=None):
        """
        Input:
            path     - str, bytes or os.PathLike object, not NoneType
            basename - str
        """
        self.model_path = Path(model_path) if model_path else None
        self.basename = basename

    def set_model_path(self, model_path):
        p = Path(model_path)
        assert p.exists(), "Model path not valid"
        self.model_path = p

    def get_model_path(self):
        return self.model_path

    def set_entities(self):
        assert self.model_path is not None, "Model path not valid!"
        self.entities = get_entities_list(self.model_path)

    def get_entities(self):
        return self.entities

    def set_basename(self, basename):
        self.basename = basename

    def get_basename(self):
        return self.basename

    def set_node_ids(self, ids_path):
        p = Path(ids_path)
        assert p.exists(), "Ids not found!"
        self.node_ids = p

    def get_node_id(self, nodes):
        """
        Node ids are not kept in memory,
        but retrieved when necessary.
        """
        f = self.node_ids.as_posix()
        for node in nodes:
            line = self.entities[node]
            yield linecache.getline(f, line + 1).rstrip()

    def load_XY(self, verbose=False):
        """
        Load produced embeddings.
        """
        assert self.model_path is not None, "Give a path first!"
        print("Loading data..")
        self.X, self.Y = load_data(self.model_path)
        classes = len(np.unique(self.Y))
        if verbose:
            print("X shape: {}".format(self.X.shape))
            print("{} classes".format(classes))

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def train(self):
        """
        ToDo
        """
        return


if __name__ == '__main__':

    emb = Embedding()
    model_path = Path('/Users/nico/Desktop/Neural_Networks/biggraph/data/models/indochina-2004')
    emb.set_model_path(model_path)
    print(emb.get_model_path())
    # gen = emb.get_node_id([1, 2])
    # for i in gen:
    #     print(i)
    emb.load_XY()
