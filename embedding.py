from utils.data_utils import *
from pathlib import Path
import linecache


class Embedding(object):

    def __init__(self, path=None):
        try:
            self.model_path = Path(path)
        except TypeError:
            self.model_path = None

    def set_path(path):
        self.model_path = Path(path)
        assert self.model_path.exists(), "Path not valid!"

    def set_node_ids(path):
        self.node_ids = Path(path)
        assert self.node_ids.exists(), "File not found!"

    def load_XY(self):
        """

        """
        assert self.model_path is not None, "Give a path first!"
        print("Loading data..")
        self.X, self.Y = load_data(self.model_path)
        classes = len(np.unique(Y))
        print("X shape: {}".format(X.shape))
        print("{} classes".format(classes))


