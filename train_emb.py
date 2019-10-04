import os
import random
import json
import h5py
import attr

import torchbiggraph as tbg
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir

def random_split_file(fpath, train_frac=0.9, shuffle=False):
    root = os.path.dirname(fpath)

    output_paths = [
        os.path.join(root, FILENAMES['train']),
        os.path.join(root, FILENAMES['test']),
    ]
    if all(os.path.exists(path) for path in output_paths):
        print("Found some files that indicate that the input data "
              "has already been shuffled and split, not doing it again.")
        print("These files are: %s" % ", ".join(output_paths))
        return

    print('Shuffling and splitting train/test file. This may take a while.')
    train_file = os.path.join(root, FILENAMES['train'])
    test_file = os.path.join(root, FILENAMES['test'])

    print('Reading data from file: ', fpath)
    with open(fpath, "rt") as in_tf:
        lines = in_tf.readlines()
    
    if shuffle:
        print('Shuffling data')
        random.shuffle(lines)

    split_len = int(len(lines) * train_frac)

    print('Splitting to train and test files')
    with open(train_file, "wt") as out_tf_train:
        for line in lines[:split_len]:
            out_tf_train.write(line)

    with open(test_file, "wt") as out_tf_test:
        for line in lines[split_len:]:
            out_tf_test.write(line)

def run_train_eval(data_path, config_path, train_path, split=False, eval_=False):
    if split:
        random_split_file(data_path)

    convert_input_data(
        config_path,
        edge_paths,
        lhs_col=0,
        rhs_col=1,
        rel_col=None,
    )

    train_config = parse_config(config_path)
    train_config = attr.evolve(train_config, edge_paths=train_path)
    train(train_config)
    
    print("Trained!")
    if eval_:
        eval_config = attr.evolve(train_config, edge_paths=eval_path)
        do_eval(eval_config)

def output_embedding():
    with open(os.path.join(DATA_DIR, "dictionary.json"), "rt") as tf:
        dictionary = json.load(tf)

    link = "0"
    offset = dictionary["entities"]["link"].index(link)
    print("our offset for link ", link, " is: ", offset)

    with h5py.File("model/test_1/embeddings_link_4.v10.h5", "r") as hf:
        embedding = hf["embeddings"][offset, :]

    print(f" our embedding looks like this: {embedding}")
    print(f"and has a size of: {embedding.shape}")


# Main method

DATA_PATH = "../tab_graphs/cnr2000/cnr2000.txt"
DATA_DIR = "../tab_graphs/cnr2000"
CONFIG_PATH = "config.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}

edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]
train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]



def main():
    print(train_path)
    run_train_eval(DATA_PATH, CONFIG_PATH, train_path, split=True, eval_=True)
    # output_embedding()

if __name__ == "__main__":
    main()
