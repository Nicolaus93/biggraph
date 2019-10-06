import os
import random
import json
import h5py
import attr
from pathlib import Path
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.config import add_to_sys_path, ConfigFileLoader
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train
from torchbiggraph.util import (
    set_logging_verbosity,
    setup_logging,
    SubprocessInitializer,
)


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


def run_train_eval(data_dir, config_path, basename, split=False,
                   eval_=False):
    data_path = data_dir / (basename + '.tab')
    if split:
        random_split_file(data_path)

    loader = ConfigFileLoader()
    config = loader.load_config(config_path, None)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    input_edge_paths = [data_dir / name for name in FILENAMES.values()]
    output_train_path, output_test_path = config.edge_paths

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        lhs_col=0,
        rhs_col=1,
        rel_col=None,
        dynamic_relations=False,
    )

    # update the config edge_paths and train
    train_config = attr.evolve(config, edge_paths=[output_train_path])
    train(train_config, subprocess_init=subprocess_init)
    print("Trained!")

    if eval_:
        relations = [attr.evolve(r, all_negs=True) for r in config.relations]
        eval_config = attr.evolve(
            config, edge_paths=[output_test_path],
            relations=relations,
            num_uniform_negs=0)
        do_eval(eval_config, subprocess_init=subprocess_init)


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


DATA_DIR = Path("/data/graphs/cnr-2000")
CONFIG_PATH = "config.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt'
}


def main():
    basename = 'cnr-2000'
    run_train_eval(DATA_DIR, CONFIG_PATH, basename, split=True, eval_=True)


if __name__ == "__main__":
    main()
