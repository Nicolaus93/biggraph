import os
import random
import attr
import argparse
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
    train_file = fpath.parent / 'train.txt'
    test_file = fpath.parent / 'test.txt'

    if train_file.exists() and test_file.exists():
        print("Found some files that indicate that the input data "
              "has already been shuffled and split, not doing it again.")
        print("These files are: {} and {}".format(
            train_file, test_file))

    else:
        print('Shuffling and splitting train/test file. \
            This may take a while.')

        print("Reading data from file: {}".format(fpath))
        with fpath.open("rt") as in_tf:
            lines = in_tf.readlines()

        if shuffle:
            print('Shuffling data')
            random.shuffle(lines)

        split_len = int(len(lines) * train_frac)

        print('Splitting to train and test files')
        with train_file.open("wt") as out_tf_train:
            for line in lines[:split_len]:
                out_tf_train.write(line)

        with test_file.open("wt") as out_tf_test:
            for line in lines[split_len:]:
                out_tf_test.write(line)

    return train_file, test_file


# def random_split_file(fpath, train_frac=0.9, shuffle=False):
#     root = os.path.dirname(fpath)

#     output_paths = [
#         os.path.join(root, FILENAMES['train']),
#         os.path.join(root, FILENAMES['test']),
#     ]
#     if all(os.path.exists(path) for path in output_paths):
#         print("Found some files that indicate that the input data "
#               "has already been shuffled and split, not doing it again.")
#         print("These files are: %s" % ", ".join(output_paths))
#         return

#     print('Shuffling and splitting train/test file. This may take a while.')
#     train_file = os.path.join(root, FILENAMES['train'])
#     test_file = os.path.join(root, FILENAMES['test'])

#     print('Reading data from file: ', fpath)
#     with open(fpath, "rt") as in_tf:
#         lines = in_tf.readlines()

#     if shuffle:
#         print('Shuffling data')
#         random.shuffle(lines)

#     split_len = int(len(lines) * train_frac)

#     print('Splitting to train and test files')
#     with open(train_file, "wt") as out_tf_train:
#         for line in lines[:split_len]:
#             out_tf_train.write(line)

#     with open(test_file, "wt") as out_tf_test:
#         for line in lines[split_len:]:
#             out_tf_test.write(line)


def run_train_eval(data_dir, config_path, basename, split=False,
                   eval_=False):
    tab_graph = data_dir / (basename + '.tab')
    train_f, test_f = random_split_file(tab_graph)

    loader = ConfigFileLoader()
    config = loader.load_config(config_path, None)

    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    # input_edge_paths = [data_dir / name for name in FILENAMES.values()]
    input_edge_paths = [train_f, test_f]
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


# def convert_path(fname):
#     basename, _ = os.path.splitext(fname)
#     out_dir = basename + '_partitioned'
#     return out_dir


# DATA_DIR = Path("/data/graphs/cnr-2000")
# CONFIG_PATH = "config.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt'
}


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description='Generate embeddings on given graph.')
    parser.add_argument('--data_dir', type=Path, default='data',
                        help='where to save processed data')
    parser.add_argument('--basename', type=str, default='cnr-2000',
                        help='name of the graph to use')
    parser.add_argument('--config', type=Path, default=Path('config.py'),
                        help='Path to config file')

    args = parser.parse_args()
    assert args.config.exists(), "config file not found"
    assert args.data_dir.is_dir(), "data dir not found"
    basename = args.basename
    config_path = args.config
    data_dir = args.data_dir
    run_train_eval(data_dir, config_path, basename, split=True, eval_=True)


if __name__ == "__main__":
    main()
