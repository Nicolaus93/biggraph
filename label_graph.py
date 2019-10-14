import json
from pathlib import Path
import tldextract
from tqdm import tqdm
import numpy as np
import h5py
from collections import Counter
from utils.helper import iter_partitions


def store_labels(model_path, labels_dict, override=False):
    """
    model_path (Path)  - model folder path
    labels_dict (dict) - labels stored in dict
    """
    for i, value in enumerate(iter_partitions(model_path)):
        partition, count = value
        print("Partition: {}..".format(i))
        labels = np.zeros(count, dtype=int)
        links = model_path / "entity_names_link_{}.json".format(i)
        with open(links, "rt") as tf:
            entities_list = json.load(tf)
        for pos, value in enumerate(tqdm(entities_list)):
            labels[pos] = labels_dict[int(value)]
        # save labels vector
        # TODO: infer the CORRECT name of embeddings
        h5_path = model_path / 'embeddings_link_{}.v50.h5'.format(i)
        try:
            h5f = h5py.File(h5_path, 'a')
            h5f.create_dataset('labels', data=labels)
        except RuntimeError:
            print("Labels already stored!")
            if override:
                print("Overwriting new labels..")
                h5f = h5py.File(h5_path, 'r+')
                data = h5f['labels']
                data[...] = labels
        h5f.close()


def assign_labels(basename, data_folder=Path("/data"), verbose=False):
    """
    Function to assign labels to nodes in {basename} based on the url domain.
    Input:
        basename (str)
        data_folder (Path)
        verbose (bool)
    """
    urls_path = data_folder / "graphs" / basename / (basename + ".urls")
    assert urls_path.exists(), "Urls file not found!"
    # check if labels dict already existing
    labels_path = data_folder / "models" / basename / ("labels.json")
    if labels_path.exists():
        print("Labels json already existing.")
    else:
        print("Building labels json..")
        # count number of lines in file
        num_lines = sum(1 for line in urls_path.open())
        labels_array = [0] * num_lines
        with urls_path.open() as f:
            clusters_count = Counter()
            labels = dict()
            class_index = 0
            for pos, line in enumerate(tqdm(f, total=num_lines)):
                # extract the TLD
                complete_domain = tldextract.extract(line).suffix
                # we only need the country domain now
                domain = complete_domain.split(".")[-1]
                # if domain unseen add it to class indices
                if domain not in labels:
                    class_index += 1
                    labels[domain] = class_index
                # assign label and add it to array
                y = labels[domain]
                labels_array[pos] = y
                clusters_count[domain] += 1
        labels_data = dict()
        labels_data['labels'] = labels
        labels_data['count'] = clusters_count
        labels_data['array'] = labels_array
        labels_data['num_labels'] = {v: k for k, v in labels.items()}
        if verbose:
            print("Found following labels:")
            print(labels)
        with open(labels_path, 'w', encoding='utf-8') as outfile:
            json.dump(labels_data, outfile, ensure_ascii=False, indent=4)
    return labels_path


def main(basename):
    """
    TODO: add argparse
    """
    model = Path("/data/models") / basename
    # retrieve labels
    print("Assigning labels..")
    labels_path = assign_labels(basename, data_folder=Path("/data"),
                                verbose=True)
    with labels_path.open() as f:
        labels = json.load(f)
    # save labels in h5 embedding files
    print("Associating labels to embeddings..")
    store_labels(model, labels["array"])


if __name__ == "__main__":
    basename = "indochina-2004"
    main(basename)
