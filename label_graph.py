# import re
import json
from pathlib import Path
import tldextract
from tqdm import tqdm
import numpy as np
import h5py


def store_labels(model_path, labels_dict):
    """
    model_path (Path)  - model folder path
    labels_dict (dict) - labels stored in dict
    TODO: infer the CORRECT name of embeddings
    """
    config_dict = model_path / "config.json"
    with config_dict.open() as tf:
        config = json.load(tf)

    # determine the number of partitions
    num_partitions = config['entities']['link']['num_partitions']
    for i in num_partitions:
        # count the nodes in each partition
        count_file = "entity_count_link_{}.txt".format(i)
        with open(count_file, "rt") as f:
            count = int(f.readline())
        # for each partition build the labels vector associated
        labels = np.zeros(count)
        links = "entity_names_link_{}.json".format(i)
        with open(links, "rt") as tf:
            entities_list = json.load(tf)
        for pos, value in enumerate(tqdm(entities_list)):
            labels[pos] = labels_dict[int(value)]
        # save labels vector
        h5f = h5py.File('embeddings_link_{}.v50.h5'.format(i), 'a')
        h5f.create_dataset('labels', data=labels)
        h5f.close()


def assign_labels(urls, labels):

    num_lines = sum(1 for line in urls.open())
    lab = []
    with urls.open() as f:
        n = 0
        for line in tqdm(f, total=num_lines):
            # In raw string literals,
            # backslashes have no special meaning as an escape character.
            # https://knowledge.kitchen/Raw_strings_in_Python
            # domain = re.match(r'^(http[s]?://).+\.(\w+)/', line).group(2)
            # try:
            #     lab.append(labels[domain])
            # except KeyError:
            #     n += 1

            domain = tldextract.extract(line).suffix
            temp = domain.split(".")[-1]
            try:
                lab.append(labels[temp])
            except KeyError:
                print(line)
                return
    return lab, n


def main(labels, basename):
    """
    TODO: add argparse
    """
    urls = Path("/data/graphs") / basename / (basename + ".urls")
    print("Assigning labels..")
    lab, n = assign_labels(urls, labels)
    print(len(lab), n)
    print("Associating labels to embeddings..")
    model = Path("/data/models") / basename
    model = Path("/data/models/indochina-2004")
    store_labels(model, lab)


if __name__ == "__main__":
    labels = {'kh': 1, 'la': 2, 'mm': 3, 'th': 4, 'vn': 5}
    basename = "indochina-2004"
    main(labels, basename)
