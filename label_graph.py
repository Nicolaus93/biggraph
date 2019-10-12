# import re
import json
from pathlib import Path
import tldextract
from tqdm import tqdm


def build_dict(model_path):

    if (model_path / "entities_dict").exists():
        print("entities dict already existing")
        return

    with open("config.json", "rt") as tf:
        config = json.load(tf)

    num_partitions = config['entities']['link']['num_partitions']
    entities_dict = dict()

    for i in num_partitions:

        links = "entity_names_link_{}.json".format(i)
        with open(links, "rt") as tf:
            entities_list = json.load(tf)

        for pos, value in enumerate(entities_list):
            # store node address in dict: file_num and pos
            entities_dict[int(value)] = (i, pos)

    with open('entities_dict.json', 'w') as fp:
        json.dump(entities_dict, fp)


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


if __name__ == "__main__":
    labels = {'kh': 1, 'la': 2, 'mm': 3, 'th': 4, 'vn': 5}
    # basename = Path("some_graph")
    # model = Path("some_model")
    # urls = Path("graph_urls")
    # build_dict(model)
    urls = Path("data/indochina-2004.urls")
    lab, n = assign_labels(urls, labels)
    print(len(lab), n)
