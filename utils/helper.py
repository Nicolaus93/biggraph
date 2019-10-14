import json


def iter_partitions(model_path):
    """
    Returns the path of partitions and number of nodes in them.
    model_path - Path
    """
    config_dict = model_path / "config.json"
    with config_dict.open() as tf:
        config = json.load(tf)

    # determine the number of partitions
    num_partitions = config['entities']['link']['num_partitions']
    for i in range(num_partitions):
        # count the nodes in each partition
        count_file = model_path / "entity_count_link_{}.txt".format(i)
        with open(count_file, "rt") as f:
            count = int(f.readline())
        yield model_path / 'embeddings_link_{}.v50.h5'.format(i), count
