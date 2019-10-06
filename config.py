# entities_base = '../tab_graphs/cnr2000'


def get_torchbiggraph_config(num_partitions=1):
    """
    basename (str) - name of the graph
    """
    # entities_base = os.path.join("/data/", basename)
    # checkpoints = os.path.join("/data/models", basename)
    entities_base = "/data/cnr-2000"
    checkpoints = "/data/models/cnr-2000"

    config = dict(
        # I/O data
        entity_path=entities_base,
        edge_paths=[
            "/data/graphs/cnr-2000/train_partitioned.txt",
            "/data/graphs/cnr-2000/test_partitioned.txt",
        ],
        checkpoint_path=checkpoints,  # example: 'model/cnr2000'

        # Graph structure
        entities={
            'link': {'num_partitions': 1},
        },
        relations=[{
            'name': 'follow',
            'lhs': 'link',
            'rhs': 'link',
            'operator': 'none',
        }],

        dynamic_relations=False,

        # Scoring model
        dimension=64,
        global_emb=False,

        # Training
        num_epochs=50,
        num_uniform_negs=100,
        loss_fn='softmax',
        lr=0.01,

        # Misc
        hogwild_delay=2,
    )

    return config
