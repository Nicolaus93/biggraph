from os.path import join

BASENAME = "indochina-2004"


def get_torchbiggraph_config(num_partitions=1):
    """
    basename (str) - name of the graph
    """
    model = join("/data/models", BASENAME)
    graphs = join("data/graphs", BASENAME)

    config = dict(
        # I/O data
        entity_path=model,
        edge_paths=[
            graphs / "train_partitioned",
            graphs / "test_partitioned",
        ],
        checkpoint_path=model,

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

        verbose=True,
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
