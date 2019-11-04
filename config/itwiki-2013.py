from os.path import join

BASENAME = "itwiki-2013"


def get_torchbiggraph_config():

    model = join("/data/models", BASENAME)
    graphs = join("data/graphs", BASENAME)

    config = dict(
        # I/O data
        entity_path=model,
        # where to store learnt edges
        edge_paths=[
            join(graphs, "train_partitioned"),
            join(graphs, "test_partitioned"),
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
        dimension=128,
        global_emb=False,

        # Training
        num_epochs=100,
        num_uniform_negs=100,
        loss_fn='softmax',
        lr=0.01,

        # Misc
        hogwild_delay=2,
    )

    return config
