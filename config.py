entities_base = '../tab_graphs/cnr2000'

def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path=entities_base,
        edge_paths=[],
        checkpoint_path='model/cnr2000',

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

        # Scoring model
        dimension=64,
        global_emb=False,

        # Training
        num_epochs=50,
        num_uniform_negs=100,
        loss_fn='softmax',
        lr=0.1,

        # Misc
        hogwild_delay=2,
    )

    return config
