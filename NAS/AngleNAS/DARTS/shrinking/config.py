import os
class config:
    # Basic configration
    layers = 14
    edges = 14
    model_input_size_imagenet = (1,3,224,224)

    # Candidate operators
    blocks_keys = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]
    op_num=len(blocks_keys)

    # Operators encoding
    NONE = 0
    MAX_POOLING_3x3 = 1
    AVG_POOL_3x3 = 2
    SKIP_CONNECT = 3
    SEP_CONV_3x3 = 4
    SEP_CONV_5x5 = 5
    DIL_CONV_3x3 = 6
    DIL_CONV_5x5 = 7

    # Training configuration
    warmup_epochs = 5
    # More epochs are adopted for training supernet due to its slow convergence on DARTS
    first_stage_epochs = 150
    other_stage_epochs = 20
    insert_layers = 11

    # Shrinking configuration
    net_cache = 'weight.pt'
    base_net_cache = 'base_weight.pt'
    checkpoint_cache = 'checkpoint.pth.tar'
    shrinking_finish_threshold = 1000000
    sample_num = 1000
    per_stage_drop_num = 14
    epsilon = 1e-12
    
    # Enumerate all paths of a single cell
    # Path encoding is defined by node No. (e.g., [0,5] represents a path from node 0 to node 5) 
    paths = [[0, 2, 3, 4, 5], [0, 2, 3, 5], [0, 2, 4, 5], [0, 2, 5], [0, 3, 4, 5], [0, 3, 5],[0, 4, 5],[0, 5],
             [1, 2, 3, 4, 5], [1, 2, 3, 5], [1, 2, 4, 5], [1, 2, 5], [1, 3, 4, 5], [1, 3, 5],[1, 4, 5],[1, 5]]


    


