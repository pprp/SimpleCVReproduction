import os
class config:
    blocks_keys = [
    'mobilenet_3x3_ratio_3',
    'mobilenet_3x3_ratio_6',
    'mobilenet_5x5_ratio_3',
    'mobilenet_5x5_ratio_6',
    'mobilenet_7x7_ratio_3',
    'mobilenet_7x7_ratio_6'
    ]

    flops_lookup_table = '../../op_flops_dict_gpu.pkl'
    model_input_size_imagenet = (1, 3, 224, 224)