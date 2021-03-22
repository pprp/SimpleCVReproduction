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
    
    flops_lookup_table = '../../op_flops_dict_cpu.pkl'
    model_input_size_imagenet = (1, 3, 224, 224)
    backbone_info = [ # inp, oup, img_h, img_w, stride
        (3,     32,     224,    224,    2),     #conv1
        (16,    32,     112,    112,    2),     #stride = 2
        (32,    32,     56,     56,     1),
        (32,    40,     56,     56,     2),     #stride = 2
        (40,    40,     28,     28,     1),
        (40,    40,     28,     28,     1),   
        (40,    40,     28,     28,     1),
        (40,    80,     28,     28,     2),     #stride = 2
        (80,    80,     14,     14,     1),
        (80,    80,     14,     14,     1),  
        (80,    80,     14,     14,     1),
        (80,    96,     14,     14,     1),
        (96,    96,     14,     14,     1),
        (96,    96,     14,     14,     1),
        (96,    96,     14,     14,     1),
        (96,    192,    14,     14,     2),     #stride = 2
        (192,   192,    7,      7,      1),
        (192,   192,    7,      7,      1),
        (192,   192,    7,      7,      1),  
        (192,   320,    7,      7,      1),
        (320,   1280,   7,      7,      1),     # post_processing
    ]
