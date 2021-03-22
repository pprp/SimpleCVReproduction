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

    backbone_info = [ # inp, oup, img_h, img_w, stride
        (3,     40,     224,    224,    2),     #conv1
        (24,    32,     112,    112,    2),     #stride = 2
        (32,    32,     56,     56,     1),
        (32,    32,     56,     56,     1),
        (32,    32,     56,     56,     1),
        (32,    56,     56,     56,     2),     #stride = 2
        (56,    56,     28,     28,     1),
        (56,    56,     28,     28,     1),   
        (56,    56,     28,     28,     1),
        (56,    112,    28,     28,     2),     #stride = 2
        (112,   112,    14,     14,     1),
        (112,   112,    14,     14,     1),  
        (112,   112,    14,     14,     1),
        (112,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   128,    14,     14,     1),
        (128,   256,    14,     14,     2),     #stride = 2
        (256,   256,    7,      7,      1),
        (256,   256,    7,      7,      1),
        (256,   256,    7,      7,      1), 
        (256,   432,    7,      7,      1),
        (432,   1728,   7,      7,      1),     # post_processing
    ]
