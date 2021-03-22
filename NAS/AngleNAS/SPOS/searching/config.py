import os
class config:
    limit_flops = True
    max_flops = 475 * 1e6
    blocks_keys = [
    'mobilenet_3x3_ratio_3',
    'mobilenet_3x3_ratio_6',
    'mobilenet_5x5_ratio_3',
    'mobilenet_5x5_ratio_6',
    'mobilenet_7x7_ratio_3',
    'mobilenet_7x7_ratio_6'
    ]
    host = 'huyiming_wuhu2.huyiming.ws2.wuhu-a.brainpp.cn'
    username = 'test'
    port = 5672
    test_send_pipe = 'send_pipe_hym'
    test_recv_pipe = 'recv_pipe_hym'
    random_test_send_pipe = 'huyiming_random'
    random_test_recv_pipe = 'huyiming_random'
    checkpoint_cache = './checkpoint.pth.tar'
    
    net_cache = 'weight.pt'
    flops_lookup_table = '../../op_flops_dict_gpu.pkl'
    model_input_size_imagenet = (1, 3, 224, 224)
    stage_last_id=[4,8,12,16,20]
    op_num=len(blocks_keys)
    # For SPOS, the search procedure is not stable.
    # Suggest retraining both of top-2 models, if their accuracies are very close to each other.    
    topk = 2

    epsilon = 1e-12
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
