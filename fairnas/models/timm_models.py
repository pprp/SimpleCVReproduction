""" 
FairNAS-SE models
"""
from functools import partial
from timm.models import EfficientNet
from timm.models.efficientnet_builder import decode_arch_def
from timm.models.efficientnet_blocks import resolve_bn_args


def _gen_fairnas(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """fairnas a
    """
    if kwargs.get('s_r') is not None:
        s_r = kwargs.pop('s_r')
    else:
        s_r = 0.5
    if variant == 'fairnas_a':
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k7_s2_e3_c32_se%f_nsw' % s_r],
            # stage 2, 56x56 in
            ['ir_r1_k3_s1_e3_c32_se%f_nsw' % s_r],  # swish
            # stage 3, 28x28 in
            ['ir_r1_k7_s2_e3_c40_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c40_se%f_nsw' % s_r, 'ir_r1_k7_s1_e6_c40_se%f_nsw' % s_r,
             'ir_r1_k3_s1_e3_c40_se%f_nsw' % s_r],  # swish
            # stage 4, 14x14in
            ['ir_r1_k3_s2_e3_c80_se%f_nsw' % s_r, 'ir_r2_k7_s1_e6_c80_se%f_nsw' % s_r, 'ir_r1_k5_s1_e3_c80_se%f_nsw' % s_r,
             'ir_r1_k3_s1_e6_c96_se%f_nsw' % s_r, 'ir_r2_k5_s1_e3_c96_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c96_se%f_nsw' % s_r],  # swish
            # stage 5, 7x7in
            ['ir_r1_k3_s2_e6_c192_se%f_nsw' % s_r, 'ir_r1_k7_s1_e6_c192_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c192_se%f_nsw' % s_r,
             'ir_r1_k7_s1_e6_c192_se%f_nsw' % s_r,
             'ir_r1_k5_s1_e6_c320_se%f_nsw' % s_r],  # swish
        ]
    elif variant == 'fairnas_b':
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k5_s2_e3_c32_se%f_nsw' % s_r],
            # stage 2, 56x56 in
            ['ir_r1_k3_s1_e3_c32_se%f_nsw' % s_r],  # swish
            # stage 3, 28x28 in
            ['ir_r1_k5_s2_e3_c40_se%f_nsw' % s_r, 'ir_r1_k3_s1_e3_c40_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c40_se%f_nsw' % s_r,
             'ir_r1_k5_s1_e3_c40_se%f_nsw' % s_r],  # swish
            # stage 4, 14x14in
            ['ir_r1_k7_s2_e3_c80_se%f_nsw' % s_r, 'ir_r1_k3_s1_e3_c80_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c80_se%f_nsw' % s_r, 'ir_r1_k5_s1_e3_c80_se%f_nsw' % s_r,
             'ir_r1_k3_s1_e3_c96_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c96_se%f_nsw' % s_r, 'ir_r1_k7_s1_e3_c96_se%f_nsw' % s_r, 'ir_r1_k3_s1_e3_c96_se%f_nsw' % s_r],  # swish
            # stage 5, 7x7in
            ['ir_r1_k7_s2_e6_c192_se%f_nsw' % s_r, 'ir_r1_k5_s1_e6_c192_se%f_nsw' % s_r, 'ir_r1_k7_s1_e6_c192_se%f_nsw' % s_r,
             'ir_r1_k3_s1_e6_c192_se%f_nsw' % s_r,
             'ir_r1_k5_s1_e6_c320_se%f_nsw' % s_r],  # swish
        ]
    else:
        assert variant == 'fairnas_c'
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k5_s2_e3_c32_se%f_nsw' % s_r],
            # stage 2, 56x56 in
            ['ir_r1_k3_s1_e3_c32_se%f_nsw' % s_r],  # swish
            # stage 3, 28x28 in
            ['ir_r1_k7_s2_e3_c40_se%f_nsw' % s_r, 'ir_r3_k3_s1_e3_c40_se%f_nsw' % s_r],  # swish
            # stage 4, 14x14in
            ['ir_r1_k3_s2_e3_c80_se%f_nsw' % s_r, 'ir_r2_k3_s1_e3_c80_se%f_nsw' % s_r, 'ir_r1_k3_s1_e6_c80_se%f_nsw' % s_r,
             'ir_r4_k3_s1_e3_c96_se%f_nsw' % s_r],  # swish
            # stage 5, 7x7in
            ['ir_r1_k7_s2_e6_c192_se%f_nsw' % s_r, 'ir_r1_k7_s1_e6_c192_se%f_nsw' % s_r, 'ir_r2_k3_s1_e6_c192_se%f_nsw' % s_r,
             'ir_r1_k5_s1_e6_c320_se%f_nsw' % s_r],  # swish
        ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=1280,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        # act_layer=nn.ReLU,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    model = EfficientNet(**model_kwargs)
    return model


Fairnas = {
	'fairnas_a': partial(_gen_fairnas,variant='fairnas_a'),
	'fairnas_b': partial(_gen_fairnas,variant='fairnas_b'),
	'fairnas_c': partial(_gen_fairnas,variant='fairnas_c')
}




