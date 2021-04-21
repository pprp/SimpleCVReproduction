# Code from https://github.com/simochen/model-tools.
import numpy as np
import pdb
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random


def lookup_table_flops(model, candidate_width, alphas=None, input_res=32, multiply_adds=False):
    if alphas is None:
        for n, v in model.named_parameters():
            if 'alphas' in n:
                alphas = v
    num_conv = alphas.shape[0]
    device = alphas.device

    # obtain the feature map sizes
    list_bn=[]
    def bn_hook(self, input, output):
        if input[0].ndimension() == 4:
            list_bn.append(input[0].shape)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input_ = torch.rand(1, 3, input_res, input_res).to(alphas.device)
    input_.requires_grad = True
    # print('alphas:', alphas)
    # print('inputs:', input_)
    if torch.cuda.device_count() > 1:
        model.module.register_buffer('alphas_tmp', alphas.data)
    else:
        model.register_buffer('alphas_tmp', alphas.data)
    out = model(input_)

    # TODO: only appliable for resnet_20s: 2 convs followed by 1 shortcut
    list_main_bn = []
    num_width = len(candidate_width)

    for i, b in enumerate(list_bn):
        if i//num_width == 0 or \
            ((i-num_width)//(num_width**2) >= 0 and ((i-num_width)//(num_width)**2) % 3 != 2):
            list_main_bn.append(b)
    assert len(list_main_bn) == (num_width + num_width ** 2 * (num_conv-1)), 'wrong list of feature map length'

    # start compute flops for each branch
    # first obtain the kernel shapes, a list of length: num_width + num_width**2 * num_conv

    def kernel_shape_types(candidate_width):
        kshape_types = []
        first_kshape_types = []
        for i in candidate_width:
            first_kshape_types.append((i, 3, 3, 3))

        for i in candidate_width:
            for j in candidate_width:
                kshape_types.append((i, j, 3, 3)) # [co, ci, k, k]
        return kshape_types, first_kshape_types

    kshape_types, first_kshape_types = kernel_shape_types(candidate_width)
    k_shapes = []
    layer_idx = 0
    for v in model.parameters():
        if v.ndimension() == 4 and v.shape[2] == 3:
            if layer_idx == 0:
                k_shapes += first_kshape_types
            else:
                k_shapes += kshape_types
            layer_idx += 1

    # compute flops
    flops = [] # a list of length: num_width + num_width**2 * num_conv
    for idx, a_shape in enumerate(list_main_bn):
        n, ci, h, w = a_shape
        k_shape = k_shapes[idx]
        co, ci, k, _ = k_shape
        flop = co * ci * k * k * h * w
        flops.append(flop)

    # reshape flops back to list. len == num_conv
    table_flops = []
    table_flops.append(torch.Tensor(flops[:num_width]).to(device))
    for layer_idx in range(num_conv-1):
        tmp = flops[num_width + layer_idx*num_width**2:\
            num_width + (layer_idx+1)*num_width**2]
        assert len(tmp) == num_width ** 2, 'need have %d elements in %d layer'%(num_width**2, layer_idx+1)
        table_flops.append(torch.Tensor(tmp).to(device))

    return table_flops


def print_model_param_nums(model, multiply_adds=False):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of original params: %.8fM' % (total / 1e6))
    return total


def print_model_param_flops(model, input_res, multiply_adds=False):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            # if isinstance(net, torch.nn.BatchNorm2d):
            #     net.register_forward_hook(bn_hook)
            # if isinstance(net, torch.nn.ReLU):
            #     net.register_forward_hook(relu_hook)
            # if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            #     net.register_forward_hook(pooling_hook)
            # if isinstance(net, torch.nn.Upsample):
            #     net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    model = model.cuda()
    foo(model)
    input_ = torch.rand(3, 3, input_res, input_res).cuda()
    input_.requires_grad = True
    out = model(input_)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    total_flops /= 3

    print('  + Number of FLOPs of original model: %.8fG' % (total_flops / 1e9))
    # print('list_conv', list_conv)
    # print('list_linear', list_linear)
    # print('list_bn', list_bn)
    # print('list_relu', list_relu)
    # print('list_pooling', list_pooling)

    return total_flops

