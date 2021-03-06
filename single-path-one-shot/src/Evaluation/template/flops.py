import torch
def get_flops(model,input_shape=(3,224,224)):
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels%self.groups==0

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups) 
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)


    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.autograd.Variable(torch.rand(*input_shape).unsqueeze(0), requires_grad = True)
    out = model(input)

    total_flops = sum(sum(i) for i in [list_conv,list_linear])
    return total_flops

def main():
    pass
