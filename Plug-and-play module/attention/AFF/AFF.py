# source: https://mp.weixin.qq.com/s/La6rbQpnZzjWH3psB2gD6Q
# code: https://github.com/YimianDai/open-aff

# AFF
class AXYforXplusYAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(AXYforXplusYAddFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xi = x + residual
        xl = self.local_att(xi)
        xg = self.global_att(xi)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = F.broadcast_mul(wei, residual) + x

        return xo