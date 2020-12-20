import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16
# https://blog.csdn.net/black_shuang/article/details/95384597


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # dim_head x heads = 64 x 16
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x : b n (h d)
        # x shape: 1, 65, 1024
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # dim=1024 -> innerdim x 3
        # q/k/v shape: 1, 65, 1024
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), qkv)  # inner dim = (heads x dim)
        # batch, inner dim, (heads x d) -> batch, heads, inner dim, dim
        # q/k/v.shape: 1, 16, 65, 64
        # 矩阵乘法
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * \
            self.scale  # SCALE FUNCTION
        mask_value = -torch.finfo(dots.dtype).max  # mask

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)  # softmax

        # 将attention和value进行矩阵乘法施加注意力
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # batch, heads, inner dim, dim
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)  # inner dim->dim 的linear
        return out  # shape: 1, 65, 1024


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # x: 1, 65, 1024
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size

        # 1, 65, dim=1024
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask=None):
        # img 1, 3, 256, 256
        p = self.patch_size

        x = rearrange(
            img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)  # 1, 8*8, 32*32*3

        x = self.patch_to_embedding(x)  # linear 32*32*3->dim=1024
        b, n, _ = x.shape  # x: 1, 64, 1024

        # 1,1,dim->1,1,1024
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # 1,(64+1),dim=1024
        # 1, 65, dim=1024 -> 1, 64, dim
        x += self.pos_embedding[:, :(n + 1)]  # 相加 TODO 这部分不理解
        x = self.dropout(x)  # 随机失活

        # x: 1, 65, 1024
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


if __name__ == "__main__":
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)

    # optional mask, designating which patch to attend to
    mask = torch.ones(1, 8, 8).bool()

    preds = v(img, mask=mask)  # (1, 1000)

    print(preds.shape)
