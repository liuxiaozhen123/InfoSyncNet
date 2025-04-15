import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # 判断t是否为元组


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim, patch_size):   # embed = 512
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        emb_dropout=0.1
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(   # (h p1)表示图像高度被分为h个p1大小的
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, embed_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads  # 头部特征的总维数
        project_out = not (heads == 1 and dim_head == embed_dim)
        drop_rate=0.1
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(embed_dim, inner_dim*3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(drop_rate)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # print("-----------------")
        # print(out.size())
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.size())
        return self.to_out(out)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=16, in_channels=2, embed_dim=128, patch_size=8, num_heads=8, dim_head=16, mlp_ratio=4.0, num_blocks=12, num_classes=512, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)

        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(embed_dim, Attention(embed_dim, heads=num_heads, dim_head=dim_head)),
                PreNorm(embed_dim, FeedForward(embed_dim, num_classes, dropout=dropout))
            ]))

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x [bt in_channels image_size image_size] -- 64 22 22
        # print(x.size())
        x = self.patch_embedding(x)
        # print(x.size())
        # Reshape to (batch_size, num_patches, embed_dim)
        # x = x.flatten(2).transpose(1, 2)

        # ViT Blocks
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
            # print(x.size())
        # print(x.size())
        # Global Average Pooling
        x = x.mean(dim=1)
        # print(x.size())
        # x = self.classifier(x)
        # print(x.size())

        return x

















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import nn, einsum
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange


# # class PatchEmbedding(nn.Module):
# #     def __init__(self, img_size=22, in_channels=64, embed_dim=768, patch_size=2):
# #         super(PatchEmbedding, self).__init__()
# #         self.patch_size = patch_size
# #         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
# #
# #     def forward(self, x):
# #         x = self.proj(x)
# #         return x

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)  # 判断t是否为元组

# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size, in_channels, embed_dim, patch_size):
#         super().__init__()
#         image_height, image_width = pair(img_size)
#         patch_height, patch_width = pair(patch_size)
#         emb_dropout=0.
#         assert image_height % patch_height == 0 and image_width % patch_width == 0

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = in_channels * patch_height * patch_width

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.Linear(patch_dim, embed_dim)
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.dropout = nn.Dropout(emb_dropout)


#     def forward(self, img):
#         x = self.to_patch_embedding(img)

#         b, n, _ = x.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n+1)]
#         x = self.dropout(x)
#         return x


# class ViTBlock(nn.Module):
#     def __init__(self, embed_dim=121, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.1):  # embed_dim=768
#         super(ViTBlock, self).__init__()

#         # Multi-Head Self Attention
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, bias=qkv_bias)

#         # Feedforward Neural Network (MLP)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
#             nn.Dropout(drop_rate)
#         )

#         # Layer normalization for both attention and MLP
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         # Self-Attention
#         # print(x.size())   # [1392 121 512]
#         attn_output, _ = self.attention(x, x, x)
#         x = x + attn_output
#         x = self.norm1(x)

#         # MLP
#         mlp_output = self.mlp(x)
#         x = x + mlp_output
#         x = self.norm2(x)

#         return x


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class Attention(nn.Module):
#     def __init__(self, embed_dim, heads, dim_head):
#         super().__init__()
#         inner_dim = dim_head * heads  # 头部特征的总维数
#         project_out = not (heads == 1 and dim_head == embed_dim)
#         drop_rate=0.1
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(embed_dim, inner_dim*3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, embed_dim),
#             nn.Dropout(drop_rate)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = self.attend(dots)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         # print("-----------------")
#         # print(out.size())
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         # print(out.size())
#         return self.to_out(out)


# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=22, in_channels=64, embed_dim=512, patch_size=2, num_heads=8, dim_head=64, mlp_ratio=4.0, num_blocks=4, num_classes=512, dropout=0.1):
#         super(VisionTransformer, self).__init__()

#         self.patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)

#         # self.blocks = nn.ModuleList([
#         #     ViTBlock(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(num_blocks)
#         # ])
#         self.blocks = nn.ModuleList([])
#         # for _ in range(num_blocks):
#         #     self.blocks.append(nn.ModuleList([
#         #         PreNorm(embed_dim, Attention(embed_dim, heads=num_heads, dim_head=dim_head, drop_rate=drop_rate)),
#         #         PreNorm(embed_dim, FeedForward(embed_dim, num_classes, drop_rate=drop_rate))
#         #                        ]))

#         for _ in range(num_blocks):
#             self.blocks.append(nn.ModuleList([
#                 PreNorm(embed_dim, Attention(embed_dim, heads=num_heads, dim_head=dim_head)),
#                 PreNorm(embed_dim, FeedForward(embed_dim, num_classes, dropout=dropout))
#             ]))

#         self.classifier = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # x [bt in_channels image_size image_size] -- 64 22 22
#         # print(x.size())
#         x = self.patch_embedding(x)
#         # print(x.size())
#         # Reshape to (batch_size, num_patches, embed_dim)
#         # x = x.flatten(2).transpose(1, 2)

#         # ViT Blocks
#         # for block in self.blocks:
#         #     x = block(x)
#         for attn, ff in self.blocks:
#             x = attn(x) + x
#             x = ff(x) + x
#             # print(x.size())

#         # Global Average Pooling
#         x = x.mean(dim=1)


#         return x