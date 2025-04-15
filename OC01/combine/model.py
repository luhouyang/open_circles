"""
U-Net + SETR

author: Lu Hou Yang
GitHub: https://github.com/luhouyang/open_circles.git
date: 7 April 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# Reading einops (Medium): https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Embedding(nn.Module):
    # yapf: disable
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        channels=3,
        emb_dropout=0.,
    ):
        super(Embedding, self).__init__()

        image_height, image_width = self.pair(image_size)
        patch_height, patch_width = self.pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Encoder (transforms input patches to tensors of size 'dim')
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1 = patch_height,
                p2 = patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        B, N, _ = x.shape

        x += self.pos_embedding[:, :N] # adding the positional noise
        x = self.dropout(x)

        return x

    def pair(self, t):
        """Check if image and patch size given is 2D"""
        return t if isinstance(t, tuple) else (t, t)

    # yapf: enable


class UNET(nn.Module):

    def __init__(self, channels, RESF):
        super(UNET, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
            nn.Conv2d(RESF, RESF, 3, padding=1),
            nn.BatchNorm2d(RESF),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return c1, c2, c3, c4, c5


# re-encode previous tokens into higher level features
class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            # Reading GELU (PyTorch): https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    # yapf: disable
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()

        inner_dim = dim_head * heads  # length of individual k, q, OR v
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # 1/sqrt(q) from the popular 'Attention Is All You Need' paper: https://arxiv.org/abs/1706.03762
        # Attention(Q, K, V) = softmax(QK^T√d_k)V
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # expand inner_dim into k, q, v (by multiplying 3), bias not needed to keep a consistent dimension
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # only return the same tensor if the last dimension was not expanded by either multiple heads OR a different inner_dim
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # expand into q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # extract each q, k, v and group by head(s)

        # Attention(Q, K, V) = softmax(QK^T√d_k)V
        # Q . transpose(K) / sqrt(Q)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(
            attn, v
        )  # apply attention mask to values to agument the importance of each embedding
        out = rearrange(out, 'b h n d -> b n (h d)'
                        )  # recombine the head(s) into the pre-chunked tensor

        return self.to_out(out)

    # yapf: enable


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim,
                              heads=heads,
                              dim_head=dim_head,
                              dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class Segment(nn.Module):

    def __init__(self, num_classes, dim, dropout, image_height, RESF):
        super(Segment, self).__init__()

        TOTAL = dim + RESF

        self.upsample1 = nn.Sequential(
            nn.Conv2d(dim + RESF, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(TOTAL, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
        )

        self.upsample3 = nn.Sequential(
            nn.Conv2d(TOTAL, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
        )

        self.upsample4 = nn.Sequential(
            nn.Conv2d(TOTAL, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
        )

        self.out = nn.Sequential(
            nn.Conv2d(TOTAL, TOTAL, 3, padding=1),
            nn.BatchNorm2d(TOTAL),
            nn.ReLU(inplace=True),
            nn.Conv2d(TOTAL, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout + 0.2),
            nn.Conv2d(16, num_classes, 1),
        )

        self.image_height = image_height

    def forward(self, x, c1, c2, c3, c4, c5):
        # B, N, _ = x.shape

        # cls_tokens = x[:, 0, :].reshape(B, 1, -1)
        # x = x[:, 1:, :]

        # x += cls_tokens[:, :N]

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.image_height)
        x = torch.cat([x, c5], dim=1)
        x1 = self.upsample1(x)

        x1 = torch.cat([x1, c4], dim=1)
        x2 = self.upsample2(x1)

        x2 = torch.cat([x2, c3], dim=1)
        x3 = self.upsample3(x2)

        x3 = torch.cat([x3, c2], dim=1)
        x4 = self.upsample4(x3)

        x4 = torch.cat([x4, c1], dim=1)
        x5 = self.out(x4)

        return x5, x4, x3, x2, x1


class CombineModel(nn.Module):

    def __init__(
        self,
        image_size,
        patch_size,  # divisor of input image_size (=32) | (49 patches, 64) -> (7, 7, 64)
        num_classes,
        depth,
        heads,
        dim=64,
        RESF=32,
        mlp_dim=128,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
    ):
        super(CombineModel, self).__init__()

        self.unet = UNET(channels=channels, RESF=RESF)

        self.embedding = Embedding(image_size=image_size,
                                   patch_size=patch_size,
                                   dim=dim,
                                   channels=channels,
                                   emb_dropout=emb_dropout)

        self.transformer = Transformer(dim=dim,
                                       depth=depth,
                                       heads=heads,
                                       dim_head=dim_head,
                                       mlp_dim=mlp_dim,
                                       dropout=dropout)

        self.segment = Segment(num_classes=num_classes,
                               dim=dim,
                               dropout=dropout,
                               image_height=14,
                               RESF=RESF)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.unet(x)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.segment(x, c1, c2, c3, c4, c5)

        return x


class CombineModelLoss(nn.Module):

    def __init__(self, dim, RESF):
        super(CombineModelLoss, self).__init__()

        TOTAL = dim + RESF

        self.aux1 = nn.Sequential(
            nn.Upsample(scale_factor=8, align_corners=True, mode='bilinear'),
            nn.Conv2d(TOTAL, 2, 1, bias=False),
        ).cuda()
        self.aux2 = nn.Sequential(
            nn.Upsample(scale_factor=4, align_corners=True, mode='bilinear'),
            nn.Conv2d(TOTAL, 2, 1, bias=False),
        ).cuda()
        self.aux3 = nn.Sequential(
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
            nn.Conv2d(TOTAL, 2, 1, bias=False),
        ).cuda()
        self.aux4 = nn.Sequential(nn.Conv2d(TOTAL, 2, 1, bias=False), ).cuda()

        self.creterion = F.cross_entropy

    def forward(self, pred, label, PCW):
        x5, x4, x3, x2, x1 = pred

        loss1 = self.creterion(self.aux1(x1), label, PCW)
        loss2 = self.creterion(self.aux2(x2), label, PCW)
        loss3 = self.creterion(self.aux3(x3), label, PCW)
        loss4 = self.creterion(self.aux4(x4), label, PCW)
        loss5 = self.creterion(x5, label, PCW)

        return loss5, loss4, loss3, loss2, loss1
