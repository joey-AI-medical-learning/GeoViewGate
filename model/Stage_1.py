import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=4, embed_dim=96, patch_size=(8, 8, 8)):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x



class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        )

        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)



class SwinBlock3D(nn.Module):
    def __init__(self, dim, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim)

        self.drop_path = nn.Identity()

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class Swin3D_SSS_M(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=2,
        embed_dim=96,
        depth=4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained_cfg=None,
        **kwargs   
    ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(in_chans, embed_dim)

        self.blocks = nn.Sequential(
            *[
                SwinBlock3D(
                    embed_dim,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)   # [B, N, C]
        x = self.blocks(x)
        x = self.norm(x)

        x = x.mean(dim=1)
        return self.head(x)



def _create_swin3d_sss(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        Swin3D_SSS_M,
        variant,
        pretrained,
        **kwargs
    )



@register_model
def Model(pretrained=False, **kwargs):
    return _create_swin3d_sss(
        'swin3d_sss',
        pretrained,
        **kwargs
    )


if __name__ == "__main__":
    x = torch.randn(1, 4, 128, 128, 128)
    model = Model(in_chans=3, num_classes=2)
    y = model(x)


   