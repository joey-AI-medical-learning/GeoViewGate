import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    ws = window_size
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, ws * ws * ws, C)
    return windows


def window_reverse(windows, window_size, D, H, W, B):
    ws = window_size
    C = windows.shape[-1]
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, C)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        #  (2ws-1)^3 x num_heads
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 3, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_d = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_d, coords_d, coords_d, indexing='ij'))  # [3, ws, ws, ws]
        coords_flatten = torch.flatten(coords, 1)                     # [3, ws^3]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # [N, N, 3]
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 2] += ws - 1
        relative_coords[:, :, 0] *= (2 * ws - 1) ** 2
        relative_coords[:, :, 1] *= (2 * ws - 1)
        relative_position_index = relative_coords.sum(-1)             # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: [nW*B, N, C]; mask: [nW, N, N] 或 None"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, -1)     # [N, N, heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * random_tensor


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, D, H, W, attn_mask):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)     # [nW*B, ws^3, C]
        attn_windows = self.attn(x_windows, mask=attn_mask)           # [nW*B, ws^3, C]
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W, B)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, N, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.view(B, D, H, W, C)
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, D // 2, H // 2, W // 2, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x.view(B, (D // 2) * (H // 2) * (W // 2), 2 * C)


class SwinStage3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, drop_path=(), downsample=True):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            SwinBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if len(drop_path) > i else 0.0)
            for i in range(depth)
        ])
        self.downsample = PatchMerging3D(dim) if downsample else None

    def _attn_mask(self, D, H, W, device):
        ws, ss = self.window_size, self.window_size // 2
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        d_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, ws)          
        mask_windows = mask_windows.view(-1, ws * ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask                                   

    def forward(self, x, D, H, W):
        for blk in self.blocks:
            attn_mask = self._attn_mask(D, H, W, x.device) if blk.shift_size > 0 else None
            x = blk(x, D, H, W, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, D, H, W)
            D, H, W = D // 2, H // 2, W // 2
        return x, D, H, W

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, D, H, W] -> tokens [B, N, embed_dim], 及空间尺寸
        x = self.proj(x)                          # [B, E, Dp, Hp, Wp]
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)          # [B, N, E]
        return x, Dp, Hp, Wp


class Swin3D_Hierarchical(nn.Module):
    def __init__(self, in_chans=3, num_classes=3,
                 embed_dim=96, depths=(2, 2, 2), num_heads=(4, 8, 16),
                 window_size=8, mlp_ratio=4.0, drop_rate=0.0,
                 drop_path_rate=0.1, **kwargs):
        super().__init__()
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.out_dim = embed_dim * (2 ** (self.num_stages - 1))   # 96*4 = 384

        self.patch_embed = PatchEmbed3D(in_chans, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        block_offset = 0
        for i in range(self.num_stages):
            stage = SwinStage3D(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[block_offset:block_offset + depths[i]],
                downsample=(i < self.num_stages - 1),
            )
            self.stages.append(stage)
            block_offset += depths[i]

        self.norm = nn.LayerNorm(self.out_dim)
        self.head = nn.Linear(self.out_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_tokens(self, x):
        x, D, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in self.stages:
            x, D, H, W = stage(x, D, H, W)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        return self.forward_tokens(x).mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def Model(pretrained=False, **kwargs):
    return Swin3D_Hierarchical(**kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128, 128)
    model = Model(in_chans=3, num_classes=3)
    y = model(x)

