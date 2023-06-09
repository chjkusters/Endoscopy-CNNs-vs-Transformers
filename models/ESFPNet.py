"""IMPORT PACKAGES"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import math
from mmcv.cnn import ConvModule


""""""""""""""""""""""""""""""""""""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
""""""""""""""""""""""""""""""""""""
# Adapted from https://github.com/dumyCq/ESFPNet


class ESFPNetStructure(nn.Module):
    def __init__(self, opt, inference, embedding_dim=160):
        super(ESFPNetStructure, self).__init__()

        # Define whether to do inference
        self.inference = inference

        # Backbone
        self.bb = opt.backbone
        if 'B0' in self.bb:
            self.MiT = mit_b0(opt=opt)
        if 'B1' in self.bb:
            self.MiT = mit_b1(opt=opt)
        if 'B2' in self.bb:
            self.MiT = mit_b2(opt=opt)
        if 'B3' in self.bb:
            self.MiT = mit_b3(opt=opt)
        if 'B4' in self.bb:
            self.MiT = mit_b4(opt=opt)
        if 'B5' in self.bb:
            self.MiT = mit_b5(opt=opt)

        self._init_weights()  # load pretrain

        # LP Header
        self.LP_1 = LP(input_dim=self.MiT.embed_dims[0], embed_dim=self.MiT.embed_dims[0])
        self.LP_2 = LP(input_dim=self.MiT.embed_dims[1], embed_dim=self.MiT.embed_dims[1])
        self.LP_3 = LP(input_dim=self.MiT.embed_dims[2], embed_dim=self.MiT.embed_dims[2])
        self.LP_4 = LP(input_dim=self.MiT.embed_dims[3], embed_dim=self.MiT.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.MiT.embed_dims[2] + self.MiT.embed_dims[3]),
                                        out_channels=self.MiT.embed_dims[2], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.MiT.embed_dims[1] + self.MiT.embed_dims[2]),
                                        out_channels=self.MiT.embed_dims[1], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.MiT.embed_dims[0] + self.MiT.embed_dims[1]),
                                        out_channels=self.MiT.embed_dims[0], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = LP(input_dim=self.MiT.embed_dims[0], embed_dim=self.MiT.embed_dims[0])
        self.LP_23 = LP(input_dim=self.MiT.embed_dims[1], embed_dim=self.MiT.embed_dims[1])
        self.LP_34 = LP(input_dim=self.MiT.embed_dims[2], embed_dim=self.MiT.embed_dims[2])

        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.MiT.embed_dims[0] + self.MiT.embed_dims[1] +
                                      self.MiT.embed_dims[2] + self.MiT.embed_dims[3]), 1, kernel_size=(1, 1))

        # Classification Head
        # For the sake of determinism in pytorch lightning, usage of avgpool2d is necessary
        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        if opt.imagesize == 256:
            self.avgpool = nn.AvgPool2d(kernel_size=(4, 4))
        elif opt.imagesize == 352:
            self.avgpool = nn.AvgPool2d(kernel_size=(5, 5))

        if 'B0' in self.bb:
            self.fc = nn.Linear(256 * 2 * 2, opt.num_classes)
        else:
            self.fc = nn.Linear(512 * 2 * 2, opt.num_classes)

    def _init_weights(self):

        if 'B0' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b0.pth'))
        if 'B1' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b1.pth'))
        if 'B2' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b2.pth'))
        if 'B3' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b3.pth'))
        if 'B4' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b4.pth'))
        if 'B5' in self.bb:
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'mit_b5.pth'))

        model_dict = self.MiT.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.MiT.load_state_dict(model_dict)

    def forward(self, x):

        # Go Through Backbone
        B = x.shape[0]

        # stage 1
        out_1, H, W = self.MiT.patch_embed1(x)
        for i, blk in enumerate(self.MiT.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.MiT.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.MiT.embed_dims[0], 88, 88)

        # stage 2
        out_2, H, W = self.MiT.patch_embed2(out_1)
        for i, blk in enumerate(self.MiT.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.MiT.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.MiT.embed_dims[1], 44, 44)

        # stage 3
        out_3, H, W = self.MiT.patch_embed3(out_2)
        for i, blk in enumerate(self.MiT.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.MiT.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.MiT.embed_dims[2], 22, 22)

        # stage 4
        out_4, H, W = self.MiT.patch_embed4(out_3)
        for i, blk in enumerate(self.MiT.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.MiT.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.MiT.embed_dims[3], 11, 11)

        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # linear fuse and go pass LP Header
        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
        if self.inference:
            lp_34 = self.LP_34(self.linear_fuse34(
                torch.cat([lp_3, F.interpolate(lp_4, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
            lp_23 = self.LP_23(self.linear_fuse23(
                torch.cat([lp_2, F.interpolate(lp_34, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
            lp_12 = self.LP_12(self.linear_fuse12(
                torch.cat([lp_1, F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
        else:
            lp_34 = self.LP_34(self.linear_fuse34(
                torch.cat([lp_3, F.interpolate(lp_4, scale_factor=2, mode='nearest-exact')], dim=1)))
            lp_23 = self.LP_23(self.linear_fuse23(
                torch.cat([lp_2, F.interpolate(lp_34, scale_factor=2, mode='nearest-exact')], dim=1)))
            lp_12 = self.LP_12(self.linear_fuse12(
                torch.cat([lp_1, F.interpolate(lp_23, scale_factor=2, mode='nearest-exact')], dim=1)))

        # get the final output
        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
        if self.inference:
            lp4_resized = F.interpolate(lp_4, scale_factor=8, mode='bilinear', align_corners=False)
            lp3_resized = F.interpolate(lp_34, scale_factor=4, mode='bilinear', align_corners=False)
            lp2_resized = F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            lp4_resized = F.interpolate(lp_4, scale_factor=8, mode='nearest-exact')
            lp3_resized = F.interpolate(lp_34, scale_factor=4, mode='nearest-exact')
            lp2_resized = F.interpolate(lp_23, scale_factor=2, mode='nearest-exact')
        lp1_resized = lp_12
        out = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))

        # Segmentation output
        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
        if self.inference:
            seg = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            seg = F.interpolate(out, scale_factor=4, mode='nearest-exact')

        # Classification output
        cls = self.avgpool(out_4)
        cls = cls.reshape(cls.shape[0], -1)
        cls = self.fc(cls)

        return cls, seg


""""""""""""""""""""""""""""""""""""
"""" DEFINE MIXTRANSFORMER CODE"""
""""""""""""""""""""""""""""""""""""


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b0, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b1, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b2, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b3, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b4, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, opt, **kwargs):
        super(mit_b5, self).__init__(
            img_size=opt.imagesize, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


""""""""""""""""""""""""""""""""""""
"""" DEFINE DECODER CODE"""
""""""""""""""""""""""""""""""""""""


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class LP(nn.Module):
    """
    Linear Prediction
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
