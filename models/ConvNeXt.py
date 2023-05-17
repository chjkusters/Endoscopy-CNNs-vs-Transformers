"""IMPORT PACKAGES"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


""""""""""""""""""""""""
"""" HELPER FUNCTIONS """
""""""""""""""""""""""""
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


""""""""""""""""""""""""
"""" MODEL DEFINITIONS """
""""""""""""""""""""""""


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

    #     self.apply(self._init_weights)
    #     self.head.weight.data.mul_(head_init_scale)
    #     self.head.bias.data.mul_(head_init_scale)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 0:
                low_level = x
        high_level = x
        x = self.norm(x.mean([-2, -1]))         # global average pooling, (N, C, H, W) -> (N, C)
        return x, low_level, high_level

    def forward(self, x):
        x, low_level, high_level = self.forward_features(x)
        x = self.head(x)
        return x, low_level, high_level


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, in_22k=False, num_classes=1, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, num_classes=1, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes, **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        del checkpoint['model']['head.weight']
        del checkpoint['model']['head.bias']
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


""""""""""""""""""""""""""""""
"""" DEFINE SEGMENTATION MODELS"""
""""""""""""""""""""""""""""""
# https://github.com/MLearing/Pytorch-DeepLab-v3-plus


# Class for ASPP Module -ConvNeXt Tiny & Small
class ASPP_module_TS(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_TS, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=(kernel_size, kernel_size),
                                            stride=(1, 1), padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_ConvNeXt_TS(nn.Module):
    def __init__(self, n_classes=1, os=16, inference=False):
        super(DeepLabv3_plus_ConvNeXt_TS, self).__init__()

        # Define inference or not
        self.inference = inference

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_TS(768, 256, rate=rates[0])
        self.aspp2 = ASPP_module_TS(768, 256, rate=rates[1])
        self.aspp3 = ASPP_module_TS(768, 256, rate=rates[2])
        self.aspp4 = ASPP_module_TS(768, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(768, 256, (1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(256, track_running_stats=True),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(96, 48, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48, track_running_stats=True)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)))

        # Initialize weights
        self._init_weight()

    def forward(self, img, low_level_features, high_level_features):
        x1 = self.aspp1(high_level_features)
        x2 = self.aspp2(high_level_features)
        x3 = self.aspp3(high_level_features)
        x4 = self.aspp4(high_level_features)
        x5 = self.global_avg_pool(high_level_features)

        if self.inference:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        else:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest-exact')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.inference:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)),
                                       int(math.ceil(img.size()[-1]/4))), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)), int(math.ceil(img.size()[-1]/4))), mode='nearest-exact')

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        if self.inference:
            x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=img.size()[2:], mode='nearest-exact')

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)


# Class for ASPP Module -ConvNeXt Base
class ASPP_module_B(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_B, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=(kernel_size, kernel_size),
                                            stride=(1, 1), padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_ConvNeXt_B(nn.Module):
    def __init__(self, n_classes=1, os=16, inference=False):
        super(DeepLabv3_plus_ConvNeXt_B, self).__init__()

        # Define inference or not
        self.inference = inference

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_B(1024, 256, rate=rates[0])
        self.aspp2 = ASPP_module_B(1024, 256, rate=rates[1])
        self.aspp3 = ASPP_module_B(1024, 256, rate=rates[2])
        self.aspp4 = ASPP_module_B(1024, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(256, track_running_stats=True),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48, track_running_stats=True)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)))

        # Initialize weights
        self._init_weight()

    def forward(self, img, low_level_features, high_level_features):
        x1 = self.aspp1(high_level_features)
        x2 = self.aspp2(high_level_features)
        x3 = self.aspp3(high_level_features)
        x4 = self.aspp4(high_level_features)
        x5 = self.global_avg_pool(high_level_features)

        if self.inference:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        else:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest-exact')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.inference:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)),
                                       int(math.ceil(img.size()[-1]/4))), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)), int(math.ceil(img.size()[-1]/4))), mode='nearest-exact')

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        if self.inference:
            x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=img.size()[2:], mode='nearest-exact')

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)


# Class for ASPP Module -ConvNeXt Large
class ASPP_module_L(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_L, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=(kernel_size, kernel_size),
                                            stride=(1, 1), padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_ConvNeXt_L(nn.Module):
    def __init__(self, n_classes=1, os=16, inference=False):
        super(DeepLabv3_plus_ConvNeXt_L, self).__init__()

        # Define inference or not
        self.inference = inference

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_L(1536, 256, rate=rates[0])
        self.aspp2 = ASPP_module_L(1536, 256, rate=rates[1])
        self.aspp3 = ASPP_module_L(1536, 256, rate=rates[2])
        self.aspp4 = ASPP_module_L(1536, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1536, 256, (1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(256, track_running_stats=True),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(192, 48, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48, track_running_stats=True)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)))

        # Initialize weights
        self._init_weight()

    def forward(self, img, low_level_features, high_level_features):
        x1 = self.aspp1(high_level_features)
        x2 = self.aspp2(high_level_features)
        x3 = self.aspp3(high_level_features)
        x4 = self.aspp4(high_level_features)
        x5 = self.global_avg_pool(high_level_features)

        if self.inference:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        else:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest-exact')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.inference:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)),
                                       int(math.ceil(img.size()[-1]/4))), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(int(math.ceil(img.size()[-2]/4)), int(math.ceil(img.size()[-1]/4))), mode='nearest-exact')

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        if self.inference:
            x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=img.size()[2:], mode='nearest-exact')

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
