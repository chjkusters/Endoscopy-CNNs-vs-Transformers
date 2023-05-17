"""IMPORT PACKAGES"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


""""""""""""""""""""""""""""""""""""""
"""" DEFINE RESNET BACKBONE MODELS"""
""""""""""""""""""""""""""""""""""""""
# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py


# Class for creating BottleNeck Modules
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, track_running_stats=True)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        # downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


# Class for defining Block in ResNet
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1,
                               stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1,
                               stride=(stride, stride), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


# Class for Constructing complete ResNet Model
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3, pretrained=None, url=''):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

        # Initialize weights
        self._init_weight()

        # Define URL for pretrained weights
        self.url = url

        # Load pretrained weights if pretrained is True
        if pretrained:
            self._load_pretrained_model(pretrained)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        high_level_feat = x

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x, low_level_feat, high_level_feat

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion, track_running_stats=True)
            )

        layers.append(ResBlock(self.in_channels, planes, downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrained):

        # Define initialization
        if pretrained == 'ImageNet':
            pretrain_dict = model_zoo.load_url(self.url)
        elif pretrained == 'GastroNet':
            pretrain_dict = torch.load(os.path.join(os.getcwd(), 'pretrained', 'checkpoint_200ep_teacher_adapted.pth'))
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# Functions to create various different versions of ResNet
def ResNet50(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels, pretrained, url)


def ResNet101(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels, pretrained, url)


def ResNet152(num_classes, channels=3, pretrained=None, url=''):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels, pretrained, url)


""""""""""""""""""""""""""""""
"""" DEFINE SEGMENTATION MODELS"""
""""""""""""""""""""""""""""""
# https://github.com/MLearing/Pytorch-DeepLab-v3-plus


# Class for ASPP Module
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
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


class DeepLabv3_plus(nn.Module):
    def __init__(self, n_classes=1, os=16, inference=False):
        super(DeepLabv3_plus, self).__init__()

        # Define inference or not
        self.inference = inference

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, (1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(256, track_running_stats=True),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, track_running_stats=True)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48, track_running_stats=True)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                       nn.BatchNorm2d(256, track_running_stats=True),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1)))

        # Apply initialization of weights
        self._init_weight()

    def forward(self, img, low_level_features, high_level_features):
        x1 = self.aspp1(high_level_features)
        x2 = self.aspp2(high_level_features)
        x3 = self.aspp3(high_level_features)
        x4 = self.aspp4(high_level_features)
        x5 = self.global_avg_pool(high_level_features)

        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
        if self.inference:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        else:
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest-exact')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
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

        # For the sake of determinism in pytorch lightning, usage of nearest-exact is necessary
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
