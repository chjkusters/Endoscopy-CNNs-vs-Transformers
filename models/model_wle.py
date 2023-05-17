"""IMPORT PACKAGES"""
import torch.nn as nn

# Import helper functions from other files
from ResNet import ResNet50, ResNet101, ResNet152, DeepLabv3_plus
from ConvNeXt import convnext_tiny, convnext_small, convnext_base, convnext_large
from ConvNeXt import DeepLabv3_plus_ConvNeXt_TS, DeepLabv3_plus_ConvNeXt_B, DeepLabv3_plus_ConvNeXt_L
from UNet import UNet, UNetPP
from CaraNet import CaraNet
from GMSRFNet import GMSRFNet
from FCBFormer import FCBFormer
from ESFPNet import ESFPNetStructure
from SwinUNet import SwinUnet
from SwinUperNet import SwinUperNet


""""""""""""""""""""""""""""""""""""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
""""""""""""""""""""""""""""""""""""


class Model(nn.Module):
    def __init__(self, opt, inference):
        super(Model, self).__init__()

        # Define Backbone architecture
        if opt.backbone == 'ResNet-50-ImageNet':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = ResNet50(num_classes=opt.num_classes, channels=3, pretrained='ImageNet', url=url)
        elif opt.backbone == 'ResNet-50-GastroNet':
            self.backbone = ResNet50(num_classes=opt.num_classes, channels=3, pretrained='GastroNet', url='')
        elif opt.backbone == 'ResNet-101':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = ResNet101(num_classes=opt.num_classes, channels=3, pretrained='ImageNet', url=url)
        elif opt.backbone == 'ResNet-152':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = ResNet152(num_classes=opt.num_classes, channels=3, pretrained='ImageNet', url=url)
        elif opt.backbone == 'ConvNeXt-T':
            self.backbone = convnext_tiny(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-S':
            self.backbone = convnext_small(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-B':
            self.backbone = convnext_base(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ConvNeXt-L':
            self.backbone = convnext_large(pretrained=True, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-50-UNet':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = UNet(encoder_name='resnet50', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-101-UNet':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = UNet(encoder_name='resnet101', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-152-UNet':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = UNet(encoder_name='resnet152', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'EfficientNetB0-UNet':
            self.backbone = UNet(encoder_name='efficientnet-b0', url='', num_classes=opt.num_classes)
        elif opt.backbone == 'EfficientNetB6-UNet':
            self.backbone = UNet(encoder_name='efficientnet-b6', url='', num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-50-UNet++':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = UNetPP(encoder_name='resnet50', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-101-UNet++':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = UNetPP(encoder_name='resnet101', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-152-UNet++':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = UNetPP(encoder_name='resnet152', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'EfficientNetB0-UNet++':
            self.backbone = UNetPP(encoder_name='efficientnet-b0', url='', num_classes=opt.num_classes)
        elif opt.backbone == 'EfficientNetB6-UNet++':
            self.backbone = UNetPP(encoder_name='efficientnet-b6', url='', num_classes=opt.num_classes)
        elif opt.backbone == 'CaraNet':
            self.backbone = CaraNet(inference=inference, num_classes=opt.num_classes)
        elif opt.backbone == 'GMSRFNet':
            self.backbone = GMSRFNet()
        elif 'FCBFormer' in opt.backbone:
            self.backbone = FCBFormer(opt=opt)
        elif 'ESFPNet' in opt.backbone:
            self.backbone = ESFPNetStructure(opt=opt, inference=inference)
        elif 'Swin' in opt.backbone and 'UNet' in opt.backbone:
            self.backbone = SwinUnet(opt=opt)
        elif 'Swin' in opt.backbone and 'UperNet':
            self.backbone = SwinUperNet(opt=opt, inference=inference)
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))

        # Define segmentation branch architecture
        if opt.seg_branch == 'DeepLabV3p':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus(n_classes=1, inference=inference)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-TS':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_TS(n_classes=1, inference=inference)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-B':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_B(n_classes=1, inference=inference)
        elif opt.seg_branch == 'DeepLabV3p-ConvNeXt-L':
            self.single_model = False
            self.seg_branch = DeepLabv3_plus_ConvNeXt_L(n_classes=1, inference=inference)
        elif opt.seg_branch is None and 'UNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'CaraNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'GMSRFNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'FCBFormer' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'ESFPNet' in opt.backbone:
            self.single_model = True
        elif opt.seg_branch is None and 'Swin' in opt.backbone and 'UperNet' in opt.backbone:
            self.single_model = True
        else:
            raise Exception('Unexpected Segmentation Branch {}'.format(opt.seg_branch))

    def forward(self, img):

        if self.single_model:

            # Output of single model
            cls, seg = self.backbone(img)

        else:

            # Backbone output
            cls, low_level, high_level = self.backbone(img)

            # Segmentation output
            seg = self.seg_branch(img, low_level, high_level)

        return cls, seg
