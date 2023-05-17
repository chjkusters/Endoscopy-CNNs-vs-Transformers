"""IMPORT PACKAGES"""
import torch.nn as nn
from torchinfo import summary
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


""""""""""""""""""""""""""
"""" DEFINE U-NET MODELS"""
""""""""""""""""""""""""""


def UNet(encoder_name, url, num_classes=1):

    # Define model
    if 'efficientnet' in encoder_name:
        model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=3, classes=1,
                         aux_params=dict(pooling='avg', dropout=None, activation=None, classes=num_classes))
    else:
        model = smp.Unet(encoder_name=encoder_name, in_channels=3, classes=1,
                         aux_params=dict(pooling='avg', dropout=None, activation=None, classes=num_classes))

        # Pre-process names of pre-trained weights
        pretrain_dict = model_zoo.load_url(url)
        pretrain_dict_keys = list(pretrain_dict.keys())
        for key in pretrain_dict_keys:
            if 'fc' not in key:
                pretrain_dict[key.replace('{}'.format(key), 'encoder.{}'.format(key))] = pretrain_dict[key]
                del pretrain_dict[key]
            elif 'fc' in key:
                del pretrain_dict[key]

        # Load weights to model
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict, strict=True)

    return model


""""""""""""""""""""""""""""""
"""" DEFINE U-NET++ MODELS"""
""""""""""""""""""""""""""""""


def UNetPP(encoder_name, url, num_classes=1):

    # Define model
    if 'efficientnet' in encoder_name:
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=3, classes=1,
                                 aux_params=dict(pooling='avg', dropout=None, activation=None, classes=num_classes))
    else:
        model = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=3, classes=1,
                                 aux_params=dict(pooling='avg', dropout=None, activation=None, classes=num_classes))

        # Pre-process names of pre-trained weights
        pretrain_dict = model_zoo.load_url(url)
        pretrain_dict_keys = list(pretrain_dict.keys())
        for key in pretrain_dict_keys:
            if 'fc' not in key:
                pretrain_dict[key.replace('{}'.format(key), 'encoder.{}'.format(key))] = pretrain_dict[key]
                del pretrain_dict[key]
            elif 'fc' in key:
                del pretrain_dict[key]

        # Load weights to model
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict and 'fc.weight' not in k and 'fc.bias' not in k:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict, strict=True)

    return model
