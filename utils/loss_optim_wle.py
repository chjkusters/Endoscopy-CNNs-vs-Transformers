"""IMPORT PACKAGES"""
import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

""""""""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR LOSS FUNCTION"""
""""""""""""""""""""""""""""""""""""""""""""""""


def construct_loss_function(opt):

    # Define possible choices for classification loss
    if opt.cls_criterion == 'BCE':
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.cls_criterion_weight], dtype=torch.float32))
    elif opt.cls_criterion == 'CE':
        cls_criterion = nn.CrossEntropyLoss()
    elif opt.cls_criterion == 'Focal':
        cls_criterion = FocalLoss_Cls(smooth=1e-6, alpha=opt.focal_alpha_cls, gamma=opt.focal_gamma_cls)
    else:
        raise Exception('Unexpected Classification Loss {}'.format(opt.cls_criterion))

    # Define possible choices for segmentation loss
    if opt.seg_criterion == 'Dice':
        seg_criterion = BinaryDiceLoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'DiceBCE':
        seg_criterion = DiceBCELoss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'DiceBCE_Weak':
        seg_criterion = DiceBCE_WeakSup_Loss(smooth=1e-6, p=1)
    elif opt.seg_criterion == 'IoU':
        seg_criterion = IoU_Loss(smooth=1e-6)
    elif opt.seg_criterion == 'Focal':
        seg_criterion = FocalLoss(smooth=1e-6, alpha=opt.focal_alpha_seg, gamma=opt.focal_gamma_seg)
    else:
        raise Exception('Unexpected Segmentation loss {}'.format(opt.seg_criterion))

    return cls_criterion, seg_criterion


# Custom Focal Loss for classification
class FocalLoss_Cls(nn.Module):
    def __init__(self, alpha, gamma, smooth=1e-6):
        super(FocalLoss_Cls, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute Binary Cross Entropy
        BCE = F.binary_cross_entropy(preds, target, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1. - BCE_EXP) ** self.gamma * BCE

        return focal_loss


# Custom Binary Dice Loss Function
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1. - torch.divide((2*intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        return dice_loss


# Custom DiceBCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1. - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        # Calculate BCE
        BCE = torch.mean(F.binary_cross_entropy(preds, target, reduction='none'), dim=1)
        BCE = torch.mul(BCE, has_mask) / (torch.sum(has_mask) + self.smooth)
        BCE = torch.sum(BCE)

        # Calculate combined loss
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


# Custom Jaccard/IoU Loss Function
class IoU_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoU_Loss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        total = torch.sum(preds, dim=1) + torch.sum(target, dim=1)

        # Compute the Union of the prediction and target
        union = total - intersection

        # Compute IoU Loss
        IoU = 1. - torch.divide((intersection + self.smooth), (union + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        IoU = torch.mul(IoU, has_mask) / (torch.sum(has_mask) + self.smooth)
        IoU = torch.sum(IoU)

        return IoU


# Custom Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute Binary Cross Entropy
        BCE = torch.mean(F.binary_cross_entropy(preds, target, reduction='none'), dim=1)
        BCE = torch.mul(BCE, has_mask) / (torch.sum(has_mask) + self.smooth)

        # Compute Focal Loss
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1. - BCE_EXP)**self.gamma * BCE
        focal_loss = torch.sum(focal_loss)

        return focal_loss


# Custom Weakly-Supervised Loss (Max. pixel value)
class DiceBCE_WeakSup_Loss(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(DiceBCE_WeakSup_Loss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target, has_mask, labels_cls, batch_idx):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation and maximum pixel values
        preds = self.sigmoid(preds)
        max_val, _ = torch.max(preds.view(preds.shape[0], -1), dim=1, keepdim=True)

        """SUPERVISED LOSS"""
        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target. Shape = [BS, ]
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of prediction and target. Shape = [BS, ]
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice loss of shape
        dice_loss = 1. - torch.divide((2 * intersection + self.smooth), (denominator + self.smooth))

        # Multiply with has_mask to only have loss for samples with mask. Shape = [BS]
        dice_loss = torch.mul(dice_loss, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice_loss = torch.sum(dice_loss)

        # Calculate BCE
        BCE_sup = torch.mean(F.binary_cross_entropy(preds, target, reduction='none'), dim=1)
        BCE_sup = torch.mul(BCE_sup, has_mask) / (torch.sum(has_mask) + self.smooth)
        BCE_sup = torch.sum(BCE_sup)

        # Calculate combined loss
        dice_BCE_sup = BCE_sup + dice_loss

        """SEMI-SUPERVISED LOSS"""
        BCE_semsup = torch.mean(F.binary_cross_entropy(max_val, labels_cls, reduction='none'), dim=1)
        BCE_semsup = torch.mean(BCE_semsup)

        """COMPUTE COMBINED LOSS"""
        comb_loss = dice_BCE_sup + BCE_semsup

        return comb_loss


""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR OPTIMIZER"""
""""""""""""""""""""""""""""""""""""""""""


def construct_optimizer(optim, parameters, lr):

    # Define possible choices
    if optim == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-07, amsgrad=True, weight_decay=1e-4)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise Exception('Unexpected Optimizer {}'.format(optim))

    return optimizer


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR LEARNING RATE SCHEDULER"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def construct_scheduler(schedule, optimizer, lr, metric="val_loss_combine"):

    # Define possible choices
    if schedule == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                               factor=0.1, patience=10, min_lr=lr/1000)

        return {"scheduler": scheduler,
                "monitor": metric,
                "interval": "epoch"}

    elif schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

        return {"scheduler": scheduler,
                "interval": "epoch"}

    elif schedule == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2, T_mult=2,
                                                                         eta_min=lr/1000, last_epoch=-1)

        return {"scheduler": scheduler,
                "interval": "epoch"}

    else:
        return None
