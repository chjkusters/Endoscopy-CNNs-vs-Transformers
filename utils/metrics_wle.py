"""IMPORT PACKAGES"""
import torch
from torch import nn

""""""""""""""""""""""""
"""" CUSTOM METRICS"""
""""""""""""""""""""""""
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook


# Binary Dice Coefficient for Training (Batches)
class BinaryDiceMetric(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(BinaryDiceMetric, self).__init__()
        self.smooth = smooth
        self.p = p
        self.dice_accumulator = list()

    def update(self, preds, target, has_mask):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > 0.5
        target = target.contiguous().view(target.shape[0], -1) > 0.5

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice Coefficient
        dice = torch.divide((2*intersection), (denominator + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice = torch.sum(dice)
        self.dice_accumulator.append(dice)

    def compute(self):

        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.mean(dice_accumulator)

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()


# Binary Dice Coefficient for Evaluation (Single images)
class BinaryDiceMetricEval(nn.Module):
    def __init__(self, smooth=1e-6, p=1):
        super(BinaryDiceMetricEval, self).__init__()
        self.smooth = smooth
        self.p = p
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0

    def update(self, preds, target, has_mask):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1) > 0.5
        target = target.contiguous().view(target.shape[0], -1) > 0.5

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        denominator = torch.sum(preds.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

        # Compute Dice Coefficient
        dice = torch.divide((2*intersection), (denominator + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        dice = torch.mul(dice, has_mask) / (torch.sum(has_mask) + self.smooth)
        dice = torch.sum(dice)
        self.dice_accumulator.append(dice)
        self.has_mask_accumulator += torch.any(target)

    def compute(self):

        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.sum(dice_accumulator)/self.has_mask_accumulator

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()
        self.has_mask_accumulator = 0


# Jaccard Index / Intersection over Union
class IoU_Metric(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoU_Metric, self).__init__()
        self.smooth = smooth
        self.IoU_accumulator = list()

    def update(self, preds, target, has_mask):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target shape don't match"

        # Flatten the prediction and target. Shape = [BS, c*h*w]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute intersection between prediction and target
        intersection = torch.sum(torch.mul(preds, target), dim=1)

        # Compute the sum of predictions and target
        total = torch.sum(preds, dim=1) + torch.sum(target, dim=1)

        # Compute the Union of the prediction and target
        union = total - intersection

        # Compute IoU
        IoU = torch.divide((intersection + self.smooth), (union + self.smooth))

        # Multiply with has_mask to only have coefficient for samples with mask
        IoU = torch.mul(IoU, has_mask) / (torch.sum(has_mask) + self.smooth)
        IoU = torch.sum(IoU)
        self.IoU_accumulator.append(IoU)

    def compute(self):

        # Convert list to tensor and compute mean
        IoU_accumulator = torch.FloatTensor(self.IoU_accumulator)
        avg_IoU_score = torch.mean(IoU_accumulator)

        return avg_IoU_score

    def reset(self):
        self.IoU_accumulator = list()
