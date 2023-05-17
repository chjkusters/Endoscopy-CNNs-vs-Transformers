"""IMPORT PACKAGES"""
import torch
from torch import nn
import matplotlib.pyplot as plt

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


# Interobserver Binary Dice Metric
class InterObserverBinaryDiceMetric(nn.Module):
    def __init__(self, smooth=1e-6, p=1, label_smooth=0.01):
        super(InterObserverBinaryDiceMetric, self).__init__()
        self.smooth = smooth
        self.p = p
        self.dice_accumulator = list()
        self.margin = label_smooth * 0.5

    def update(self, preds, targets, has_mask):

        # Check whether the batch sizes of prediction adn target match
        assert preds.shape[0] == targets.shape[0], "pred & target batch size don't match"

        # Loop over the batch size
        for i in range(preds.shape[0]):

            # Specify predictions [X, 256, 256] and target [X, 256, 256]
            pred = preds[i, :, :, :]
            pred = torch.cat([pred] * targets.shape[1], dim=0)
            target = torch.round(targets[i, :, :, :], decimals=2)

            # Specify has_mask [X] and initialize weights [X]
            hm = (has_mask[i, :] if preds.shape[0] > 1 else has_mask)
            weights = torch.zeros([target.shape[0]])

            # Find number of unique values in the target masks without 0.0
            unique_values = torch.unique(target).tolist()
            if 0 in unique_values:
                unique_values.remove(0)

            # Threshold the predictions according to the unique values in the mask and reshape
            if len(unique_values) > 0:

                # # Plot unique values plot
                # if len(unique_values) == 1:
                #     fig, axs = plt.subplots(2)
                #     axs[0].imshow(pred[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[0].set_title('Max: {}'.format(round(torch.max(pred[0, :, :]).item(), 4)))
                #     axs[0].axis('off')
                #     axs[1].imshow(target[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[1].set_title('Max: {}'.format(round(torch.max(target[0, :, :]).item(), 4)))
                #     axs[1].axis('off')
                #     fig.suptitle('Non-Thresholded')
                #     # plt.tight_layout()
                #     plt.show()
                # else:
                #     fig, axs = plt.subplots(2, len(unique_values))
                #     for y in range(len(unique_values)):
                #         axs[0, y].imshow(pred[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[0, y].set_title('Max: {}'.format(round(torch.max(pred[y, :, :]).item(), 4)))
                #         axs[0, y].axis('off')
                #         axs[1, y].imshow(target[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[1, y].set_title('Max: {}'.format(round(torch.max(target[y, :, :]).item(), 4)))
                #         axs[1, y].axis('off')
                #     fig.suptitle('Non-Thresholded')
                #     # plt.tight_layout()
                #     plt.show()

                for j in range(len(unique_values)):
                    pred[j, :, :] = (pred[j, :, :] >= unique_values[j])
                    target[j, :, :] = (target[j, :, :] >= unique_values[j])
                    weights[j] = unique_values[j]

                # # Plot unique values plot
                # if len(unique_values) == 1:
                #     fig, axs = plt.subplots(2)
                #     axs[0].imshow(pred[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[0].set_title('Max: {}'.format(round(torch.max(pred[0, :, :]).item(), 4)))
                #     axs[0].axis('off')
                #     axs[1].imshow(target[0, :, :], vmin=0, vmax=1, cmap='gray')
                #     axs[1].set_title('Max: {}'.format(round(torch.max(target[0, :, :]).item(), 4)))
                #     axs[1].axis('off')
                #     fig.suptitle('Thresholded')
                #     # plt.tight_layout()
                #     plt.show()
                # else:
                #     fig, axs = plt.subplots(2, len(unique_values))
                #     for y in range(len(unique_values)):
                #         axs[0, y].imshow(pred[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[0, y].set_title('Max: {}'.format(round(torch.max(pred[y, :, :]).item(), 4)))
                #         axs[0, y].axis('off')
                #         axs[1, y].imshow(target[y, :, :], vmin=0, vmax=1, cmap='gray')
                #         axs[1, y].set_title('Max: {}'.format(round(torch.max(target[y, :, :]).item(), 4)))
                #         axs[1, y].axis('off')
                #     fig.suptitle('Thresholded')
                #     # plt.tight_layout()
                #     plt.show()

                pred = pred.contiguous().view(pred.shape[0], -1)
                target = target.contiguous().view(target.shape[0], -1)

                # Compute intersection between prediction and target
                intersection = torch.sum(torch.mul(pred, target), dim=1)

                # Compute the sum of predictions and target
                denominator = torch.sum(pred.pow(self.p), dim=1) + torch.sum(target.pow(self.p), dim=1)

                # Compute Dice Coefficient
                dice = torch.divide((2 * intersection), (denominator + self.smooth))
                # print(dice)

                # Optionally weight the individual dice scores
                # dice = torch.mul(dice, weights)

                # Multiply with has_mask to only have coefficient for samples with mask
                dice = torch.mul(dice, hm) / (torch.sum(hm) + self.smooth)
                dice = torch.sum(dice)
                # print(dice)

                # Update dice total accumulator
                self.dice_accumulator.append(dice)

                return dice
        else:
            return 0

    def compute(self):

        # Convert list to tensor and compute mean
        dice_accumulator = torch.FloatTensor(self.dice_accumulator)
        avg_dice_score = torch.mean(dice_accumulator)

        return avg_dice_score

    def reset(self):
        self.dice_accumulator = list()
