"""IMPORT PACKAGES"""
import os
import argparse
import json
import random
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import torch
import torchmetrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from data.dataset_public import read_inclusion_kvasir, augmentations_kvasir
from data.dataset_public import read_inclusion_giana, augmentations_giana
from train_public import check_cuda, find_best_model_kvasir, find_best_model_giana
from models.model_wle import Model
from utils.metrics_wle import BinaryDiceMetricEval


""""""""""""""""""""""""
"""" HELPER FUNCTIONS """
""""""""""""""""""""""""


# Make function for defining parameters
def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)

    # EXTRACT INFORMATION FROM PARAMETERS USED IN EXPERIMENT
    f = open(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'params.json'))
    data = json.load(f)
    parser.add_argument('--backbone', type=str, default=data['backbone'])
    parser.add_argument('--seg_branch', type=str, default=data['seg_branch'])
    parser.add_argument('--imagesize', type=int, default=data['imagesize'])
    parser.add_argument('--num_classes', type=str, default=data['num_classes'])
    parser.add_argument('--label_smoothing', type=float, default=data['label_smoothing'])

    args = parser.parse_args()

    return args


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria['dev'] = {'dataset': ['validation'],
                       'min_height': None,
                       'min_width': None
                       }

    criteria['test'] = {'dataset': ['test'],
                        'min_height': None,
                        'min_width': None
                        }

    criteria['test-corrupt'] = {'dataset': ['test-corrupt'],
                                'min_height': None,
                                'min_width': None
                                }

    return criteria


""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR INFERENCE """
""""""""""""""""""""""""""""""


def run_kvasir(opt):

    # Test Device
    device = check_cuda()

    # Construct data
    criteria = get_data_inclusion_criteria()
    if DEFINE_SET == 'Val':
        val_inclusion = read_inclusion_kvasir(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test':
        val_inclusion = read_inclusion_kvasir(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test-Corrupt':
        val_inclusion = read_inclusion_kvasir(path=CACHE_PATH, criteria=criteria['test-corrupt'])
        print('Found {} images...'.format(len(val_inclusion)))
    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(DEFINE_SET))

    # Construct transforms
    data_transforms = augmentations_kvasir(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt, inference=True)
    best_index = find_best_model_kvasir(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    checkpoint = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    checkpoint_keys = list(checkpoint.keys())
    for key in checkpoint_keys:
        checkpoint[key.replace('model.', '')] = checkpoint[key]
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=True)

    # Save final model as .pt file
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize metrics
    dice_score = BinaryDiceMetricEval()

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():

        # Loop over the data
        for img in val_inclusion:

            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']
            mask = img['mask']

            # Construct Opening print line
            print('\nOpening image: {}'.format(img_name))

            # Open Image
            image = Image.open(file).convert('RGB')

            # By default set has_mask to zero
            has_mask = 1

            # Open mask for neoplasia cases
            if len(mask) > 0:
                maskpath = random.choice(mask)
                mask_gt = Image.open(maskpath).convert('1')

                # Address mismatches
                if mask_gt.size != image.size:
                    mask_gt = mask_gt.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
                else:
                    mask_gt = mask_gt.crop((roi[2], roi[0], roi[3], roi[1]))

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t, mask_gt = data_transforms['test'](image, mask_gt, has_mask)
            image_t = image_t.unsqueeze(0).cuda()
            mask_dice = mask_gt.unsqueeze(0)

            # Get prediction of model and perform Sigmoid activation
            # cls_pred, seg_pred = model(image_t)
            out1, out2 = model(image_t)
            cls_pred = (out1 if out1.dim() == 2 else out2)
            seg_pred = (out2 if out2.dim() == 4 else out1)
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)
            print('Maximal Segmentation Score: {:.4f}'.format(mask_cls_logit.item()))

            # Process segmentation prediction; Average Dice Score
            dice_score.update(seg_pred, mask_dice, torch.tensor(has_mask))

            # Process predicted mask and save to specified folder
            mask = mask.permute(1, 2, 0)
            maskpred = np.array(mask * 255, dtype=np.uint8)
            maskpred_pil = Image.fromarray(cv2.cvtColor(maskpred, cv2.COLOR_GRAY2RGB), mode='RGB')

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'masks')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'masks'))

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(mask_gt[0, :, :], cmap='gray')
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            ax2.imshow(maskpred_pil)
            ax2.axis('off')
            ax2.set_title('Generated Mask')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, 'output', 'masks', img_name + '.png'), bbox_inches='tight')
            plt.close()

    # Compute Avg Dice for segmentation
    avg_dice = dice_score.compute()
    print('\nSegmentation Performance')
    print('avg_dice_seg: {:.4f}'.format(avg_dice.item()))


def run_giana(opt):

    # Test Device
    device = check_cuda()

    # Create model output database
    df = pd.DataFrame(columns=['Case', 'CLS', 'SEG', 'CLS Correct', 'SEG Correct'])
    logi = 0

    # Construct data
    criteria = get_data_inclusion_criteria()
    if DEFINE_SET == 'Val':
        val_inclusion = read_inclusion_giana(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test':
        val_inclusion = read_inclusion_giana(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test-Corrupt':
        val_inclusion = read_inclusion_giana(path=CACHE_PATH, criteria=criteria['test-corrupt'])
        print('Found {} images...'.format(len(val_inclusion)))
    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(DEFINE_SET))

    # Construct transforms
    data_transforms = augmentations_giana(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt, inference=True)
    best_index = find_best_model_giana(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    checkpoint = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index))['state_dict']

    # Adapt state_dict keys (remove model. from the key and save again)
    checkpoint_keys = list(checkpoint.keys())
    for key in checkpoint_keys:
        checkpoint[key.replace('model.', '')] = checkpoint[key]
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=True)

    # Save final model as .pt file
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize metrics
    dice_score = BinaryDiceMetricEval()
    auc_score = torchmetrics.AUROC(num_classes=3)
    acc_score = torchmetrics.Accuracy(num_classes=3)
    tp_seg, tn_seg, fp_seg, fn_seg = 0., 0., 0., 0.
    y_true, y_mask_pred = list(), list()

    # Push model to GPU and set in evaluation mode
    model.cuda()
    model.eval()
    with torch.no_grad():

        # Loop over the data
        for img in val_inclusion:

            # Extract information from cache
            file = img['file']
            img_name = os.path.splitext(os.path.split(file)[1])[0]
            roi = img['roi']
            mask = img['mask']

            # Construct target
            label = img['label']
            if label:
                target = True
                y_true.append(target)
            else:
                target = False
                y_true.append(target)

            # Construct Opening print line
            print('\nOpening image: {}'.format(img_name))

            # Open Image
            image = Image.open(file).convert('RGB')

            # By default set has_mask to zero
            has_mask = 1

            # Open mask for neoplasia cases
            if len(mask) > 0:
                maskpath = random.choice(mask)
                mask_gt = Image.open(maskpath).convert('1')

                # Address size mismatch
                if mask_gt.size != image.size:
                    mask_gt = mask_gt.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
                else:
                    mask_gt = mask_gt.crop((roi[2], roi[0], roi[3], roi[1]))

            # Crop the image to the ROI
            image = image.crop((roi[2], roi[0], roi[3], roi[1]))

            # Apply transforms to image and mask
            image_t, mask_gt = data_transforms['test'](image, mask_gt, has_mask)
            image_t = image_t.unsqueeze(0).cuda()
            mask_dice = mask_gt.unsqueeze(0)

            # Get prediction of model and perform Sigmoid activation
            # cls_pred, seg_pred = model(image_t)
            out1, out2 = model(image_t)
            cls_pred = (out1 if out1.dim() == 2 else out2)
            seg_pred = (out2 if out2.dim() == 4 else out1)
            cls_pred = torch.softmax(cls_pred, dim=1).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process classification prediction
            cls_ind, cls_val = torch.argmax(cls_pred), torch.max(cls_pred)
            print('Classification Score: {:.4f}, Classification Index: {}, Ground Truth Label: {}'.format(cls_val.item(), cls_ind.item(), label.item()))

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)
            mask_cls = (torch.max(mask) > 0.5).item()
            print('Maximal Segmentation Score: {:.4f}'.format(mask_cls_logit.item()))
            y_mask_pred.append(mask_cls_logit.item())

            # Process segmentation prediction; Average Dice Score
            dice_score.update(seg_pred, mask_dice, torch.tensor(has_mask))

            # Update classification metrics
            auc_score.update(cls_pred, torch.tensor([label.item()]))
            acc_score.update(cls_pred, torch.tensor([label.item()]))

            # Update segmentation metrics
            tp_seg += (target * mask_cls)
            tn_seg += ((1 - target) * (1 - mask_cls))
            fp_seg += ((1 - target) * mask_cls)
            fn_seg += (target * (1 - mask_cls))

            # Add values to the dataframe
            cls_result = (cls_ind.item() == label.item())
            seg_result = (mask_cls == target)
            df.loc[logi] = [img_name,
                            round(cls_val.item(), 5),
                            round(mask_cls_logit.item(), 5),
                            cls_result,
                            seg_result]
            logi += 1

            # Process predicted mask and save to specified folder
            mask = mask.permute(1, 2, 0)
            maskpred = np.array(mask * 255, dtype=np.uint8)
            maskpred_pil = Image.fromarray(cv2.cvtColor(maskpred, cv2.COLOR_GRAY2RGB), mode='RGB')

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'figures')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'figures'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'masks')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'masks'))

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(mask_gt[0, :, :], cmap='gray')
            ax1.set_title('Ground Truth')
            ax1.axis('off')
            ax2.imshow(maskpred_pil)
            ax2.axis('off')
            ax2.set_title('Generated Mask')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, 'output', 'masks', img_name + '.png'), bbox_inches='tight')
            plt.close()

    # Print accuracy
    accuracy_cls = acc_score.compute()
    print('\nClassification Performance')
    print('accuracy_cls: {:.4f}'.format(accuracy_cls))

    # Compute AUC for classification
    auc = auc_score.compute()
    print('auc_cls: {:.4f}'.format(auc))

    # Compute accuracy, sensitivity and specificity for segmentation
    accuracy_seg = (tp_seg + tn_seg) / (tp_seg + fn_seg + tn_seg + fp_seg)
    sensitivity_seg = tp_seg / (tp_seg + fn_seg)
    specificity_seg = tn_seg / (tn_seg + fp_seg)

    # Print accuracy, sensitivity and specificity for segmentation
    print('\nSegmentation Performance')
    print('accuracy_seg: {:.4f}'.format(accuracy_seg))
    print('sensitivity_seg: {:.4f}'.format(sensitivity_seg))
    print('specificity_seg: {:.4f}'.format(specificity_seg))

    # Compute AUC for segmentation
    auc = roc_auc_score(y_true, y_mask_pred)
    print('auc_seg: {:.4f}'.format(auc))
    fpr, tpr, _ = roc_curve(y_true, y_mask_pred)

    # Compute Avg Dice for segmentation
    avg_dice = dice_score.compute()
    print('avg_dice_seg: {:.4f}'.format(avg_dice.item()))

    # Plot ROC curve for segmentation results and save to specified folder
    plt.plot(fpr, tpr, marker='.', label='Max segmentation Value')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    major_ticks = np.arange(0., 1.01, 0.05)
    plt.xticks(major_ticks, fontsize='x-small')
    plt.yticks(major_ticks)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.grid(True)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.title('ROC AUC')
    plt.savefig(os.path.join(OUTPUT_PATH, 'figures', 'auc_curve.jpg'))
    plt.close()

    # Save dataframe as csv file
    df.to_excel(os.path.join(OUTPUT_PATH, 'output', 'cls_scores.xlsx'))


""""""""""""""""""
"""" EXECUTION """
""""""""""""""""""


if __name__ == '__main__':

    """"ADJUSTABLE PARAMETERS"""
    EXPERIMENT_NAME = 'Experiment1'
    DEFINE_SET = 'Val'

    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), 'experiments')

    """SPECIFY PATH FOR CACHE"""
    if 'kvasir' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Val':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Validation Set')
    elif 'kvasir' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Test Set')
    elif 'kvasir' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test-Corrupt':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Corrupt Test Set')
    elif 'sysucc' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Val':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Validation Set')
    elif 'sysucc' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Test Set')
    elif 'sysucc' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test-Corrupt':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Corrupt Test Set')
    elif 'giana' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Val':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Validation Set')
    elif 'giana' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Test Set')
    elif 'giana' in EXPERIMENT_NAME.lower() and DEFINE_SET == 'Test-Corrupt':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Corrupt Test Set')
    else:
        raise ValueError

    """EXTRACT PARAMETERS"""
    opt = get_params()

    """EXECUTE FUNCTIONS"""
    if 'kvasir' in EXPERIMENT_NAME.lower():
        run_kvasir(opt=opt)
    elif 'giana' in EXPERIMENT_NAME.lower():
        run_giana(opt=opt)
    else:
        raise ValueError
