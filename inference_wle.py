"""IMPORT PACKAGES"""
import os
import argparse
import json
import random
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from data.dataset_wle import read_inclusion, augmentations
from train_wle import check_cuda, find_best_model
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

    criteria['dev'] = {'modality': ['wle'],
                       'dataset': ['validation'],
                       'min_height': None,
                       'min_width': None
                       }

    criteria['test'] = {'modality': ['wle'],
                        'dataset': ['test'],
                        'min_height': None,
                        'min_width': None
                        }

    criteria['test-corrupt'] = {'dataset': ['test-corrupt'],
                                'min_height': None,
                                'min_width': None
                                }

    criteria['rejected'] = {'modality': ['wle'],
                            'min_height': None,
                            'min_width': None
                            }

    criteria['born'] = {'modality': ['wle'],
                        'min_height': None,
                        'min_width': None,
                        'mask_only': True
                        }

    return criteria


""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR INFERENCE """
""""""""""""""""""""""""""""""


def run(opt):

    # Test Device
    device = check_cuda()

    # Create model output database
    df = pd.DataFrame(columns=['Case', 'CLS', 'SEG', 'CLS Correct', 'SEG Correct'])
    logi = 0

    # Construct data
    criteria = get_data_inclusion_criteria()
    if DEFINE_SET == 'Val':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['dev'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Test-Corrupt':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['test-corrupt'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'Rejected-Quality' or DEFINE_SET == 'Rejected-Criteria' or DEFINE_SET == 'Rejected-AllFrames':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['rejected'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'BORN':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['born'])
        print('Found {} images...'.format(len(val_inclusion)))
    elif DEFINE_SET == 'ARGOS':
        val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['rejected'])
        print('Found {} images...'.format(len(val_inclusion)))
    else:
        raise Exception('Unrecognized DEFINE_SET: {}'.format(DEFINE_SET))

    # Construct transforms
    data_transforms = augmentations(opt=opt)

    # Construct Model and load weights
    model = Model(opt=opt, inference=True)
    best_index = find_best_model(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME), finetune=True)
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
    tp_cls, tn_cls, fp_cls, fn_cls = 0., 0., 0., 0.
    tp_seg, tn_seg, fp_seg, fn_seg = 0., 0., 0., 0.
    y_true, y_pred, y_mask_pred = list(), list(), list()

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
            has_mask = 0

            # Set has_mask for NDBE cases
            if label == np.array([0], dtype=np.float32):
                has_mask = 1

            # Open mask for neoplasia cases
            if len(mask) > 0:
                maskpath = random.choice(mask)
                mask_gt = Image.open(maskpath).convert('1')

                # IN THE TEST SET 2 MISMATCHES ARE OCCURRING WHAT TO DO WITH THIS!!!!!
                if mask_gt.size != image.size:
                    mask_gt = mask_gt.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
                else:
                    mask_gt = mask_gt.crop((roi[2], roi[0], roi[3], roi[1]))
                has_mask = 1
            # Create mask with all zeros when there are no available ones
            else:
                mask_np = np.zeros(image.size)
                mask_gt = Image.fromarray(mask_np, mode='RGB').convert('1')
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
            cls_pred = torch.sigmoid(cls_pred).cpu()
            seg_pred = torch.sigmoid(seg_pred).cpu()

            # Process classification prediction; positive prediction if exceed threshold = 0.5
            # print('Classification Score: {:.4f}'.format(cls_pred.item()))
            cls = cls_pred > 0.5
            cls = cls.squeeze(axis=0).item()

            # Process segmentation prediction; positive prediction if 1 pixel exceeds threshold = 0.5
            mask = seg_pred.squeeze(axis=0)
            mask_cls_logit = torch.max(mask)
            mask_cls = (torch.max(mask) > 0.5).item()
            # print('Maximal Segmentation Score: {:.4f}'.format(mask_cls_logit.item()))

            # Process segmentation prediction; Average Dice Score
            dice_score.update(seg_pred, mask_dice, torch.tensor(has_mask))

            # Append values to list
            y_pred.append(cls_pred.item())
            y_mask_pred.append(mask_cls_logit.item())

            # Update classification metrics
            tp_cls += (target * cls)
            tn_cls += ((1 - target) * (1 - cls))
            fp_cls += ((1 - target) * cls)
            fn_cls += (target * (1 - cls))

            # Update segmentation metrics
            tp_seg += (target * mask_cls)
            tn_seg += ((1 - target) * (1 - mask_cls))
            fp_seg += ((1 - target) * mask_cls)
            fn_seg += (target * (1 - mask_cls))

            # Add values to the dataframe
            cls_result = (cls == target)
            seg_result = (mask_cls == target)
            df.loc[logi] = [img_name,
                            round(cls_pred.item(), 5),
                            round(mask_cls_logit.item(), 5),
                            cls_result,
                            seg_result]
            logi += 1

            # Process predicted mask and save to specified folder
            mask = mask.permute(1, 2, 0)
            maskpred = np.array(mask * 255, dtype=np.uint8)
            maskpred_pil = Image.fromarray(cv2.cvtColor(maskpred, cv2.COLOR_GRAY2RGB), mode='RGB')

            # Make folders
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'wrong')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'wrong'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr'))
            if not os.path.exists(os.path.join(OUTPUT_PATH, 'output', 'correct')):
                os.makedirs(os.path.join(OUTPUT_PATH, 'output', 'correct'))
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

            #  Transform to heatmap
            heatmap = cv2.cvtColor(cv2.applyColorMap(maskpred, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmap = heatmap / 255.

            # Define alpha value
            alphavalue = 0.5
            alpha = np.where(maskpred > 0.1, alphavalue, 0.)

            # Process heatmap to PIL image, resize and convert to RGB
            heatmap = np.array(np.concatenate((heatmap, alpha), axis=-1) * 255, dtype=np.uint8)
            heatmap_pil = Image.fromarray(heatmap, mode='RGBA')
            w = int(image.size[0])
            h = int(image.size[1])
            heatmap_pil = heatmap_pil.resize(size=(w, h), resample=Image.NEAREST)
            heatmap_pil = heatmap_pil.convert('RGB')

            # Create original image with heatmap overlay
            composite = Image.blend(heatmap_pil, image, 0.6)
            draw = ImageDraw.Draw(composite)
            font = ImageFont.truetype('C:/Users/s157128/Documents/Roboto/Roboto-Regular.ttf', size=48)
            draw.text((0, 0), "Cls: {:.3f}, Seg:{:.3f}".format(cls_pred.item(), mask_cls_logit.item()),
                      (255, 255, 255), font=font)

            # Save the composite images in folders for wrong and correct classifications
            if mask_cls != target and cls != target:
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'wrong', '{:04d}_'.format(
                        int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'wrong', img_name + '.jpg'))
            elif mask_cls != target and cls == target:
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong',
                                 '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong', img_name + '.jpg'))
            elif mask_cls == target and cls != target:
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr',
                                 '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr', img_name + '.jpg'))
            else:
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'correct',
                                 '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'correct', img_name + '.jpg'))

    # Compute accuracy, sensitivity and specificity for classification
    accuracy_cls = (tp_cls + tn_cls) / (tp_cls + fn_cls + tn_cls + fp_cls)
    sensitivity_cls = tp_cls / (tp_cls + fn_cls)
    specificity_cls = tn_cls / (tn_cls + fp_cls + 1e-16)

    # Print accuracy, sensitivity and specificity
    print('\nClassification Performance')
    print('accuracy_cls: {:.4f}'.format(accuracy_cls))
    print('sensitivity_cls: {:.4f}'.format(sensitivity_cls))
    print('specificity_cls: {:.4f}'.format(specificity_cls + 1e-16))

    # Compute AUC for classification
    auc = roc_auc_score(y_true, y_pred)
    print('auc_cls: {:.4f}'.format(auc))
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot ROC curve for classification results and save to specified folder
    plt.plot(fpr, tpr, marker='.', label='Classification head')

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
    if DEFINE_SET == 'Val':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Validation Set')
    elif DEFINE_SET == 'Test':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Test Set')
    elif DEFINE_SET == 'Test-Corrupt':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Corrupt Test Set')
    elif DEFINE_SET == 'Rejected-Quality':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Rejected-Quality Set')
    elif DEFINE_SET == 'Rejected-Criteria':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Rejected-Criteria Set')
    elif DEFINE_SET == 'BORN':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'BORN Module Set')
    elif DEFINE_SET == 'ARGOS':
        CACHE_PATH = os.path.join(os.getcwd(), 'cache', '')
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'ARGOS Fuji Set')
    else:
        raise ValueError

    """EXTRACT PARAMETERS"""
    opt = get_params()

    """EXECUTE FUNCTIONS"""
    run(opt=opt)



