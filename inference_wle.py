"""IMPORT PACKAGES"""
import os
import argparse
import time
import json
import copy
import random
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.metrics import roc_curve, roc_auc_score

from dataset_wle import read_inclusion, augmentations
from train_wle import check_cuda, find_best_model
from model_wle import Model
# from metrics_wle import BinaryDiceMetric, InterObserverBinaryDiceMetric
from metrics_wle import BinaryDiceMetricEval
from interobserver_segmentation import InterObserverBinaryDiceMetric, MSEMetric, InterObserverMSEMetric

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
                # composite.save(
                #     os.path.join(OUTPUT_PATH, 'output', 'wrong', '{:04d}_'.format(
                #         int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'wrong', img_name + '.jpg'))
            elif mask_cls != target and cls == target:
                # composite.save(
                #     os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong',
                #                  '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls corr-seg wrong', img_name + '.jpg'))
            elif mask_cls == target and cls != target:
                # composite.save(
                #     os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr',
                #                  '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
                composite.save(
                    os.path.join(OUTPUT_PATH, 'output', 'cls wrong-seg corr', img_name + '.jpg'))
            else:
                # composite.save(
                #     os.path.join(OUTPUT_PATH, 'output', 'correct',
                #                  '{:04d}_'.format(int(mask_cls_logit.item() * 1000)) + img_name + '.jpg'))
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


def run_interobserver(opt):

    # Construct data
    criteria = get_data_inclusion_criteria()
    # if DEFINE_SET == 'Val':
    #     val_inclusion = read_inclusion(path=CACHE_PATH_AM, criteria=criteria['dev'])
    # elif DEFINE_SET == 'Test':
    #     val_inclusion = read_inclusion(path=CACHE_PATH_AM, criteria=criteria['test'])
    # elif DEFINE_SET == 'Rejected-Quality' or DEFINE_SET == 'Rejected-Criteria' or DEFINE_SET == 'Rejected-AllFrames':
    #     val_inclusion = read_inclusion(path=CACHE_PATH_AM, criteria=criteria['rejected'])
    # elif DEFINE_SET == 'BORN':
    #     val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['born'])
    # elif DEFINE_SET == 'ARGOS':
    #     val_inclusion = read_inclusion(path=CACHE_PATH, criteria=criteria['rejected'])
    #     print('Found {} images...'.format(len(val_inclusion)))
    # else:
    #     raise Exception('Unrecognized DEFINE_SET: {}'.format(DEFINE_SET))

    path = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/example cache'
    val_inclusion = read_inclusion(path=path, criteria=criteria['rejected'])
    print('Found {} images...'.format(len(val_inclusion)))

    # Construct transforms
    data_transforms = augmentations(opt=opt)
    tensor_to_pil = transforms.ToPILImage()

    # Construct Model and load weights
    model = Model(opt=opt, inference=True)
    weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Initialize metric
    dice_score = InterObserverBinaryDiceMetric(label_smooth=opt.label_smoothing)

    # Amount of channels for comparison
    comp_channels = 3

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
            masklist = img['mask']
            label = img['label']

            # Open Image
            image = Image.open(file).convert('RGB')

            # By default set has_mask to zero
            has_mask = torch.zeros([comp_channels])

            # Set the first entry of has_mask for NDBE cases
            if label == np.array([0], dtype=np.float32):
                has_mask[0] = 1

            # Open mask for neoplasia cases
            if len(masklist) > 0:

                # Create empty average mask
                mask_avg = 0

                # Extract information on expert
                expert_list = list(set([os.path.split(os.path.split(masklist[i])[0])[1] for i in range(len(masklist))]))
                expert_list.sort()

                # Load all the individual masks
                lower0, higher0, lower1, higher1 = False, False, False, False
                ll0, hl0, ll1, hl1 = 0, 0, 0, 0
                for i in range(len(masklist)):
                    expert = os.path.split(os.path.split(masklist[i])[0])[1]
                    likelihood = os.path.split(os.path.split(os.path.split(masklist[i])[0])[0])[1]
                    if expert_list.index(expert) == 0 and 'Lower' in likelihood:
                        lower0 = True
                        ll0 = Image.open(masklist[i]).convert('1')
                        if ll0.size != image.size:
                            ll0 = np.array(ll0.resize(image.size, resample=Image.NEAREST).
                                           crop((roi[2], roi[0], roi[3], roi[1])))
                        else:
                            ll0 = np.array(ll0.crop((roi[2], roi[0], roi[3], roi[1])))
                    elif expert_list.index(expert) == 0 and 'Higher' in likelihood:
                        hl0 = Image.open(masklist[i]).convert('1')
                        higher0 = True
                        if hl0.size != image.size:
                            hl0 = np.array(hl0.resize(image.size, resample=Image.NEAREST).
                                           crop((roi[2], roi[0], roi[3], roi[1])))
                        else:
                            hl0 = np.array(hl0.crop((roi[2], roi[0], roi[3], roi[1])))
                    elif expert_list.index(expert) == 1 and 'Lower' in likelihood:
                        ll1 = Image.open(masklist[i]).convert('1')
                        lower1 = True
                        if ll1.size != image.size:
                            ll1 = np.array(ll1.resize(image.size, resample=Image.NEAREST).
                                           crop((roi[2], roi[0], roi[3], roi[1])))
                        else:
                            ll1 = np.array(ll1.crop((roi[2], roi[0], roi[3], roi[1])))
                    elif expert_list.index(expert) == 1 and 'Higher' in likelihood:
                        hl1 = Image.open(masklist[i]).convert('1')
                        higher1 = True
                        if hl1.size != image.size:
                            hl1 = np.array(hl1.resize(image.size, resample=Image.NEAREST).
                                           crop((roi[2], roi[0], roi[3], roi[1])))
                        else:
                            hl1 = np.array(hl1.crop((roi[2], roi[0], roi[3], roi[1])))
                    else:
                        raise ValueError('More than 2 experts...')

                # Replace LL with LL U HL if they both exist to enforce the protocol
                if lower0 and higher0:
                    ll0 = np.bitwise_or(ll0, hl0)
                if lower1 and higher1:
                    ll1 = np.bitwise_or(ll1, hl1)

                # Produce average mask
                mask_avg += ll0 / len(masklist)
                mask_avg += hl0 / len(masklist)
                mask_avg += ll1 / len(masklist)
                mask_avg += hl1 / len(masklist)

                # # Plot averaged mask
                # plt.figure()
                # plt.imshow(mask_avg, vmin=0, vmax=1, cmap='gray')
                # plt.title('Averaged Mask')
                # plt.axis('off')
                # plt.tight_layout()
                # plt.show()

                # fig, axs = plt.subplots(1, 3)
                # axs[0].imshow(image.crop((roi[2], roi[0], roi[3], roi[1])))
                # axs[0].axis('off')
                # axs[1].imshow(mask_avg, vmin=0, vmax=1, cmap='gray')
                # axs[1].axis('off')

                # Create compound X-channel mask and adapt has_mask
                mask = np.zeros([comp_channels, mask_avg.shape[0], mask_avg.shape[1]])
                unique_values = list(np.unique(mask_avg))
                unique_values.remove(0)
                if 1 in unique_values:
                    unique_values.remove(1)
                for j in range(len(unique_values)):
                    mask[j, :, :] = (mask_avg >= unique_values[j]) * unique_values[j]
                    has_mask[j] = 1

                # Convert back to tensor
                mask = torch.from_numpy(mask)

                # Crop the image to the ROI
                image = image.crop((roi[2], roi[0], roi[3], roi[1]))

                # Prepare Image and Mask for inference
                mask = tensor_to_pil(mask)
                image_t, mask_gt = data_transforms['test'](image, mask, has_mask)
                image_t = image_t.unsqueeze(0).cuda()
                mask_dice = mask_gt.unsqueeze(0)

                # Get prediction of model and perform Sigmoid activation
                out1, out2 = model(image_t)
                seg_pred = (out2 if out2.dim() == 4 else out1)
                seg_pred = torch.sigmoid(seg_pred).cpu()

                """INTEROBSERVER DICE SCORE COMPUTATION"""
                int_dice = dice_score.update(seg_pred, mask_dice, has_mask)

                # # Process predicted mask and save to specified folder
                # mask = seg_pred.squeeze(axis=0)
                # mask_cls_logit = torch.max(mask)
                # mask = mask.permute(1, 2, 0)
                # maskpred = np.array(mask * 255, dtype=np.uint8)
                #
                # #  Transform to heatmap
                # heatmap = cv2.cvtColor(cv2.applyColorMap(maskpred, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                # heatmap = heatmap / 255.
                #
                # # Define alpha value
                # alphavalue = 0.5
                # alpha = np.where(maskpred > 0.1, alphavalue, 0.)
                #
                # # Process heatmap to PIL image, resize and convert to RGB
                # heatmap = np.array(np.concatenate((heatmap, alpha), axis=-1) * 255, dtype=np.uint8)
                # heatmap_pil = Image.fromarray(heatmap, mode='RGBA')
                # w = int(image.size[0])
                # h = int(image.size[1])
                # heatmap_pil = heatmap_pil.resize(size=(w, h), resample=Image.NEAREST)
                # heatmap_pil = heatmap_pil.convert('RGB')
                #
                # # Create original image with heatmap overlay
                # composite = Image.blend(heatmap_pil, image, 0.6)
                # draw = ImageDraw.Draw(composite)
                # font = ImageFont.truetype('C:/Users/s157128/Documents/Roboto/Roboto-Regular.ttf', size=48)
                # draw.text((0, 0), "Seg:{:.3f}".format(mask_cls_logit.item()),
                #           (255, 255, 255), font=font)

                # plt.imshow(composite)
                # plt.tight_layout()
                # plt.axis('off')
                # plt.show()

                # save_path = 'C:/Users/s157128/OneDrive - TU Eindhoven/PhD/Vervolg Projecten/Brainstorm/Metric/examples gelaagd Dice'
                # axs[2].imshow(composite)
                # plt.axis('off')
                # plt.tight_layout()
                # plt.suptitle('Gelaagd Dice: {:.4f}'.format(int_dice))
                # # plt.show()
                # plt.savefig(os.path.join(save_path, img_name + '.jpg'))

    """INTEROBSERVER DICE SCORE"""
    avg_dice = dice_score.compute()
    print('interobserver_dice: {:.4f}'.format(avg_dice.item()))


def fps_check(opt, use_cuda=True):

    # Test Device
    if use_cuda:
        device = check_cuda()

    # Construct Model and load weights
    model = Model(opt=opt, inference=True)
    weights = torch.load(os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'final_pytorch_model.pt'))
    model.load_state_dict(weights, strict=True)

    # Create random dummy input
    num_samples = 500
    if use_cuda:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize).cuda()
    else:
        dummy = torch.rand(1, 3, opt.imagesize, opt.imagesize)

    # Push model to GPU and set in evaluation mode
    if use_cuda:
        model.cuda()
    model.eval()
    with torch.no_grad():

        starttime = time.time()
        for i in range(num_samples):
            cls_pred, seg_pred = model(dummy)
        stoptime = time.time()

    inference_time = stoptime - starttime
    avg_inference_sample = inference_time / num_samples
    avg_inference_fps = 1/avg_inference_sample

    print('Average Inference Time per sample: {} sec.'.format(avg_inference_sample))
    print('Average fps: {}'.format(avg_inference_fps))


""""""""""""""""""
"""" EXECUTION """
""""""""""""""""""


if __name__ == '__main__':

    """"ADJUSTABLE PARAMETERS"""
    EXPERIMENT_NAME = 'Real-Exp19-CNN-EfficientNetB6-UNet++-5050-12.5%'
    DEFINE_SET = 'Test-Corrupt'

    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/experiments'

    """SPECIFY PATH FOR CACHE"""
    if DEFINE_SET == 'Val':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_plausible_nodup'
        CACHE_PATH_AM = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_all_masks_nodup'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Validation Set')
    elif DEFINE_SET == 'Test':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_plausible_nodup'
        CACHE_PATH_AM = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_all_masks_nodup'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Test Set')
    elif DEFINE_SET == 'Test-Corrupt':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_corrupt_test'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Corrupt Test Set')
    elif DEFINE_SET == 'Rejected-Quality':
        CACHE_PATH = 'D:/Python Scripts - Open Research/training+rejected_cache_no_duplicates/cache_wle_rejected_quality_plausible_nodup'
        CACHE_PATH_AM = 'D:/Python Scripts - Open Research/training+rejected_cache_no_duplicates/cache_wle_rejected_quality_all_masks_nodup'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Rejected-Quality Set')
    elif DEFINE_SET == 'Rejected-Criteria':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_rejected_criteria_plausible_nodup'
        CACHE_PATH_AM = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_rejected_criteria_all_masks_nodup'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Rejected-Criteria Set')
    elif DEFINE_SET == 'Rejected-AllFrames':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_rejected_allframes_plausible_nodup'
        CACHE_PATH_AM = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_rejected_allframes_all_masks_nodup'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'Rejected-All Frames Set')
    elif DEFINE_SET == 'BORN':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_born_sweet'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'BORN Module Set')
    elif DEFINE_SET == 'ARGOS':
        CACHE_PATH = 'D:/Python Scripts - Open Research/WLE-Transformers-PL/cache_wle_argos_soft'
        OUTPUT_PATH = os.path.join(SAVE_DIR, EXPERIMENT_NAME, 'Image Inference', 'ARGOS Fuji Set')
    else:
        raise ValueError

    """EXTRACT PARAMETERS"""
    opt = get_params()

    """EXECUTE FUNCTIONS"""
    if DEFINE_SET == 'Val' or DEFINE_SET == 'Test':
        run(opt=opt)
        # run_interobserver(opt=opt)
    else:
        run(opt=opt)



