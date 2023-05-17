"""IMPORT PACKAGES"""
import os
import re
import argparse
import json
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset_public import DATASET_TRAIN_TEST_KVASIR, DATASET_VAL_KVASIR, read_inclusion_kvasir, augmentations_kvasir
from data.dataset_public import DATASET_TRAIN_TEST_GIANA, DATASET_VAL_GIANA, read_inclusion_giana, augmentations_giana
from data.dataset_public import DATASET_TRAIN_TEST_SYSUCC, DATASET_VAL_SYSUCC, read_inclusion_sysucc, augmentations_sysucc, sample_weights_sysucc
from utils.loss_optim_wle import construct_optimizer, construct_scheduler, construct_loss_function
from utils.metrics_wle import BinaryDiceMetric
from models.model_wle import Model

""""""""""""""""""""""""
"""" HELPER FUNCTIONS """
""""""""""""""""""""""""


# Make function for defining parameters - KVASIR
def get_params_kvasir():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)
    parser.add_argument('--seed', type=int, default=7)

    # DEFINE MODEL
    parser.add_argument('--backbone', type=str, default='ESFPNet-B0')
    parser.add_argument('--seg_branch', type=str, default=None)

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Plateau')
    parser.add_argument('--cls_criterion', type=str, default='BCE')
    parser.add_argument('--seg_criterion', type=str, default='DiceBCE')
    parser.add_argument('--cls_criterion_weight', type=int, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.01)

    # AUGMENTATION PARAMS
    parser.add_argument('--imagesize', type=int, default=256)   # Should be 256, but for ViT 224
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1)

    # TRAINING PARAMETERS
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--train_lr', type=float, default=1e-3)

    args = parser.parse_args()

    return args


# Make function for defining parameters - SYSUCC
def get_params_sysucc():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)
    parser.add_argument('--seed', type=int, default=7)

    # DEFINE MODEL
    parser.add_argument('--backbone', type=str, default='ESFPNet-B0')
    parser.add_argument('--seg_branch', type=str, default=None)

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Plateau')
    parser.add_argument('--cls_criterion', type=str, default='BCE')
    parser.add_argument('--seg_criterion', type=str, default='DiceBCE')
    parser.add_argument('--cls_criterion_weight', type=int, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.01)

    # AUGMENTATION PARAMS
    parser.add_argument('--imagesize', type=int, default=256)   # Should be 256, but for ViT 224
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1)

    # TRAINING PARAMETERS
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_lr', type=float, default=1e-4)

    args = parser.parse_args()

    return args


# Make function for defining parameters - GIANA
def get_params_giana():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DEFINE EXPERIMENT NAME
    parser.add_argument('--experimentname', type=str, default=EXPERIMENT_NAME)
    parser.add_argument('--seed', type=int, default=7)

    # DEFINE MODEL
    parser.add_argument('--backbone', type=str, default='ESFPNet-B0')
    parser.add_argument('--seg_branch', type=str, default=None)

    # DEFINE OPTIMIZER, CRITERION, SCHEDULER
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Plateau')
    parser.add_argument('--cls_criterion', type=str, default='CE')
    parser.add_argument('--seg_criterion', type=str, default='DiceBCE')
    parser.add_argument('--cls_criterion_weight', type=int, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.01)

    # AUGMENTATION PARAMS
    parser.add_argument('--imagesize', type=int, default=256)   # Should be 256, but for ViT 224
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=3)

    # TRAINING PARAMETERS
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--train_lr', type=float, default=1e-4)

    args = parser.parse_args()

    return args


# Specify function for defining inclusion criteria for training, finetuning and development set
def get_data_inclusion_criteria():
    criteria = dict()

    criteria['train'] = {'dataset': ['train'],
                         'min_height': None,
                         'min_width': None
                         }

    criteria['dev'] = {'dataset': ['validation'],
                       'min_height': None,
                       'min_width': None
                       }

    return criteria


# Function for checking whether GPU or CPU is being used
def check_cuda():

    print('\nExtract Device...')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
        device_count = torch.cuda.device_count()
        torch.cuda.empty_cache()
        print('Using device: {}'.format(device))
        print('Device name: {}'.format(device_name))
        print('Device count: {}\n'.format(device_count))
    else:
        device = torch.device('cpu')
        print('Using device: cpu\n')


# Find best checkpoint model for sysucc
def find_best_model_sysucc(path):

    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            val = re.findall(r'\d+\.\d+', file)
            auc = val[0]
            files.append(file)
            values.append(auc)

    # Find file with highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


# Find best checkpoint model for kvasir
def find_best_model_kvasir(path):

    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            val = re.findall(r'\d+\.\d+', file)
            dice = val[0]
            files.append(file)
            values.append(dice)

    # Find file with highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


# Find best checkpoint model for giana
def find_best_model_giana(path):

    # Append all files
    files = list()
    values = list()

    # List files with certain extension
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            val = re.findall(r'\d+\.\d+', file)
            acc, auc, dice = val[0], val[1], val[2]
            value = auc
            files.append(file)
            values.append(value)

    # Find file with highest value
    max_val = max(values)
    indices = [i for i, x in enumerate(values) if x == max_val]
    max_index = indices[-1]

    return files[max_index]


# Remove keys from checkpoint for finetuning
def remove_keys(opt, ckpt_path):

    # Extract checkpoint name
    filename = os.path.splitext((os.path.split(ckpt_path)[1]))[0]

    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Unpack the keys of the checkpoint
    checkpoint_keys = list(checkpoint['state_dict'].keys())

    # Loop over the keys
    for key in checkpoint_keys:

        # Exclude layers that are to be preserved
        if 'ResNet' in opt.backbone or 'FCBFormer' in opt.backbone or 'ESFPNet' in opt.backbone:
            if 'backbone.fc' in key:
                del checkpoint['state_dict'][key]
                print('Deleted key: {}'.format(key))
        elif 'ConvNeXt' in opt.backbone:
            if 'backbone.head' in key:
                del checkpoint['state_dict'][key]
                print('Deleted key: {}'.format(key))
        elif 'UNet' in opt.backbone:
            if 'Swin' in opt.backbone:
                if 'backbone.swin_unet.fc' in key:
                    del checkpoint['state_dict'][key]
                    print('Deleted key: {}'.format(key))
            else:
                if 'backbone.classification_head.3' in key:
                    del checkpoint['state_dict'][key]
                    print('Deleted key: {}'.format(key))
        elif 'Swin' in opt.backbone and 'UperNet' in opt.backbone:
            if 'backbone.fc' in key:
                del checkpoint['state_dict'][key]
                print('Deleted key: {}'.format(key))
        elif 'CaraNet' in opt.backbone:
            if 'backbone.resnet.fc' in key:
                del checkpoint['state_dict'][key]
                print('Deleted key: {}'.format(key))

    # Save new checkpoint
    new_filename = filename + '-removehead.ckpt'
    torch.save(checkpoint, os.path.join(SAVE_DIR, opt.experimentname, new_filename))

    return new_filename


""""""""""""""""""""""""""""""""""""""""""""
"""" DATA: PYTORCH LIGHTNING DATAMODULES KVASIR """
""""""""""""""""""""""""""""""""""""""""""""
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#why-do-i-need-a-datamodule


class WLEDataModuleKvasir(pl.LightningDataModule):
    def __init__(self, data_dir, criteria, transforms, opt):
        super().__init__()
        self.data_dir = data_dir
        self.criteria = criteria
        self.transforms = transforms
        self.train_set = None
        self.val_set_train = None
        self.val_set_test = None

    def setup(self, stage: Optional[str] = None):

        # Find data that satisfies the inclusion criteria
        train_inclusion = read_inclusion_kvasir(path=self.data_dir, criteria=self.criteria['train'])
        val_inclusion = read_inclusion_kvasir(path=self.data_dir, criteria=self.criteria['dev'])

        # Construct datasets
        self.train_set = DATASET_TRAIN_TEST_KVASIR(inclusion=train_inclusion,
                                                   transform=self.transforms['train'],
                                                   random_noise=True)
        self.val_set_train = DATASET_VAL_KVASIR(inclusion=val_inclusion,
                                                transform=self.transforms['val'])
        self.val_set_test = DATASET_TRAIN_TEST_KVASIR(inclusion=val_inclusion,
                                                      transform=self.transforms['test'],
                                                      random_noise=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=opt.batchsize, shuffle=False, num_workers=4,
                          pin_memory=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_set_train, batch_size=opt.batchsize, num_workers=4,
                          pin_memory=True, prefetch_factor=4)

    def test_dataloader(self):
        return DataLoader(self.val_set_test, batch_size=opt.batchsize, num_workers=4)


""""""""""""""""""""""""""""""""""""""""""""
"""" DATA: PYTORCH LIGHTNING DATAMODULES GIANA """
""""""""""""""""""""""""""""""""""""""""""""
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#why-do-i-need-a-datamodule


class WLEDataModuleGiana(pl.LightningDataModule):
    def __init__(self, data_dir, criteria, transforms, opt):
        super().__init__()
        self.data_dir = data_dir
        self.criteria = criteria
        self.transforms = transforms
        self.train_set = None
        self.val_set_train = None
        self.val_set_test = None

    def setup(self, stage: Optional[str] = None):

        # Find data that satisfies the inclusion criteria
        train_inclusion = read_inclusion_giana(path=self.data_dir, criteria=self.criteria['train'])
        val_inclusion = read_inclusion_giana(path=self.data_dir, criteria=self.criteria['dev'])

        # Construct datasets
        self.train_set = DATASET_TRAIN_TEST_GIANA(inclusion=train_inclusion,
                                                  transform=self.transforms['train'],
                                                  random_noise=True)
        self.val_set_train = DATASET_VAL_GIANA(inclusion=val_inclusion,
                                               transform=self.transforms['val'])
        self.val_set_test = DATASET_TRAIN_TEST_GIANA(inclusion=val_inclusion,
                                                     transform=self.transforms['test'],
                                                     random_noise=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=opt.batchsize, shuffle=False, num_workers=4,
                          pin_memory=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_set_train, batch_size=opt.batchsize, num_workers=4,
                          pin_memory=True, prefetch_factor=4)

    def test_dataloader(self):
        return DataLoader(self.val_set_test, batch_size=opt.batchsize, num_workers=4)


""""""""""""""""""""""""""""""""""""""""""""""""""
"""" MODEL: PYTORCH LIGHTNING & PYTORCH MODULE KVASIR """
""""""""""""""""""""""""""""""""""""""""""""""""""
# https://www.pytorchlightning.ai/
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
# https://medium.com/aimstack/how-to-tune-hyper-params-with-fixed-seeds-using-pytorch-lightning-and-aim-c61c73f75c7c
# https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html
# https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html


class WLEModelKvasir(pl.LightningModule):
    def __init__(self, opt, finetune):
        super(WLEModelKvasir, self).__init__()

        # Fix seed for reproducibility
        pl.seed_everything(seed=opt.seed, workers=True)

        # Define whether the stage is training or finetuning
        self.finetune = finetune

        # Define label smoothing
        self.label_smoothing = opt.label_smoothing

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # Define loss functions for classification and segmentation
        self.cls_criterion, self.seg_criterion = construct_loss_function(opt=opt)

        # Define model
        self.model = Model(opt=opt, inference=False)

        # Specify metrics
        self.train_dice = BinaryDiceMetric()
        self.val_dice = BinaryDiceMetric()
        self.test_dice = BinaryDiceMetric()

    def forward(self, x):

        # # Extract outputs of the model
        # cls_out, mask_out = self.model(x)

        # Extract outputs of the model: Segmentation [BS, 1, h, w], Classification [BS, 1]
        out1, out2 = self.model(x)
        cls_out = (out1 if out1.dim() == 2 else out2)
        mask_out = (out2 if out2.dim() == 4 else out1)

        return cls_out, mask_out

    def configure_optimizers(self):

        # Define learning rate
        if not self.finetune:
            learning_rate = opt.train_lr
        else:
            learning_rate = opt.finetune_lr

        # Define optimizer
        optimizer = construct_optimizer(optim=opt.optimizer, parameters=self.parameters(), lr=learning_rate)

        # Define learning rate scheduler
        scheduler = construct_scheduler(schedule=opt.scheduler, optimizer=optimizer, lr=learning_rate, metric='val_loss_seg')

        if scheduler is not None:
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, mask, has_mask = train_batch

        # Extract predictions of the network
        _, seg = self.forward(img)

        # Perform label smoothing
        mask_smooth = (1.-self.label_smoothing)*mask + self.label_smoothing*0.5

        # Compute Loss for segmentation output
        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, _, batch_idx)
        self.log('train_loss_seg', seg_loss.item())

        # Update metric
        logits_seg = self.sigmoid(seg)
        self.train_dice.update(logits_seg, mask, has_mask)

        return seg_loss

    def training_epoch_end(self, training_step_outputs):

        # Compute metric
        train_dice = self.train_dice.compute()

        # Log and print metric value
        self.log('train_dice', train_dice)
        print('\n' + 120 * "=")
        print(f"Training Set: Avg. Dice Score: {train_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.train_dice.reset()

    def validation_step(self, val_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, mask, has_mask = val_batch

        # Extract predictions of the network
        _, seg = self.forward(img)

        # Perform label smoothing
        mask_smooth = (1. - self.label_smoothing) * mask + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, _, batch_idx)
        self.log('val_loss_seg', seg_loss.item())

        # Update metrics
        logits_seg = self.sigmoid(seg)
        self.val_dice.update(logits_seg, mask, has_mask)

        return seg_loss

    def validation_epoch_end(self, validation_step_outputs):

        # Compute metric values
        val_dice = self.val_dice.compute()

        # Log and print values
        self.log('val_dice', val_dice)
        print('\n\n' + 120 * "=")
        print(f"Validation Set: Avg. Dice Score: {val_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.val_dice.reset()

    def test_step(self, test_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, mask, has_mask = test_batch

        # Extract predictions of the network
        _, seg = self.forward(img)

        # Update metrics
        logits_seg = self.sigmoid(seg)
        self.test_dice.update(logits_seg, mask, has_mask)

    def test_epoch_end(self, test_step_outputs):

        # Execute metric computation
        test_dice = self.test_dice.compute()

        # Print results
        print('\n\n' + 120 * "=")
        print(f"Test Set: Avg. Dice Score: {test_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.test_dice.reset()


""""""""""""""""""""""""""""""""""""""""""""""""""
"""" MODEL: PYTORCH LIGHTNING & PYTORCH MODULE GIANA """
""""""""""""""""""""""""""""""""""""""""""""""""""
# https://www.pytorchlightning.ai/
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
# https://medium.com/aimstack/how-to-tune-hyper-params-with-fixed-seeds-using-pytorch-lightning-and-aim-c61c73f75c7c
# https://pytorch-lightning.readthedocs.io/en/1.4.3/common/weights_loading.html
# https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html


class WLEModelGiana(pl.LightningModule):
    def __init__(self, opt, finetune):
        super(WLEModelGiana, self).__init__()

        # Fix seed for reproducibility
        pl.seed_everything(seed=opt.seed, workers=True)

        # Define whether the stage is training or finetuning
        self.finetune = finetune

        # Define label smoothing
        self.label_smoothing = opt.label_smoothing

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # Define loss functions for classification and segmentation
        self.cls_criterion, self.seg_criterion = construct_loss_function(opt=opt)

        # Define model
        self.model = Model(opt=opt, inference=False)

        # Specify metrics
        self.train_auc = torchmetrics.AUROC(num_classes=3)
        self.train_dice = BinaryDiceMetric()

        self.val_acc = torchmetrics.Accuracy(num_classes=3)
        self.val_auc = torchmetrics.AUROC(num_classes=3)
        self.val_dice = BinaryDiceMetric()

        self.test_acc = torchmetrics.Accuracy(num_classes=3)
        self.test_auc = torchmetrics.AUROC(num_classes=3)
        self.test_dice = BinaryDiceMetric()

    def forward(self, x):

        # # Extract outputs of the model
        # cls_out, mask_out = self.model(x)

        # Extract outputs of the model: Segmentation [BS, 1, h, w], Classification [BS, 1]
        out1, out2 = self.model(x)
        cls_out = (out1 if out1.dim() == 2 else out2)
        mask_out = (out2 if out2.dim() == 4 else out1)

        return cls_out, mask_out

    def configure_optimizers(self):

        # Define learning rate
        if not self.finetune:
            learning_rate = opt.train_lr
        else:
            learning_rate = opt.finetune_lr

        # Define optimizer
        optimizer = construct_optimizer(optim=opt.optimizer, parameters=self.parameters(), lr=learning_rate)

        # Define learning rate scheduler
        scheduler = construct_scheduler(schedule=opt.scheduler, optimizer=optimizer, lr=learning_rate, metric='val_loss_combine')

        if scheduler is not None:
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = train_batch
        lab = lab.type(torch.LongTensor).cuda()

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Perform label smoothing
        mask_smooth = (1.-self.label_smoothing)*mask + self.label_smoothing*0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab)
        self.log('train_loss_cls', cls_loss.item())

        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, lab, batch_idx)
        self.log('train_loss_seg', seg_loss.item())

        summed_loss = cls_loss + seg_loss
        self.log('train_loss_combine', summed_loss.item())

        # Update metrics
        logits_cls = self.softmax(preds)
        logits_seg = self.sigmoid(seg)
        self.train_auc.update(logits_cls, lab.to(torch.int32))
        self.train_dice.update(logits_seg, mask, has_mask)

        return summed_loss

    def training_epoch_end(self, training_step_outputs):

        # Compute metrics
        train_auc = self.train_auc.compute()
        train_dice = self.train_dice.compute()

        # Log and print metric value
        self.log('train_auc', train_auc)
        self.log('train_dice', train_dice)
        print('\n' + 120 * "=")
        print(f"Training Set:  AUC Cls: {train_auc:.4}, Avg. Dice Score: {train_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.train_auc.reset()
        self.train_dice.reset()

    def validation_step(self, val_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = val_batch
        lab = lab.type(torch.LongTensor).cuda()

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Perform label smoothing
        mask_smooth = (1. - self.label_smoothing) * mask + self.label_smoothing * 0.5

        # Compute Loss for both outputs
        cls_loss = self.cls_criterion(preds, lab)
        self.log('val_loss_cls', cls_loss.item())

        seg_loss = self.seg_criterion(seg, mask_smooth, has_mask, lab, batch_idx)
        self.log('val_loss_seg', seg_loss.item())

        summed_loss = cls_loss + seg_loss
        self.log('val_loss_combine', summed_loss.item())

        # Update metrics
        logits_cls = self.softmax(preds)
        logits_seg = self.sigmoid(seg)
        self.val_acc.update(logits_cls, lab.to(torch.int32))
        self.val_auc.update(logits_cls, lab.to(torch.int32))
        self.val_dice.update(logits_seg, mask, has_mask)

        return summed_loss

    def validation_epoch_end(self, validation_step_outputs):

        # Compute metric values
        val_acc = self.val_acc.compute()
        val_auc = self.val_auc.compute()
        val_dice = self.val_dice.compute()

        # Log and print values
        self.log('val_acc', val_acc)
        self.log('val_auc', val_auc)
        self.log('val_dice', val_dice)
        print('\n\n' + 120 * "=")
        print(f"Validation Set: Accuracy: {val_acc:.4}, "
              f"AUC Cls: {val_auc:.4} "
              f"Avg. Dice Score: {val_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_dice.reset()

    def test_step(self, test_batch, batch_idx):

        # Extract images, labels, mask and has_mask
        img, lab, mask, has_mask = test_batch
        lab = lab.type(torch.LongTensor).cuda()

        # Extract predictions of the network
        preds, seg = self.forward(img)

        # Update metrics
        logits_cls = self.softmax(preds)
        logits_seg = self.sigmoid(seg)
        self.test_acc.update(logits_cls, lab.to(torch.int32))
        self.test_auc.update(logits_cls, lab.to(torch.int32))
        self.test_dice.update(logits_seg, mask, has_mask)

    def test_epoch_end(self, test_step_outputs):

        # Execute metric computation
        test_acc = self.test_acc.compute()
        test_auc = self.test_auc.compute()
        test_dice = self.test_dice.compute()

        # Print results
        print('\n\n' + 120 * "=")
        print(f"Test Set: Accuracy: {test_acc:.4} "
              f"AUC Cls: {test_auc:.4} "
              f"Avg. Dice Score: {test_dice:.4}")
        print(120 * "=" + '\n')

        # Reset metric values
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_dice.reset()


""""""""""""""""""""""""""""""
"""" FUNCTION FOR EXECUTION KVASIR """
""""""""""""""""""""""""""""""


def run_kvasir(opt):

    """TEST DEVICE"""
    check_cuda()
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    """SETUP PYTORCH LIGHTNING DATAMODULE"""
    print('Starting PyTorch Lightning DataModule...')
    criteria = get_data_inclusion_criteria()
    data_transforms = augmentations_kvasir(opt)
    dm_train = WLEDataModuleKvasir(data_dir=CACHE_PATH, criteria=criteria, transforms=data_transforms, opt=opt)

    """SETUP PYTORCH LIGHTNING MODEL"""
    print('Starting PyTorch Lightning Model...')

    # Construct Loggers for PyTorch Lightning
    wandb_logger_train = WandbLogger(name='Train-{}'.format(EXPERIMENT_NAME), project='Endoscopy-CNNs-vs-Transformers',
                                     save_dir=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    lr_monitor_train = LearningRateMonitor(logging_interval='step')

    # Construct callback used for training the model
    checkpoint_callback_train = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        dirpath=os.path.join(SAVE_DIR, EXPERIMENT_NAME),
        filename='model-{epoch:02d}-{val_dice:.4f}',
        save_top_k=3,
        save_weights_only=True
    )

    """TRAINING PHASE"""

    # Construct PyTorch Lightning Trainer
    pl_model = WLEModelKvasir(opt=opt, finetune=False)
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         max_epochs=opt.num_epochs,
                         logger=wandb_logger_train,
                         callbacks=[checkpoint_callback_train,
                                    lr_monitor_train],
                         check_val_every_n_epoch=1,
                         log_every_n_steps=10,
                         deterministic=True)

    # Start Training
    trainer.fit(model=pl_model, datamodule=dm_train)
    wandb_logger_train.experiment.finish()

    """INFERENCE PHASE"""
    best_index = find_best_model_kvasir(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    trainer.test(model=pl_model,
                 datamodule=dm_train,
                 ckpt_path=os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index))


""""""""""""""""""""""""""""""
"""" FUNCTION FOR EXECUTION GIANA """
""""""""""""""""""""""""""""""


def run_giana(opt):

    """TEST DEVICE"""
    check_cuda()
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    """SETUP PYTORCH LIGHTNING DATAMODULE"""
    print('Starting PyTorch Lightning DataModule...')
    criteria = get_data_inclusion_criteria()
    data_transforms = augmentations_giana(opt)
    dm_train = WLEDataModuleGiana(data_dir=CACHE_PATH, criteria=criteria, transforms=data_transforms, opt=opt)

    """SETUP PYTORCH LIGHTNING MODEL"""
    print('Starting PyTorch Lightning Model...')

    # Construct Loggers for PyTorch Lightning
    wandb_logger_train = WandbLogger(name='Train-{}'.format(EXPERIMENT_NAME), project='Endoscopy-CNNs-vs-Transformers',
                                     save_dir=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    lr_monitor_train = LearningRateMonitor(logging_interval='step')

    # Construct callback used for training the model
    checkpoint_callback_train = ModelCheckpoint(
        monitor='val_auc',
        mode='max',
        dirpath=os.path.join(SAVE_DIR, EXPERIMENT_NAME),
        filename='model-{epoch:02d}-{val_acc:.4f}-{val_auc:.4f}-{val_dice:.4f}',
        save_top_k=3,
        save_weights_only=True
    )

    """TRAINING PHASE"""

    # Construct PyTorch Lightning Trainer
    pl_model = WLEModelGiana(opt=opt, finetune=False)
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         max_epochs=opt.num_epochs,
                         logger=wandb_logger_train,
                         callbacks=[checkpoint_callback_train,
                                    lr_monitor_train],
                         check_val_every_n_epoch=1,
                         log_every_n_steps=10,
                         deterministic=True)

    # Start Training
    trainer.fit(model=pl_model, datamodule=dm_train)
    wandb_logger_train.experiment.finish()

    """INFERENCE PHASE"""
    best_index = find_best_model_giana(path=os.path.join(SAVE_DIR, EXPERIMENT_NAME))
    trainer.test(model=pl_model,
                 datamodule=dm_train,
                 ckpt_path=os.path.join(SAVE_DIR, EXPERIMENT_NAME, best_index))


""""""""""""""""""""""""""
"""EXECUTION OF FUNCTIONS"""
""""""""""""""""""""""""""

if __name__ == '__main__':

    """DEFINE EXPERIMENT NAME"""
    EXPERIMENT_NAME = 'Experiment1'

    """SPECIFY PATH FOR SAVING"""
    SAVE_DIR = os.path.join(os.getcwd(), 'experiments')

    """"""""""""
    """KVASIR"""
    """"""""""""

    # Specify cache path
    CACHE_PATH = os.path.join(os.getcwd(), 'cache')

    # Specify parameters and inclusion criteria
    opt = get_params_kvasir()

    # Check if direction for logging the information already exists; otherwise make direction
    if not os.path.exists(os.path.join(SAVE_DIR, opt.experimentname)):
        os.mkdir(os.path.join(SAVE_DIR, opt.experimentname))

    # Save params from opt as a dictionary in a json file 'params.json'
    with open(os.path.join(SAVE_DIR,  opt.experimentname, 'params.json'), 'w') as fp:
        json.dump(opt.__dict__, fp, indent=4)

    # Save inclusion criteria (already dictionary) in a json file 'datacriteria.json'
    with open(os.path.join(SAVE_DIR, opt.experimentname, 'datacriteria.json'), 'w') as fp:
        json.dump(get_data_inclusion_criteria(), fp, indent=4)

    """EXECUTE FUNCTION"""
    run_kvasir(opt)

    """"""""""""
    """SYSUCC"""
    """"""""""""

    # Specify cache path
    CACHE_PATH = os.path.join(os.getcwd(), 'cache')

    # Specify parameters and inclusion criteria
    opt = get_params_sysucc()

    # Check if direction for logging the information already exists; otherwise make direction
    if not os.path.exists(os.path.join(SAVE_DIR, opt.experimentname)):
        os.mkdir(os.path.join(SAVE_DIR, opt.experimentname))

    # Save params from opt as a dictionary in a json file 'params.json'
    with open(os.path.join(SAVE_DIR,  opt.experimentname, 'params.json'), 'w') as fp:
        json.dump(opt.__dict__, fp, indent=4)

    # Save inclusion criteria (already dictionary) in a json file 'datacriteria.json'
    with open(os.path.join(SAVE_DIR, opt.experimentname, 'datacriteria.json'), 'w') as fp:
        json.dump(get_data_inclusion_criteria(), fp, indent=4)

    """EXECUTE FUNCTION"""
    run_sysucc(opt)

    """"""""""""
    """GIANA"""
    """"""""""""

    # Specify cache path
    CACHE_PATH = os.path.join(os.getcwd(), 'cache')

    # Specify parameters and inclusion criteria
    opt = get_params_giana()

    # Check if direction for logging the information already exists; otherwise make direction
    if not os.path.exists(os.path.join(SAVE_DIR, opt.experimentname)):
        os.mkdir(os.path.join(SAVE_DIR, opt.experimentname))

    # Save params from opt as a dictionary in a json file 'params.json'
    with open(os.path.join(SAVE_DIR,  opt.experimentname, 'params.json'), 'w') as fp:
        json.dump(opt.__dict__, fp, indent=4)

    # Save inclusion criteria (already dictionary) in a json file 'datacriteria.json'
    with open(os.path.join(SAVE_DIR, opt.experimentname, 'datacriteria.json'), 'w') as fp:
        json.dump(get_data_inclusion_criteria(), fp, indent=4)

    """EXECUTE FUNCTION"""
    run_giana(opt)





