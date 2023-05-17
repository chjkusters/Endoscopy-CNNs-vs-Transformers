"""IMPORT PACKAGES"""
import random
import os
import json
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms

"""DEFINE POSSIBLE EXTENSIONS"""
EXT_VID = ['.mp4', '.m4v', '.avi', '.MP4']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp']

"""DEFINE VARIABLES OF GASTRONET NORMALIZATION"""
KVASIR_MEAN = (0.51129418, 0.34449831, 0.18483892)       # KVASIR RGB VALUES
KVASIR_STD = (0.15095082, 0.11362306, 0.08710852)        # KVASIR RGB VALUES
GIANA_MEAN = (0.66016361, 0.38281666, 0.28119857)       # GIANA RGB VALUES
GIANA_STD = (0.21073012, 0.17948747, 0.15878445)        # GIANA RGB VALUES
IMAGENET_MEAN = (0.485, 0.456, 0.406)                 # RGB VALUES
IMAGENET_STD = (0.229, 0.224, 0.225)                  # RGB VALUES


""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR KVASIR DATASET """
""""""""""""""""""""""""""""""""""""""""""


def read_inclusion_kvasir(path, criteria):

    # Initialize lists
    img_list = list()

    # Initialize empty dictionary
    cache = dict()

    # Loop over cachefiles and check for inclusion criteria
    cache_files = os.listdir(path)
    for cachefile in cache_files:
        with open(os.path.join(path, cachefile)) as json_file:
            data = json.load(json_file)
            cache[cachefile] = data

    # Obtain min_height, min_width and mask_only from inclusion criteria
    min_height = criteria.pop('min_height', None)
    min_width = criteria.pop('min_width', None)

    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        # Loop over keys and values in criteria
        for k_ic, v_ic in criteria.items():

            # Check whether the value specified in ic is present in cachefile
            if not v_cache[k_ic] in v_ic:
                include = False
                break

        # Check whether min_height from inclusion criteria is not None
        if min_height is not None:
            if v_cache['height'] < min_height:
                include = False

        # Check whether min_width from inclusion criteria is not None
        if min_width is not None:
            if v_cache['width'] < min_width:
                include = False

        # Check whether include is true
        if include:
            info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'roi': v_cache['roi']}
            img_list.append(info)

    return img_list


def augmentations_kvasir(opt):

    # Initialize lists and dictionary
    train_transforms = list()
    val_transforms = list()
    test_transforms = list()
    data_transforms = dict()

    # Specify augmentation techniques for training
    train_technique1 = random.choice([Identity(),
                                      Identity(),
                                      GaussianBlur(),
                                      RandomAdjustSharpness(sharpness_factor=2, p=1),
                                      RandomAffine(max_rotate=25, max_translate=5, max_shear=15)
                                      ])
    train_technique2 = random.choice([Resize([opt.imagesize, opt.imagesize]),
                                      RandomResizedCrop((opt.imagesize, opt.imagesize)),
                                      RandomResizedCrop((opt.imagesize, opt.imagesize), scale=(0.7, 1.1))
                                      ])
    train_technique3 = random.choice([Identity(),
                                      Grayscale(num_output_channels=3),
                                      ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
                                      ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
                                      ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)
                                      ])
    train_transforms.extend([train_technique1,
                             train_technique2,
                             train_technique3,
                             RandomHorizontalFlip(p=0.5),
                             RandomVerticalFlip(p=0.5),
                             Rotate([0, 90, 180, 270, 360]),
                             ToTensor(),
                             Normalize(mean=[KVASIR_MEAN[0], KVASIR_MEAN[1], KVASIR_MEAN[2]],
                                       std=[KVASIR_STD[0], KVASIR_STD[1], KVASIR_STD[2]])
                             ])

    # Specify augmentation techniques for validation set
    val_technique1 = random.choice([Resize([opt.imagesize, opt.imagesize]),
                                    RandomResizedCrop((opt.imagesize, opt.imagesize))])
    val_technique2 = random.choice([Identity(),
                                    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
                                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)])
    val_transforms.extend([val_technique1,
                           val_technique2,
                           RandomHorizontalFlip(p=0.5),
                           RandomVerticalFlip(p=0.5),
                           Rotate([0, 90, 180, 270, 360]),
                           ToTensor(),
                           Normalize(mean=[KVASIR_MEAN[0], KVASIR_MEAN[1], KVASIR_MEAN[2]],
                                     std=[KVASIR_STD[0], KVASIR_STD[1], KVASIR_STD[2]])])

    # Specify augmentation techniques for test set
    test_transforms.extend([Resize([opt.imagesize, opt.imagesize]),
                            ToTensor(),
                            Normalize(mean=[KVASIR_MEAN[0], KVASIR_MEAN[1], KVASIR_MEAN[2]],
                                      std=[KVASIR_STD[0], KVASIR_STD[1], KVASIR_STD[2]])])

    # Compose transforms and place into dictionary
    data_transforms['train'] = Compose(train_transforms)
    data_transforms['val'] = Compose(val_transforms)
    data_transforms['test'] = Compose(test_transforms)

    return data_transforms


class DATASET_TRAIN_TEST_KVASIR(Dataset):
    def __init__(self, inclusion, transform=None, random_noise=False):
        self.inclusion = inclusion
        self.transform = transform
        self.random_noise = random_noise

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        image = Image.open(img_name).convert('RGB')

        # By default set has_mask to zero
        has_mask = 1

        # Open mask or create one artificially
        masklist = self.inclusion[idx]['mask']

        # Open mask for neoplasia cases
        if len(masklist) > 0:
            maskpath = random.choice(masklist)
            mask = Image.open(maskpath).convert('1')

            # Check if shapes are equal, if not discard mask (i.e. set has_mask = 0, do not set has_mask = 1)
            if mask.size != image.size:
                mask = mask.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
            else:
                mask = mask.crop((roi[2], roi[0], roi[3], roi[1]))

        # Crop the image to the ROI
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        if self.transform:
            image, mask = self.transform(image, mask, has_mask)

        if self.random_noise:
            ch, row, col = image.shape
            mean = 0
            var = random.choice([0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.05])
            sigma = var ** 0.5
            gauss = torch.tensor(np.random.normal(mean, sigma, (ch, row, col)), dtype=torch.float32)
            image = image + gauss

        return image, mask, has_mask


class DATASET_VAL_KVASIR(Dataset):
    def __init__(self, inclusion, transform=None):

        # For robustness do 4 times the validation set, with different augmentations
        self.inclusion = inclusion + inclusion + inclusion + inclusion
        self.transform = transform

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        image = Image.open(img_name).convert('RGB')

        # By default set has_mask to zero
        has_mask = 1

        # Open mask or create one artificially
        masklist = self.inclusion[idx]['mask']

        # Open mask for neoplasia cases
        if len(masklist) > 0:
            maskpath = random.choice(masklist)
            mask = Image.open(maskpath).convert('1')

            # Check if shapes are equal, if not discard mask (i.e. set has_mask = 0, do not set has_mask = 1)
            if mask.size != image.size:
                mask = mask.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
            else:
                mask = mask.crop((roi[2], roi[0], roi[3], roi[1]))

        # Crop the image to the ROI
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        if self.transform:
            image, mask = self.transform(image, mask, has_mask)

        return image, mask, has_mask


""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTIONS FOR GIANA DATASET """
""""""""""""""""""""""""""""""""""""""""""


def read_inclusion_giana(path, criteria):

    # Initialize lists
    img_list = list()

    # Initialize empty dictionary
    cache = dict()

    # Loop over cachefiles and check for inclusion criteria
    cache_files = os.listdir(path)
    for cachefile in cache_files:
        with open(os.path.join(path, cachefile)) as json_file:
            data = json.load(json_file)
            cache[cachefile] = data

    # Obtain min_height, min_width and mask_only from inclusion criteria
    min_height = criteria.pop('min_height', None)
    min_width = criteria.pop('min_width', None)

    # Loop over keys and values in cache files
    for k_cache, v_cache in cache.items():

        # By default set include to True
        include = True

        # Loop over keys and values in criteria
        for k_ic, v_ic in criteria.items():

            # Check whether the value specified in ic is present in cachefile
            if not v_cache[k_ic] in v_ic:
                include = False
                break

        # Check whether min_height from inclusion criteria is not None
        if min_height is not None:
            if v_cache['height'] < min_height:
                include = False

        # Check whether min_width from inclusion criteria is not None
        if min_width is not None:
            if v_cache['width'] < min_width:
                include = False

        # Check whether include is true
        if include:
            if v_cache['class'] == 'normal':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array(0),
                        'roi': v_cache['roi']}
                img_list.append(info)
            elif v_cache['class'] == 'inflammatory':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array(1),
                        'roi': v_cache['roi']}
                img_list.append(info)
            elif v_cache['class'] == 'vascularlesions':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array(2),
                        'roi': v_cache['roi']}
                img_list.append(info)
            else:
                print('Unrecognized class..')
                raise ValueError

    return img_list


def augmentations_giana(opt):

    # Initialize lists and dictionary
    train_transforms = list()
    val_transforms = list()
    test_transforms = list()
    data_transforms = dict()

    # Specify augmentation techniques for training
    train_technique1 = random.choice([Identity(),
                                      Identity(),
                                      GaussianBlur(),
                                      RandomAdjustSharpness(sharpness_factor=2, p=1),
                                      RandomAffine(max_rotate=25, max_translate=5, max_shear=15)])
    train_technique2 = random.choice([Resize([opt.imagesize, opt.imagesize]),
                                      RandomResizedCrop((opt.imagesize, opt.imagesize)),
                                      RandomResizedCrop((opt.imagesize, opt.imagesize), scale=(0.7, 1.1))])
    train_technique3 = random.choice([Identity(),
                                      Grayscale(num_output_channels=3),
                                      ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0),
                                      ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
                                      ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)])
    train_transforms.extend([train_technique1,
                             train_technique2,
                             train_technique3,
                             RandomHorizontalFlip(p=0.5),
                             RandomVerticalFlip(p=0.5),
                             Rotate([0, 90, 180, 270, 360]),
                             ToTensor(),
                             Normalize(mean=[GIANA_MEAN[0], GIANA_MEAN[1], GIANA_MEAN[2]],
                                       std=[GIANA_STD[0], GIANA_STD[1], GIANA_STD[2]])
                             ])

    # Specify augmentation techniques for validation set
    val_technique1 = random.choice([Resize([opt.imagesize, opt.imagesize]),
                                    RandomResizedCrop((opt.imagesize, opt.imagesize))])
    val_technique2 = random.choice([Identity(),
                                    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
                                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0)])
    val_transforms.extend([val_technique1,
                           val_technique2,
                           RandomHorizontalFlip(p=0.5),
                           RandomVerticalFlip(p=0.5),
                           Rotate([0, 90, 180, 270, 360]),
                           ToTensor(),
                           Normalize(mean=[GIANA_MEAN[0], GIANA_MEAN[1], GIANA_MEAN[2]],
                                     std=[GIANA_STD[0], GIANA_STD[1], GIANA_STD[2]])])

    # Specify augmentation techniques for test set
    test_transforms.extend([Resize([opt.imagesize, opt.imagesize]),
                            ToTensor(),
                            Normalize(mean=[GIANA_MEAN[0], GIANA_MEAN[1], GIANA_MEAN[2]],
                                      std=[GIANA_STD[0], GIANA_STD[1], GIANA_STD[2]])])

    # Compose transforms and place into dictionary
    data_transforms['train'] = Compose(train_transforms)
    data_transforms['val'] = Compose(val_transforms)
    data_transforms['test'] = Compose(test_transforms)

    return data_transforms


class DATASET_TRAIN_TEST_GIANA(Dataset):
    def __init__(self, inclusion, transform=None, random_noise=False):
        self.inclusion = inclusion
        self.transform = transform
        self.random_noise = random_noise

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        lab = self.inclusion[idx]['label']
        image = Image.open(img_name).convert('RGB')

        # # One-hot encoding for the label
        # num_classes = 3
        # label = np.zeros([num_classes])
        # label[int(lab[0])] = 1.

        # No one-hot encoding
        label = lab

        # By default set has_mask to one
        has_mask = 1

        # Open mask or create one artificially
        masklist = self.inclusion[idx]['mask']

        # Open mask for neoplasia cases
        if len(masklist) > 0:
            maskpath = random.choice(masklist)
            mask = Image.open(maskpath).convert('1')

            # Check if shapes are equal, if not discard mask (i.e. set has_mask = 0, do not set has_mask = 1)
            if mask.size != image.size:
                mask = mask.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
            else:
                mask = mask.crop((roi[2], roi[0], roi[3], roi[1]))

        # Crop the image to the ROI
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        if self.transform:
            image, mask = self.transform(image, mask, has_mask)

        if self.random_noise:
            ch, row, col = image.shape
            mean = 0
            var = random.choice([0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.05])
            sigma = var ** 0.5
            gauss = torch.tensor(np.random.normal(mean, sigma, (ch, row, col)), dtype=torch.float32)
            image = image + gauss

        return image, label, mask, has_mask


class DATASET_VAL_GIANA(Dataset):
    def __init__(self, inclusion, transform=None):

        # For robustness do 4 times the validation set, with different augmentations
        self.inclusion = inclusion + inclusion + inclusion + inclusion
        self.transform = transform

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        lab = self.inclusion[idx]['label']
        image = Image.open(img_name).convert('RGB')

        # By default set has_mask to one
        has_mask = 1

        # # One-hot encoding for the label
        # num_classes = 3
        # label = np.zeros([num_classes])
        # label[int(lab[0])] = 1.

        # No one-hot encoding
        label = lab

        # Open mask or create one artificially
        masklist = self.inclusion[idx]['mask']

        # Open mask for neoplasia cases
        if len(masklist) > 0:
            maskpath = random.choice(masklist)
            mask = Image.open(maskpath).convert('1')

            # Check if shapes are equal, if not discard mask (i.e. set has_mask = 0, do not set has_mask = 1)
            if mask.size != image.size:
                mask = mask.resize(image.size, resample=Image.NEAREST).crop((roi[2], roi[0], roi[3], roi[1]))
            else:
                mask = mask.crop((roi[2], roi[0], roi[3], roi[1]))

        # Crop the image to the ROI
        image = image.crop((roi[2], roi[0], roi[3], roi[1]))

        if self.transform:
            image, mask = self.transform(image, mask, has_mask)

        return image, label, mask, has_mask


""""""""""""""""""""""""""
"""" DATA AUGMENTATION """
""""""""""""""""""""""""""


# Custom Resize class
class Resize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img, mask, has_mask):
        img = transforms.functional.resize(img, size=self.target_size,
                                           interpolation=transforms.InterpolationMode.LANCZOS)
        mask = transforms.functional.resize(mask, size=self.target_size,
                                            interpolation=transforms.InterpolationMode.NEAREST)

        return img, mask, has_mask


# Custom Rotation transform
class Rotate:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img, mask, has_mask):
        angle = random.choice(self.angles)
        img = transforms.functional.rotate(img, angle)

        if has_mask:
            mask = transforms.functional.rotate(mask, angle)

        return img, mask, has_mask


# Custom Horizontal Flipping
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, has_mask):
        do_horizontal = random.random() < self.p
        if do_horizontal:
            img = transforms.functional.hflip(img)
            if has_mask:
                mask = transforms.functional.hflip(mask)

        return img, mask, has_mask


# Custom Vertical Flipping
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, has_mask):
        do_vertical = random.random() < self.p
        if do_vertical:
            img = transforms.functional.vflip(img)
            if has_mask:
                mask = transforms.functional.vflip(mask)

        return img, mask, has_mask


# Custom Class for NO augmentation at all
class Identity:
    def __init__(self):
        self.identity = None

    def __call__(self, img, mask, has_mask):

        return img, mask, has_mask


# Custom Class for Grayscale transform
class Grayscale:
    def __init__(self, num_output_channels=3):
        self.grayscale = transforms.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, img, mask, has_mask):
        img = self.grayscale(img)

        return img, mask, has_mask


# Custom Class for ColorJitter
class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue=0.0):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                             saturation=saturation, hue=hue)

    def __call__(self, img, mask, has_mask):
        img = self.jitter(img)

        return img, mask, has_mask


# Custom Random Adjusting Sharpness
class RandomAdjustSharpness:
    def __init__(self, sharpness_factor, p=1.):
        self.sharpness = transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p)

    def __call__(self, img, mask, has_mask):
        img = self.sharpness(img)

        return img, mask, has_mask


# Custom Class for Gaussian Blurring images
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=1.0, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, mask, has_mask):
        do_it = random.random() <= self.prob
        if not do_it:
            return img, mask, has_mask

        img_blurr = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

        return img_blurr, mask, has_mask


# Custom Class for making random resized crops for input images, and resize to desired size afterwards
class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.9, 1.1), ratio=(7. / 8., 8. / 7.),
                 interpolation=torchvision.transforms.InterpolationMode.LANCZOS):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, mask, has_mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        img = torchvision.transforms.functional.resized_crop(
            img=img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=self.size,
            interpolation=self.interpolation)

        mask = torchvision.transforms.functional.resized_crop(
            img=mask,
            top=i,
            left=j,
            height=h,
            width=w,
            size=self.size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        return img, mask, has_mask


# Custom Class for random affine transformations with predefined ranges for rotation, translation and shearing
class RandomAffine:
    def __init__(self, max_rotate, max_translate, max_shear):
        self.max_rotate = max_rotate
        self.max_translate = max_translate
        self.max_shear = max_shear

    def __call__(self, img, mask, has_mask):
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        translate = [int(random.uniform(-self.max_translate, self.max_translate)),
                     int(random.uniform(-self.max_translate, self.max_translate))]
        shear = [random.uniform(-self.max_shear, self.max_shear)]

        img = transforms.functional.affine(
            img=img,
            angle=angle,
            translate=translate,
            scale=1.,
            shear=shear,
            interpolation=transforms.InterpolationMode.BICUBIC)

        if has_mask:
            mask = transforms.functional.affine(
                img=mask,
                angle=angle,
                translate=translate,
                scale=1.,
                shear=shear,
                interpolation=transforms.InterpolationMode.NEAREST)

        return img, mask, has_mask


# Custom Class for Normalizing images
class Normalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img, mask, has_mask):
        img = self.normalize(img)

        return img, mask, has_mask


# Custom Class for PIL to Tensor
class ToTensor:
    def __init__(self):
        self.ToTensor = transforms.ToTensor()

    def __call__(self, img, mask, has_mask):
        img = self.ToTensor(img)
        mask = self.ToTensor(mask)

        return img, mask, has_mask


# Custom Class for composing
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, has_mask):
        for t in self.transforms:
            img, mask, has_mask = t(img, mask, has_mask)

        return img, mask
