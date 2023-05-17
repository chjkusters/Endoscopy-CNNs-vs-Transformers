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
WLE_MEAN = (0.64041256, 0.36125767, 0.31330117)       # WLE RGB VALUES
WLE_STD = (0.18983584, 0.15554344, 0.14093774)        # WLE RGB VALUES
IMAGENET_MEAN = (0.485, 0.456, 0.406)                 # RGB VALUES
IMAGENET_STD = (0.229, 0.224, 0.225)                  # RGB VALUES


""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTION FOR FINDING INCLUSION """
""""""""""""""""""""""""""""""""""""""""""


def read_inclusion(path, criteria):

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
    mask_only = criteria.pop('mask_only', None)

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

        # Check whether mask only is True
        if mask_only is True:

            # Exclude images if there are no masks
            if len(v_cache['masks']) == 0 and v_cache['class'] == 'neo':
                include = False

        # Check whether include is true
        if include:
            if v_cache['class'] == 'neo':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array([1], dtype=np.float32),
                        'roi': v_cache['roi'], 'subtlety': v_cache['subtlety'], 'quality': v_cache['quality']}
                img_list.append(info)
            elif v_cache['class'] == 'ndbe':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array([0], dtype=np.float32),
                        'roi': v_cache['roi'], 'subtlety': v_cache['subtlety'], 'quality': v_cache['quality']}
                img_list.append(info)
            else:
                print('Unrecognized class..')
                raise ValueError

    return img_list


def read_inclusion_split(path, criteria, split_perc=1.0):

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
    mask_only = criteria.pop('mask_only', None)

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

        # Check whether mask only is True
        if mask_only is True:

            # Exclude images if there are no masks
            if len(v_cache['masks']) == 0 and v_cache['class'] == 'neo':
                include = False

        # Check whether include is true
        if include:
            if v_cache['class'] == 'neo':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array([1], dtype=np.float32),
                        'roi': v_cache['roi'], 'subtlety': v_cache['subtlety'], 'quality': v_cache['quality']}
                img_list.append(info)
            elif v_cache['class'] == 'ndbe':
                info = {'file': v_cache['file'], 'mask': v_cache['masks'], 'label': np.array([0], dtype=np.float32),
                        'roi': v_cache['roi'], 'subtlety': v_cache['subtlety'], 'quality': v_cache['quality']}
                img_list.append(info)
            else:
                print('Unrecognized class..')
                raise ValueError

    # Only take the percentage of data that is specified
    if split_perc < 1.0:

        # Instantiate lst of all patient IDs
        patids = list()
        patids_ndbe = list()
        patids_neo = list()
        counter_ndbe = 0
        counter_neo = 0
        counter_masks = 0

        # Loop over the keys of included images, extract patient ID and append to list
        for img in img_list:
            imgname = os.path.split(img['file'])[1]
            patid = '_'.join(imgname.split('_')[:2])
            if patid not in patids:
                patids.append(patid)

        # Sort and shuffle the list of patient IDs
        patids = sorted(patids)
        random.seed(7)
        random.shuffle(patids)

        # Define the percentage of patient IDs
        splitidx = round(len(patids) * split_perc)
        incl_patids = patids[:splitidx]

        # Define the final image list
        img_list_split = list()

        # Loop over the keys of included images, extract patient ID and append to list
        for img in img_list:
            imgname = os.path.split(img['file'])[1]
            patid = '_'.join(imgname.split('_')[:2])
            if patid in incl_patids:
                img_list_split.append(img)

                # Find the number of patients and images for each class
                if img['label'] == np.array([0], dtype=np.float32):
                    counter_ndbe += 1
                    if patid not in patids_ndbe:
                        patids_ndbe.append(patid)
                elif img['label'] == np.array([1], dtype=np.float32):
                    counter_neo += 1
                    if patid not in patids_neo:
                        patids_neo.append(patid)
                    if len(img['mask']) > 0:
                        counter_masks += 1

        print('Image Set Size: {}, Patient Set Size: {}'.format(len(img_list_split), len(incl_patids)))
        print('NDBE Image Set Size: {}, NDBE Patient Set Size: {}'.format(counter_ndbe, len(patids_ndbe)))
        print('NEO Image Set Size: {}, NEO Patient Set Size: {}, NEO masks: {}'.format(counter_neo, len(patids_neo), counter_masks))
        print('Check Images: {}, Check Patients: {}, Overlap: {}'
              .format(counter_neo+counter_ndbe, len(patids_ndbe)+len(patids_neo),
                      len(list(set(patids_neo) & set(patids_ndbe)))))

        return img_list_split

    else:

        # Instantiate lst of all patient IDs
        patids = list()
        patids_ndbe = list()
        patids_neo = list()
        counter_ndbe = 0
        counter_neo = 0
        counter_masks = 0

        # Loop over the keys of included images, extract patient ID and append to list
        for img in img_list:
            imgname = os.path.split(img['file'])[1]
            patid = '_'.join(imgname.split('_')[:2])

            # # For BORN Set
            # if 'IWGCO-BORN' in imgname:
            #     patid = imgname.split('_')[0].split('-')[2]
            # else:
            #     patid = imgname.split('_')[0]

            if patid not in patids:
                patids.append(patid)

            # Find the number of patients and images for each class
            if img['label'] == np.array([0], dtype=np.float32):
                counter_ndbe += 1
                if patid not in patids_ndbe:
                    patids_ndbe.append(patid)
            elif img['label'] == np.array([1], dtype=np.float32):
                counter_neo += 1
                if patid not in patids_neo:
                    patids_neo.append(patid)
                if len(img['mask']) > 0:
                    counter_masks += 1

        print('Image Set Size: {}, Patient Set Size: {}'.format(len(img_list), len(patids)))
        print('NDBE Image Set Size: {}, NDBE Patient Set Size: {}'.format(counter_ndbe, len(patids_ndbe)))
        print('NEO Image Set Size: {}, NEO Patient Set Size: {}, NEO masks: {}'.format(counter_neo, len(patids_neo), counter_masks))
        print('Check Images: {}, Check Patients: {}, Overlap: {}'
              .format(counter_neo + counter_ndbe, len(patids_ndbe) + len(patids_neo),
                      len(list(set(patids_neo) & set(patids_ndbe)))))

        return img_list


def sample_weights(img_list):

    # Initialize indices from 0 - length of dataset [0, 1, ..., length]
    indices = list(range(len(img_list)))

    # Initialize empty list for weights
    weights = list()

    # Initialize counters for number of benign and malignant images
    n_benign = 0.
    n_malign = 0.

    # Initialize counters for neoplastic masks and subtlety
    n_masks = 0.
    n_subtle_0 = 0.
    n_subtle_1 = 0.
    n_subtle_2 = 0.

    # Loop over the indices
    for ind in indices:

        # Extract label from image list
        label = img_list[ind]['label']
        mask = img_list[ind]['mask']
        subtlety = img_list[ind]['subtlety']

        # Count numbers of interest
        if not label:
            n_benign += 1.
        elif label:
            n_malign += 1.
            if len(mask) > 0:
                n_masks += 1.
            if subtlety == 0:
                n_subtle_0 += 1.
            elif subtlety == 1:
                n_subtle_1 += 1.
            elif subtlety == 2:
                n_subtle_2 += 1.

    # print('Benign: {}, Malign: {}'.format(n_benign, n_malign))
    # print('Number of neoplastic masks: {}'.format(n_masks))
    # print('Subtlety Neo 0: {}, 1: {}, 2: {}, Total: {}'.format(n_subtle_0, n_subtle_1, n_subtle_2,
    #                                                            n_subtle_0 + n_subtle_1 + n_subtle_2))

    # Loop over the indices
    for ind in indices:

        # Extract label from image list
        label = img_list[ind]['label']
        subtlety = img_list[ind]['subtlety']

        """Assign weights to all the indices for 50:50 sampling NDBE:NEO"""
        if not label:
            weight = (1.0 / 2.0) * (1.0 / n_benign)
            assert weight > 0.
        elif label:
            weight = (1.0 / 2.0) * (1.0 / n_malign)
            assert weight > 0.
        weights.append(weight)

        """Assign weights to all the indices 50:50 NDBE:NEO (with 12.5:12.5:25 Subtlety 0:1:2"""
        # if not label:
        #     weight = (1.0 / 2.0) * (1.0 / n_benign)
        #     assert weight > 0.
        # elif label:
        #     if subtlety == 0:
        #         weight = (1.0 / 8.0) * (1.0 / n_subtle_0)
        #         assert weight > 0.
        #     elif subtlety == 1:
        #         weight = (1.0 / 8.0) * (1.0 / n_subtle_1)
        #         assert weight > 0.
        #     elif subtlety == 2:
        #         weight = (1.0 / 4.0) * (1.0 / n_subtle_2)
        #         assert weight > 0.
        # weights.append(weight)

    # Convert weights to numpy array in float64 format
    weights = np.array(weights, dtype=np.float64)
    # print('Length:  {}, Sum: {}'.format(len(weights), np.sum(weights)))

    return weights


""""""""""""""""""""""""""""""""""""""""""
"""" FUNCTION FOR CREATING TRANSFORMS """
""""""""""""""""""""""""""""""""""""""""""


def augmentations(opt):

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
                             Normalize(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                       std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

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
                           Normalize(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                     std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

    # Specify augmentation techniques for test set
    test_transforms.extend([Resize([opt.imagesize, opt.imagesize]),
                            ToTensor(),
                            Normalize(mean=[WLE_MEAN[0], WLE_MEAN[1], WLE_MEAN[2]],
                                      std=[WLE_STD[0], WLE_STD[1], WLE_STD[2]])])

    # Compose transforms and place into dictionary
    data_transforms['train'] = Compose(train_transforms)
    data_transforms['val'] = Compose(val_transforms)
    data_transforms['test'] = Compose(test_transforms)

    return data_transforms


""""""""""""""""""""""""""""""""""""""""""
"""" DATASET FOR TRAINING AND TESTING """
""""""""""""""""""""""""""""""""""""""""""


class DATASET_TRAIN_TEST(Dataset):
    def __init__(self, inclusion, transform=None, random_noise=False):
        self.inclusion = inclusion
        self.transform = transform
        self.random_noise = random_noise

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        label = self.inclusion[idx]['label']
        image = Image.open(img_name).convert('RGB')

        # By default set has_mask to zero
        has_mask = 0

        # Set has_mask for NDBE cases
        if label == np.array([0], dtype=np.float32):
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
                has_mask = 1

        # Create mask with all zeros when there are no available ones
        else:
            mask_np = np.zeros(image.size)
            mask = Image.fromarray(mask_np, mode='RGB').convert('1')
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


""""""""""""""""""""""""""""""""
"""" DATASET FOR VALIDATION """
""""""""""""""""""""""""""""""""


class DATASET_VAL(Dataset):
    def __init__(self, inclusion, transform=None):

        # For robustness do 4 times the validation set, with different augmentations
        self.inclusion = inclusion + inclusion + inclusion + inclusion
        self.transform = transform

    def __len__(self):
        return len(self.inclusion)

    def __getitem__(self, idx):
        img_name = self.inclusion[idx]['file']
        roi = self.inclusion[idx]['roi']
        label = self.inclusion[idx]['label']
        image = Image.open(img_name).convert('RGB')

        # By default set has_mask to zero
        has_mask = 0

        # Set has_mask for NDBE cases
        if label == np.array([0], dtype=np.float32):
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
                has_mask = 1

        # Create mask with all zeros when there are no available ones
        else:
            mask_np = np.zeros(image.size)
            mask = Image.fromarray(mask_np, mode='RGB').convert('1')
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
