"""IMPORT PACKAGES"""
import os
import json
import random
import numpy as np
import pandas as pd
from skimage.measure import label
from scipy import ndimage
from PIL import Image
from numba import jit
from collections import Counter
import shutil

"""SPECIFY EXTENSIONS AND DATA ROOTS"""
EXT_VID = ['.mp4', '.m4v',  '.avi']
EXT_IMG = ['.jpg', '.png', '.tiff', '.tif', '.bmp', '.jpeg']

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE SET OF FUNCTIONS FOR SELECTING ROI IN RAW ENDOSCOPE IMAGES"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Define function for minimum pooling the images
@jit(nopython=True)
def min_pooling(img, g=8):

    # Copy Image
    out = img.copy()

    # Determine image shape and compute step size for pooling
    h, w = img.shape
    nh = int(h / g)
    nw = int(w / g)

    # Perform minimum pooling
    for y in range(nh):
        for x in range(nw):
            out[g * y:g * (y + 1), g * x:g * (x + 1)] = np.min(out[g * y:g * (y + 1), g * x:g * (x + 1)])

    return out


# Define function for finding largest connected region in images
def getlargestcc(segmentation):

    # Use built-in label method, to label connected regions of an integer array
    labels = label(segmentation)

    # Assume at least 1 CC
    assert (labels.max() != 0)  # assume at least 1 CC

    # Find the largest of connected regions, return as True and False
    largestcc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largestcc


# Define function for finding bounding box coordinates for ROI in images
def bbox(img):

    # Find rows and columns where a True Bool is encountered
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    # Find the first and last row/column for the bounding box coordinates
    # cmin = left border, cmax = right border, rmin = top border, rmax = bottom border
    # Usage Image.crop((left, top, right, bottom))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


# Define Complete function for finding ROI bounding box coordinates, by combining previous functions
def find_roi(img):

    # Open image as numpy array
    image = np.array(img, dtype=np.uint8)

    # Compute L1 norm of the image
    norm_org = np.linalg.norm(image, axis=-1)

    # Use Gaussian Filter to capture low-frequency information
    img_gauss = ndimage.gaussian_filter(norm_org, sigma=5)

    # Scale pixel values
    img_scaled = ((norm_org - np.min(img_gauss)) / (np.max(img_gauss) - np.min(img_gauss))) * 255

    # Use minimum pooling
    img_norm = min_pooling(img_scaled, g=8)

    # Find largest connected region with threshold image as input
    th = 10
    largestcc = getlargestcc(img_norm >= th)

    # Obtain cropping coordinates
    rmin, rmax, cmin, cmax = bbox(largestcc)

    return rmin, rmax, cmin, cmax


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE FUNCTION FOR GENERATING RANDOM SPLIT TRAIN/VALIDATION/TEST"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def random_split_kvasir(root_dir):

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Sort and shuffle list
    img_files = sorted(img_files)
    random.seed(7)
    random.shuffle(img_files)

    # Define percentage for each set
    train_perc = 0.5
    val_perc = 0.25
    split_idx_train = round(len(img_files) * train_perc)
    split_idx_val = round(len(img_files) * (val_perc+train_perc))

    # Split the lists
    incl_train = img_files[:split_idx_train]
    incl_val = img_files[split_idx_train:split_idx_val]
    incl_test = img_files[split_idx_val:]

    # Move Training set
    for i in range(len(incl_train)):
        src_path = incl_train[i]
        imagename = os.path.split(incl_train[i])[1]
        folder_name = os.path.split(incl_train[i])[0]
        dest_path = os.path.join(folder_name, 'train', imagename)
        shutil.move(src_path, dest_path)

    # Move Validation Set
    for i in range(len(incl_val)):
        src_path = incl_val[i]
        imagename = os.path.split(incl_val[i])[1]
        folder_name = os.path.split(incl_val[i])[0]
        dest_path = os.path.join(folder_name, 'validation', imagename)
        shutil.move(src_path, dest_path)

    # Move Test Set
    for i in range(len(incl_test)):
        src_path = incl_test[i]
        imagename = os.path.split(incl_test[i])[1]
        folder_name = os.path.split(incl_test[i])[0]
        dest_path = os.path.join(folder_name, 'test', imagename)
        shutil.move(src_path, dest_path)


def random_split_giana(root_dir):

    # Create empty dictionary for image files
    img_files_inflam = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(os.path.join(root_dir, 'inflammatory')):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files_inflam.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for image files
    img_files_vasc = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(os.path.join(root_dir, 'vascularlesions')):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files_vasc.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Sort and shuffle list
    img_files_inflam = sorted(img_files_inflam)
    img_files_vasc = sorted(img_files_vasc)
    random.seed(7)
    random.shuffle(img_files_inflam)
    random.shuffle(img_files_vasc)

    # Define percentage for each set
    train_perc = 0.5
    val_perc = 0.25
    split_idx_train_inflam = round(len(img_files_inflam) * train_perc)
    split_idx_val_inflam = round(len(img_files_inflam) * (train_perc+val_perc))
    split_idx_train_vasc = round(len(img_files_vasc) * train_perc)
    split_idx_val_vasc = round(len(img_files_inflam) * (train_perc + val_perc))

    # Split the lists
    incl_train_inflam = img_files_inflam[:split_idx_train_inflam]
    incl_val_inflam = img_files_inflam[split_idx_train_inflam:split_idx_val_inflam]
    incl_test_inflam = img_files_inflam[split_idx_val_inflam:]
    incl_train_vasc = img_files_vasc[:split_idx_train_vasc]
    incl_val_vasc = img_files_vasc[split_idx_train_vasc:split_idx_val_vasc]
    incl_test_vasc = img_files_vasc[split_idx_val_vasc:]

    # Move Training set Inflammatory
    for i in range(len(incl_train_inflam)):
        src_path = incl_train_inflam[i]
        imagename = os.path.split(incl_train_inflam[i])[1]
        folder_name = os.path.split(os.path.split(incl_train_inflam[i])[0])[0]
        dest_path = os.path.join(folder_name, 'train', 'inflammatory', imagename)
        shutil.move(src_path, dest_path)

    # Move Validation set Inflammatory
    for i in range(len(incl_val_inflam)):
        src_path = incl_val_inflam[i]
        imagename = os.path.split(incl_val_inflam[i])[1]
        folder_name = os.path.split(os.path.split(incl_val_inflam[i])[0])[0]
        dest_path = os.path.join(folder_name, 'validation', 'inflammatory', imagename)
        shutil.move(src_path, dest_path)

    # Move Test set Inflammatory
    for i in range(len(incl_test_inflam)):
        src_path = incl_test_inflam[i]
        imagename = os.path.split(incl_test_inflam[i])[1]
        folder_name = os.path.split(os.path.split(incl_test_inflam[i])[0])[0]
        dest_path = os.path.join(folder_name, 'test', 'inflammatory', imagename)
        shutil.move(src_path, dest_path)

    # Move Training set Vascular
    for i in range(len(incl_train_vasc)):
        src_path = incl_train_vasc[i]
        imagename = os.path.split(incl_train_vasc[i])[1]
        folder_name = os.path.split(os.path.split(incl_train_vasc[i])[0])[0]
        dest_path = os.path.join(folder_name, 'train', 'vascularlesions', imagename)
        shutil.move(src_path, dest_path)

    # Move Validation set Vascular
    for i in range(len(incl_val_vasc)):
        src_path = incl_val_vasc[i]
        imagename = os.path.split(incl_val_vasc[i])[1]
        folder_name = os.path.split(os.path.split(incl_val_vasc[i])[0])[0]
        dest_path = os.path.join(folder_name, 'validation', 'vascularlesions', imagename)
        shutil.move(src_path, dest_path)

    # Move Test set Vascular
    for i in range(len(incl_test_vasc)):
        src_path = incl_test_vasc[i]
        imagename = os.path.split(incl_test_vasc[i])[1]
        folder_name = os.path.split(os.path.split(incl_test_vasc[i])[0])[0]
        dest_path = os.path.join(folder_name, 'test', 'vascularlesions', imagename)
        shutil.move(src_path, dest_path)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""DEFINE FUNCTION FOR CREATING CACHE WITH METADATA FOR AVAILABLE IMAGES"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def cache_giana(root_dir, mask_dir, storing_folder):

    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), storing_folder))

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for mask files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):

        # Loop over filenames in files
        for file in files:

            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    """"""""""""""""""""""""""""""""""""""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """"""""""""""""""""""""""""""""""""""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General folder structure: inflammatory/normal/vascular lesions
        data['file'] = img  # path to file
        data['class'] = os.path.split(os.path.split(img)[0])[1]  # inflammatory/normal/vascular lesions
        data['dataset'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]  # train/validation/test

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_giana_test_corrupt(root_dir, mask_dir, conversion_file, img_to_class, storing_folder):

    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), storing_folder))

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for mask files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):

        # Loop over filenames in files
        for file in files:

            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    # Read conversion file
    df = pd.read_excel(conversion_file)
    original_files = df['original file'].values.tolist()
    corrupted_files = df['corrupted file'].values.tolist()

    # Read images to class file
    df_class = pd.read_excel(img_to_class)
    images_to_class_files = df_class['image'].values.tolist()
    class_list = df_class['class'].values.tolist()

    """"""""""""""""""""""""""""""""""""""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """"""""""""""""""""""""""""""""""""""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Extract original image name from list
        original_file = original_files[corrupted_files.index(imgname)]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General folder structure: inflammatory/normal/vascular lesions
        data['file'] = img  # path to file
        data['class'] = class_list[images_to_class_files.index(original_file)]
        data['dataset'] = 'test-corrupt'
        data['original'] = original_file

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_sysucc(root_dir, storing_folder):

    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), storing_folder))

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    """"""""""""""""""""""""""""""""""""""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """"""""""""""""""""""""""""""""""""""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        # General folder structure: inflammatory/normal/vascular lesions
        data['file'] = img  # path to file
        data['class'] = os.path.split(os.path.split(img)[0])[1]                      # Cancer/Normal
        data['dataset'] = os.path.split(os.path.split(os.path.split(img)[0])[0])[1]  # train/test/validation

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_kvasir_seg(root_dir, mask_dir, storing_folder):

    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), storing_folder))

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for mask files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):

        # Loop over filenames in files
        for file in files:

            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    """"""""""""""""""""""""""""""""""""""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """"""""""""""""""""""""""""""""""""""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file
        data['dataset'] = os.path.split(os.path.split(img)[0])[1]       # train/validation/test

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def cache_kvasir_seg_test_corrupt(root_dir, mask_dir, conversion_file, storing_folder):

    # Create directory
    print('Generating cache...')
    os.makedirs(os.path.join(os.getcwd(), storing_folder))

    # Create empty dictionary for image files
    img_files = list()

    # Loop over roots (folders in root_dir), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(root_dir):

        # Loop over filenames in files
        for name in files:

            # Check for .extension of the files; append to video files or image files accordingly
            # os.path.splitext splits name in filename and .ext; [1] will correspond to .ext
            if os.path.splitext(name.lower())[1] in EXT_IMG:
                img_files.append(os.path.join(root, name))
            elif name == 'Thumbs.db':
                os.remove(os.path.join(root, name))
            else:
                print('FILE NOT SUPPORTED: {}'.format(os.path.join(root, name)))

    # Create empty dictionary for mask files
    maskdict = dict()

    # Loop over roots (folders in OTMASKDIR), dirs (folders in roots), files (files in dirs)
    for root, dirs, files in os.walk(mask_dir):

        # Loop over filenames in files
        for file in files:

            # Extract filename before .ext
            maskcase = os.path.splitext(file)[0]

            # Append filename to mask dictionary if already existing key; otherwise create key and append to list
            if maskcase in maskdict.keys():
                maskdict[maskcase].append(os.path.join(root, file))
            else:
                maskdict[maskcase] = list()
                maskdict[maskcase].append(os.path.join(root, file))

    # Read conversion file
    df = pd.read_excel(conversion_file)
    original_files = df['original file'].values.tolist()
    corrupted_files = df['corrupted file'].values.tolist()

    """"""""""""""""""""""""""""""""""""""
    """CREATE JSON FILE FOR EVERY IMAGE """
    """"""""""""""""""""""""""""""""""""""

    # Loop over list of img_files, which was created by appending the roots of all images in root_dir
    for img in img_files:
        print('Reading image: ', img)

        # Extract imagename from the path to image (img)
        imgname = os.path.split(img)[1]

        # Extract original image name from list
        original_file = original_files[corrupted_files.index(imgname)]

        # Create empty dictionary for metadata
        data = dict()

        # Extract information from filename and place in dictionary with corresponding key
        data['file'] = img  # path to file
        data['dataset'] = 'test-corrupt'
        data['original'] = original_file

        # Open the image as numpy array; extract height and width and place in data dictionary
        frame = np.array(Image.open(img))
        data['height'] = frame.shape[0]
        data['width'] = frame.shape[1]
        print('Frame shape: ', frame.shape)

        # Extract bounding box coordinates, store in dictionary as rmin, rmax, cmin, cmax
        roi = find_roi(frame)
        data['roi'] = [float(x) for x in roi]

        # Create new key in data dictionary for masks and initialize as list; instantiate with maskdict list
        data['masks'] = list()
        if os.path.splitext(os.path.split(img)[1])[0] in maskdict.keys():
            data['masks'] = maskdict[os.path.splitext(os.path.split(img)[1])[0]]

        # Print final version of data dictionary with all keys and values in there
        print('Data: ', data)

        # Check whether there is already a json file for this particular image; otherwise create file
        if not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_2.json')
        elif not os.path.exists(os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')):
            jsonfile = os.path.join(os.getcwd(), storing_folder, os.path.splitext(imgname)[0] + '_3.json')
        else:
            raise ValueError

        # For every img in img_files write dictionary data into corresponding json file
        with open(jsonfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)


""""""""""""""""""""""""""
"""EXECUTION OF FUNCTIONS"""
""""""""""""""""""""""""""
if __name__ == '__main__':

    # Define Paths for GIANA dataset
    giana_img_dir = 'D:/Public Endoscopic Datasets/GIANA/Images'
    giana_mask_dir = 'D:/Public Endoscopic Datasets/GIANA/Masks'
    giana_store = 'cache_giana'

    giana_img_dir_corrupt = 'D:/Public Endoscopic Datasets/GIANA-Test-C/Images'
    giana_mask_dir_corrupt = 'D:/Public Endoscopic Datasets/GIANA-Test-C/Masks'
    giana_conv_file = 'D:/Public Endoscopic Datasets/GIANA-Test-C/conversion_file.xlsx'
    giana_img_to_class = 'D:/Public Endoscopic Datasets/GIANA-Test-C/images_to_class.xlsx'
    giana_store_corrupt = 'cache_giana_test_corrupt'

    # Define paths for Sysucc dataset
    sysucc_img_dir = 'D:/Public Endoscopic Datasets/SYSUCC/Preprocessed/Images'
    sysucc_store = 'cache_sysucc'

    # Define paths for Kvasir-SEG dataset
    kvasir_seg_img_dir = 'D:/Public Endoscopic Datasets/HYPERKVASIR/segmented_images_preprocessed/Images'
    kvasir_seg_mask_dir = 'D:/Public Endoscopic Datasets/HYPERKVASIR/segmented_images_preprocessed/Masks'
    kvasir_seg_store = 'cache_kvasir_seg'

    kvasir_seg_img_dir_corrupt = 'D:/Public Endoscopic Datasets/KVASIR-Test-C/Images'
    kvasir_seg_mask_dir_corrupt = 'D:/Public Endoscopic Datasets/KVASIR-Test-C/Masks'
    kvasir_seg_conv_file_dir = 'D:/Public Endoscopic Datasets/KVASIR-Test-C/conversion_file.xlsx'
    kvasir_seg_store_corrupt = 'cache_kvasir_seg_test_corrupt'

    # Generate Cache
    # cache_giana(root_dir=giana_img_dir, mask_dir=giana_mask_dir, storing_folder=giana_store)
    # cache_sysucc(root_dir=sysucc_img_dir, storing_folder=sysucc_store)
    # cache_kvasir_seg(root_dir=kvasir_seg_img_dir, mask_dir=kvasir_seg_mask_dir, storing_folder=kvasir_seg_store)

    # Generate Cache Corrupt
    # cache_giana_test_corrupt(root_dir=giana_img_dir_corrupt, mask_dir=giana_mask_dir_corrupt,
    #                          conversion_file=giana_conv_file, img_to_class=giana_img_to_class,
    #                          storing_folder=giana_store_corrupt)
    # cache_kvasir_seg_test_corrupt(root_dir=kvasir_seg_img_dir_corrupt, mask_dir=kvasir_seg_mask_dir_corrupt,
    #                               conversion_file=kvasir_seg_conv_file_dir, storing_folder=kvasir_seg_store_corrupt)

