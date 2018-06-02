import cv2
import math
import scipy
from scipy import ndimage
import numpy as np
from math import ceil
import glob, os
from sklearn import preprocessing
import h5py

## Preprocess data and store in hdf5 format
## Read batch of data, random sample (not actually random, just close)
## We cannot randomize over all dataset cause of sequential reading/processing
## From each batch, read a small portion to serve as validation and test data

#### USER INPUT PARAMETERS ####

# First, select the folder which we want to search
# If subfolder list is filled, search only in mentioned subfolders
# Otherwise take main folder and use it

## OPTION 1 - Oxford with subfolders
# data_folder = 'E:\\Datasets\\oxbuild_images'
# sub_folders = ['all_souls', 'asmolean', 'balliol', 'bodleian', 'christ_church', \
#                'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
## OPTION 2 - VOC - all images in 1 folder (no correspondencies)
data_folder = 'E:\\Datasets\\VOC12\\JPEGImages' 
sub_folders = None

# Next, specify the file in which to store data
store_file = 'E:\\Datasets\\32_32_VOC_zero_cent_noscale_final.h5'

# Variables to control the train/valid/test dataset relative size
# In percentages

train_size = 0.85
valid_size = 0.10
test_size = 1 - train_size - valid_size

# Batch size
# Define the batch size of image patches to process each time.
# The actual value may be different cause to unknown size of keypoints per image

batch_size = 10000

# EARLY STOP SIZE
# Define maximum size of dataset (when we want to extract all, define ti as -1

max_size = -1  # 1000000

# percentage of noise
# Define percentage of noise added to image for fine  tuning phase

_p = 15.

# Extraction neighbourhood
# X times the meaningful radius of patch

n_mul = 10

# Size of patch
# Same for width / heigth - Will be resized to the size below after extracting

p_size = 32

# OPTIONS FOR DATASET
_not_scale = True # If selected, not meaningful region will be selected based on the keypoint meaningful size. region will be set to p_size * p_size from original image
_not_rotation = False # if selected - patches wont be rotated to dominant orientation
_add_noise = False
_add_rotation = False

_zero_centered = True # just center to -128:127 by substracting constant
_zero_mean_one_std = False
_unit_norm = False
_zero_one = False

# Open file and prepare for append
h5f = h5py.File(store_file, 'a')
train_h = h5f.create_dataset("train_dataset", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
valid_h = h5f.create_dataset("valid_dataset", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
test_h = h5f.create_dataset("test_dataset", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
if _add_noise:
    train_n = h5f.create_dataset("train_dataset_noise", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    valid_n = h5f.create_dataset("valid_dataset_noise", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    test_n = h5f.create_dataset("test_dataset_noise", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
if _add_rotation:
    train_r1 = h5f.create_dataset("train_dataset_r1", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    valid_r1 = h5f.create_dataset("valid_dataset_r1", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    test_r1 = h5f.create_dataset("test_dataset_r1", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    train_r2 = h5f.create_dataset("train_dataset_r2", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    valid_r2 = h5f.create_dataset("valid_dataset_r2", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))
    test_r2 = h5f.create_dataset("test_dataset_r2", (0, p_size, p_size, 1), maxshape=(None, p_size, p_size, 1))

n_num = 5  # define noise levels (how many additional datasets for noise
# Issue is that we needto shuffle noise the same way as the train/valid dataset (unison shuffling)


#### END OF USER INPUT ####

# Definitions
sift = cv2.xfeatures2d.SIFT_create()


def sobel_two(image):
    results = np.zeros(shape=(image.shape[0], image.shape[1], 2))
    for i in range(2):
        results[:, :, i] = ndimage.sobel(image, axis=i)

    return results



def shuffle_in_unison(a, b, c=None, d=None):
    if c is None and d is None:
        assert len(a) == len(b)
    elif d is None:
        assert len(a) == len(b) == len(c)
    else:
        assert len(a) == len(b) == len(c) == len(d)

    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    if c is not None:
        shuffled_c = np.empty(c.shape, dtype=c.dtype)
    if d is not None:
        shuffled_d = np.empty(d.shape, dtype=d.dtype)

    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        if c is not None:
            shuffled_c[new_index] = c[old_index]
        if d is not None:
            shuffled_d[new_index] = d[old_index]

    if c is None and d is None:
        return shuffled_a, shuffled_b
    elif d is None:
        return shuffled_a, shuffled_b, shuffled_c
    else:
        return shuffled_a, shuffled_b, shuffled_c, shuffled_d


def extract_patches(data_fodler):
    # Declare local variables:
    num_kp = 0  # number of extracted keypoints
    im_patches_all = None
    im_patches_all = np.zeros(shape=(0, p_size, p_size, 1))
    if _add_noise:
        im_patches_all_noisy = None
        im_patches_all_noisy = np.zeros(shape=(0, p_size, p_size, 1))
    if _add_rotation:
        im_patches_all_rotated = None
        im_patches_all_rotated = np.zeros(shape=(0, p_size, p_size, 1))
        im_patches_all_rotated_c = None
        im_patches_all_rotated_c = np.zeros(shape=(0, p_size, p_size, 1))
    # First load all images in the folder
    for file in os.listdir(data_fodler):
        if file.endswith(".jpg"):
            # print(os.path.join(data_fodler, file))
            # Load the image
            img = cv2.imread(os.path.join(data_fodler, file))
            # Convert it to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect keypoints
            kp = sift.detect(img, None)
            # Get shape of image
            cols, rows = img.shape
            # Declare memory for patches
            im_patches = None
            im_patches = np.zeros(shape=(len(kp), p_size, p_size))

            if _add_noise:
                # Declare memory for noise patches
                im_patches_noisy = None
                im_patches_noisy = np.zeros(shape=(len(kp), p_size, p_size))
            if _add_rotation:
                # Declare memory for rotated patches (clockwise and counter clockwise
                im_patches_rotated = None
                im_patches_rotated = np.zeros(shape=(len(kp), p_size, p_size))
                im_patches_rotated_c = None
                im_patches_rotated_c = np.zeros(shape=(len(kp), p_size, p_size))

            # Variable to store the index of good keypoints
            u = 0
            # Extract patches and store them
            for i in range(len(kp)):
                # Get meaningful radius of keypoint
                octave = kp[i].octave & 255
                octave = octave if (octave < 128) else (-128 | octave);
                scale = 1. / (1 << octave) if octave >= 0 else (float)(1 << -octave)
                radius = kp[i].size * scale
                # Check if we can extract all information from the patch (in range of image)
                if not _not_scale:
                    if round((kp[i].pt[0]) - round(math.sqrt(2 * math.pow(radius * n_mul / 2, 2)))) < 0:
                        continue
                    if round((kp[i].pt[1]) - round(math.sqrt(2 * math.pow(radius * n_mul / 2, 2)))) < 0:
                        continue
                    if round((kp[i].pt[0]) + round(math.sqrt(2 * math.pow(radius * n_mul / 2, 2)))) > img.shape[0]:
                        continue
                    if round((kp[i].pt[1]) + round(math.sqrt(2 * math.pow(radius * n_mul / 2, 2)))) > img.shape[1]:
                        continue
                else:
                    # no rescaling to meaningful region
                    if round((kp[i].pt[0]) - round(math.sqrt(2 * math.pow(p_size / 2, 2)))) < 0:
                        continue
                    if round((kp[i].pt[1]) - round(math.sqrt(2 * math.pow(p_size / 2, 2)))) < 0:
                        continue
                    if round((kp[i].pt[0]) + round(math.sqrt(2 * math.pow(p_size / 2, 2)))) > img.shape[0]:
                        continue
                    if round((kp[i].pt[1]) + round(math.sqrt(2 * math.pow(p_size / 2, 2)))) > img.shape[1]:
                        continue

                # when rotation, do affine transform
                if not _not_rotation:
                    # Get rotation matrix
                    M = cv2.getRotationMatrix2D((kp[i].pt[0], kp[i].pt[1]), kp[i].angle, 1)
                    # Transform based on rotation from keypoint
                    dst = cv2.warpAffine(img, M, (cols, rows))
                else:
                    dst = img
                # Get patch
                if not _not_scale:
                    roi = dst[round(kp[i].pt[1] - radius * n_mul / 2):round(kp[i].pt[1] + radius * n_mul / 2),
                          round(kp[i].pt[0] - radius * n_mul / 2):round(kp[i].pt[0] + radius * n_mul / 2)]
                else:
                    roi = dst[round(kp[i].pt[1] - p_size / 2):round(kp[i].pt[1] + p_size / 2),
                          round(kp[i].pt[0] - p_size / 2):round(kp[i].pt[0] + p_size / 2)]
                # Resize patch to selected size
                dst_resize = cv2.resize(roi, (p_size, p_size));
                # Store image & scale (zero mean 1 variance)
                im_patches[u, :, :] = dst_resize
                if _zero_centered:
                    im_patches[u, :, :] = (im_patches[u, :, :] / 255 - 0.5)
                if _zero_mean_one_std:
                    im_patches[u, :, :] = preprocessing.scale(im_patches[u, :, :].reshape(p_size * p_size)) \
                        .reshape(p_size, p_size)  # UPDATE MJ 23.1.2018
                if _unit_norm:
                    im_patches[u, :, :] = preprocessing.normalize((im_patches[u, :, :].reshape(p_size * p_size)) \
                                                                  .reshape(1, -1))[0].reshape(p_size,
                                                                                              p_size)  # UPDATE MJ 17.4.2018
                if _zero_one:
                    im_patches[u, :, :] = preprocessing.minmax_scale(im_patches[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)

                # Random noise for fine tuning (noise dataset)
                if _add_noise:
                    if _zero_centered:
                        im_patches_noisy[u, :, :] = (im_patches_noisy[u, :, :] / 255 - 0.5)
                    noise_train = np.random.choice([0, 1], size=dst_resize.shape, p=[_p / 100, (100 - _p) / 100])
                    im_patches_noisy[u, :, :] = np.multiply(dst_resize, noise_train)

                    if _zero_mean_one_std:
                        im_patches_noisy[u, :, :] = preprocessing.scale(
                            im_patches_noisy[u, :, :].reshape(p_size * p_size)).reshape(
                            p_size,
                            p_size)
                    if _unit_norm:
                        im_patches_noisy[u, :, :] = preprocessing.normalize((im_patches_noisy[u, :, :] \
                            .reshape(p_size * p_size)).reshape(1, -1))[0].reshape(p_size, p_size)
                    if _zero_one:
                        im_patches_noisy[u, :, :] = preprocessing.minmax_scale(
                            im_patches_noisy[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)

                if _add_rotation:
                    # rotation dataset
                    # Get rotation matrix
                    M = cv2.getRotationMatrix2D((kp[i].pt[0], kp[i].pt[1]), kp[i].angle + 5, 1)  # counter clockwise
                    # Transform based on rotation from keypoint
                    dst = cv2.warpAffine(img, M, (cols, rows))
                    # Get patch
                    if not _not_scale:
                        roi = dst[round(kp[i].pt[1] - radius * n_mul / 2):round(kp[i].pt[1] + radius * n_mul / 2),
                            round(kp[i].pt[0] - radius * n_mul / 2):round(kp[i].pt[0] + radius * n_mul / 2)]
                    else:
                        roi = dst[round(kp[i].pt[1] - p_size / 2):round(kp[i].pt[1] + p_size / 2),
                              round(kp[i].pt[0] - p_size / 2):round(kp[i].pt[0] + p_size / 2)]
                    # Resize patch to selected size
                    dst_resize = cv2.resize(roi, (p_size, p_size));
                    # Store image & scale (zero mean 1 variance)
                    im_patches_rotated[u, :, :] = dst_resize
                    if _zero_centered:
                        im_patches_rotated[u, :, :] = (im_patches_rotated[u, :, :] / 255 - 0.5)
                    if _zero_mean_one_std:
                        im_patches_rotated[u, :, :] = preprocessing.scale(
                            im_patches_rotated[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)
                    if _unit_norm:
                        im_patches_rotated[u, :, :] = preprocessing.normalize((im_patches_rotated[u, :, :] \
                            .reshape(p_size * p_size)).reshape(1, -1))[0].reshape(p_size, p_size)
                    if _zero_one:
                        im_patches_rotated[u, :, :] = preprocessing.minmax_scale(
                            im_patches_rotated[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)

                    M = cv2.getRotationMatrix2D((kp[i].pt[0], kp[i].pt[1]), kp[i].angle - 5, 1)  # clockwise
                    # Transform based on rotation from keypoint
                    dst = cv2.warpAffine(img, M, (cols, rows))
                    # Get patch
                    if not _not_scale:
                        roi = dst[round(kp[i].pt[1] - radius * n_mul / 2):round(kp[i].pt[1] + radius * n_mul / 2),
                            round(kp[i].pt[0] - radius * n_mul / 2):round(kp[i].pt[0] + radius * n_mul / 2)]
                    else:
                        roi = dst[round(kp[i].pt[1] - p_size / 2):round(kp[i].pt[1] + p_size / 2),
                              round(kp[i].pt[0] - p_size / 2):round(kp[i].pt[0] + p_size / 2)]
                    # Resize patch to selected size
                    dst_resize = cv2.resize(roi, (p_size, p_size));
                    # Store image & scale (zero mean 1 variance)
                    im_patches_rotated_c[u, :, :] = dst_resize
                    if _zero_centered:
                        im_patches_rotated_c[u, :, :] = (im_patches_rotated_c[u, :, :] / 255 - 0.5)
                    if _zero_mean_one_std:
                        im_patches_rotated_c[u, :, :] = preprocessing.scale(
                            im_patches_rotated_c[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)
                    if _unit_norm:
                        im_patches_rotated_c[u, :, :] = preprocessing.normalize((im_patches_rotated_c[u, :, :] \
                            .reshape(p_size * p_size)).reshape(1, -1))[0].reshape(p_size, p_size)
                    if _zero_one:
                        im_patches_rotated_c[u, :, :] = preprocessing.minmax_scale(
                            im_patches_rotated_c[u, :, :].reshape(p_size * p_size)).reshape(p_size, p_size)

                # If the process comes to this point, we know we got the good patch
                # Increment u variable to reshape finished dataset later

                # CHECK FOR ZEROS
                if np.any(im_patches[u, :, :]):
                    u = u + 1
                else:
                    print('zly')
            # Remove the empty list at the end
            im_patches = im_patches[1:u, :, :]
            im_patches = im_patches.reshape((-1, p_size, p_size, 1)).astype(float)
            # Random shuffle the patches from single image
            # np.random.shuffle(im_patches)
            if _add_noise:
                im_patches_noisy = im_patches_noisy[1:u, :, :]
                im_patches_noisy = im_patches_noisy.reshape((-1, p_size, p_size, 1)).astype(float)
                # np.random.shuffle(im_patches_noisy)
            if _add_rotation:
                im_patches_rotated = im_patches_rotated[1:u, :, :]
                im_patches_rotated = im_patches_rotated.reshape((-1, p_size, p_size, 1)).astype(float)
                im_patches_rotated_c = im_patches_rotated_c[1:u, :, :]
                im_patches_rotated_c = im_patches_rotated_c.reshape((-1, p_size, p_size, 1)).astype(float)
                # np.random.shuffle(im_patches_rotated)
                # np.random.shuffle(im_patches_rotated_c)

            # Unison shuffle of dataset - we need to have correspondencies in indices for different data from original
            if _add_noise and _add_rotation:
                im_patches, im_patches_noisy, im_patches_rotated, im_patches_rotated_c = \
                    shuffle_in_unison(im_patches, im_patches_noisy, im_patches_rotated, im_patches_rotated_c)
            elif _add_noise:
                im_patches, im_patches_noisy = \
                    shuffle_in_unison(im_patches, im_patches_noisy)
            elif _add_rotation:
                im_patches, im_patches_rotated, im_patches_rotated_c = \
                    shuffle_in_unison(im_patches, im_patches_rotated, im_patches_rotated_c)
            else:
                np.random.shuffle(im_patches)

            # Append to the whole batch
            im_patches_all = np.append(im_patches_all, im_patches, axis=0)
            if _add_noise:
                im_patches_all_noisy = np.append(im_patches_all_noisy, im_patches_noisy, axis=0)
            if _add_rotation:
                im_patches_all_rotated = np.append(im_patches_all_rotated, im_patches_rotated, axis=0)
                im_patches_all_rotated_c = np.append(im_patches_all_rotated_c, im_patches_rotated_c, axis=0)
            # Check if batch size has been overshot
            if len(im_patches_all) > batch_size:
                # Shuffle whole batch
                # Unison shuffle of dataset - we need to have correspondencies in indices for different data from original
                if _add_noise and _add_rotation:
                    im_patches_all, im_patches_all_noisy, im_patches_all_rotated, im_patches_all_rotated_c = \
                        shuffle_in_unison(im_patches_all, im_patches_all_noisy, im_patches_all_rotated,
                                          im_patches_all_rotated_c)
                elif _add_noise:
                    im_patches_all, im_patches_all_noisy = \
                        shuffle_in_unison(im_patches_all, im_patches_all_noisy)
                elif _add_rotation:
                    im_patches_all, im_patches_all_rotated, im_patches_all_rotated_c = \
                        shuffle_in_unison(im_patches_all, im_patches_all_rotated, im_patches_all_rotated_c)
                else:
                    np.random.shuffle(im_patches_all)

                # Split into training / valid / test datasets
                end_t = round(len(im_patches_all) * train_size)
                train_set = im_patches_all[1:end_t, :, :, :]
                end_v = end_t + round(len(im_patches_all) * valid_size)
                valid_set = im_patches_all[end_t:end_v, :, :, :]
                test_set = im_patches_all[end_v:, :, :, :]

                # noisy
                if _add_noise:
                    end_t = round(len(im_patches_all_noisy) * train_size)
                    train_set_n = im_patches_all_noisy[1:end_t, :, :, :]
                    end_v = end_t + round(len(im_patches_all_noisy) * valid_size)
                    valid_set_n = im_patches_all_noisy[end_t:end_v, :, :, :]
                    test_set_n = im_patches_all_noisy[end_v:, :, :, :]

                # rotated
                if _add_rotation:
                    end_t = round(len(im_patches_all_rotated) * train_size)
                    train_set_r = im_patches_all_rotated[1:end_t, :, :, :]
                    end_v = end_t + round(len(im_patches_all_rotated) * valid_size)
                    valid_set_r = im_patches_all_rotated[end_t:end_v, :, :, :]
                    test_set_r = im_patches_all_rotated[end_v:, :, :, :]

                    end_t = round(len(im_patches_all_rotated_c) * train_size)
                    train_set_r_c = im_patches_all_rotated_c[1:end_t, :, :, :]
                    end_v = end_t + round(len(im_patches_all_rotated_c) * valid_size)
                    valid_set_r_c = im_patches_all_rotated_c[end_t:end_v, :, :, :]
                    test_set_r_c = im_patches_all_rotated_c[end_v:, :, :, :]

                # Store as an append file
                # First reshape
                train_h.resize(train_h.shape[0] + len(train_set), axis=0)
                valid_h.resize(valid_h.shape[0] + len(valid_set), axis=0)
                test_h.resize(test_h.shape[0] + len(test_set), axis=0)

                if _add_noise:
                    train_n.resize(train_n.shape[0] + len(train_set_n), axis=0)
                    valid_n.resize(valid_n.shape[0] + len(valid_set_n), axis=0)
                    test_n.resize(test_n.shape[0] + len(test_set_n), axis=0)

                if _add_rotation:
                    train_r1.resize(train_r1.shape[0] + len(train_set_r), axis=0)
                    valid_r1.resize(valid_r1.shape[0] + len(valid_set_r), axis=0)
                    test_r1.resize(test_r1.shape[0] + len(test_set_r), axis=0)

                    train_r2.resize(train_r2.shape[0] + len(train_set_r_c), axis=0)
                    valid_r2.resize(valid_r2.shape[0] + len(valid_set_r_c), axis=0)
                    test_r2.resize(test_r2.shape[0] + len(test_set_r_c), axis=0)

                # Now append
                train_h[-len(train_set):] = train_set
                valid_h[-len(valid_set):] = valid_set
                test_h[-len(test_set):] = test_set

                if _add_noise:
                    train_n[-len(train_set_n):] = train_set_n
                    valid_n[-len(valid_set_n):] = valid_set_n
                    test_n[-len(test_set_n):] = test_set_n

                if _add_rotation:
                    train_r1[-len(train_set_r):] = train_set_r
                    valid_r1[-len(valid_set_r):] = valid_set_r
                    test_r1[-len(test_set_r):] = test_set_r

                    train_r2[-len(train_set_r_c):] = train_set_r_c
                    valid_r2[-len(valid_set_r_c):] = valid_set_r_c
                    test_r2[-len(test_set_r_c):] = test_set_r_c

                # Reset the batch holders
                im_patches_all = None
                im_patches_all = np.zeros(shape=(0, p_size, p_size, 1))
                train_set = None
                valid_set = None
                test_set = None

                if _add_noise:
                    im_patches_all_noisy = None
                    im_patches_all_noisy = np.zeros(shape=(0, p_size, p_size, 1))
                    train_set_n = None
                    valid_set_n = None
                    test_set_n = None
                if _add_rotation:
                    im_patches_all_rotated = None
                    im_patches_all_rotated = np.zeros(shape=(0, p_size, p_size, 1))
                    train_set_r = None
                    valid_set_r = None
                    test_set_r = None

                    im_patches_all_rotated_c = None
                    im_patches_all_rotated_c = np.zeros(shape=(0, p_size, p_size, 1))
                    train_set_r_c = None
                    valid_set_r_c = None
                    test_set_r_c = None

                if max_size > 0 and max_size <= train_h.shape[0]:
                    # Close file
                    h5f.close()
                    # exit
                    exit(0)

    # Last patches which didnt fulfill minimum batch
    if im_patches_all is not None and len(im_patches_all) > 0:
        # Shuffle whole batch
        # Unison shuffle of dataset - we need to have correspondencies in indices for different data from original
        if _add_noise and _add_rotation:
            im_patches_all, im_patches_all_noisy, im_patches_all_rotated, im_patches_all_rotated_c = \
                shuffle_in_unison(im_patches_all, im_patches_all_noisy, im_patches_all_rotated,
                                  im_patches_all_rotated_c)
        elif _add_noise:
            im_patches_all, im_patches_all_noisy = \
                shuffle_in_unison(im_patches_all, im_patches_all_noisy)
        elif _add_rotation:
            im_patches_all, im_patches_all_rotated, im_patches_all_rotated_c = \
                shuffle_in_unison(im_patches_all, im_patches_all_rotated, im_patches_all_rotated_c)
        else:
            np.random.shuffle(im_patches_all)

        # Split into training / valid / test datasets
        end_t = round(len(im_patches_all) * train_size)
        train_set = im_patches_all[1:end_t, :, :, :]
        end_v = end_t + round(len(im_patches_all) * valid_size)
        valid_set = im_patches_all[end_t:end_v, :, :, :]
        test_set = im_patches_all[end_v:, :, :, :]

        # noisy
        if _add_noise:
            end_t = round(len(im_patches_all_noisy) * train_size)
            train_set_n = im_patches_all_noisy[1:end_t, :, :, :]
            end_v = end_t + round(len(im_patches_all_noisy) * valid_size)
            valid_set_n = im_patches_all_noisy[end_t:end_v, :, :, :]
            test_set_n = im_patches_all_noisy[end_v:, :, :, :]

        # rotated
        if _add_noise:
            end_t = round(len(im_patches_all_rotated) * train_size)
            train_set_r = im_patches_all_rotated[1:end_t, :, :, :]
            end_v = end_t + round(len(im_patches_all_rotated) * valid_size)
            valid_set_r = im_patches_all_rotated[end_t:end_v, :, :, :]
            test_set_r = im_patches_all_rotated[end_v:, :, :, :]

            end_t = round(len(im_patches_all_rotated_c) * train_size)
            train_set_r_c = im_patches_all_rotated_c[1:end_t, :, :, :]
            end_v = end_t + round(len(im_patches_all_rotated_c) * valid_size)
            valid_set_r_c = im_patches_all_rotated_c[end_t:end_v, :, :, :]
            test_set_r_c = im_patches_all_rotated_c[end_v:, :, :, :]

        # Store as an append file
        # First reshape
        train_h.resize(train_h.shape[0] + len(train_set), axis=0)
        valid_h.resize(valid_h.shape[0] + len(valid_set), axis=0)
        test_h.resize(test_h.shape[0] + len(test_set), axis=0)

        if _add_noise:
            train_n.resize(train_n.shape[0] + len(train_set_n), axis=0)
            valid_n.resize(valid_n.shape[0] + len(valid_set_n), axis=0)
            test_n.resize(test_n.shape[0] + len(test_set_n), axis=0)

        if _add_rotation:
            train_r1.resize(train_r1.shape[0] + len(train_set_r), axis=0)
            valid_r1.resize(valid_r1.shape[0] + len(valid_set_r), axis=0)
            test_r1.resize(test_r1.shape[0] + len(test_set_r), axis=0)

            train_r2.resize(train_r2.shape[0] + len(train_set_r_c), axis=0)
            valid_r2.resize(valid_r2.shape[0] + len(valid_set_r_c), axis=0)
            test_r2.resize(test_r2.shape[0] + len(test_set_r_c), axis=0)

        # Now append
        train_h[-len(train_set):] = train_set
        valid_h[-len(valid_set):] = valid_set
        test_h[-len(test_set):] = test_set

        if _add_noise:
            train_n[-len(train_set_n):] = train_set_n
            valid_n[-len(valid_set_n):] = valid_set_n
            test_n[-len(test_set_n):] = test_set_n

        if _add_rotation:
            train_r1[-len(train_set_r):] = train_set_r
            valid_r1[-len(valid_set_r):] = valid_set_r
            test_r1[-len(test_set_r):] = test_set_r

            train_r2[-len(train_set_r_c):] = train_set_r_c
            valid_r2[-len(valid_set_r_c):] = valid_set_r_c
            test_r2[-len(test_set_r_c):] = test_set_r_c

        # Reset the batch holders
        im_patches_all = None
        im_patches_all = np.zeros(shape=(0, p_size, p_size, 1))
        train_set = None
        valid_set = None
        test_set = None

        if _add_noise:
            im_patches_all_noisy = None
            im_patches_all_noisy = np.zeros(shape=(0, p_size, p_size, 1))
            train_set_n = None
            valid_set_n = None
            test_set_n = None

        if _add_rotation:
            im_patches_all_rotated = None
            im_patches_all_rotated = np.zeros(shape=(0, p_size, p_size, 1))
            train_set_r = None
            valid_set_r = None
            test_set_r = None

            im_patches_all_rotated_c = None
            im_patches_all_rotated_c = np.zeros(shape=(0, p_size, p_size, 1))
            train_set_r_c = None
            valid_set_r_c = None
            test_set_r_c = None

    # Run program


if sub_folders != None and len(sub_folders) > 0:
    for sfolders in sub_folders:
        extract_patches(data_folder + '\\' + sfolders)
else:
    extract_patches(data_folder)
# Close file
h5f.close()
