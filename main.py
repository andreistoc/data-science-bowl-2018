import pathlib
import imageio
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy import ndimage
import pandas as pd
from glob import glob
import ntpath

ntpath.basename("a/b/c")


def rle_encoding(x):
    """

    :param x: numpy array of shape (height, width), 1 - mask, 0 - background
    :return: run length as list
    """

    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


# Glob the training data
training_paths = np.array(glob("input/stage1_train/*/images/*.png"))

# Glob the training targets
training_targets_paths = np.array(glob("input/stage1_train/*/masks/*.png"))

training_dict = {}

for training_path in training_paths:
    filename = path_leaf(training_path).replace('.png', '')
    masks = []
    for target_path in training_targets_paths:
        if filename in target_path:
            masks.append(target_path)
    training_dict[training_path] = masks

for item in training_dict:
    print(item, training_dict[item])

# Glob the testing data
testing = np.array(glob("input/stage1_test/*images/*.png"))
