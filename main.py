import imageio
import numpy as np
from glob import glob
import ntpath
from keras.preprocessing import image
from tqdm import tqdm
from PIL import Image
import tensorflow

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
    """
    Get the filename from a path
    :param path: string path
    :return: string filename
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def pad_image(old_im, new_size):
    """
    Pad an image to a desired size
    :param old_im: two element array with dimensions
    :param new_size: new desired size
    :return: new image padded with black
    """
    old_size = old_im.size
    new_im = Image.new('RGB', new_size)
    new_im.paste(old_im, ((new_size[0] - old_size[0]) / 2,
                          (new_size[1] - old_size[1]) / 2))

    return new_im


def path_to_tensor(img_path, size):
    """
    Takes in the path to an image and converts it to a 4D Tensor
    :param img_path: string path to image
    :param size: desired size
    :return:
    """
    img = image.load_img(img_path)
    img = pad_image(img, size)
    # convert PIL.Image.Image type to 3D tensor
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    Converts a list of images to an array of tensors
    :param img_paths: paths to images
    :return:
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# TODO implement target merging function


# Glob the training data
training_paths = np.array(glob("input/stage1_train/*/images/*.png"))

print("Loaded training images. \n")

# Glob the training targets
training_targets_paths = np.array(glob("input/stage1_train/*/masks/*.png"))

training_dict = {}

print("Loaded training masks. \n")

for training_path in training_paths:
    filename = path_leaf(training_path).replace('.png', '')
    masks = []
    for target_path in training_targets_paths:
        if filename in target_path:
            masks.append(target_path)
    training_dict[training_path] = masks

max_train_size = [0, 0]
for path in training_paths:
    im = imageio.imread(path)
    if im.shape[0] > max_train_size[0]:
        max_train_size[0] = im.shape[0]
    if im.shape[1] > max_train_size[1]:
        max_train_size[1] = im.shape[0]

nn_image_side = max(max_train_size)
nn_image_size = [nn_image_side, nn_image_side]

print("Formatted training data into dictionary. \n")

# TODO merge targets

# TODO pad images with black

# TODO create autoencoder

# TODO train autoencoder

# TODO test autoencoder

# TODO detect nuclei
