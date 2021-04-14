import cv2
import torch
import numpy as np
import random
from PIL import Image

from functools import wraps
from time import time

from control import *


def load_data(directory, n_class):

    data_paths = []
    data_labels = []

    for cls in range(n_class):

        paths = [os.path.join(directory, f"class {cls}", file_name) for file_name in
                 os.listdir(os.path.join(directory, f"class {cls}"))]

        data_paths += paths
        data_labels += [cls] * len(paths)

    return data_paths, data_labels


def distance(a, b):
    """
    Compute distance of 2 equal shape vectors
    :param a:
    :param b:
    :return: euclidean distance between them
    """
    return (a - b).pow(2).sum(-1)


def make_label(cls, n_class):
    """
    Return label from a specific class
    :param cls: The class of the image
    :param n_class: Total number of classes
    :return: A tensor of shape (1, n_class)
    """
    label = torch.zeros((1, n_class))
    label[0][cls] = 1

    return label


def timing(f):
    """
    Time the underlying function
    :param f: The function to be timed
    :return: The decorator
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('Time elapsed: %2.4f sec' % (te - ts))
        return result

    return wrap
