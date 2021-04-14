import os
from PIL import Image
import random
import numpy as np
import torch

from scipy import spatial
from trainer import get_embeddings

from control import *

TRIPLET_MODEL_NAME = "2DwinCNNL2_TripletLoss_v4.pt"
CLASSF_MODEL_NAME = "2dCNNL2_TripletLoss_classf.pt"


def load_data(directory, n_class, img_size, cls=-1):

    label = torch.zeros((1, n_class))
    label[0][cls] = 1

    data = []

    if cls == -1:
        paths = [os.path.join(directory, file_name) for file_name in
                 os.listdir(os.path.join(directory))]
        data_labels = [label] * len(paths)
    else:
        paths = [os.path.join(directory, f"class {cls}", file_name) for file_name in
                 os.listdir(os.path.join(directory, f"class {cls}"))]
        data_labels = [label] * len(paths)

    for path in paths:
        data.append(np.asarray(Image.open(path).resize((img_size, img_size))))
        data_labels.append(label)

    data = transform(data, img_size)

    return data, data_labels


def load_glued_image(path, n_class, cls, img_size=256):
    """
    Load glued vase image (Assume width > height)
    :param path: Path to the glued vase image
    :param n_class: Number of classes to predict
    :param cls: The designated class to load the data
    :param img_size: Size of the output
    :return: data - Image of (img_size x img_size) with data_labels as their corresponding labels
    """

    data = []
    data_labels = []

    label = torch.zeros((1, n_class))
    label[0][cls] = 1

    glued = np.asarray(Image.open(path))

    height = glued.shape[0]
    width = glued.shape[1]

    for i in range(width - height):
        temp = glued[:, i:i + height, :]
        temp = Image.fromarray(temp).resize((img_size, img_size))
        temp = np.asarray(temp)

        data.append(temp)
        data_labels.append(label)

    data = transform(data, img_size)

    return shuffle(data, data_labels)


def shuffle(data, data_labels):
    # Shuffle the feature vectors and labels
    t = list(zip(data, data_labels))
    random.shuffle(t)

    return zip(*t)


def transform(data, img_size):
    """
    Normalize the data loaded and return
    :param data: The list of data containing the window of image
    :param img_size: size of the image
    :return:
    """

    data = [torch.from_numpy(arr).float().view(1, 3, img_size, img_size) for arr in data]
    # data = [tensor / 255 for tensor in data]

    return data

training, training_labels = load_glued_image(GLUED_IMAGE_PATH, NUM_CLASS, cls=0, img_size=256)
negative, negative_labels = load_data(NEGATIVE_DIR, NUM_CLASS, img_size=256)

# Initialize Models
triplet_model = torch.load(os.path.join(MODEL_ROOT_DIR, TRIPLET_MODEL_NAME))
embedding_net = triplet_model.embedding_net.cuda()

# Get feature embeddings
train_embeddings = get_embeddings(embedding_net, training)
negative_embeddings = get_embeddings(embedding_net, negative)

# Compute the representative of the class
train_average = np.mean(train_embeddings, axis=0)
negative_average = np.mean(negative_embeddings, axis=0)

# Testing phase
test_path = os.path.join(DATA_ROOT_DIR, "Testing/", f"Testing2", "class 0")

testing, testing_labels = load_data(test_path, NUM_CLASS,  img_size=256)
test_embeddings = get_embeddings(embedding_net, testing)

cor = 0
tot = 0

for i in range(test_embeddings.shape[0]):

    s1 = 1 - spatial.distance.cosine(test_embeddings[i], train_average)
    s2 = 1 - spatial.distance.cosine(test_embeddings[i], negative_average)

    print(f"{i}, {s1}, {s2}, {'Right' if s1 > s2 else 'Wrong'}")

    cor += 1 if s1 > s2 else 0
    tot += 1

print(f"Accuracy: {cor/tot}")
