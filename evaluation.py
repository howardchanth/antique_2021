"""
Evaluation of models
References:

https://arxiv.org/pdf/1503.03832.pdf

"""

import os
import pickle
import torch
from torch import nn

from models import CNNEmbeddingNet, CNNEmbeddingNetL2, TripletNet, ClassificationNet
from losses import TripletLoss
from trainer import train_triplet, test_model_OUTDATED, train_triplet
from utils import load_data


# Control variables
BATCH_SIZE = 16
NUM_CLASS = 6
WINDOW_SIZE = 32
IMAGE_SIZE = 256
# Margin for triplet loss model
MARGIN = 0.1
LR = 1e-2
N_EPOCH = 5
MODEL_NAME = "2dCNNL2_TripletLoss_classf_v4.pt"

DATA_ROOT_DIR = os.path.join("./", "Data/")
TESTING_DIRS = [os.path.join(DATA_ROOT_DIR, "Testing/", f"Testing{i}") for i in range(3)]
MODEL_ROOT_DIR = os.path.join("./", "Model/")

# Testing settings
# Classification of 6 nets
try:
    testing_model = torch.load(os.path.join(MODEL_ROOT_DIR, MODEL_NAME))
except ValueError:
    testing_model = pickle.load(open(os.path.join(MODEL_ROOT_DIR, MODEL_NAME), 'rb'))
classf_model = ClassificationNet(testing_model.embedding_net, 6).cuda()
testing_criterion = nn.BCELoss()

# Load Testing Data
for testing_dir in TESTING_DIRS:
    testing = []
    testing_labels = []
    for cls in range(6):
        dir = os.path.join(testing_dir, f"class {cls}")

        if not os.listdir(dir):
            continue

        t, tl = load_data(dir, cls)
        testing += t
        testing_labels += tl

    """Testing Phase"""
    # Perform testing using the testing set
    test_model_OUTDATED(classf_model, testing_criterion, testing, testing_labels)
