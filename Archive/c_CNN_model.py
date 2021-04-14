"""
Simple CNN model using 1d convolutional network
References:

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

https://github.com/cfotache/pytorch_imageclassifier/blob/master/PyTorch_Image_Training.ipynb
"""

import os
import torch
from torch import nn
from torch import optim

from models import CNNEmbeddingNet
from trainer import train_model, test_model_OUTDATED
from utils import load_data

from control import *


""" Load Training data"""
# Load training data to list
training, training_labels = load_data(TRAINING_DIR, n_class=6)
print("Data vectors loaded!!")
print("-" * 30)

# Construct and train the model
model = CNNEmbeddingNet().cuda()
# criterion = SupConLoss(temperature=0.7)
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

"""Training Phase"""
model = training(model, criterion, optimizer, training, training_labels, n_epoch=N_EPOCH)

# Save the trained model
torch.save(model, os.path.join(MODEL_ROOT_DIR, MODEL_NAME))

# Load Testing
for i, testing_dir in enumerate(TESTING_DIRS):
    print(f"Performing testing on the {i+1}-th testing set")
    testing, testing_labels = load_data(testing_dir, n_class=6)

    """Testing Phase"""
    # Perform testing using the testing set
    testing(model, testing, testing_labels)
