import os
import torch
from torch import nn
from torch import optim

from models import CNNEmbeddingNet, CNNEmbeddingNetL2, TripletNet, ClassificationNet
from trainer import train_triplet, train_model
from utils import load_glued_image, clear_cache
from control import *


DATA_ROOT_DIR = os.path.join("./", "Data/")
TRAINING_DIR = os.path.join(DATA_ROOT_DIR, "Training/")
TESTING_DIRS = [os.path.join(DATA_ROOT_DIR, "Testing/", f"Testing{i}") for i in range(3)]
MODEL_ROOT_DIR = os.path.join("./", "Model/")
NEGATIVE_DIR = os.path.join(DATA_ROOT_DIR, "Negatives/")
GLUED_IMAGE_PATH = os.path.join(DATA_ROOT_DIR, "glued_surr.jpg")

training, training_labels = load_glued_image(GLUED_IMAGE_PATH, IMAGE_SIZE)

triplet_model = torch.load(os.path.join(MODEL_ROOT_DIR, TRIPLET_MODEL_NAME))
classf_model = ClassificationNet(triplet_model.embedding_net, 6).cuda()

criterion = nn.BCELoss()
optimizer = optim.SGD(triplet_model.embedding_net.parameters(), lr=LR, momentum=0.9)

# Train the model
classf_model = train_model(classf_model, criterion, optimizer, training, training_labels, n_epoch=N_EPOCH)

# Save the trained model
torch.save(classf_model, os.path.join(MODEL_ROOT_DIR, CLASSF_MODEL_NAME))
