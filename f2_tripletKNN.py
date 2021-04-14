import os
import numpy as np
import torch
import pickle

from sklearn.neighbors import KNeighborsClassifier
from trainer import get_embeddings
from utils import load_glued_image

from control import *

TRIPLET_MODEL_NAME = "2Dwin_TripletLoss_v4.pt"
CLASSF_MODEL_NAME = "2dCNNL2_TripletLoss_classf.pt"

training, training_labels = load_glued_image(GLUED_IMAGE_PATH, IMAGE_SIZE)

# Define KNN Classifier

triplet_model = torch.load(os.path.join(MODEL_ROOT_DIR, TRIPLET_MODEL_NAME))
embedding_net = triplet_model.embedding_net.cuda()

knn = KNeighborsClassifier(N_NEIBOUR)

"""
Training the KNN 
using fit(X,y)
X: {array-like, sparse matrix} of shape (n_samples, n_features) - n embedding vectors
y: {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
"""

# Need to train vase 0 and others simultaneuosly
train_embeddings, targets = get_embeddings(embedding_net, training, training_labels)

knn.fit(train_embeddings, targets)

with open(os.path.join(MODEL_ROOT_DIR, "2dCNNL2_TripletLoss_classf_v5.pt"), "wb") as file:
    pickle.dump(knn, file)

knn.predict(train_embeddings)
