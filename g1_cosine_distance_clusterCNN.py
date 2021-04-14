import numpy as np
import torch

from scipy import spatial
from trainer import get_embeddings
from utils import load_data

from dataset import VaseDataset
from torch.utils.data import DataLoader

from control import *

TRIPLET_MODEL_VERSION = 9
TRIPLET_MODEL_NAME = "2Dwin_TripletLoss"

all_embeddings = {
    "pattern": None,
    "color": None,
    "shape": None
}
# ["pattern", "color", "shape"]
# Wrap the pipeline into a function
for model_type in ["pattern"]:

    # Load paths and create Pytorch dataset
    training_paths, training_labels = load_data(TRAINING_DIR, NUM_CLASS)
    training = VaseDataset(training_paths, training_labels, IMAGE_SIZE, CROP_SIZE, model_type)

    # Make data loaders for the clustered CNN
    training_loader = DataLoader(training, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

    model = torch.load(os.path.join(MODEL_ROOT_DIR, f"{TRIPLET_MODEL_NAME}_{model_type}_v{TRIPLET_MODEL_VERSION}.pt"))
    emb_net = model.embedding_net.cuda()

    # Get embeddings of each class
    training_embeddings = get_embeddings(emb_net, training_loader, NUM_CLASS)
    all_embeddings[model_type] = training_embeddings

    # TODO: Temporary set to 2 classes
    training_embeddings = training_embeddings[:2]
    training_reprs = [np.mean(emb, axis=0) for emb in training_embeddings]

    """ Testing Phase"""
    test_path = os.path.join(DATA_ROOT_DIR, "Testing/", "Testing2")

    testing_paths, testing_labels = load_data(test_path, NUM_CLASS)
    testing_dataset = VaseDataset(testing_paths, testing_labels, IMAGE_SIZE, CROP_SIZE, model_type)
    testing_loader = DataLoader(testing_dataset, batch_size=TESTING_BATCH_SIZE, shuffle=True)

    testing_embeddings = get_embeddings(emb_net, testing_loader, NUM_CLASS)
    testing_reprs = [np.mean(emb, axis=0) for emb in testing_embeddings[:2]]

    print (f"Testing {model_type} model:")
    idx = 0
    cor = 0
    tot = 0

    for cls in range(2):
        for emb in testing_embeddings[cls]:
            similarities = []
            idx += 1
            for train_cls in range(2):  # ONLY COMPUTE 2 CLASSES FOR NOW
                similarities.append(1 - spatial.distance.cosine(emb, training_reprs[train_cls]))

            pred = np.argmax(similarities)
            print(f"{idx}, {similarities}, {pred}, {cls}, {'Right' if pred == cls else 'Wrong'}")

            cor += 1 if pred == cls else 0
            tot += 1


    # # Vertically concatenate the embeddings
    # train_embeddings = np.concatenate(train_embeddings, axis=1)
    # negative_embeddings = np.concatenate(negative_embeddings, axis=1)

    # test_embeddings = np.concatenate(test_embeddings, axis=1)