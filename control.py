import os


TRAINING_BATCH_SIZE = 8  # >1: Online training
TESTING_BATCH_SIZE = 1
NUM_CLASS = 7
IMAGE_SIZE = 300
CROP_SIZE = 256
# Margin for triplet loss model
MARGIN = 0.1
LR = 1e-2
N_EPOCH = 3
N_TRIPLET_TRAINING_EPOCH = 10


DATA_ROOT_DIR = os.path.join("./", "Data/")
TRAINING_DIR = os.path.join(DATA_ROOT_DIR, "Training/")
TESTING_DIRS = [os.path.join(DATA_ROOT_DIR, "Testing/", f"Testing{i}") for i in range(3)]
NEGATIVE_DIR = os.path.join(DATA_ROOT_DIR, "Negatives/")

GLUED_IMAGE_PATH = os.path.join(DATA_ROOT_DIR, "glued_surr.jpg")
MODEL_ROOT_DIR = os.path.join("./", "Model/")
