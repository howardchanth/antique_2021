"""
Triplet Loss model
References:

https://arxiv.org/pdf/1503.03832.pdf

"""

from torch import optim

from models import CNNEmbeddingNet, CNNEmbeddingNetL2, TripletNet
from losses import TripletLoss
from trainer import train_triplet

from control import *
from utils import *
from dataset import VaseDataset

""" Define Model Names"""
TRIPLET_MODEL_NAME = "2Dwin_TripletLoss_v5.pt"
CLASSF_MODEL_NAME = "2dCNNL2_TripletLoss_classf.pt"

""" Load Training data"""
# Load training data to list
training, training_labels = load_glued_image(GLUED_IMAGE_PATH, NUM_CLASS, 0, IMAGE_SIZE)
# Load the negative set of images
negative, negative_labels = load_data(NEGATIVE_DIR, n_class=NUM_CLASS, cls=5, img_size=IMAGE_SIZE)
print("-" * 30)
print("Data loaded!!")
print("-" * 30)

# 3 models extracting the pattern feature
triplet_mdoel = TripletNet(CNNEmbeddingNetL2()).cuda()

# pattern_model = TripletNet(
#     torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
# ).cuda()

criterion = TripletLoss(margin=MARGIN)

# optimizer = optim.Adam(triplet_model.parameters(), lr=LR)
optimizer = optim.SGD(triplet_mdoel.embedding_net.parameters(), lr=LR, momentum=0.9)

"""Training Phase"""

""" Stage 1: Train the Embedding Network"""
triplet_model = train_triplet(triplet_mdoel, criterion, optimizer, training, training_labels, negative,
                              n_epoch=N_EPOCH, batch_size=TRAINING_BATCH_SIZE)

# Save the trained model
torch.save(triplet_model, os.path.join(MODEL_ROOT_DIR, TRIPLET_MODEL_NAME))


