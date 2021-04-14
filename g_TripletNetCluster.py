"""
Triplet Loss model with cluster based CNN on shape, color and pattern

"""

from torch import optim
from torch.utils.data import DataLoader

from models import CNNEmbeddingNetL2, TripletNet
from losses import TripletLoss
from trainer import train_triplet
from dataset import VaseDataset, TripletVaseDataset

from utils import *

# TODO: Plot classification results in 2D
# TODO: Consider rotating the data to generate more variations
# TODO: Tune the hyper-parameters using dev set
# TODO: Use the uncertainty index to determine declassification
# TODO: Use pretrained image models to be the embedding net
# TODO: Better training and testing design


""" Define Model Names"""
TRIPLET_MODEL_VERSION = 9
TRIPLET_MODEL_NAME = "2Dwin_TripletLoss"
CLASSF_MODEL_NAME = "2dCNNL2_TripletLoss_classf.pt"

""" Load Training data"""
""" Train the Embedding Network with 3 features"""

# ["pattern", "color", "shape"]

for model_type in ["pattern", "color", "shape"]:

    # Load paths and create Pytorch dataset
    training_paths, training_labels = load_data(TRAINING_DIR, NUM_CLASS)
    training = TripletVaseDataset(VaseDataset(training_paths, training_labels, IMAGE_SIZE, CROP_SIZE, model_type))

    # Make data loaders for the clustered CNN
    training_loader = DataLoader(training, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

    print("-" * 30)
    print(f"{model_type} data loaded!!")
    print("-" * 30)

    in_channel = 2 if model_type == "color" else 3

    model = TripletNet(CNNEmbeddingNetL2(in_channel, 128)).cuda()
    # pattern_model = TripletNet(
    #     torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    # ).cuda()

    """Training Phase"""
    print("-" * 30)
    print(f"Training {model_type} Model")
    print("-" * 30)

    criterion = TripletLoss(margin=MARGIN)

    # TODO: Can trial on ADAM optimizers
    optimizer = optim.SGD(model.embedding_net.parameters(), lr=LR, momentum=0.9)
    model = train_triplet(model, criterion, optimizer, training_loader, n_epoch=N_EPOCH)
    torch.save(model, os.path.join(MODEL_ROOT_DIR, f"{TRIPLET_MODEL_NAME}_{model_type}_v{TRIPLET_MODEL_VERSION}.pt"))
    torch.cuda.empty_cache()

