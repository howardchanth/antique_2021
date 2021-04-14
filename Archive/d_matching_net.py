"""
References:
https://github.com/BoyuanJiang/matching-networks-pytorch

"""
import os
import numpy as np
import tqdm
import torch

from PIL import Image
from matching_network import MatchingNetwork
from torch.autograd import Variable
from models import CNNEmbeddingNet

DATA_ROOT_DIR = os.path.join("./", "Data/")
MODEL_DIR = os.path.join("./", "Model/")
IMAGE_SIZE = 512
TRAIN_BATCH_SIZE = 1
LR = 1e-3
WEIGHT_DECAY = 0.9

# Load training data
# Training data: Glued image with all features included
# Training shape: (512, 512)
data = Image.open(os.path.join(DATA_ROOT_DIR, "glued_surr.jpg"))
data = data.resize((IMAGE_SIZE, IMAGE_SIZE))
data = np.asarray(data)

# Initialize matching network model
matchNet = MatchingNetwork(
    keep_prob=0.8,
    batch_size=1,
    num_channels=3,
    learning_rate=LR,
    fce=True,
    num_classes_per_set=6,
    num_samples_per_class=1,
    image_size=IMAGE_SIZE,
    use_cuda=False
)

optimizer = torch.optim.Adam(matchNet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Train the model with supporting set
with tqdm.tqdm(total=TRAIN_BATCH_SIZE) as pbar:
    # Load the supporting set
    x_support_set = Image.open(os.path.join(DATA_ROOT_DIR, "glued_surr.jpg")).resize((IMAGE_SIZE, IMAGE_SIZE))
    x_support_set = Variable(torch.from_numpy(np.asarray(x_support_set))).float()

    y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
    x_target = Variable(torch.from_numpy(x_target)).float()
    y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

    # convert to one hot encoding
    y_support_set = y_support_set.unsqueeze(2)
    sequence_length = y_support_set.size()[1]
    batch_size = y_support_set.size()[0]
    y_support_set_one_hot = Variable(
        torch.zeros(batch_size, sequence_length, classes_per_set).scatter_(2,y_support_set.data,
                                                                                1), requires_grad=False)

    # reshape channels and change order
    size = x_support_set.size()
    x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
    x_target = x_target.permute(0, 3, 1, 2)

    if torch.cuda.is_available():
        acc, c_loss = matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                    y_target.cuda())
    else:
        acc, c_loss = matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

    optimizer.zero_grad()
    c_loss.backward()
    optimizer.step()

    iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss.data[0], acc.data[0])
    pbar.set_description(iter_out)
    pbar.update(1)
    total_c_loss += c_loss.data[0]
    total_accuracy += acc.data[0]



