import torch
import numpy as np
from utils import timing, distance


@timing
def train_model(model, criterion, optimizer, training, training_labels, n_epoch):
    """

    :param model: The model to be trained
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param training: Training data in list
    :param training_labels: Corresponding training labels
    :param n_epoch: Number of epochs to be trained
    :return:
    """
    for epoch in range(n_epoch):  # loop over the training multiple times
        running_loss = 0.0
        for idx, (train, label) in enumerate(zip(training, training_labels)):

            # Zero the gradient buffers of all parameters and back-propagate with random gradients
            optimizer.zero_grad()

            if torch.cuda.is_available():
                label = label.cuda()
                out = model(train.cuda())
            else:
                out = model(train)

            model.zero_grad()

            # Back-propagate the loss back into the model
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % 200 == 199:
                print('[%d, %d] loss: %.8f' %
                      (epoch + 1, idx + 1, running_loss / 200))
                running_loss = 0

    return model


@timing
def train_triplet(model, criterion, optimizer, data_loader, n_epoch):
    """
    Training phase for triplet net
    When batch_size = 1, equivalent to normal triplet training
    :param model: The triplet model to be trained
    :param criterion: The loss function
    :param optimizer: Optimizer
    :param data_loader: Data loader containing the training data
    :param n_epoch: Number of training epochs
    :return: Trained model
    """
    for epoch in range(n_epoch):
        running_loss = 0.0
        for idx, (anc_img, pos_img, neg_img) in enumerate(data_loader):

            # Zero the gradient buffers of all parameters and back-propagate with random gradients
            optimizer.zero_grad()

            if torch.cuda.is_available():
                out_anc, out_pos, out_neg = model(anc_img.cuda(), pos_img.cuda(), neg_img.cuda())
            else:
                out_anc, out_pos, out_neg = model(anc_img, pos_img, neg_img)

            # TODO: Check if really okay to use mean
            anc_mean = torch.mean(out_anc, axis=0)
            # Find the hard positive with largest distance
            # Find the hard negative with smallest distance
            pos_distances = [distance(anc_mean, pos) for _, pos in enumerate(out_pos)]
            neg_distances = [distance(anc_mean, neg) for _, neg in enumerate(out_neg)]

            # out_pos = out+pos[int(np.argmax(pos_distances))]
            out_pos = out_pos[int(np.random.choice(len(pos_distances)))]
            out_neg = out_neg[int(np.argmin(neg_distances))]

            # Back-propagate the loss back into the model
            loss = criterion(out_anc, out_pos, out_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print statistics
            if idx % 200 == 199:
                print('[%d, %d] loss: %.8f' %
                      (epoch + 1, idx + 1, running_loss / 200))
                running_loss = 0

    return model


def get_embeddings(embedding_net, data_loader, n_class):
    embeddings = [None] * n_class

    for idx, (data, label) in enumerate(data_loader):
        if label.shape[0] > 1:
            label = label[0]
            data = data[:1, :, :, :]

        cls = label

        with torch.no_grad():
            if embeddings[cls] is None:
                embeddings[cls] = embedding_net(data.cuda())
            else:
                out = embedding_net(data.cuda())
                embeddings[cls] = torch.cat((embeddings[cls], out), dim=0)

    for idx, emb in enumerate(embeddings):
        if emb is None:
            embeddings[idx] = None
        else:
            embeddings[idx] = np.array(embeddings[idx].cpu())

    return embeddings


def classify_single(out):
    t = torch.zeros(out.shape)
    t[:, torch.argmax(out)] = 1

    return t.cuda() if torch.cuda.is_available() else t
