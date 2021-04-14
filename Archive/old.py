import os
import torch


def load_data_windowed(dir, n_class=0):
    #
    """
    (OUTDATED!)
    :param dir: Directory to training data
    :param n_class: Number of classes in the subdirectory
    :return: Training data windows and their corresponding labels
    """
    data, data_labels = [], []
    label = torch.zeros((1, n_class))
    if n_class == 0:
        # Default to be unclassified (class 6)
        label[0][5] = 1
        paths = [os.path.join(dir, file_name) for file_name in
                 os.listdir(os.path.join(dir))]

        data, data_labels = load_images(paths, data, data_labels, label=label)
    else:
        # Load the classes and labels as training data
        for cls in range(n_class):
            # Label for the vase
            label = torch.zeros((1, 6))
            label[0][cls] = 1

            paths = [os.path.join(dir, f"class {cls}", file_name) for file_name in
                     os.listdir(os.path.join(dir, f"class {cls}"))]

            data, data_labels = load_images(paths, data, data_labels, label=label)

    return normalize(data, data_labels)


def load_images(paths, data, data_labels, window_size=32, img_size=256, stride=4, label=None):
    """
    Load image from paths, and use a convolutional sliding window to slide the images
    :param paths: paths to the individual images
    :param window_size: size of th sliding window
    :param img_size: Size of the uniform scaled image
    :param label: Label of the image class
    :return: List of image array with shape (window_size, window_size, 3) and their corresponding
    """

    for path in paths:
        temp = np.asarray(Image.open(path).resize((img_size, img_size)))

    # Use a square sliding window with a stride of 4
        for i in range(0, img_size - window_size, stride):
            for j in range(0, img_size - window_size, stride):
                data.append(temp[i:i+window_size, j:j+window_size, :])
                data_labels.append(label)

    return data, data_labels

def clear_cache():
    """
    Clear cache at the end of every run
    :return: None
    """
    print("\nClearing Cache...")
    filelist = [f for f in os.listdir(CACHE_DIR)]
    print(filelist)

    for f in filelist:
        os.remove(os.path.join(CACHE_DIR, f))

    print("Cache Cleared!")

def load_glued_to_cache(path, n_class, cls):
    """
    Load glued vase image (Assume width > height)
    :param path: Path to the glued vase image
    :param n_class: Number of classes to predict
    :param cls: The designated class to load the data
    :param img_size: Size of the output
    :param save: Whether or not winows need to be saved
    :return: data - Image of (img_size x img_size) with data_labels as their corresponding labels
    """

    cache_paths = []
    data_labels = []

    label = torch.zeros((1, n_class))
    label[0][cls] = 1

    glued = np.asarray(Image.open(path))

    height = glued.shape[0]
    width = glued.shape[1]

    for i in range(width - height):
        target_path = os.path.join(CACHE_DIR, f"{str(len(os.listdir(CACHE_DIR)))}.jpg")

        temp = glued[:, i:i+height, :]
        temp = Image.fromarray(temp)  # .resize((img_size, img_size))
        # Save the window to cache
        temp.save(target_path)

        cache_paths.append(target_path)
        data_labels.append(label)

    return cache_paths, data_labels

def load_cache(dir, n_class):
    """
    Load Training and Testing data to cache
    :param dir: The training or testing directory to be loaded
    :return: The paths and labels to be fed into the dataloader
    """

    cache_paths = []
    labels = []

    for cls in range(n_class):
        subdir = os.path.join(dir, f"class {cls}")

        paths = [os.path.join(subdir, file_name) for file_name in
                 os.listdir(os.path.join(subdir))]

        for path in paths:
            if "glued" in path:
                c_paths, lbs = load_glued_to_cache(path, n_class, cls)
            else:
                c_paths = os.path.join(CACHE_DIR, str(len(os.listdir(CACHE_DIR))))
                copy(path, c_paths)
                c_paths = [c_paths]

            cache_paths += c_paths
            labels += [make_label(cls, n_class)]

    return cache_paths, labels

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.embedding_net(x)
        x = self.nonlinear(x)
        x = self.fc1(x)
        x = self.nonlinear(x)
        scores = F.softmax(self.fc2(x), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))




@timing
def test_model_OUTDATED(model, criterion, testing, testing_labels):
    """
    (OUTDATED)
    :param model:
    :param criterion:
    :param testing:
    :param testing_labels:
    :return:
    """

    testing_loss = 0
    correct_preds = 0
    total_preds = 0

    for idx, (test, label) in enumerate(zip(testing, testing_labels)):
        # Use cuda if available
        if torch.cuda.is_available():
            test = test.cuda()
            label = label.cuda()

        with torch.no_grad():
            out = model(test)

        loss = criterion(out, label)

        testing_loss += loss.item()
        # Make predictions with the largest probability
        preds = classify_single(out)

        correct_preds += torch.all(preds.eq(label)).item()
        total_preds += 1
        accuracy = correct_preds / total_preds

        # Print testing results
        if idx % 5 == 4:
            print(out)
            print(preds)
            print(label)
            print('[%d] loss: %f' %
                  (idx + 1, testing_loss / idx))
            print(f"[{idx + 1}] Accuracy: {accuracy}")

@timing
def train_triplet(model, criterion, optimizer, training, training_labels, negative, n_epoch, batch_size):
    """
    Training phase for triplet net (Online when batch size > 1)
    When batch_size = 1, equivalent to normal triplet training
    :param model: The triplet model to be trained
    :param criterion: The loss function
    :param optimizer: Optimizer
    :param training: List of training data
    :param training_labels:
    :param negative: The set of negatives to be
    :param n_epoch: Number of training epochs
    :param batch_size: The batch of hard mining
    :return: Trained model
    """
    for epoch in range(n_epoch):
        running_loss = 0.0
        for idx, (_, _) in enumerate(zip(training, training_labels)):

            optimizer.zero_grad()

            out_anc, out_pos, out_neg = find_triplet(model, idx, training, training_labels, negative, batch_size)

            loss = criterion(out_anc, out_pos, out_neg)
            # Back-propagate the loss back into the model
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            # print statistics
            if idx % 200 == 199:
                print('[%d, %d] loss: %.8f' %
                      (epoch + 1, idx + 1, running_loss / 2000))
                running_loss = 0

    return model


def feature_color(data: list, img_size: int):
    """
    Get the vase color for every vase in the image
    :param data: The list of data in RGB tensors
    :param img_size: Size of the image
    :return: The list of images in HSV
    """

    data = [img.numpy().reshape(img_size, img_size, 3).astype(np.uint8)
            for img in data]
    data = [np.asarray(Image.fromarray(img).convert("HSV"))
            for img in data]

    data = transform(data, img_size)

    return data


def feature_shape(data: list, img_size: int):
    """
    Get the vase color for every vase in the image
    :param data: The list of data in RGB tensors
    :param img_size: Size of the image
    :return: An array of transformed data
    """
    # Transform the data to cv2 images
    data = [img.numpy().reshape(img_size, img_size, 3).astype(np.uint8)
            for img in data]
    # Transform from RGB to BGR
    data = [img[:, :, ::-1].copy() for img in data]

    for idx, img in enumerate(data):
        # Convert data to grayscale images
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_img = np.zeros(img.shape)
        data[idx] = cv2.drawContours(new_img, contours, -1, (0, 255, 0), 3)

    data = transform(data, img_size)

    return data

def load_glued_image(path, n_class, cls, img_size=256):
    """
    Load glued vase image (Assume width > height)
    :param path: Path to the glued vase image
    :param n_class: Number of classes to predict
    :param cls: The designated class to load the data
    :param img_size: Size of the output
    :return: data - Image of (img_size x img_size) with data_labels as their corresponding labels
    """

    data = []
    data_labels = []

    label = torch.zeros((1, n_class))
    label[0][cls] = 1

    glued = np.asarray(Image.open(path))

    height = glued.shape[0]
    width = glued.shape[1]

    for i in range(width - height):
        temp = glued[:, i:i + height, :]
        temp = Image.fromarray(temp).resize((img_size, img_size))
        temp = np.asarray(temp)

        data.append(temp)
        data_labels.append(label)

    data = transform(data, img_size)

    return shuffle(data, data_labels)

def find_triplet(model, idx, training, training_labels, negative, batch_size):
    # Generate triplet for batch_size times and find the hard triplets
    positive_embs = []
    negative_embs = []
    out_anc = None

    for _ in range(batch_size):
        # Choose one positive example
        pos_idx = idx
        while pos_idx == idx and torch.all(training_labels[pos_idx].eq(training_labels[idx])):
            pos_idx = np.random.choice(len(training))
        # Choose one negative example
        neg_idx = np.random.choice(len(negative))

        if torch.cuda.is_available():
            out_anc, out_pos, out_neg = model(training[idx].cuda(), training[pos_idx].cuda(), negative[neg_idx].cuda())
        else:
            out_anc, out_pos, out_neg = model(training[idx], training[pos_idx], negative[neg_idx])

        positive_embs.append(out_pos)
        negative_embs.append(out_neg)

    # Find the hard positive with largest distance
    # Find the hard negative with smallest distance
    pos_distances = [distance(out_anc, pos) for pos in positive_embs]
    neg_distances = [distance(out_anc, neg) for neg in negative_embs]

    # out_pos = positive_embs[int(np.argmax(pos_distances))]
    out_pos = positive_embs[int(np.random.choice(len(pos_distances)))]
    out_neg = negative_embs[int(np.argmin(neg_distances))]

    return out_anc, out_pos, out_neg



@timing
def train_triplet(model, criterion, optimizer, training, training_labels, negative, n_epoch, batch_size=1):
    """
    Training phase for triplet net
    When batch_size = 1, equivalent to normal triplet training
    :param model: The triplet model to be trained
    :param criterion: The loss function
    :param optimizer: Optimizer
    :param training: List of training data
    :param training_labels:
    :param negative: The set of negatives to be
    :param n_epoch: Number of training epochs
    :param batch_size:
    :return: Trained model
    """
    for epoch in range(n_epoch):
        running_loss = 0.0
        for idx, (train, label) in enumerate(zip(training, training_labels)):

            # Generate triplet for batch_size times and find the hard triplets
            positive_embs = []
            negative_embs = []
            out_anc = None
            # Zero the gradient buffers of all parameters and back-propagate with random gradients
            optimizer.zero_grad()

            for _ in range(batch_size):
                # Choose one positive example
                pos_idx = idx
                while pos_idx == idx and torch.all(training_labels[idx].eq(label)):
                    pos_idx = np.random.choice(len(training))
                # Choose one negative example
                neg_idx = np.random.choice(len(negative))

                if torch.cuda.is_available():
                    out_anc, out_pos, out_neg = model(train.cuda(), training[pos_idx].cuda(),
                                                      negative[neg_idx].cuda())
                else:
                    out_anc, out_pos, out_neg = model(train, training[pos_idx], negative[neg_idx])

                positive_embs.append(out_pos)
                negative_embs.append(out_neg)

            # Find the hard positive with largest distance
            # Find the hard negative with smallest distance
            pos_distances = [distance(out_anc, pos) for pos in positive_embs]
            neg_distances = [distance(out_anc, neg) for neg in negative_embs]

            # out_pos = positive_embs[int(np.argmax(pos_distances))]
            out_pos = positive_embs[int(np.random.choice(len(pos_distances)))]
            out_neg = negative_embs[int(np.argmin(neg_distances))]

            # Back-propagate the loss back into the model
            loss = criterion(out_anc, out_pos, out_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print statistics
            if idx % 200 == 199:
                print('[%d, %d] loss: %.8f' %
                      (epoch + 1, idx + 1, running_loss / 2000))
                running_loss = 0

    return model

def shuffle(data, data_labels):
    # Shuffle the feature vectors and labels
    t = list(zip(data, data_labels))
    random.shuffle(t)

    return zip(*t)

def transform(data, img_size):
    """
    Normalize the data loaded and return
    :param data: The list of data containing the window of image
    :param img_size: size of the image
    :return:
    """

    data = [torch.from_numpy(arr).float()
                .view(1, 3, img_size, img_size) for arr in data]
    # data = [tensor / 255 for tensor in data]

    return data