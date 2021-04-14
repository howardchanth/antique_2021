import numpy as np
import cv2
import torch
from PIL import Image


from torch.utils.data import Dataset
from torchvision import transforms


class VaseDataset(Dataset):
    def __init__(self, data_paths, data_labels, img_size, crop_size, type):
        """
        :param data_paths: A list of data paths
        :param data_labels: A list of labels of the data
        :param img_size: size of the image
        :param crop_size: size of the image after being cropped
        :param type of the dataset
        """
        super(Dataset, self).__init__()
        self.type = type

        self.data_paths = data_paths
        self.data_labels = data_labels

        self.img_size = img_size
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize images to size (IMAGE_SIZE, IMAGE_SIZE)
            transforms.ToTensor(),  # Convert PIL Image (HxWxC) to tensor (CxHxW)
            # (CxHxW) format required by Conv2d layer in PyTorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop((crop_size, crop_size))
            # Normalize RGB values from [0, 255]
            # to have specified mean/variance values.
            # This setting usually used with image normalization
            # in PyTorch.
        ])

    def __getitem__(self, index):
        img_path, label = self.data_paths[index], self.data_labels[index]
        img = Image.open(img_path)
        img.draft('RGB', (self.img_size, self.img_size))

        if self.transform is not None:
            img = self.transform(img)

        if self.type == "color":
            img = self.get_color(img)

        if self.type == "shape":
            img = self.get_shape(img)

        return img, label

    def get_color(self, img):
        img = img.numpy().reshape(self.crop_size, self.crop_size, 3).astype(np.uint8)
        img = np.asarray(Image.fromarray(img).convert("HSV"))
        # Load only H, S color features
        img = img[:, :, :2]
        img = torch.Tensor(img).reshape(2, self.crop_size, self.crop_size)

        return img

    def get_shape(self, img):

        img = img.numpy().reshape(self.crop_size, self.crop_size, 3).astype(np.uint8)
        # Convert from RGB to BGR
        img = img[:, :, ::-1].copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        new_img = np.zeros(img.shape)
        new_img = cv2.drawContours(new_img, contours, -1, (0, 255, 0), 3)
        new_img = torch.Tensor(new_img).reshape(3, self.crop_size, self.crop_size)

        return new_img

    def __len__(self):
        return len(self.data_paths)


class TripletVaseDataset(VaseDataset):

    def __init__(self, vase_dataset):
        super(TripletVaseDataset, self).__init__(
            vase_dataset.data_paths,
            vase_dataset.data_labels,
            vase_dataset.img_size,
            vase_dataset.crop_size,
            vase_dataset.type
        )

    def __getitem__(self, index):

        img1, label1 = super(TripletVaseDataset, self).__getitem__(index)
        pos_index = index
        neg_index = index

        while pos_index == index or self.data_labels[pos_index] != label1:
            pos_index = np.random.choice(len(self.data_labels))
        while neg_index == index or self.data_labels[neg_index] == label1:
            neg_index = np.random.choice(len(self.data_labels))

        img2, label2 = super(TripletVaseDataset, self).__getitem__(pos_index)
        img3, label3 = super(TripletVaseDataset, self).__getitem__(neg_index)

        return (img1, img2, img3)

