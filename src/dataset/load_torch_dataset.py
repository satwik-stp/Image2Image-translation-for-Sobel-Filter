import torch
from torch.utils.data import Dataset


class Img2ImgDataset(Dataset):

    """
    Inherited Class of Pytorch's Dataset tailored for this project
    """

    def __init__(self, images,ground_truths,transform):
        """
        :param images: input images
        :param ground_truths: output filtered images
        :param transform: pytorch transforms
        """
        self.images=images
        self.ground_truths=ground_truths
        self.transform = transform

    def __len__(self):
        """
        Overridden function for returning the size of the dataset
        :return: size of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.ground_truths[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y




