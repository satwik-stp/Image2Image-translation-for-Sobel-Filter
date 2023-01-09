import os
import shutil
import gdown
from torchvision.datasets import CIFAR100,OxfordIIITPet


class DownloadDataset:
    """
    Class that structures downloading and extracting the datasets experimented in this project
    -- Mini CoCo dataset : Downloaded and extracted from an github repo
    -- CIFAR-100 dataset : Downloaded through torchvision library
    -- OxfordIIITPet dataset : Downloaded through torchvision library
    """
    
    def __init__(self,dataset_name):
        """
        :param dataset_name: can be "mini_coco","cifar","oxford_pet"
        """
        self.dataset_name=dataset_name

    def download_from_url(self,url):
        """
        :func to download datasets that require installation from an url

        :param url: url to be downloading
        :return: None
        """
        if self.dataset_name=="mini_coco":
            print("Mini coco dataset from https://github.com/giddyyupp/coco-minitrain repo")

            # Installing in data->raw and unpacking it
            # "gdown" is a python package for downloading public google drive links
            gdown.download(id=url, output="data/raw/dataset.zip", quiet=False)
            shutil.unpack_archive("data/raw/dataset.zip", "data/raw/")
            os.remove("data/raw/dataset.zip")

            print("\n Finished downloading and unpacking Mini coco dataset")

    def download_from_torch(self):
        """
        :func to download datasets using torchvision libraries

        :return: None
        """
        if self.dataset_name=="cifar":
            print("CIFAR-100 dataset using torchvision package \n\n")
            # Installing in data->raw and unpacking it using torchvision dataset
            CIFAR100(download=True,root="./data/raw/")
            print("\n Finished downloading and unpacking CIFAR100 dataset")

        elif self.dataset_name=="oxford_pet":
            print("OxfordIIITPet dataset using torchvision package \n\n")
            # Installing in data->raw and unpacking it using torchvision dataset
            OxfordIIITPet(download=True,root="./data/raw/")
            print("\n Finished downloading and unpacking OxfordIIITPet dataset")