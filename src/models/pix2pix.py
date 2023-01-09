import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import segmentation_models_pytorch as smp


class Generator(nn.Module):
    def __init__(self, dropout_p=0.4,input_channel=3,output_channel=1):
        super(Generator, self).__init__()

        self.dropout_p = dropout_p
        # Load Unet with Resnet34 Embedding from SMP library. Pre-trained on Imagenet
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                             in_channels=input_channel, classes=output_channel, activation=None)
        # Adding two layers of Dropout as the original Unet doesn't have any
        # This will be used to feed noise into the networking during both experiment and evaluation
        # These extra layers will be added on decoder part where 2D transposed convolution is occured
        for idx in range(1, 3):
            self.unet.decoder.blocks[idx].conv1.add_module('3', nn.Dropout2d(p=self.dropout_p))
        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        for module in self.unet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x):
        x = self.unet(x)
        x = F.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dropout_p=0.4,input_channel=4):
        super(Discriminator, self).__init__()
        self.dropout_p = dropout_p
        self.model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(4, 128, 3, stride=2, padding=2)),
          ('bn1', nn.BatchNorm2d(128)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(128, 256, 3, stride=2, padding=2)),
          ('bn2', nn.BatchNorm2d(256)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(256, 512, 3)),
          ('dropout3', nn.Dropout2d(p=self.dropout_p)),
          ('bn3', nn.BatchNorm2d(512)),
          ('relu3', nn.ReLU()),
          ('conv4', nn.Conv2d(512, 1024, 3)),
          ('dropout4', nn.Dropout2d(p=self.dropout_p)),
          ('bn4', nn.BatchNorm2d(1024)),
          ('relu4', nn.ReLU()),
          ('conv5', nn.Conv2d(1024, 512, 3, stride=2, padding=2)),
          ('dropout5', nn.Dropout2d(p=self.dropout_p)),
          ('bn5', nn.BatchNorm2d(512)),
          ('relu5', nn.ReLU()),
          ('conv6', nn.Conv2d(512, 256, 3, stride=2, padding=2)),
          ('dropout6', nn.Dropout2d(p=self.dropout_p)),
          ('bn6', nn.BatchNorm2d(256)),
          ('relu6', nn.ReLU()),
          ('conv7', nn.Conv2d(256, 128, 3)),
          ('bn7', nn.BatchNorm2d(128)),
          ('relu7', nn.ReLU()),
          ('conv8', nn.Conv2d(128, 3, 3)),
        ]))

    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return  x