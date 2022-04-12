from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# model network class
class Model_Net(nn.Module):
    def __init__(self, img_width, img_height) -> None:
        super(Model_Net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.fully_connected1 = nn.Linear(in_features = self.img_width * self.img_height, out_features = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        x1 = F.relu(self.maxpool1(self.conv1(inputs1)))
        x1 = F.relu(self.maxpool1(self.conv2(x1)))
        x2 = F.relu(self.maxpool1(self.conv1(inputs2)))
        x2 = F.relu(self.maxpool1(self.conv2(x2)))
        x = torch.multiply(x1, x2)
        x = x.view(self.img_width * self.img_height, -1)
        x = self.fully_connected1(x)
        x = self.sigmoid(x)
        return x
