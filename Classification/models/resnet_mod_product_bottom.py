import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from models.resnet50_torch import ResNet50, ResNetBlock50
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch
from torchsummary import summary

class MultiHeadResNet_Bottoms(nn.Module):
    def __init__(self, pre_trained, requires_grad):
        super(MultiHeadResNet_Bottoms, self).__init__()
        if pre_trained == True:
            self.model = ResNet50(ResNetBlock50, 3, 1000, include_top = False)
        else:
            self.model = ResNet50(ResNetBlock50, 3, 1000, include_top = False, weights = None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # for 13 articles
        self.l0 = nn.Linear(2048, 10)
        # for 38 colours
        self.l1 = nn.Linear(2048, 34)
        # for 3 genders
        self.l2 = nn.Linear(2048, 3)
        # for 5 usage
        self.l3 = nn.Linear(2048, 6)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # get model output before the final layer
        x = self.model(x)
        # apply adaptive average pooling
        x = self.avgpool(x)
        # flatten the output
        x = torch.flatten(x, 1)
        # apply linear layer
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        l3 = self.l3(x)
        # apply softmax as it is a multi-class classification problem
        l0 = self.softmax(l0)
        l1 = self.softmax(l1)
        l2 = self.softmax(l2)
        l3 = self.softmax(l3)
        return l0, l1, l2, l3