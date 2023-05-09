import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from models.resnet50_torch import ResNet50, ResNetBlock50
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch
from torchsummary import summary

class MultiHeadResNet(nn.Module):
    def __init__(self, pre_trained, requires_grad):
        super(MultiHeadResNet, self).__init__()
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
        # for 19 categories
        self.l0 = nn.Linear(2048, 19)
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
        # apply softmax as it is a multi-class classification problem
        l0 = self.softmax(l0)
        return l0

# model = MultiHeadResNet(pre_trained = True, requires_grad = False)
# model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# # compare with torchvision.models.resnet50
# resnet50_torch = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
# resnet50_torch.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# # print model summary
# summary(model, (3, 224, 224))
# summary(resnet50_torch, (3, 224, 224))
# # print weights of the first layer
# print(model.model.conv1.weight[0][0])
# print(resnet50_torch.conv1.weight[0][0])