import torch
from torch import nn
from torchsummary import summary
# import torch resnet50
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import models
import pretrainedmodels

resnet50_url = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'

class ResNetBlock50(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResNetBlock50, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, block, image_channels, num_classes, include_top = True, weights='imagenet'):
        super(ResNet50, self).__init__()

        self.expansion = 4
        layers = [3, 4, 6, 3]
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(block, layers[3], intermediate_channels=512, stride=2)

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * self.expansion, num_classes)
        else:
            self.avgpool = nn.Identity()
            self.fc = nn.Identity()

        if weights == 'imagenet':
            state_dict = torch.hub.load_state_dict_from_url(resnet50_url, progress=True)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Uncomment for non custom Network
        # x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
    
# # ResNet50
# model = ResNet50(ResNetBlock50, 3, 1000, include_top = True)
# model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# # compare with torchvision.models.resnet50
# resnet50_torch = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
# resnet50_torch.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(model, (3, 224, 224))
# summary(resnet50_torch, (3, 224, 224))