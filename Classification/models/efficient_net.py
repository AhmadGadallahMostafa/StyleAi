import torch
import torch.nn as nn
from math import ceil
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import re

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    # default groups = 1 is normal convolution
    # groups = in_channels is depthwise convolution
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # same as swish activation
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # how much is the channel important to the output
        return x * self.se(x)       


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * expand_ratio
        reduced_dim = int(in_channels / reduction)
        self.expand = in_channels != hidden_dim
        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class OurEfficientNet(nn.Module):
    def __init__(self, version, num_classes, pretrained=False):
        super(OurEfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)

        self.classifier0 = nn.Sequential( # for gender
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, 5),
            nn.Softmax(dim=1)
        )
        self.classifier1 = nn.Sequential( # for subCategory
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, 11),
            nn.Softmax(dim=1)
        )
        self.classifier2 = nn.Sequential( # for articleType
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, 46),
            nn.Softmax(dim=1)
        )
        self.classifier3 = nn.Sequential( # for baseColour
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, 46),
            nn.Softmax(dim=1)
        )
        self.classifier4 = nn.Sequential( # for usage
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, 7),
            nn.Softmax(dim=1)
        )

    def create_features(self, width_factor, depth_factor, last_channels):
        #channels = int(32 * width_factor)
        channels =  4*ceil(int(32*width_factor) / 4)
        number_of_channels_from_backbone = 1280
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4) # must be divisible by 4
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride if layer == 0 else 1,
                        padding=kernel_size // 2,
                        expand_ratio=expand_ratio,
                    )
                )
                in_channels = out_channels
        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*features)

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x0 = self.classifier0(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        return x0, x1, x2, x3, x4



# Preprocess image
tfms = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = tfms(Image.open('Classification\models\dog.jpg')).unsqueeze(0)
# get labels from labels.txt
labels_map = []
with open('Classification\models\labels.txt', 'r') as f:
    #remove comma and newline
    labels_map = [line.strip().split(',')[0] for line in f.readlines()]

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
custom_model = OurEfficientNet(version='b5', num_classes=1000, pretrained=True).to(device)

# # loop through all layers and copy weights in order
# # for i = 1 in actual_model and custom_model
# # replace v with v
# custom_dict = custom_model.state_dict()
# ac_dict = actual_model.state_dict()
# for (k, v), (k2, v2) in zip(custom_dict.items(), ac_dict.items()):
#     custom_dict[k] = v2
# custom_model.load_state_dict(custom_dict)
# custom_model.eval()
# with torch.no_grad():
#     outputs = custom_model(img)

# print('-----')
# for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
#     prob = torch.softmax(outputs, dim=1)[0, idx].item()
#     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))