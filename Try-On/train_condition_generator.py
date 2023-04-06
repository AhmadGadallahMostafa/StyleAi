import torch 
import torch.nn as nn
import numpy as np
from torchvision import transforms

from Dataset import TryOnDataset, DataLoader

transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = TryOnDataset(root='Try-On\data', mode='train', data_list='train_pairs.txt', transform=transform)
print(dataset[0])

