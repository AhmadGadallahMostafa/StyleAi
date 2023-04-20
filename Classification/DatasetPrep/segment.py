import os
import pandas as pd

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET
device = "cuda"

images_dir = "Classification\DatasetPrep\\fashion-dataset\images"
result_dir = "Classification\DatasetPrep\\fashion-dataset\seg_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

# read csv file styles cleaned
data = pd.read_csv('fashion-dataset\styles_cleaned.csv', error_bad_lines=False, warn_bad_lines=False)

# read each image from csv file
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    # read image
    img = Image.open(os.path.join(images_dir, row['id'] + '.jpg'))
    # resize image
    img = img.resize((768, 768), Image.ANTIALIAS)
    # convert image to tensor
    img = transform_rgb(img)
    # add batch dimension
    img = img.unsqueeze(0)
    # move to device
    img = img.to(device)
    # forward pass
    d1, d2, d3, d4, d5, d6, d7 = net(img)
    # get prediction
    pred = d1[:, 0, :, :]
    # apply sigmoid
    pred = F.sigmoid(pred)
    # get prediction
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()
    # save prediction
    im = Image.fromarray(pred * 255).convert('RGB')
    im.save(os.path.join(result_dir, row['id'] + '.jpg'))