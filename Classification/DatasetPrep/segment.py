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

images_dir = "fashion-dataset\images"
result_dir = "fashion-dataset\seg_images"
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
data = pd.read_csv('fashion-dataset\styles_cleaned.csv')

# read each image from csv file
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    # read image
    # flush cache
    torch.cuda.empty_cache()
    img = Image.open(os.path.join(images_dir, str(row['id']) + '.jpg'))
    # check subCategory
    img = img.resize((768, 768), Image.BILINEAR)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    if row['subCategory'] == 'Topwear':
        # get mask for upper body
        upper_body_mask = np.where(output_arr == 1, 255, 0).astype("uint8")
        upper_body_mask = Image.fromarray(upper_body_mask, mode="L")
        # Apply mask to original image
        img = np.array(img)
        img = np.where(np.expand_dims(upper_body_mask, axis=2) == 0, 255, img)
        # change background to white
        img = np.where(img == 0, 255, img)
        img = Image.fromarray(img.astype("uint8"), mode="RGB")
        img.save(os.path.join(result_dir, str(row['id']) + '.jpg'))
    
    elif row['subCategory'] == 'Bottomwear':
        # get mask for lower body
        lower_body_mask = np.where(output_arr == 2, 255, 0).astype("uint8")
        lower_body_mask = Image.fromarray(lower_body_mask, mode="L")
        # Apply mask to original image
        img = np.array(img)
        img = np.where(np.expand_dims(lower_body_mask, axis=2) == 0, 0, img)
        # change background to white
        img = np.where(img == 0, 255, img)
        img = Image.fromarray(img.astype("uint8"), mode="RGB")
        img.save(os.path.join(result_dir, str(row['id']) + '.jpg'))

    elif row['subCategory'] == 'Dress':
        # get mask for full body
        full_body_mask = np.where(output_arr == 3, 255, 0).astype("uint8")
        full_body_mask = Image.fromarray(full_body_mask, mode="L")
        # Apply mask to original image
        img = np.array(img)
        img = np.where(np.expand_dims(full_body_mask, axis=2) == 0, 0, img)
        # change background to white
        img = np.where(img == 0, 255, img)
        img = Image.fromarray(img.astype("uint8"), mode="RGB")
        img.save(os.path.join(result_dir, str(row['id']) + '.jpg'))

    else:
        # read image
        img = Image.open(os.path.join(images_dir, str(row['id']) + '.jpg'))
        # check subCategory
        img = img.resize((768, 768), Image.BILINEAR)
        # save image without segmentation
        img.save(os.path.join(result_dir, str(row['id']) + '.jpg'))