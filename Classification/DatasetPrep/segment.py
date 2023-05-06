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

images_dir = "Classification/DatasetPrep/DeepFashion/"
result_dir = "Classification/DatasetPrep/DeepFashion/seg_images"
checkpoint_path = os.path.join("", "Classification/DatasetPrep/trained_checkpoint/cloth_segm_u2net_latest.pth")

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_channels=3, out_channels=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

# read csv file styles cleaned
data = pd.read_csv('Classification/DatasetPrep/DeepFashion/Anno_coarse/all_data.csv')

# try and except is used to skip the images which are not present in the directory
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    try:
        img = Image.open(os.path.join(images_dir, str(row['image_name'])))
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

        if row['category_type'] == 1:
            # get mask for upper body
            upper_body_mask = np.where(output_arr == 1, 255, 0).astype("uint8")
            upper_body_mask = Image.fromarray(upper_body_mask, mode="L")
            # Apply mask to original image
            img = np.array(img)
            img = np.where(np.expand_dims(upper_body_mask, axis=2) == 0, 255, img)
            # change background to white
            img = np.where(img == 0, 255, img)
            img = Image.fromarray(img.astype("uint8"), mode="RGB")
            # keep the image name until last slash this is the directory name
            directory_name = str(row['image_name']).rsplit('/', 1)[0]
            # check if directory which is the same as image name exists
            if not os.path.exists(os.path.join(result_dir, directory_name)):
                os.makedirs(os.path.join(result_dir, directory_name))
            img.save(os.path.join(result_dir, str(row['image_name'])))
        
        elif row['category_type'] == 2:
            # get mask for lower body
            lower_body_mask = np.where(output_arr == 2, 255, 0).astype("uint8")
            lower_body_mask = Image.fromarray(lower_body_mask, mode="L")
            # Apply mask to original image
            img = np.array(img)
            img = np.where(np.expand_dims(lower_body_mask, axis=2) == 0, 0, img)
            # change background to white
            img = np.where(img == 0, 255, img)
            img = Image.fromarray(img.astype("uint8"), mode="RGB")
            directory_name = str(row['image_name']).rsplit('/', 1)[0]
            # check if directory which is the same as image name exists
            if not os.path.exists(os.path.join(result_dir, directory_name)):
                os.makedirs(os.path.join(result_dir, directory_name))
            img.save(os.path.join(result_dir, str(row['image_name'])))

        elif row['category_type'] == 3:
            # get mask for full body
            full_body_mask = np.where(output_arr == 3, 255, 0).astype("uint8")
            full_body_mask = Image.fromarray(full_body_mask, mode="L")
            # Apply mask to original image
            img = np.array(img)
            img = np.where(np.expand_dims(full_body_mask, axis=2) == 0, 0, img)
            # change background to white
            img = np.where(img == 0, 255, img)
            img = Image.fromarray(img.astype("uint8"), mode="RGB")
            directory_name = str(row['image_name']).rsplit('/', 1)[0]
            # check if directory which is the same as image name exists
            if not os.path.exists(os.path.join(result_dir, directory_name)):
                os.makedirs(os.path.join(result_dir, directory_name))
            img.save(os.path.join(result_dir, str(row['image_name'])))

        else:
            # read image
            img = Image.open(os.path.join(images_dir, str(row['image_name'])))
            # check subCategory
            img = img.resize((768, 768), Image.BILINEAR)
            # save image without segmentation

            directory_name = str(row['image_name']).rsplit('/', 1)[0]
            # check if directory which is the same as image name exists
            if not os.path.exists(os.path.join(result_dir, directory_name)):
                os.makedirs(os.path.join(result_dir, directory_name))
            img.save(os.path.join(result_dir, str(row['image_name'])))
    except:
        print("Image not found id: ", row['image_name'])
        continue