import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.aligned_dataset import Normalize
from utils import load_ckpt

from network.u2net import U2NET
device = "cuda"

images_dir = "Segmentation\\test_images\inputs"
result_dir = "Segmentation\\test_images\outputs"
checkpoint_path = os.path.join("", "Segmentation\\trained\\best.pth")

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_channels=3, out_channels=4)
net = load_ckpt(net, checkpoint_path)
net = net.to(device)
net = net.eval()

images = os.listdir(images_dir)

# try and except is used to skip the images which are not present in the directory
for image in tqdm(images):
    image_name = image.split(".")[0]
    img = Image.open(os.path.join(images_dir, image))
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

    img_copy = img.copy()
    # get mask for upper body
    upper_body_mask = np.where(output_arr == 1, 255, 0).astype("uint8")
    upper_body_mask = Image.fromarray(upper_body_mask, mode="L")
    # Apply mask to original image
    img = np.array(img)
    img = np.where(np.expand_dims(upper_body_mask, axis=2) == 0, 255, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    # save upper body image
    img.save(os.path.join(result_dir, image_name + "_upper_body.jpg"))

    # get mask for lower body
    lower_body_mask = np.where(output_arr == 2, 255, 0).astype("uint8")
    lower_body_mask = Image.fromarray(lower_body_mask, mode="L")
    # Apply mask to original image
    img = np.array(img_copy)
    img = np.where(np.expand_dims(lower_body_mask, axis=2) == 0, 255, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    # save lower body image
    img.save(os.path.join(result_dir, image_name + "_lower_body.jpg"))

    # get mask for full body
    full_body_mask = np.where(output_arr == 3, 255, 0).astype("uint8")
    full_body_mask = Image.fromarray(full_body_mask, mode="L")
    # Apply mask to original image
    img = np.array(img_copy)
    img = np.where(np.expand_dims(full_body_mask, axis=2) == 0, 255, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    # save full body image
    img.save(os.path.join(result_dir, image_name + "_full_body.jpg"))

