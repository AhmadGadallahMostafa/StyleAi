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

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET
device = "cuda"

image_dir = "input_images"
result_dir = "output_images"
checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

images_list = sorted(os.listdir(image_dir))
pbar = tqdm(total=len(images_list))
for image_name in images_list:
    img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    # resize image to 768x768
    img = img.resize((768, 768), Image.BILINEAR)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    # output image now consists of classes
    # get masks for each class
    # 0 - background
    # 1 - upper body
    # 2 - lower body
    # 3 - full body

    # get mask for upper body
    upper_body_mask = np.where(output_arr == 1, 255, 0).astype("uint8")
    upper_body_mask = Image.fromarray(upper_body_mask, mode="L")
    # Apply mask to original image
    img_cp = img.copy()
    img = np.array(img)
    img = np.where(np.expand_dims(upper_body_mask, axis=2) == 0, 255, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    img.save(os.path.join(result_dir, "upper_body_" + image_name))

    # get mask for lower body
    lower_body_mask = np.where(output_arr == 2, 255, 0).astype("uint8")
    lower_body_mask = Image.fromarray(lower_body_mask, mode="L")
    # Apply mask to original image
    img = np.array(img_cp)
    img = np.where(np.expand_dims(lower_body_mask, axis=2) == 0, 0, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    img.save(os.path.join(result_dir, "lower_body_" + image_name))

    # get mask for full body
    full_body_mask = np.where(output_arr == 3, 255, 0).astype("uint8")
    full_body_mask = Image.fromarray(full_body_mask, mode="L")
    # Apply mask to original image
    img = np.array(img_cp)
    img = np.where(np.expand_dims(full_body_mask, axis=2) == 0, 0, img)
    # change background to white
    img = np.where(img == 0, 255, img)
    img = Image.fromarray(img.astype("uint8"), mode="RGB")
    img.save(os.path.join(result_dir, "full_body_" + image_name))

    pbar.update(1)

pbar.close()

# import os

# from tqdm import tqdm
# from PIL import Image
# import numpy as np

# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms

# from data.base_dataset import Normalize_image
# from utils.saving_utils import load_checkpoint_mgpu

# from networks import U2NET

# torch.cuda.empty_cache()
# device = "cuda"

# image_dir = "input_images"
# result_dir = "output_images"
# checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
# do_palette = True


# def get_palette(num_cls):
#     """Returns the color map for visualizing the segmentation mask.
#     Args:
#         num_cls: Number of classes
#     Returns:
#         The color map
#     """
#     n = num_cls
#     palette = [0] * (n * 3)
#     for j in range(0, n):
#         lab = j
#         palette[j * 3 + 0] = 0
#         palette[j * 3 + 1] = 0
#         palette[j * 3 + 2] = 0
#         i = 0
#         while lab:
#             palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
#             palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
#             palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
#             i += 1
#             lab >>= 3
#     return palette


# transforms_list = []
# transforms_list += [transforms.ToTensor()]
# transforms_list += [Normalize_image(0.5, 0.5)]
# transform_rgb = transforms.Compose(transforms_list)

# net = U2NET(in_ch=3, out_ch=4)
# net = load_checkpoint_mgpu(net, checkpoint_path)
# net = net.to(device)
# net = net.eval()

# palette = get_palette(4)

# images_list = sorted(os.listdir(image_dir))
# pbar = tqdm(total=len(images_list))
# for image_name in images_list:
#     img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
#     # resize to 768x768
#     img = img.resize((768, 768), Image.BILINEAR)
#     image_tensor = transform_rgb(img)
#     image_tensor = torch.unsqueeze(image_tensor, 0)

#     output_tensor = net(image_tensor.to(device))
#     output_tensor = F.log_softmax(output_tensor[0], dim=1)
#     output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
#     output_tensor = torch.squeeze(output_tensor, dim=0)
#     output_tensor = torch.squeeze(output_tensor, dim=0)
#     output_arr = output_tensor.cpu().numpy()

#     output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
#     if do_palette:
#         output_img.putpalette(palette)
#     output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

#     pbar.update(1)

# pbar.close()