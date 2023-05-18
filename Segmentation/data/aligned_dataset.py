import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import cv2
import collections
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data

class AlignedDataset(data.Dataset):

    def __init__(self):
        super(AlignedDataset, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.image_dir = opt.IMAGE_FOLDER
        self.df_path = opt.CSV_PATH
        self.width = opt.WIDTH
        self.height = opt.HEIGHT

        # for rgb imgs

        transforms_list = []
        # we need to make sure that the transform is tensor first
        transforms_list += [transforms.ToTensor()]
        # then we normalize using mean and std of 0.5
        transforms_list += [Normalize(0.5, 0.5)]
        self.transform_rgb = transforms.Compose(transforms_list)

        # reading the csv file and creating a dataframe
        self.df = pd.read_csv(self.df_path)
        # creating a dictionary with image_id as key and labels and encoded pixels as values
        self.image_info = collections.defaultdict(dict)
        # creating the dictionary with image_id as key and image_path as value
        self.df["CategoryId"] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = (self.df.groupby("ImageId")["EncodedPixels", "CategoryId"].agg(lambda x: list(x)).reset_index())
        size_df = self.df.groupby("ImageId")["Height", "Width"].mean().reset_index()
        temp_df = temp_df.merge(size_df, on="ImageId", how="left")

        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row["ImageId"]
            image_path = os.path.join(self.image_dir, image_id)
            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]["orig_height"] = row["Height"]
            self.image_info[index]["orig_width"] = row["Width"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]

        self.dataset_size = len(self.image_info)

    def __getitem__(self, index):
        # load images ad masks
        idx = index
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        image_tensor = self.transform_rgb(img)

        info = self.image_info[idx]
        mask = np.zeros(
            (len(info["annotations"]), self.width, self.height), dtype=np.uint8
        )
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            sub_mask = self.rle_decode(
                annotation, (info["orig_height"], info["orig_width"])
            )
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in upperbody:
                first_channel += new_masks[i]
            elif labels[i] in lowerbody:
                second_channel += new_masks[i]
            elif labels[i] in wholebody:
                third_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3
        conflict_mask = (final_label <= 3).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        target_tensor = torch.as_tensor(final_label, dtype=torch.int64)

        return image_tensor, target_tensor

    def __len__(self):
        return len(self.image_info)

    def name(self):
        return "AlignedDataset"

    def rle_decode(self, mask_rle, shape):
        """
        mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
        shape: (height,width) of array to return
        Returns numpy array according to the shape, 1 - mask, 0 - background
        """
        shape = (shape[1], shape[0])
        s = mask_rle.split()
        # gets starts & lengths 1d arrays
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        # gets ends 1d array
        ends = starts + lengths
        # creates blank mask image 1d array
        temp_shape = int(shape[0] * shape[1])
        img = np.zeros(temp_shape, dtype=np.uint8)
        # sets mark pixles
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # reshape as a 2d mask image
        temp_shape = (int(shape[0]), int(shape[1]))
        return img.reshape(temp_shape).T  # Needed to align to RLE direction
    
class Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (int, float, tuple, list))
        assert isinstance(std, (int, float, tuple, list))
        self.mean = mean
        self.std = std

        self.normalize_scale_1 = transforms.Normalize(mean, std)
        self.normalize_scale_3 = transforms.Normalize([mean] * 3, [std] * 3)
        self.normalize_scale_18 = transforms.Normalize([mean] * 18, [std] * 18)
        
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            return self.normalize_scale_1(tensor)
        elif tensor.shape[0] == 3:
            return self.normalize_scale_3(tensor)
        elif tensor.shape[0] == 18:
            return self.normalize_scale_18(tensor)
        else:
            raise ValueError("Tensor shape not supported: {}".format(tensor.shape))