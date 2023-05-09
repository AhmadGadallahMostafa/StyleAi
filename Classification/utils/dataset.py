# File to prepare the dataset for training and testing

import os
from torch.utils.data import Dataset
from utils.utils import clean_data
import torch
import joblib
import math
import cv2
import torchvision.transforms as transforms

from utils.utils import clean_data

IMG_PATH = 'Classification\DatasetPrep\DeepFashion'
LABEL_PATH = 'Classification\inputs\labels'

class FashionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df

        # for deep fashion dataset
        self.num_list_category = joblib.load(LABEL_PATH + '/category.pkl')
        self.is_train = is_train
        # the training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomAffine(30),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # the validation transforms
        if not self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(IMG_PATH, str(self.df['image_path'][index])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        cat_category = self.df['category'][index]

        label_category = self.num_list_category[cat_category]
        
        image = torch.tensor(image, dtype=torch.float32)
        # change label to tensor
        label_category = torch.tensor(label_category, dtype=torch.long)

        return {
            'image': image,
            'category': label_category
        }