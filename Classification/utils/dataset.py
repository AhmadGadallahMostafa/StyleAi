# File to prepare the dataset for training and testing

import os
from torch.utils.data import Dataset
import torch
import joblib
import math
import cv2
import torchvision.transforms as transforms

IMG_PATH = 'Classification\DatasetPrep\DeepFashion'
LABEL_PATH_TOP = 'Classification\inputs\labels/shoes'
LABEL_PATH = 'Classification\inputs\labels'


IMG_PATH_PRODUCT = 'Classification\DatasetPrep\\fashion-dataset\seg_images'

def train_valid_split(df):
    # shuffle the dataframe with a random seed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 90% for training and 10% for validation
    num_train_samples = math.floor(len(df) * 0.90)
    num_val_samples = math.floor(len(df) * 0.10)
    train_df = df[:num_train_samples].reset_index(drop=True)
    val_df = df[-num_val_samples:].reset_index(drop=True)
    return train_df, val_df

class FashionDataset_Deep(Dataset):
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
    
class FashionDataset_Product(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.num_list_gender = joblib.load(LABEL_PATH_TOP + '/gender.pkl')
        self.num_list_color = joblib.load(LABEL_PATH_TOP + '/baseColour.pkl')
        self.num_list_article = joblib.load(LABEL_PATH_TOP + '/articleType.pkl')
        self.num_list_usage = joblib.load(LABEL_PATH_TOP + '/usage.pkl')
        self.is_train = is_train
        # the training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
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
        image = cv2.imread(IMG_PATH_PRODUCT + f"/{self.df['id'][index]}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        cat_gender = self.df['gender'][index]
        label_gender = self.num_list_gender[cat_gender]

        cat_article = self.df['articleType'][index]
        label_article = self.num_list_article[cat_article]
        
        cat_color = self.df['baseColour'][index]
        label_color = self.num_list_color[cat_color]

        cat_usage = self.df['usage'][index]
        label_usage = self.num_list_usage[cat_usage]
        
        # image to float32 tensor
        image = torch.tensor(image, dtype=torch.float32)
        # labels to long tensors
        label_gender = torch.tensor(label_gender, dtype=torch.long)
        label_article = torch.tensor(label_article, dtype=torch.long)
        label_color = torch.tensor(label_color, dtype=torch.long)
        label_usage = torch.tensor(label_usage, dtype=torch.long)
        return {
            'image': image,
            'article': label_article,
            'color': label_color,
            'gender': label_gender,
            'usage': label_usage
        }