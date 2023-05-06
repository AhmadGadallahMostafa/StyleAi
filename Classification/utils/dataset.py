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

def train_valid_split(df):
    # shuffle the dataframe with a random seed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 90% for training and 10% for validation
    num_train_samples = math.floor(len(df) * 0.90)
    num_val_samples = math.floor(len(df) * 0.10)
    train_df = df[:num_train_samples].reset_index(drop=True)
    val_df = df[-num_val_samples:].reset_index(drop=True)
    return train_df, val_df

class FashionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        # self.num_list_gender = joblib.load(LABEL_PATH + '/gender.pkl')
        # self.num_list_usage = joblib.load(LABEL_PATH + '/usage.pkl')
        # self.num_list_sub_category = joblib.load(LABEL_PATH + '/subCategory.pkl')
        # self.num_list_base_colour = joblib.load(LABEL_PATH + '/baseColour.pkl')
        # self.num_list_article_type = joblib.load(LABEL_PATH + '/articleType.pkl')

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
        
        # cat_gender = self.df['gender'][index]
        # cat_usage = self.df['usage'][index]
        # cat_sub = self.df['subCategory'][index]
        # cat_article = self.df['articleType'][index]
        # cat_color = self.df['baseColour'][index]

        cat_category = self.df['category'][index]

        # label_gender = self.num_list_gender[cat_gender]
        # label_usage = self.num_list_usage[cat_usage]
        # label_sub = self.num_list_sub_category[cat_sub]
        # label_article = self.num_list_article_type[cat_article]
        # label_color = self.num_list_base_colour[cat_color]

        label_category = self.num_list_category[cat_category]
        
        image = torch.tensor(image, dtype=torch.float32)
        # change label to tensor
        # label_gender = torch.tensor(label_gender, dtype=torch.long)
        # label_usage = torch.tensor(label_usage, dtype=torch.long)
        # label_sub = torch.tensor(label_sub, dtype=torch.long)
        # label_article = torch.tensor(label_article, dtype=torch.long)
        # label_color = torch.tensor(label_color, dtype=torch.long)

        label_category = torch.tensor(label_category, dtype=torch.long)

        return {
            'image': image,
            'category': label_category
        }