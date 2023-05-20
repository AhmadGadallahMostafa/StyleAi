import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class CatDataset(Dataset):
    def __init__(self,
                root = "C:/Users/Ahmed/Documents/GitHub/fashion_compatibility_mcn/data/images",
                data_dir = "../data",
                data = 'train_no_dup_with_category_3more_name.json',
                transform = None,
                use_mean_img=True,
                neg_samples=True):
        
        self.root = root
        self.data_dir = data_dir
        self.data = json.load(open(os.path.join(data_dir, data)))
        self.data = [(x, y) for x, y in self.data.items()]
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.neg_samples = neg_samples

        self.categories = []
        self.categ2idx = {}
        self.categ2idx['UNK'] = 0
        self.categories.append('UNK')
        with open(os.path.join(self.data_dir, 'final_word_dict.txt')) as f:
            for i, line in enumerate(f):
                category = line.strip().split()[0]
                if category not in self.categ2idx:
                    self.categories.append(category)
                    self.categ2idx[category] = i + 1
    
    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        idx_list = []
        for w in name.split():
            if w in self.categ2idx:
                idx_list.append(self.categ2idx[w])
            else:
                idx_list.append(self.categ2idx['UNK'])
        return idx_list
    
    def __getitem__(self, index):
        item_categories = ['upper', 'bottom', 'shoe', 'bag', 'accessory']
        outfit_id, items = self.data[index]
        possible_negatives = []
        if random.randint(0, 1) and self.neg_samples:
            # sample a negative item
            possible_negatives = list(items.keys())
        
        images = []
        labels = []
        names = []
        for item in item_categories:
            if item in possible_negatives:
                neg_item = random.choice(self.data)
                while (neg_item[0] == outfit_id) or (item not in neg_item[1].keys()):
                    neg_item = random.choice(self.data)
                image_path = os.path.join(self.root, str(neg_item[0]), str(neg_item[1][item]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(neg_item[1][item]['name'])))
                labels.append('{}_{}'.format(neg_item[0], neg_item[1][item]['index']))
            elif item in items.keys():
                image_path = os.path.join(self.root, str(outfit_id), str(items[item]['index'])+'.jpg')
                names.append(torch.LongTensor(self.str_to_idx(items[item]['name'])))
                labels.append('{}_{}'.format(outfit_id, items[item]['index']))
            elif self.use_mean_img:
                image_path = os.path.join(self.data_dir, item+'.png')
                names.append(torch.LongTensor([])) 
                labels.append('{}_{}'.format(item, 'mean'))
            else:   
                continue
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        total_images = torch.stack(images)
        is_compat = len(possible_negatives) == 0 
        offsets = torch.LongTensor(list(itertools.accumulate([0] + [len(n) for n in names[:-1]])))
        return total_images, names, offsets, outfit_id, labels, is_compat

    def get_fitb_quesiton(self, index):
        item_categories = ['upper', 'bottom', 'shoe', 'bag', 'accessory']
        outfit_id, items = self.data[index]
        question_item = random.choice(list(items))
        question_id = f"{outfit_id}_{items[question_item]['index']}"

        images = []
        labels = []
        for item in item_categories:
            if item in items.keys():
                image_path = os.path.join(self.root, str(outfit_id), f"{items[item]['index']}.jpg")
                labels.append(f"{outfit_id}_{items[item]['index']}")
                image = Image.open(image_path).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
            elif self.use_mean_img:
                image_path = os.path.join(self.data_dir, f"{item}.png")
                labels.append(f"{item}_mean")
                image = Image.open(image_path).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
        total_images = torch.stack(images)

        possible_items_ids = [outfit_id]
        possible_items = []
        possible_items_labels = []
        while len(possible_items_ids) < 4:
            possible_item = random.choice(self.data)
            if possible_item[0] not in possible_items_ids and question_item in possible_item[1]:
                possible_items_ids.append(possible_item[0])
                image_path = os.path.join(self.root, str(possible_item[0]), f"{possible_item[1][question_item]['index']}.jpg")
                image = Image.open(image_path).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                possible_items.append(image)
                possible_items_labels.append(f"{possible_item[0]}_{possible_item[1][question_item]['index']}")
        
        return total_images, labels, question_item, question_id, possible_items, possible_items_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, offsets, set_ids, labels, is_compat = zip(*data)
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (
        lengths,
        images,
        names,
        offsets,
        set_ids,
        labels,
        is_compat
    )