# here we will define the class that will be used to load the dataset
# that datset is the same as the one used in the paper which belongs to CP-VITON paper

import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import json

# For visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class TryOnDataset(Dataset):
    def __init__(self, root, mode, data_list, transform=None):
        super(TryOnDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.data_list = data_list
        self.transform = transform
        self.height = 256
        self.width = 192
        self.semantic_channels = 13
        self.transform = transform
        if self.mode == 'train':
            self.root = osp.join(self.root, 'train')
        else:
            self.root = osp.join(self.root, 'test')
        # now we will read the image names from the data_list
        self.image_paired = []
        with open(osp.join(root, data_list), 'r') as f:
            for line in f.readlines():
                self.image_paired.append(line.split(' ')[0])


    # we now overide the function __getitem__ to return the image and the clothes
    def __getitem__(self, index):
        #1- we will get the image and then resize it 
        image_name = self.image_paired[index]
        person_image = Image.open(osp.join(self.root, 'image', image_name))
        person_image = transforms.Resize(self.width, interpolation = 2)(person_image)
        person_image = self.transform(person_image)
        #2 get clothes image
        clothes_image = Image.open(osp.join(self.root, 'cloth', image_name)).convert('RGB')
        clothes_image = transforms.Resize(self.width, interpolation = 2)(clothes_image)
        clothes_image = self.transform(clothes_image)
        #3 clothes image mask
        clothes_mask = Image.open(osp.join(self.root, 'cloth-mask', image_name))
        clothes_mask = transforms.Resize(self.width, interpolation = 0)(clothes_mask)
        clothes_mask_array = np.array(clothes_mask)
        # float threshold
        clothes_mask_array = (clothes_mask_array >= 128).astype(np.float32)
        clothes_mask = torch.from_numpy(clothes_mask_array)
        clothes_mask.unsqueeze_(0)
        # get parse-v3 image which has the sama name as image_name 
        # but it is in the folder image-parse-v3 and it is a png image
        parse_v3_image = Image.open(osp.join(self.root, 'image-parse-v3', image_name.replace('.jpg', '.png')))
        parse_v3_image = transforms.Resize(self.width, interpolation = 0)(parse_v3_image)
        parse_v3_image_array = torch.from_numpy(np.array(parse_v3_image)[None]).long()
        
        # parse map
        labels = {
            0:  ['background',  [0, 10]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]]
        }
        
        parse_map = torch.FloatTensor(20, self.height, self.width).zero_()
        parse_map = parse_map.scatter_(0, parse_v3_image_array, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_channels, self.height, self.width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]
                
        parse_onehot = torch.FloatTensor(1, self.height, self.width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # get parse-v3.2 image which has the sama name as image_name
        # but it is in the folder image-parse-agnostic-v3.2 and it is a png image
        parse_agnostic_v3_2_image = Image.open(osp.join(self.root, 'image-parse-agnostic-v3.2', image_name.replace('.jpg', '.png')))
        parse_agnostic_v3_2_image = transforms.Resize(self.width, interpolation = 0)(parse_agnostic_v3_2_image)
        parse_agnostic_v3_2_image_array = torch.from_numpy(np.array(parse_agnostic_v3_2_image)[None]).long()

        # now we generate the parse agnostic map
        parse_agnostic_map = torch.FloatTensor(20, self.height, self.width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic_v3_2_image_array, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_channels, self.height, self.width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        
        # generate the parse cloth and parse cloth mask
        parse_cloth_mask = new_parse_map[3:4]
        parse_cloth = person_image * parse_cloth_mask + (1 - parse_cloth_mask)

        # load the pose points image 
        # which is in file openpose_img and it has the same name as image_name 
        # but with externsion _rendered.png
        pose_points_image = Image.open(osp.join(self.root, 'openpose_img', image_name.replace('.jpg', '_rendered.png')))
        pose_points_image = transforms.Resize(self.width, interpolation = 2)(pose_points_image)
        pose_points_image = self.transform(pose_points_image)

        # now we load pose points from json file 
        pose_points = image_name.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
        with open(osp.join(self.root, "openpose_json", pose_points), 'r') as f:
            pose_name = json.load(f)
            pose_points = pose_name['people'][0]['pose_keypoints_2d']
            pose_points = np.array(pose_points).reshape(-1, 3)[:, :2]

        # we load the dense pose 
        # which is in file image-densepose and it has the same name as image_name
        densepose_image = Image.open(osp.join(self.root, 'image-densepose', image_name))
        densepose_image = transforms.Resize(self.width, interpolation = 2)(densepose_image)
        densepose_image = self.transform(densepose_image)

        # We are generate the last image which will be input to the imaga generator 
        # but we will use the images generated by the author of the paper
        
        # we load the image generated by the author of the paper
        # which is in file image-parse-agnostic-v3.2 and it has the same name as image_name
        agnostic_image = Image.open(osp.join(self.root, 'agnostic-v3.2', image_name))
        agnostic_image = transforms.Resize(self.width, interpolation = 2)(agnostic_image)
        agnostic_image = self.transform(agnostic_image)


        # return the result dict
        return {
            # input 1 which is the cloth and cloth mask 
            'cloth': clothes_image,
            'cloth_mask': clothes_mask,
            # input 2 which is the densepose image and the new parse agnostic map
            'dense_pose': densepose_image,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_points_image, 
            # image generotr input
            'agnostic': agnostic_image,
            # losses 
            # cross entropy loss for the parse one hot
            'parse_one_hot': parse_onehot,
            # GAN LOSS 
            'parse' : new_parse_map,
            # L1 loss 
            'parse_cloth_mask': parse_cloth_mask,
            # VGG loss
            'parse_cloth': parse_cloth,
            # for showing imagse 
            'image': person_image,
            'cloth_name': image_name,
        }
    
    # override len function
    def __len__(self):
        return len(self.image_paired)
    


        
class DataLoader(object):
    def __init__(self, dataset, shuffle=True, batch_size=8):
        super(DataLoader, self).__init__()
        if shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:   
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size = batch_size, shuffle=(train_sampler is None), pin_memory = True, drop_last = True, sampler = train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch



    

        
