import torch
import torch.nn as nn
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from ConditionGeneratorNetwork import ConditionGenerator
from Dataset import TryOnDataset, DataLoader
from train_image_generator import get_options
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from ConditionGeneratorNetwork import make_grid
import torchgeometry as tgm
from matplotlib import pyplot as plt

def main():
    opt = get_options()
    # Creat condition generator and load weights
    cg = ConditionGenerator(16, 4, 13)
    cg.load_state_dict(torch.load('condition_generator.pth'), strict=False)
    cg.cuda()
    cg.eval()
    print("Condition Generator loaded")
    
    # Create image generator and load weights
    ig = torch.load("generator.pt").cuda()
    ig.eval()
    print("Image Generator loaded")
    
    # Create a Dataset object and a DataLoader object
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = TryOnDataset(root=opt.dataset, mode='test', data_list=opt.test_list, transform=transform, height=1024, width=768)
    dt = DataLoader(dataset, shuffle=True, batch_size=1)

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()
    for input in dt.data_loader:
        with torch.no_grad():
            # Clothes Input for condition generator
            cloth = input['cloth'].cuda()
            cloth_mask = input['cloth_mask']
            cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
            # Pose Input for condition generator
            dense_pose = input['dense_pose'].cuda()
            parse_agnostic = input['parse_agnostic'].cuda()
            # we load the agnositic image 
            agnostic_image = input['agnostic'].cuda()
            # we load the original image
            real_image = input['image'].cuda()
            
            # First we generate the fake map and the warped cloth using the condition generator
            # We need to resize the cloth, cloth_mask, dense_pose and parse_agnostic to the same size used in the training
            cloth_resized = F.interpolate(cloth, size=(256, 192), mode='bilinear', align_corners=True)
            cloth_mask_resized = F.interpolate(cloth_mask, size=(256, 192), mode='nearest')
            dense_pose_resized = F.interpolate(dense_pose, size=(256, 192), mode='bilinear', align_corners=True)
            parse_agnostic_resized = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
            
            # we generate the fake map and the warped cloth
            fake_map, warped_cloth, warped_cloth_mask, flow_list = cg(torch.cat((cloth_resized, cloth_mask_resized), dim=1), torch.cat((parse_agnostic_resized, dense_pose_resized), dim=1))

            tmp = torch.ones_like(fake_map.detach())
            tmp[:, 3:4, :, :] = warped_cloth_mask
            fake_map = fake_map * tmp

            # now we will upsample the resolution of fake map to 1024 x 768
            fake_map = F.interpolate(fake_map, size=(1024, 768), mode='bilinear')
            # we apply gassian blur to the fake map
            fake_map_gaussian = gauss(fake_map)
            fake_map = fake_map_gaussian.argmax(dim=1)[:, None] # gets max value of each pixel across all 13 channels 


            # we need to reduce the channels of the fake map from 13 to 7 so we can feed it to the image generator
            # parse map
            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }

            parse_map = torch.FloatTensor(opt.batch_size, 13, 1024, 768).zero_().cuda()
            parse_map = parse_map.scatter_(1, fake_map, 1.0)
            new_parse_map = torch.FloatTensor(opt.batch_size, 7, 1024, 768).zero_().cuda()

            for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_map[:, i] += parse_map[:, label]
            
            #new_parse_map = new_parse_map.detach()


            # we need to do warping as the resolution of the image generator is higher than the condition generator
            N, C, H, W = cloth.size()
            # make grid
            grid = make_grid(N, H, W)
            flow = flow_list[-1]
            FW, FH = flow.size(2), flow.size(1)
            flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
            hor = 2 * flow[:, :, :, 0:1] / (FW / 2 - 1)
            ver = 2 * flow[:, :, :, 1:2] / (FH / 2 - 1)
            # we then concatenate the horizontal and vertical flow components
            flow_norm = torch.cat([hor, ver], 3)
            flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
            # we then add the grid to the flow
            grid = grid + flow_norm
            # we then warp the cloth
            warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
            # we then warp the cloth mask
            warped_cloth_mask = F.grid_sample(cloth_mask, grid, padding_mode='border')

            # Show the warped cloth
            # warped_cloth_np = warped_cloth[0].cpu().detach().numpy().transpose(1, 2, 0)
            # plt.imshow(warped_cloth_np)
            # plt.show()

            # occulusion removal    
            tmp = torch.softmax(fake_map_gaussian, dim=1)
            cloth_mask_with_body_removed = warped_cloth_mask - ((torch.cat([tmp[:, 1:3, :, :], tmp[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)) * warped_cloth_mask
            cloth_with_body_removed = warped_cloth * cloth_mask_with_body_removed + torch.ones_like(warped_cloth) * (1 - cloth_mask_with_body_removed)

            # show cloth with body removed
            # cloth_with_body_removed_np = cloth_with_body_removed[0].cpu().detach().numpy().transpose(1, 2, 0)
            # plt.imshow(cloth_with_body_removed_np)
            # plt.show()

            # now wwe need to call the forward function of the image generator to get the output
            fake_image = ig(torch.cat((agnostic_image, dense_pose, cloth_with_body_removed), dim = 1), new_parse_map)

            # show the fake image
            fake_image_np = fake_image[0].cpu().detach().numpy().transpose(1, 2, 0)
            plt.imshow(fake_image_np)
            plt.show()
            print("wedw")




if __name__ == '__main__':
    main()
