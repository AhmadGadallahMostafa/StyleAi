import torch
import torch.nn as nn
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
import argparse
import os
import time
from ConditionGeneratorNetwork import ConditionGenerator
from Dataset import TryOnDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from ConditionGeneratorNetwork import make_grid
from ImageGeneratorNetworkOG import ImageGeneratorNetwork
import torchgeometry as tgm
from matplotlib import pyplot as plt
from PIL import Image
from PreProcessing import PreProcessing
from tqdm import tqdm

def get_options():
    parser = argparse.ArgumentParser(
        description='Image Generator Training')
    # the path to the dataset
    parser.add_argument('--dataset', type=str, default='Try-On\data', help='path to dataset')
    # the path to the train_pairs.txt file
    parser.add_argument('--train_list', type=str, default='train_pairs.txt', help='path to train list')
    # the path to the test_pairs.txt file
    parser.add_argument('--test_list', type=str, default='test_pairs.txt', help='path to test list')
    # batch size
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    return parser.parse_args()

def ndim_tensor2im(image_tensor, imtype=np.uint8, batch=0):
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)

def visualize_segmap(input, multi_channel=True, tensor_out=True, batch=0) :
    palette = [
        0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
        254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
        85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
        0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
    ]
    input = input.detach()
    if multi_channel :
        input = ndim_tensor2im(input,batch=batch)
    else :
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out :
        trans = transforms.ToTensor()
        return trans(input.convert('RGB'))

    return input

def main():
    opt = get_options()
    # Preprocess the images
    input_path_image = "Try-On/InputImages"
    input_path_cloth = "Try-On/InputClothesImages"
    preProcessing = PreProcessing()
    preProcessing.run()

    # Creat condition generator and load weights
    cg = ConditionGenerator(16, 4, 13)
    cg.load_state_dict(torch.load('condition_generator.pth'), strict=False)
    cg.cuda()
    cg.eval()
    print("Condition Generator loaded")
    
    # Create image generator and load weights
    ig = ImageGeneratorNetwork(9).cuda()
    ig.load_state_dict(torch.load('image_generator_og.pth'))
    ig.cuda()
    ig.eval()
    
    print("Image Generator loaded")
    
    # Create a Dataset object and a DataLoader object
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #dataset = TryOnDataset(root=opt.dataset, mode='test', data_list=opt.test_list, transform=transform , height=1024, width=768)
    #dt = DataLoader(dataset, shuffle=True, batch_size=1)

    # preprocessed images
    agnostic_path = "Try-On/PreProcessedImages/Agnostic"
    cloth_mask_path = "Try-On/PreprocessedImages/ClothMask"
    dense_pose_path = "Try-On/PreprocessedImages/DensePose"
    parse_agnostic_path = "Try-On/PreprocessedImages/ParseWithoutUpper"
    parse_path = "Try-On/PreprocessedImages/ConvertedParse"



    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()
    j = 0
    for im_name in tqdm(os.listdir("Try-On/InputImages")):
        for cloth_name in os.listdir("Try-On/InputClothesImages"):
            with torch.no_grad():
                # Get Original Image (.jpg)
                real_image = Image.open(os.path.join(input_path_image, im_name))
                real_image = transforms.Resize((1024, 768), interpolation = 2)(real_image)
                real_image = transform(real_image).cuda()
                real_image = real_image.unsqueeze(0)

                # Get Clothes Image (.jpg)
                cloth = Image.open(os.path.join(input_path_cloth, cloth_name))
                cloth = transforms.Resize((1024, 768), interpolation = 2)(cloth)
                cloth = transform(cloth).cuda()
                cloth = cloth.unsqueeze(0)
                
                # Get Cloth Mask (.png)
                temp = cloth_name.replace('.jpg', '.png')
                cloth_mask = Image.open(os.path.join(cloth_mask_path, temp))
                cloth_mask = transforms.Resize((1024, 768), interpolation = 0)(cloth_mask)
                clothes_mask_array = np.array(cloth_mask)
                # float threshold
                clothes_mask_array = (clothes_mask_array >= 128).astype(np.float32)
                cloth_mask = torch.from_numpy(clothes_mask_array)
                cloth_mask.unsqueeze_(0)
                cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
                cloth_mask = cloth_mask.unsqueeze(0)

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

                # Get Parse Agnostic (.png)
                temp = im_name.replace('.jpg', '.png')
                parse_agnostic = Image.open(os.path.join(parse_agnostic_path, temp))
                parse_agnostic = transforms.Resize((1024, 768), interpolation = 0)(parse_agnostic)
                parse_agnostic_array = torch.from_numpy(np.array(parse_agnostic)[None]).long()
                # we need to reduce the channels of the parse agnostic from 20 to 13 so we can feed it to the condition generator
                parse_agnostic_map = torch.FloatTensor(20, 1024, 768).zero_()
                parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic_array, 1.0)
                new_parse_agnostic_map = torch.FloatTensor(13, 1024, 768).zero_()
                for i in range(len(labels)):
                    for label in labels[i][1]:
                        new_parse_agnostic_map[i] += parse_agnostic_map[label]
                parse_agnostic = new_parse_agnostic_map.cuda()
                parse_agnostic = parse_agnostic.unsqueeze(0)

                # Get Dense Pose (.jpg)
                dense_pose = Image.open(os.path.join(dense_pose_path, im_name))
                dense_pose = transforms.Resize((1024, 768), interpolation = 2)(dense_pose)
                dense_pose = transform(dense_pose).cuda()
                dense_pose = dense_pose.unsqueeze(0)

                # Get original parse (.png)
                temp = im_name.replace('.jpg', '.png')
                original_parse = Image.open(os.path.join(parse_path, temp))
                original_parse = transforms.Resize((1024, 768), interpolation = 0)(original_parse)
                original_parse_array = torch.from_numpy(np.array(original_parse)[None]).long()
                # we need to reduce the channels of the parse from 20 to 13 so we can feed it to the condition generator
                parse_map = torch.FloatTensor(20, 1024, 768).zero_()
                parse_map = parse_map.scatter_(0, original_parse_array, 1.0)
                new_parse_map = torch.FloatTensor(13, 1024, 768).zero_()
                for i in range(len(labels)):
                    for label in labels[i][1]:
                        new_parse_map[i] += parse_map[label]
                original_parse = new_parse_map.cuda()
                original_parse = original_parse.unsqueeze(0)

                # Get Agnostic Image (.png)
                temp = im_name.replace('.jpg', '.png')
                agnostic_image = Image.open(os.path.join(agnostic_path, temp))
                agnostic_image = transforms.Resize((1024, 768), interpolation = 2)(agnostic_image)
                agnostic_image = transform(agnostic_image).cuda()
                agnostic_image = agnostic_image.unsqueeze(0)

                # # Clothes Input for condition generator
                # cloth = batch['cloth'].cuda()
                # cloth_mask = batch['cloth_mask']
                # cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
                
                # batch = dt.next_batch()
                
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
                #flow_norm = torch.cat([hor, ver], 3)
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
                original_image_np = real_image[0].cpu().detach().numpy().transpose(1, 2, 0)
                # plt.imshow(original_image_np)
                # plt.show()

                # show clothes 
                # cloth_np = cloth[0].cpu().detach().numpy().transpose(1, 2, 0)
                # plt.imshow(cloth_np)
                # plt.show()

                # show the fake image
                fake_image_np = fake_image[0].cpu().detach().numpy().transpose(1, 2, 0)
                # plt.imshow(fake_image_np)
                # plt.show()
                temp = im_name.replace('.jpg', '') + cloth_name.replace('.jpg', '')
                real_image = F.interpolate(real_image, size=(512, 384), mode='bilinear', align_corners=True)
                cloth_temp = F.interpolate(cloth, size=(512, 384), mode='bilinear', align_corners=True)
                grid = make_image_grid([real_image[0].cpu() / 2 + 0.5, cloth_temp[0].cpu() / 2 + 0.5, fake_image[0].cpu() / 2 + 0.5], nrow=3)
                save_image(grid, os.path.join('output_image_generator/', temp + '.png'))
                grid = make_image_grid([cloth[0].cpu(), cloth_with_body_removed[0].cpu(), visualize_segmap(fake_map_gaussian).detach().cpu(), visualize_segmap(original_parse).detach().cpu()], nrow=2)
                save_image(grid, os.path.join('output_condition_generator/', temp + '.png'))



if __name__ == '__main__':
    main()
