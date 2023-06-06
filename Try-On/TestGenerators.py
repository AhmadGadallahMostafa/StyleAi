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
import torchgeometry as tgm
from matplotlib import pyplot as plt
from PIL import Image
from ImageGeneratorNetworkOG import ImageGeneratorNetwork
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import structural_similarity_index_measure

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
    dataset = TryOnDataset(root=opt.dataset, mode='test', data_list=opt.test_list, transform=transform , height=1024, width=768)
    dt = DataLoader(dataset, shuffle=True, batch_size=1)

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()
    j = 0
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips.cuda()
    
    lpips_loss_image = 0
    lpips_loss_parse = 0
    ssim_loss_image = 0
    ssim_loss_parse = 0

    for j in range(0, 2000):
        with torch.no_grad():
            # Clothes Input for condition generator
            batch = dt.next_batch()
            cloth = batch['cloth'].cuda()
            cloth_mask = batch['cloth_mask']
            cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
            
            #batch = dt.next_batch()
            # Pose Input for condition generator
            dense_pose = batch['dense_pose'].cuda()
            parse_agnostic = batch['parse_agnostic'].cuda()
            original_parse = batch['parse'].cuda()
            # we load the agnositic image 
            agnostic_image = batch['agnostic'].cuda()
            # we load the original image
            real_image = batch['image'].cuda()
            
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
            # original_image_np = real_image[0].cpu().detach().numpy().transpose(1, 2, 0)
            # plt.imshow(original_image_np)
            # plt.show()

            # show clothes 
            # cloth_np = cloth[0].cpu().detach().numpy().transpose(1, 2, 0)
            # plt.imshow(cloth_np)
            # plt.show()

            # show the fake image
            #fake_image_np = fake_image[0].cpu().detach().numpy().transpose(1, 2, 0)
            #plt.imshow(fake_image_np)
            #plt.show()
            real_image = F.interpolate(real_image, size=(512, 384), mode='bilinear', align_corners=True)
            cloth_temp = F.interpolate(cloth, size=(512, 384), mode='bilinear', align_corners=True)
            lpips_loss_image += lpips(fake_image, real_image)
            fake_map_visualize = visualize_segmap(fake_map_gaussian).cuda()
            original_parse_visualize = visualize_segmap(original_parse).cuda()
            lpips_loss_parse += lpips(fake_map_visualize.unsqueeze(0), original_parse_visualize.unsqueeze(0))
            print("LPIPS Loss Image: ", lpips_loss_image)
            print("LPIPS Loss Parse: ", lpips_loss_parse)
            
            # fid_loss_image = fid(fake_image, real_image)
            # print("FID Loss Image: ", fid_loss_image)
            # fid_loss_parse = fid(fake_map_visualize.unsqueeze(0), original_parse_visualize.unsqueeze(0))
            # print("FID Loss Parse: ", fid_loss_parse)

            ssim_loss_image += structural_similarity_index_measure(fake_image, real_image)
            print("SSIM Loss Image: ", ssim_loss_image)
            ssim_loss_parse += structural_similarity_index_measure(fake_map_visualize.unsqueeze(0), original_parse_visualize.unsqueeze(0))
            print("SSIM Loss Parse: ", ssim_loss_parse)


            grid = make_image_grid([real_image[0].cpu() / 2 + 0.5, cloth_temp[0].cpu() / 2 + 0.5, fake_image[0].cpu() / 2 + 0.5], nrow=3)
            save_image(grid, os.path.join('output_image_generator/', str(j) + '.png'))

            grid = make_image_grid([cloth[0].cpu() / 2 + 0.5, cloth_with_body_removed[0].cpu() / 2 + 0.5, visualize_segmap(fake_map_gaussian).detach().cpu(), visualize_segmap(original_parse).detach().cpu()], nrow=2)
            save_image(grid, os.path.join('output_condition_generator/', str(j) + '.png'))
            j += 1
            print("Finished iteration: ", j)
    
    print("Final LPIPS Loss Image: ", lpips_loss_image / 2000)
    print("Final LPIPS Loss Parse: ", lpips_loss_parse / 2000)
    print("Final SSIM Loss Image: ", ssim_loss_image / 2000)
    print("Final SSIM Loss Parse: ", ssim_loss_parse / 2000)
    # save the losses
    f = open("losses tryon/losses.txt", "a")
    f.write("Final LPIPS Loss Image: " + str(lpips_loss_image / 2000) + "\n")
    f.write("Final LPIPS Loss Parse: " + str(lpips_loss_parse / 2000) + "\n")
    f.write("Final SSIM Loss Image: " + str(ssim_loss_image / 2000) + "\n")
    f.write("Final SSIM Loss Parse: " + str(ssim_loss_parse / 2000) + "\n")
    f.close()



if __name__ == '__main__':
    main()
