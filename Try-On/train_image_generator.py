import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import argparse
from Dataset import TryOnDataset, DataLoader
from ImageGeneratorNetwork import ImageGeneratorNetwork, EncapsulatedDiscriminator
from PIL import Image
from ConditionGeneratorNetwork import ConditionGenerator
from LossesConditionGenerator import GANLoss, LossVGG
# imprt tensorboard
from torch.utils.tensorboard import SummaryWriter
import time
from ConditionGeneratorNetwork import make_grid
import torch.nn.functional as F
import os
import os.path as osp
from tqdm import tqdm
from torch.utils.data import Subset
from matplotlib import pyplot as plt
# import gassiuaan blur 
from scipy.ndimage.filters import gaussian_filter
import torchgeometry as tgm




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


# clear the cache
torch.cuda.empty_cache()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# in this function we will define some paramters passed as command line arguments that wll be used in the training process
def get_options():
    parser = argparse.ArgumentParser(
        description='Image Generator Training')
    # the path to the dataset
    parser.add_argument('--dataset', type=str,
                        default='Try-On\data', help='path to dataset')
    # the path to the train_pairs.txt file
    parser.add_argument('--train_list', type=str,
                        default='train_pairs.txt', help='path to train list')
    # number of epochs
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    # the path to the test_pairs.txt file
    parser.add_argument('--test_list', type=str,
                        default='test_pairs.txt', help='path to test list')
    # batch size
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    return parser.parse_args()


def train_generator(data_loader_train, condition_generator, image_generator, discriminator, epochs, data_loader_val, writer, batch_size):
    # setting the image generator and the discriminator to train mode
    image_generator.train()
    discriminator.train()
    # setting the condition generator to eval mode
    condition_generator.eval()
    # send the models to the GPU
    image_generator.cuda()
    discriminator.cuda()
    condition_generator.cuda()
    # now we define the optimizers for the image generator and the discriminator
    optimizer_image_generator = torch.optim.Adam(image_generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
    # we can also use the learning rate scheduler to decrease the learning rate as the training process goes on
    scheduler_image_generator = torch.optim.lr_scheduler.StepLR(optimizer_image_generator, step_size=1000, gamma=0.1)
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=1000, gamma=0.1)
    # we define the loss functions
    criterion_GAN = GANLoss()
    criterion_VGG = LossVGG()
    criterion_L1 = nn.L1Loss()

    # we start the training loop
    for step in tqdm(range(0, 200000)):
        start_time = time.time()
        # load batch from the data loader 
        batch = data_loader_train.next_batch() 
        # we get the inputs of our models from the batch
        # Clothes Input for condition generator
        cloth = batch['cloth'].cuda()
        cloth_mask = batch['cloth_mask']
        cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
        # Pose Input for condition generator
        dense_pose = batch['dense_pose'].cuda()
        parse_agnostic = batch['parse_agnostic'].cuda()
        # we load the agnositic image 
        agnostic_image = batch['agnostic'].cuda()
        # we load the original image
        real_image = batch['image'].cuda()

        with torch.no_grad():
            # now we call the condition generator to get the output
            # the image gene works on higher resolution images so we need to resize the input so we can feed it to the condition generator\
            cloth_resized = F.interpolate(cloth, size=(256, 192), mode='bilinear', align_corners=True)
            cloth_mask_resized = F.interpolate(cloth_mask, size=(256, 192), mode='nearest')
            dense_pose_resized = F.interpolate(dense_pose, size=(256, 192), mode='bilinear', align_corners=True)
            parse_agnostic_resized = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')


            fake_map, warped_cloth, warped_cloth_mask, flow_list = condition_generator(torch.cat((cloth_resized, cloth_mask_resized), dim=1), torch.cat((parse_agnostic_resized, dense_pose_resized), dim=1))


            tmp = torch.ones_like(fake_map.detach())
            tmp[:, 3:4, :, :] = warped_cloth_mask
            fake_map = fake_map * tmp

            # we need to do warping as the resoltion of the image generator is higher than the condition generator
            N, C, H, W = cloth.size()
            # make grid
            grid = make_grid(N, H, W)
            flow = flow_list[-1]
            FW, FH = flow.size(2), flow.size(1)
            flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            hor = 2 * flow[:, :, :, 0:1] / (FW / 2 - 1)
            ver = 2 * flow[:, :, :, 1:2] / (FH / 2 - 1)
            # we then concatenate the horizontal and vertical flow components
            flow_norm = torch.cat([hor, ver], 3)
            # we then add the grid to the flow
            grid = grid + flow_norm
            # we then warp the cloth
            warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
            # we then warp the cloth mask
            warped_cloth_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')

            # Show the warped cloth
            warped_cloth_np = warped_cloth[0].cpu().detach().numpy().transpose(1, 2, 0)
            plt.imshow(warped_cloth_np)
            plt.show()


            # now we will upsample the resolution of fake map to 1024 x 768
            fake_map = F.interpolate(fake_map, size=(1024, 768), mode='bilinear', align_corners=True)
            # we apply gassian blur to the fake map
            fake_map_gaussian = tgm.image.gaussian_blur(fake_map, (13, 13), (3, 3))
            fake_map = fake_map_gaussian.argmax(dim=1)[:, None] # gets max value of each pixel across all 13 channels 
            

            # save the image
            fake_map_np = fake_map[0].cpu().detach().numpy().transpose(1, 2, 0)
            plt.imshow(fake_map_np)
            plt.show()


            # occulusion removal    
            tmp = torch.softmax(fake_map_gaussian, dim=1)
            cloth_mask_with_body_removed = warped_cloth_mask - ((torch.cat([tmp[:, 1:3, :, :], tmp[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)) * warped_cloth_mask
            cloth_with_body_removed = warped_cloth * cloth_mask_with_body_removed + torch.ones_like(warped_cloth) * (1 - cloth_mask_with_body_removed)

            # show cloth with body removed
            cloth_with_body_removed_np = cloth_with_body_removed[0].cpu().detach().numpy().transpose(1, 2, 0)
            plt.imshow(cloth_with_body_removed_np)
            plt.show()

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
            
            parse_map = torch.FloatTensor(batch_size, 13, 1024, 768).zero_().cuda()
            parse_map = parse_map.scatter_(1, fake_map, 1.0)
            new_parse_map = torch.FloatTensor(batch_size, 7, 1024, 768).zero_().cuda()

 
            for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_map[:, i] += parse_map[:, label]


            new_parse_map = new_parse_map.detach()

        # now wwe need to call the forward function of the image generator to get the output
        fake_image = image_generator(torch.cat((agnostic_image, dense_pose, warped_cloth), dim = 1), new_parse_map)
        # now we will call the discriminator to get the output
        pred_fake_1, pred_fake_2 = discriminator(torch.cat((fake_image, new_parse_map), dim=1))
        pred_real_1, pred_real_2 = discriminator(torch.cat((real_image, new_parse_map), dim=1))

        # here we wil lstart the losses calculations for the image generator
        # we now calculate the gan loss
        loss_gan = criterion_GAN(pred_fake_1, True)
        # fearture matching loss
        loss_feature_matching = criterion_L1(pred_real_1, pred_fake_1) + criterion_L1(pred_real_2, pred_fake_2)
        loss_feature_matching = loss_feature_matching * 5
        # vgg loss
        loss_vgg = criterion_VGG(fake_image, real_image) * 10
        # take mean of all the losses
        total_loss_gen = (loss_gan + loss_feature_matching + loss_vgg) / 3

        # now will calculate the losses for the discriminator
        loss_fake = criterion_GAN(pred_fake_1, False)
        loss_real = criterion_GAN(pred_real_1, True)
        total_loss_disc = (loss_fake + loss_real) / 2


        optimizer_image_generator.zero_grad()
        total_loss_gen.backward(retain_graph=True)
        optimizer_image_generator.step()

        optimizer_discriminator.zero_grad()
        total_loss_disc.backward()
        optimizer_discriminator.step()

        if (step + 1) % 100 == 0:
            t = time.time() - start_time
            print("step: %d, time: %d, loss_G: %f, loss_D: %f" % (step + 1, t, total_loss_gen.item(), total_loss_disc.item()))

        if (step + 1) % 100 == 0:
            writer.add_scalar('loss_G', total_loss_gen.item(), step)
            writer.add_scalar('loss_D', total_loss_disc.item(), step)

        if (step + 1) % 1000 == 0:
            # save the model
            print("saving the model")
            torch.save(condition_generator.state_dict(),'image_generator.pth')
            torch.save(discriminator.state_dict(), 'discriminator_image_generator.pth')        


def main():
        # get the command line arguments
        opt = get_options()
        # create the dataset
        train_dataset = TryOnDataset(root=opt.dataset, mode='train',
                                   data_list=opt.train_list, transform=transform, height = 1024 , width = 768)
        # validation dataset
        test_dataset = TryOnDataset(root=opt.dataset, mode='val', data_list=opt.test_list, transform=transform)
        val_dataset = Subset(test_dataset, range(0, 2000))
        # data_loader for training
        data_loader_train = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size)    
        # data_loader for validation
        data_loader_val = DataLoader(val_dataset, shuffle=True, batch_size=opt.batch_size)    
        # create the condtion generator model taht will feed the input to the image generator
        condition_generator = ConditionGenerator(pose_channels= 16, cloth_channels = 4, output_channels = 13)
        # Image generator model
        image_generator = ImageGeneratorNetwork(input_channels = 9)
        # discriminator model
        discriminator = EncapsulatedDiscriminator(input_channels = 10)
        # we start the training process
        print('Start training the image generator')
        # Tensorboard writer
        writer = SummaryWriter(log_dir="logs")
        # read the model if it exists
        if os.path.exists('condition_generator.pth'):
            print('loading the condition generator model')
            condition_generator.load_state_dict(torch.load('condition_generator.pth'))
        

        train_generator(data_loader_train, condition_generator, image_generator, discriminator, opt.epochs, data_loader_val, writer, opt.batch_size)


if __name__ == '__main__':
    main()