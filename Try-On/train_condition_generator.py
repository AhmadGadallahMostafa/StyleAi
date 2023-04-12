import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import argparse
from Dataset import TryOnDataset, DataLoader
from ConditionGeneratorNetwork import ConditionGenerator, EncapsulatedDiscriminator
from LossesConditionGenerator import GANLoss, LossVGG
# imprt tensorboard
from torch.utils.tensorboard import SummaryWriter
import time
from ConditionGeneratorNetwork import make_grid
import torch.nn.functional as F
import os
import os.path as osp

# clear the cache
torch.cuda.empty_cache()



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def cross_entropy_2d(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, ignore_index=250
    )
    return loss


# in this function we will define some paramters passed as command line arguments that wll be used in the training process
def get_options():
    parser = argparse.ArgumentParser(
        description='Condition Generator Training')
    # the path to the dataset
    parser.add_argument('--dataset', type=str,
                        default='Try-On\data', help='path to dataset')
    # the path to the train_pairs.txt file
    parser.add_argument('--train_list', type=str,
                        default='train_pairs.txt', help='path to train list')
    # batch size
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    # number of epochs
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    return parser.parse_args()

# in this function we will define the training process
def train_condition(data_loader, condition_generator, discriminator, num_epochs):
    # we first set te models to train mode and send them to the GPU
    condition_generator.train()
    discriminator.train()
    condition_generator.cuda()
    discriminator.cuda()
    # we define the optimizers for the models
    optimizer_condition = torch.optim.Adam(condition_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # we define the loss functions
    criterion_gan = GANLoss()
    criterion_vgg = LossVGG()
    criterion_l1 = nn.L1Loss()
    # we define the tensorboard writer
    writer = SummaryWriter('runs/condition_generator')
    start_time = time.time()
    # we start the training process
    for epoch in range(num_epochs):
        # loop over batches 
        for i, batch in enumerate(data_loader):
            # we get the inputs of our models from the batch 
            # Clothes Input 
            cloth = batch['cloth'].cuda()
            cloth_mask = batch['cloth_mask']
            cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
            # Pose Input
            dense_pose = batch['dense_pose'].cuda()
            parse_agnostic = batch['parse_agnostic'].cuda()
            # Now we get the tensors that will be used for losses calculation
            parse_cloth = batch['parse_cloth'].cuda()     # vgg loss
            parse = batch['parse'].cuda()                 # gan loss 
            parse_one_hot = batch['parse_one_hot'].cuda() # cross entropy loss
            parse_cloth_mask = batch['parse_cloth_mask'].cuda() # l1 loss
            # now we call the forward function of the condition generator
            fake_map, warped_cloth, warped_cloth_mask, flow_list = condition_generator(torch.cat((cloth, cloth_mask), dim = 1), torch.cat((parse_agnostic, dense_pose), dim = 1))
            # now we calculate the losses
            # first we get the one hot encoding of warped_cloth_mask
            tmp = torch.ones_like(fake_map.detach()) # we create a tensor with ones
            tmp[:, 3:4, :, :] = warped_cloth_mask # we set the 4th channel to be the warped_cloth_mask
            fake_map = fake_map * tmp       # element wise multiplication
            # handling occlusion of cloth with body parts 
            tmp = torch.softmax(fake_map, dim = 1)
            cloth_mask_with_body_removed = warped_cloth_mask - ((torch.cat([tmp[:, 0:3, :, :], tmp[:, 5:, :, :]], dim = 1)).sum(dim = 1, keepdim=True)) * warped_cloth_mask
            cloth_with_body_removed = warped_cloth * cloth_mask_with_body_removed + torch.ones_like(warped_cloth) * (1 - cloth_mask_with_body_removed)

            # get the one hot encoding of warped cloth mask
            warped_cloth_mask_one_hot = torch.FloatTensor((warped_cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
            # now we will make the any pixel in the upper body of the fake map to be 1
            fake_cloth_mask = (torch.argmax(fake_map, dim = 1, keepdim = True) == 3).long()
            # now we find the missaligned pixels
            missaligned_pixels = (fake_cloth_mask != warped_cloth_mask_one_hot).float()
            # L1 loss
            l1_loss = criterion_l1(cloth_mask_with_body_removed, parse_cloth_mask)
            # VGG loss
            vgg_loss = criterion_vgg(cloth_with_body_removed, parse_cloth)

            #total varitaion loss which will take the last flow map and calculate the total variation loss
            # This is done for smoothness of the flow map
            tv_loss = 0
            for flow in flow_list[-1:]:
                tv_h = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                tv_w = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                tv_loss += tv_h + tv_w
            
            CN, _, CH, CW = cloth.size()
            weights = [16, 8, 4, 2, 1]
            j = 0
            # intermedate flow loss 
            for flow in flow_list[:-1]:
                FN, FH, FW, _ = flow.size()
                grid = make_grid(CN, CH, CW)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(CH, CW), mode='bilinear').permute(0, 2, 3, 1)
                hor = 2 * flow[:, :, :, 0:1] / (FW / 2 - 1)
                ver = 2 * flow[:, :, :, 1:2] / (FH / 2 - 1)
                # we then concatenate the horizontal and vertical flow components
                flow_norm = torch.cat([hor, ver], 3)
                warped_cloth = F.grid_sample(cloth, grid + flow_norm, padding_mode='border')
                warped_cloth_mask = F.grid_sample(cloth_mask, grid + flow_norm, padding_mode='border')
                # overalap remove
                tmp = torch.softmax(fake_map, dim = 1)
                warped_cloth_mask = warped_cloth_mask - ((torch.cat([tmp[:, 0:3, :, :], tmp[:, 5:, :, :]], dim = 1)).sum(dim = 1, keepdim=True)) * warped_cloth_mask
                # l1 loss
                l1_loss += criterion_l1(warped_cloth_mask, parse_cloth_mask) / weights[j]
                # vgg loss
                vgg_loss += criterion_vgg(warped_cloth, parse_cloth) / weights[j]
                j += 1
            
            # now we calculate the cross entropy loss
            cross_entropy_loss = cross_entropy_2d(fake_map, parse_one_hot.transpose(0, 1)[0].long())

            # gan loss
            # we first get the fake image
            fake_map_softmax = torch.softmax(fake_map, dim = 1)
            # we now call the discriminator
            discriminator_input = torch.cat((cloth.detach(), cloth_mask.detach(), parse_agnostic.detach(), dense_pose.detach(), fake_map_softmax), dim = 1)
            pred_image = discriminator(discriminator_input)
            gan_loss = criterion_gan(pred_image, True)
            # we now calculate the gan loss
            fake_image_pred = discriminator(torch.cat((cloth.detach(), cloth_mask.detach(), parse_agnostic.detach(), dense_pose.detach(), fake_map_softmax.detach()), dim = 1))
            real_image_pred = discriminator(torch.cat((cloth.detach(), cloth_mask.detach(), parse_agnostic.detach(), dense_pose.detach(), parse), dim = 1))
            loss_fake = criterion_gan(fake_image_pred, False)
            loss_real = criterion_gan(real_image_pred, True)

            # now we calculate the total loss
            generator_loss = 10 * l1_loss + vgg_loss + 2 * tv_loss + 10 * cross_entropy_loss + gan_loss
            discriminator_loss = (loss_fake + loss_real)

            # step 
            optimizer_condition.zero_grad()
            generator_loss.backward()
            optimizer_condition.step()

            optimizer_discriminator.zero_grad()
            discriminator_loss.backward()
            optimizer_discriminator.step()
            
            # print the losses
            if (i + 1) % 10 == 0:
                t = time.time() - start_time
                print("epoch: %d ,step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV loss: %.4f CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f"
                        % (epoch, i + 1, t, generator_loss.item(), l1_loss.item(), vgg_loss.item(), tv_loss.item(), cross_entropy_loss.item(), gan_loss.item(), discriminator_loss.item(), loss_real.item(), loss_fake.item()), flush=True)

            # save the model
            if (i + 1) % 1000 == 0:
                print('saving the model')
                torch.save(condition_generator.state_dict(), 'condition_generator.pth')
                torch.save(discriminator.state_dict(), 'discriminator.pth')


                
                



def main():
        # get the command line arguments
        opt = get_options()
        # create the dataset
        dataset = TryOnDataset(root=opt.dataset, mode='train',
                                   data_list=opt.train_list, transform=transform)
        # create the dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=2)        
        # create the condtion generator model
        condition_generator = ConditionGenerator(pose_channels= 16, cloth_channels = 4, output_channels = 13)
        # print the number of parameters in the model
        print('The number of parameters in the condition generator is: ', sum(p.numel() for p in condition_generator.parameters() if p.requires_grad))
        # create the discriminator model
        discriminator = EncapsulatedDiscriminator(input_channels = 4 + 16 + 13)
        # print the number of parameters in the model
        print('The number of parameters in the discriminator is: ', sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
        # we start the training process
        print('Start training the condition generator')
        train_condition(dataloader.data_loader, condition_generator, discriminator, opt.epochs)


if __name__ == '__main__':
    main()