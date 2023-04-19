import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import argparse
from Dataset import TryOnDataset, DataLoader
from ImageGeneratorNetwork import ImageGeneratorNetwork, EncapsulatedDiscriminator
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
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

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
        agnostic_image = batch['agnostic_image'].cuda()

        with torch.no_grad():
            # now we call the condition generator to get the output
            fake_map, warped_cloth, warped_cloth_mask, flow_list = condition_generator(torch.cat((cloth, cloth_mask), dim=1), torch.cat((parse_agnostic, dense_pose), dim=1))

            # occulusion 
            tmp = torch.softmax(fake_map, dim=1)
            cloth_mask_with_body_removed = warped_cloth_mask - ((torch.cat([tmp[:, 1:3, :, :], tmp[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)) * warped_cloth_mask
            cloth_with_body_removed = warped_cloth * cloth_mask_with_body_removed + torch.ones_like(warped_cloth) * (1 - cloth_mask_with_body_removed)


            cloth_with_body_removed_np = cloth_with_body_removed[0].detach().cpu().numpy().transpose(1, 2, 0)
            tmp = torch.ones_like(fake_map.detach())
            tmp[:, 3:4, :, :] = warped_cloth_mask
            fake_map = fake_map * tmp

        # now wwe need to call the forward function of the image generator to get the output
        fake_image = image_generator(agnostic_image, fake_map, flow_list)









def main():
        # get the command line arguments
        opt = get_options()
        # create the dataset
        train_dataset = TryOnDataset(root=opt.dataset, mode='train',
                                   data_list=opt.train_list, transform=transform)
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