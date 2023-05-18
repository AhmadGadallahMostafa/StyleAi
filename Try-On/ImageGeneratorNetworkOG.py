import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import enum
import functools
import os
from torch.nn.utils import spectral_norm


# in this file we define the network architecture of the image generator 
# This network takes 4 inputs 
# 1. the fake segmentation map generated bt the condition generator
# 2. Densepose map 
# 3. Parse Agnostic image
# 4. warped cloth image
# This network will consists of  series of SPADE rensent blocks 


# first we define the SPADE normalization layer
class SPADENormalization(nn.Module):
    def __init__(self, noise_channels, segmentation_map_channels):
        super(SPADENormalization, self).__init__()
        # we will use instance normalization to normalize the input
        self.instance_norm = nn.InstanceNorm2d(noise_channels, affine=False)
        # first convolution layer to generate the gamma and beta parameters
        self.conv = nn.Conv2d(segmentation_map_channels, 128, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        # second convolution layer to generate gamma
        self.gamma = nn.Conv2d(128, noise_channels, kernel_size = 3, padding = 1)
        # second convolution layer to generate beta
        self.beta = nn.Conv2d(128, noise_channels, kernel_size = 3, padding = 1)
        # noise optimization layer
        self.noise_optimization = nn.Parameter(torch.zeros(noise_channels))

    def forward(self, x, segmentation_map):
        # sample noise from x 
        b, c, h, w = x.size()
        noise = (torch.randn(b, w, h, 1).cuda() * self.noise_optimization).transpose(1, 3)
        conv_result_one = self.conv(segmentation_map)
        conv_result_one = self.relu(conv_result_one)
        gamma = self.gamma(conv_result_one)
        beta = self.beta(conv_result_one)
        # normalize the input with the noise 
        normalized_input = self.instance_norm(x + noise)
        # apply the gamma and beta parameters to the normalized input
        output = normalized_input * (gamma + 1) + beta
        return output
    

# next we define the SPADE rensent block which will be used in the image generator network
class SPADEResnetBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPADEResnetBlock, self).__init__()
        # we will use spectral normalization in the convolution layers which is proven to work well for gans 
        self.conv1 = spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1))
        self.conv2 = spectral_norm(nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1))
        self.norm1 = SPADENormalization(input_channels, 7)
        self.norm2 = SPADENormalization(output_channels, 7)
        if input_channels != output_channels:
            self.conv_shortcut = spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size = 1, bias=False))
            self.norm_shortcut = SPADENormalization(input_channels, 7)
        self.LRelu = nn.LeakyReLU(0.2)
   


    def forward(self, x, segmentation_map):
        # first we define the shortcut connection
        segmentation_map = F.interpolate(segmentation_map, size=x.size()[2:], mode='nearest')
        shortcut = x
        if hasattr(self, 'conv_shortcut'):
            shortcut = self.norm_shortcut(x, segmentation_map)
            shortcut = self.conv_shortcut(shortcut)
        # first convolution layer
        conv1 = self.norm1(x, segmentation_map)
        conv1 = self.LRelu(conv1)
        conv1 = self.conv1(conv1)
        #####
        conv2 = self.norm2(conv1, segmentation_map)
        conv2 = self.LRelu(conv2)
        conv2 = self.conv2(conv2)
        # add the input to the output of the second convolution layer
        output = shortcut + conv2
        return output


# next we define the image generator network
class ImageGeneratorNetwork(nn.Module):
    def __init__(self, input_channels):
        super(ImageGeneratorNetwork, self).__init__()
        # we define the resnet blocks that form our network 
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.SpadeResnetBlock1 = SPADEResnetBlock(1024, 1024)
        self.SpadeResnetBlock2 = SPADEResnetBlock(1024 + 16, 1024)
        self.SpadeResnetBlock3 = SPADEResnetBlock(1024 + 16, 1024)
        self.SpadeResnetBlock4 = SPADEResnetBlock(1024 + 16, 512)
        self.SpadeResnetBlock5 = SPADEResnetBlock(512 + 16, 256)
        self.SpadeResnetBlock6 = SPADEResnetBlock(256 + 16, 128)
        self.SpadeResnetBlock7 = SPADEResnetBlock(128 + 16, 64)
        self.SpadeResnetBlock8 = SPADEResnetBlock(64 + 16, 32)
        # we define the convolution layers that form our network    
        self.conv1 = nn.Conv2d(input_channels, 1024, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        #self.conv_array = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        # last convolution layer to generate the output image wwith 3 channels
        self.conv9 = nn.Conv2d(32 , 3, kernel_size = 3, padding = 1)
        # last activation layer to generate the output image
        self.LRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()


    def forward(self, x, segmentation_map):
        feature_pyramid = []
        for i in range(8):
            features = F.interpolate(x, size = (4 * 2**i, 3 * 2**i), mode = 'nearest')
            feature_pyramid.append(features)
        
        # first convolution layer
        x = self.SpadeResnetBlock1(self.conv1(feature_pyramid[0]), segmentation_map)
        # first upsampling layer
        x = self.up_sample(x)
        # second convolution layer
        x = self.SpadeResnetBlock2(torch.cat((x, self.conv2(feature_pyramid[1])), 1), segmentation_map) 
        # second upsampling layer
        x = self.up_sample(x)
        # third convolution layer
        x = self.SpadeResnetBlock3(torch.cat((x, self.conv3(feature_pyramid[2])), 1), segmentation_map)
        # third upsampling layer
        x = self.up_sample(x)
        # fourth convolution layer
        x = self.SpadeResnetBlock4(torch.cat((x, self.conv4(feature_pyramid[3])), 1), segmentation_map)
        # fourth upsampling layer
        x = self.up_sample(x)
        # fifth convolution layer
        x = self.SpadeResnetBlock5(torch.cat((x, self.conv5(feature_pyramid[4])), 1), segmentation_map)
        # fifth upsampling layer
        x = self.up_sample(x)
        # sixth convolution layer
        x = self.SpadeResnetBlock6(torch.cat((x, self.conv6(feature_pyramid[5])), 1), segmentation_map)
        # sixth upsampling layer
        x = self.up_sample(x)
        # seventh convolution layer
        x = self.SpadeResnetBlock7(torch.cat((x, self.conv7(feature_pyramid[6])), 1), segmentation_map)
        # seventh upsampling layer
        x = self.up_sample(x)
        # eighth convolution layer
        x = self.SpadeResnetBlock8(torch.cat((x, self.conv8(feature_pyramid[7])), 1), segmentation_map)
        # last convolution layer
        x = self.LRelu(x)
        x = self.conv9(x)
        # last activation layer
        x = self.tanh(x)
        return x

class DiscriminatorNetwork(nn.Module):
    def __init__(self, input_channels):
        super(DiscriminatorNetwork, self).__init__()
        # first layer of the discriminator
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size = 4, stride = 2, padding = 2)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        # second layer of the discriminator
        # spectral normalization is used to stabilize the training process and make the only learnable paramter the lipschitz constant
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 2, bias = False))
        self.n2 = nn.InstanceNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        # third layer of the discriminator
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 2, bias = False))
        self.n3 = nn.InstanceNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        # fourth layer of the discriminator
        self.conv4 = nn.Conv2d(256, 1, kernel_size = 4, stride = 1, padding = 2)

        

    def forward(self, x):
        # first layer of the discriminator
        x = self.leaky_relu1(self.conv1(x))
        # second layer of the discriminator
        x = self.leaky_relu2(self.n2(self.conv2(x)))
        # third layer of the discriminator
        x = self.leaky_relu3(self.n3(self.conv3(x)))
        # fourth layer of the discriminator
        x = self.conv4(x)
        return x


# we now will write an encapsuation discriminator that will be contain multiple discriminators
# this will be used to train the generator to generate realistic results
class EncapsulatedDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(EncapsulatedDiscriminator, self).__init__()
        # we will now define the discriminators
        self.discriminator1 = DiscriminatorNetwork(input_channels)
        self.discriminator2 = DiscriminatorNetwork(input_channels)
        self.downsample = nn.AvgPool2d(3, stride = 2, padding = [1, 1], count_include_pad = False)

        
    def forward(self, x):
        # we will now define the forward pass
        # now we will get the output from the first discriminator
        out1 = self.discriminator1(x)
        # now we will down sample the input again
        x = self.downsample(x)
        # now we will get the output from the second discriminator
        out2 = self.discriminator2(x)
        # now we will return the output
        return out1, out2

def main():
    image_generator_network = ImageGeneratorNetwork(9)
    d = EncapsulatedDiscriminator(10)
    print('Number of parameters in the ImageGeneratorNetwork network: ', sum(p.numel() for p in image_generator_network.parameters() if p.requires_grad))
    print('Number of parameters in the EncapsulatedDiscriminator network: ', sum(p.numel() for p in d.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()









        
    


