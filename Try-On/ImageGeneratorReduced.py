import torch.nn.functional as F
import torch 
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import spectral_norm

# clear the cache
torch.cuda.empty_cache()



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
        noise = (torch.randn(x.shape[0], x.shape[2], x.shape[3], 1).cuda() * self.noise_optimization).permute(0, 3, 1, 2)
        conv_result_one = self.conv(segmentation_map)
        conv_result_one = self.relu(conv_result_one)
        gamma = self.gamma(conv_result_one)
        beta = self.beta(conv_result_one)
        # normalize the input with the noise 
        normalized_input = self.instance_norm(x + noise)
        # apply the gamma and beta parameters to the normalized input
        output = (gamma + 1) * normalized_input + beta
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
        shortcut = x
        # make the segmentation map the same size as the input
        segmentation_map = F.interpolate(segmentation_map, size = x.shape[2:], mode = 'nearest')
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


# we wil define the a reduced version of the image generator network
# we will use the same architecture as the image generator network but we will reduce the number of layers
# we will also reduce the number of channels in the layers
# we will also reduce the number of residual blocks
class RImageGeneratorNetwork(nn.Module):
    def __init__(self, input_channels):
        super(RImageGeneratorNetwork, self).__init__()
        # we define the resnet blocks that form our network 
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.SpadeResnetBlock1 = SPADEResnetBlock(128, 128)
        self.SpadeResnetBlock2 = SPADEResnetBlock(128 + 16, 128)
        #self.SpadeResnetBlock3 = SPADEResnetBlock(128 + 16, 128)
        self.SpadeResnetBlock4 = SPADEResnetBlock(128 + 16, 64)
        self.SpadeResnetBlock5 = SPADEResnetBlock(64 + 16, 32)
        self.SpadeResnetBlock6 = SPADEResnetBlock(32 + 16, 16)
        self.SpadeResnetBlock7 = SPADEResnetBlock(16 + 16, 16)
        #self.SpadeResnetBlock8 = SPADEResnetBlock(16 + 16, 16)
        # we define the convolution layers that form our network    
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        #self.conv3 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        #self.conv8 = nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 1)
        # last convolution layer to generate the output image wwith 3 channels
        self.conv9 = nn.Conv2d(16 , 3, kernel_size = 3, padding = 1)
        # last activation layer to generate the output image
        self.LRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()


    def forward(self, x, segmentation_map):
        feature_pyramid = []
        for i in range(8):
            features = F.interpolate(x, size = (8 * 2**i, 6 * 2**i), mode = 'nearest')
            feature_pyramid.append(features)
        # first convolution layer
        x = self.SpadeResnetBlock1(self.conv1(feature_pyramid[0]), segmentation_map)
        # first upsampling layer
        x = self.up_sample(x)
        # second convolution layer
        x = self.SpadeResnetBlock2(torch.cat((self.conv2(feature_pyramid[1]), x), 1), segmentation_map) 
        # second upsampling layer
        x = self.up_sample(x)
        # # third convolution layer
        # x = self.SpadeResnetBlock3(torch.cat((self.conv3(feature_pyramid[2]), x), 1), segmentation_map)
        # third upsampling layer
        x = self.up_sample(x)
        # fourth convolution layer
        x = self.SpadeResnetBlock4(torch.cat((self.conv4(feature_pyramid[3]), x), 1), segmentation_map)
        # fourth upsampling layer
        x = self.up_sample(x)
        # fifth convolution layer
        x = self.SpadeResnetBlock5(torch.cat((self.conv5(feature_pyramid[4]), x), 1), segmentation_map)
        # fifth upsampling layer
        x = self.up_sample(x)
        # sixth convolution layer
        x = self.SpadeResnetBlock6(torch.cat((self.conv6(feature_pyramid[5]), x), 1), segmentation_map)
        # sixth upsampling layer
        x = self.up_sample(x)
        # seventh convolution layer
        x = self.SpadeResnetBlock7(torch.cat((self.conv7(feature_pyramid[6]), x), 1), segmentation_map)
        # seventh upsampling layer
        x = self.up_sample(x)
        # # eighth convolution layer
        # x = self.SpadeResnetBlock8(torch.cat((self.conv8(feature_pyramid[7]), x), 1), segmentation_map)
        # last convolution layer
        x = self.LRelu(x)
        x = self.conv9(x)
        # last activation layer
        x = self.tanh(x)
        return x
    

def main():
    image_generator_network = RImageGeneratorNetwork(9)
    print('Number of parameters in the ImageGeneratorNetwork network: ', sum(p.numel() for p in image_generator_network.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()