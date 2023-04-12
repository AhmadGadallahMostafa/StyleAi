import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import enum
import functools
import os


# we define an enum for the sampling type 
# generally we either up sample or down sample the imaga by a factor of 2
class SamplingType(enum.Enum):
    down = -1
    same = 0
    up = 1

# convolution fomrula for the convolutional layer
# out = (in - kernel + 2 * padding) / stride + 1
# in = N X C X H X W
# in = 8 X 3 X 256 X 192
# kernel = 3 X 3
# 256 - 3 + 2 * 1 / 2 + 1 = 128
# 192 - 3 + 2 * 1 / 2 + 1 = 96
# out = 8 X 96 X 128 X 96


# This class will inherit from nn.Module and it will be used as
# building block for encoder inside the condition generator
class ResNetBlock(nn.Module):
    # defining the constructor
    def __init__(self, in_channels, out_channels, scale):
        super(ResNetBlock, self).__init__()
        # the scale here is from the architecture of the High Resolution condition generator
        if scale == SamplingType.down:
            self.scale = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
        elif scale == SamplingType.same:
            self.scale = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        elif scale == SamplingType.up:
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
            )

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # now we define the activation function
        self.activation = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.scale(x)
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.activation(x)
        return x
    


# we need to define the encoder which will be used inside the condition generator twice     
class Encoder(nn.Module):
    def __init__(self, in_channels, scale = SamplingType.down):
        super(Encoder, self).__init__()
        self.resnet_block1 = ResNetBlock(in_channels = in_channels, out_channels = 96, scale = scale)
        self.resnet_block2 = ResNetBlock(in_channels = 96, out_channels = 192, scale = scale)
        self.resnet_block3 = ResNetBlock(in_channels = 192, out_channels = 384, scale = scale)
        self.resnet_block4 = ResNetBlock(in_channels = 384, out_channels = 384, scale = scale)
        self.resnet_block5 = ResNetBlock(in_channels = 384, out_channels = 384, scale = scale)
        
    def forward(self, x):
        feature_map = []
        x = self.resnet_block1(x)
        feature_map.append(x)
        x = self.resnet_block2(x)
        feature_map.append(x)
        x = self.resnet_block3(x)
        feature_map.append(x)
        x = self.resnet_block4(x)
        feature_map.append(x)
        x = self.resnet_block5(x)
        feature_map.append(x)
        return feature_map


class FusionBlockParams():
    def __init__(self):
        pass
    
    # conv1
    def get_conv_cloth(self, index):
        input_channels = [384, 384, 192, 96]
        output_channels = [384, 384, 384, 384] 
        return input_channels[index], output_channels[index]
    
    # bottleneck layer
    def get_conv_pose(self, index):
        input_channels = [384, 384, 192, 96]
        output_channels = [384, 384, 384, 384]
        return input_channels[index], output_channels[index]    
    
    # conf flow 
    def get_conv_warped_T2(self, index):
        input_channels = [768, 768, 768, 768]
        output_channels = [2, 2, 2, 2]
        return input_channels[index], output_channels[index]    
    
    #seg_decoder
    def get_resnet_block(self, index):
        input_channels = [1152, 1152, 768, 576]
        output_channels = [384, 192, 96, 96]
        return input_channels[index], output_channels[index]    




# we now define the fusion block which will be used inside the condition generator
# after the two encoders
class FusionBlock(nn.Module):
    def __init__(self, index, params):
        super(FusionBlock, self).__init__()
        self.index = index
        self.params = params

        # #conv1 
        in_channels, out_channels = self.params.get_conv_cloth(self.index)
        self.conv_cloth = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)

        # bottleneck layer
        in_channels, out_channels = self.params.get_conv_pose(self.index)
        self.conv_pose = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU(inplace = False) # this is applied after conv_pose 

        # # conv_flow
        in_channels, out_channels = self.params.get_conv_warped_T2(self.index)
        self.conv_warped_T2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True)

        # seg decoder
        in_channels, out_channels = self.params.get_resnet_block(self.index)
        self.res_block = ResNetBlock(in_channels = in_channels, out_channels = out_channels, scale = SamplingType.up)

    # The forward function takes in 5 inputs
    def forward(self, T1, T2, pose_encoder_i, clothes_encoder_i, flow):
        # we make the grid to be used in the grid_sample function from size of clothes_encoder_i
        n_channels, height, width = clothes_encoder_i.size(0), clothes_encoder_i.size(2), clothes_encoder_i.size(3)
        grid = make_grid(n_channels, height, width)
        # we upsample the T1
        T1 = F.interpolate(T1, scale_factor = 2, mode = 'bilinear')
        T1 = T1 + self.conv_cloth(clothes_encoder_i)
        
        # next we upsampkle the flow, keep in mind that the flow is convolved before the first fusion block 
        # the permation is done be nature of optical flows which are in the form of (h, w, 2)
        flow = F.interpolate(flow.permute(0, 3, 1, 2), scale_factor = 2, mode = 'bilinear').permute(0, 2, 3, 1)  
        # we then normalize the horizontal and vertical flow components to be in the range of [-1, 1]
        # shallow copy of the flow
        hor = 2 * flow[:, :, :, 0:1] / (width / 2 - 1)
        ver = 2 * flow[:, :, :, 1:2] / (height / 2 - 1)
        # we then concatenate the horizontal and vertical flow components
        flow_norm = torch.cat([hor, ver], 3)
        # we then sample the T1 using the grid and calc the warped T2
        warped_T1 = F.grid_sample(T1, grid + flow_norm, padding_mode='border')
        # continuing the flow pipeline
        flow = flow + self.conv_warped_T2(torch.cat([warped_T1, self.relu(self.conv_pose(T2))], 1)).permute(0, 2, 3, 1)
        # then we conc T2 with the pose encoderi and warpred T1
        T2 = self.res_block(torch.cat([T2, pose_encoder_i, warped_T1], 1))
        return T1, T2, flow

# this function is used in the warping 
def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    grid = torch.cat([grid_x, grid_y], 3).cuda()
    return grid


# Now we need to define the actual condition generator which 
# will consist of two encoders one for the clothes and one for the pose
# also it will be composed of 4 fusion blocks which will fuse the output of the encoders together
# This fusion enables the network to learn the relationship between the clothes and the pose
# which in turn will help the network to generate the correct clothes for the pose and avoid occlusion 
class ConditionGenerator(nn.Module):
    # defining the constructor
    def __init__(self, pose_channels, cloth_channels, output_channels):
        super(ConditionGenerator, self).__init__()
        # defining the pose encoder
        self.pose_encoder = Encoder(in_channels = pose_channels, scale = SamplingType.down)
        # defining the cloth encoder
        self.cloth_encoder = Encoder(in_channels = cloth_channels, scale = SamplingType.down)
        # flow pre conv
        self.flow_pre_conv = nn.Conv2d(768, 2, kernel_size = 3, stride = 1, padding = 1, bias = True)
        # now we define two res block for T2 TO pass through before fusion block 1
        self.T2_res_block1 = ResNetBlock(in_channels = 384, out_channels = 768, scale = SamplingType.same) 
        self.T2_res_block2 = ResNetBlock(in_channels = 768, out_channels = 384, scale = SamplingType.up)
        # defining the fusion blocks
        params = FusionBlockParams()
        self.fusion_block1 = FusionBlock(index = 0, params = params)
        self.fusion_block2 = FusionBlock(index = 1, params = params)
        self.fusion_block3 = FusionBlock(index = 2, params = params)
        self.fusion_block4 = FusionBlock(index = 3, params = params)
        # define the last layer 
        self.out_layer = ResNetBlock(in_channels = 96 + pose_channels + cloth_channels, out_channels = output_channels, scale = SamplingType.same)

    def forward(self, clothes, pose):
        # first we define 2 lists for the pose and clothes encoders
        clothes_encoder_list = self.cloth_encoder(clothes)
        pose_encoder_list = self.pose_encoder(pose)
        flow_list = []

        # now we define thep pre processing of the flow 
        flow = self.flow_pre_conv(torch.cat([clothes_encoder_list[4], pose_encoder_list[4]], 1)).permute(0, 2, 3, 1)
        flow_list.append(flow)
        # now we define the pre processing of the T2
        T2 = self.T2_res_block1(pose_encoder_list[4])
        T2 = self.T2_res_block2(T2)
        # T1 does not need any pre processing
        T1 = clothes_encoder_list[4]
        # now we define the first fusion block
        T1, T2, flow = self.fusion_block1(T1, T2, pose_encoder_list[3], clothes_encoder_list[3], flow)
        flow_list.append(flow)
        # now we define the second fusion block
        T1, T2, flow = self.fusion_block2(T1, T2, pose_encoder_list[2], clothes_encoder_list[2], flow)
        flow_list.append(flow)
        # now we define the third fusion block
        T1, T2, flow = self.fusion_block3(T1, T2, pose_encoder_list[1], clothes_encoder_list[1], flow)
        flow_list.append(flow)
        # now we define the fourth fusion block
        T1, T2, flow = self.fusion_block4(T1, T2, pose_encoder_list[0], clothes_encoder_list[0], flow)
        flow_list.append(flow)

        # we now do the final warping 
        # we first define the grid
        grid = make_grid(clothes.shape[0], clothes.shape[2], clothes.shape[3])
        # up sample the flow
        flow = F.interpolate(flow.permute(0, 3, 1, 2), scale_factor = 2, mode = 'bilinear').permute(0, 2, 3, 1)
        # we then normalize the horizontal and vertical flow components to be in the range of [-1, 1]
        # shallow copy of the flow
        hor = 2 * flow[:, :, :, 0:1] / (clothes.shape[3] / 2 - 1)
        ver = 2 * flow[:, :, :, 1:2] / (clothes.shape[2] / 2 - 1)
        # we then concatenate the horizontal and vertical flow components
        flow_norm = torch.cat([hor, ver], 3)
        # we then sample the T1 using the grid and calc the warped T2
        last_warped_T1 = F.grid_sample(clothes, grid + flow_norm, padding_mode='border')

        # now we define the output
        T2 = self.out_layer(torch.cat([T2, pose, last_warped_T1], 1))

        # finally we get the warped cloth and warped cloth mask
        warped_c = last_warped_T1[:, :-1, :, :]
        warped_c_mask = last_warped_T1[:, -1:, :, :]

        return T2, warped_c, warped_c_mask, flow_list



# we will now define the discriminator that will be used to train the generator
# this discriminator will be used to train the generator to generate realistic results
# dropout is used to avoid overfitting
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        # normalzaition 
        self.norm_layer = functools.partial(nn.InstanceNorm2d, affine = False)
        #first layer 
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size = 4, stride = 2, padding = 2, bias = True)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace = True)
        #second layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 2, bias = True)
        self.n2 = self.norm_layer(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace = True)
        self.dropout2 = nn.Dropout(0.5)
        #third layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 2, bias = True)
        self.n3 = self.norm_layer(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace = True)
        self.dropout3 = nn.Dropout(0.5)
        #fourth layer
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 1, padding = 2, bias = True)
        self.n4 = self.norm_layer(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace = True)
        # last layer
        self.conv5 = nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 2, bias = True)

    def forward(self, x):
        # first layer
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        # second layer
        x = self.conv2(x)
        x = self.n2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        # third layer
        x = self.conv3(x)
        x = self.n3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)
        # fourth layer
        x = self.conv4(x)
        x = self.n4(x)
        x = self.leaky_relu4(x)
        # last layer
        x = self.conv5(x)
        return x
    
# we now will write an encapsuation discriminator that will be contain multiple discriminators
# this will be used to train the generator to generate realistic results
class EncapsulatedDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(EncapsulatedDiscriminator, self).__init__()
        # we will now define the discriminators
        self.discriminator1 = Discriminator(input_channels)
        self.discriminator2 = Discriminator(input_channels)
        self.downsample = nn.AvgPool2d(3, stride = 2, padding = [1, 1], count_include_pad = False)

        
    def forward(self, x):
        # we will now define the forward pass
        # first we will down sample the input
        x = self.downsample(x)
        # now we will get the output from the first discriminator
        out1 = self.discriminator1(x)
        # now we will down sample the input again
        x = self.downsample(x)
        # now we will get the output from the second discriminator
        out2 = self.discriminator2(x)
        # now we will return the output
        return out1, out2
    


