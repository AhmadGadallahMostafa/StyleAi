import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import enum
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


class Params():
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
        input_channels = [1152, 1152, 786, 576]
        output_channels = [384, 192, 96, 96]
        return input_channels[index], output_channels[index]    




# we now define the fusion block which will be used inside the condition generator
# after the two encoders
class FusionBlock(nn.Module):
    def __init_(self, index, params):
        super(FusionBlock, self).__init__()
        self.index = index
        self.params = params

        # #conv1 
        in_channels, out_channels = self.params.get_conv_cloth(self.index)
        self.conv_cloth = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)

        # bottleneck layer
        in_channels, out_channels = self.params.get_conv_pose(self.index)
        self.conv_pose = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU(inplace = True) # this is applied after conv_pose 

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
    def __init__(self, pose_channels, cloth_channels, ):
        super(ConditionGenerator, self).__init__()
        # defining the pose encoder
        self.pose_encoder = Encoder(in_channels = pose_channels, scale = SamplingType.down)
        # defining the cloth encoder
        self.cloth_encoder = Encoder(in_channels = cloth_channels, scale = SamplingType.down)
