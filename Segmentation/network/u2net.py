import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv batch norm relu
class REBNCONV(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, dilation=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
    
def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear')

class RSU7(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation = 1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation = 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        for i in range(2, 6):
            setattr(self, f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilation = 1))
            setattr(self, f'pool{i}', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.rebnconv6 = REBNCONV(mid_channels, mid_channels, dilation = 1)

        self.rebnconv7 = REBNCONV(mid_channels, mid_channels, dilation = 2)

        # decoder loop
        for i in range(6, 1, -1):
            setattr(self, f'rebnconv{i}d', REBNCONV(mid_channels*2, mid_channels, dilation = 1))

        self.rebnconv1d = REBNCONV(mid_channels*2, out_channels, dilation = 1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = getattr(self, f'rebnconv{2}')(hx)
        hx = getattr(self, f'pool{2}')(hx2)

        hx3 = getattr(self, f'rebnconv{3}')(hx)   
        hx = getattr(self, f'pool{3}')(hx3)

        hx4 = getattr(self, f'rebnconv{4}')(hx)
        hx = getattr(self, f'pool{4}')(hx4)

        hx5 = getattr(self, f'rebnconv{5}')(hx)
        hx = getattr(self, f'pool{5}')(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = getattr(self, f'rebnconv{6}d')(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = getattr(self, f'rebnconv{5}d')(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = getattr(self, f'rebnconv{4}d')(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = getattr(self, f'rebnconv{3}d')(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = getattr(self, f'rebnconv{2}d')(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
    
class RSU6(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation = 1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation = 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        for i in range(2, 5):
            setattr(self, f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilation = 1))
            setattr(self, f'pool{i}', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.rebnconv5 = REBNCONV(mid_channels, mid_channels, dilation = 1)

        self.rebnconv6 = REBNCONV(mid_channels, mid_channels, dilation = 2)

        # decoder loop
        for i in range(5, 1, -1):
            setattr(self, f'rebnconv{i}d', REBNCONV(mid_channels*2, mid_channels, dilation = 1))

        self.rebnconv1d = REBNCONV(mid_channels*2, out_channels, dilation = 1)

    def forward(self, x):
        
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = getattr(self, f'rebnconv{2}')(hx)
        hx = getattr(self, f'pool{2}')(hx2)

        hx3 = getattr(self, f'rebnconv{3}')(hx)
        hx = getattr(self, f'pool{3}')(hx3)

        hx4 = getattr(self, f'rebnconv{4}')(hx)
        hx = getattr(self, f'pool{4}')(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = getattr(self, f'rebnconv{5}d')(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = getattr(self, f'rebnconv{4}d')(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = getattr(self, f'rebnconv{3}d')(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = getattr(self, f'rebnconv{2}d')(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin 
    
class RSU5(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation = 1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation = 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        for i in range(2, 4):
            setattr(self, f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilation = 1))
            setattr(self, f'pool{i}', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation = 1)

        self.rebnconv5 = REBNCONV(mid_channels, mid_channels, dilation = 2)

        # decoder loop
        for i in range(4, 1, -1):
            setattr(self, f'rebnconv{i}d', REBNCONV(mid_channels*2, mid_channels, dilation = 1))

        self.rebnconv1d = REBNCONV(mid_channels*2, out_channels, dilation = 1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = getattr(self, f'rebnconv{2}')(hx)
        hx = getattr(self, f'pool{2}')(hx2)

        hx3 = getattr(self, f'rebnconv{3}')(hx)
        hx = getattr(self, f'pool{3}')(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = getattr(self, f'rebnconv{4}d')(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = getattr(self, f'rebnconv{3}d')(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = getattr(self, f'rebnconv{2}d')(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
    
class RSU4(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation = 1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation = 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        for i in range(2, 3):
            setattr(self, f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilation = 1))
            setattr(self, f'pool{i}', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation = 1)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation = 2)

        # decoder loop
        for i in range(3, 1, -1):
            setattr(self, f'rebnconv{i}d', REBNCONV(mid_channels*2, mid_channels, dilation = 1))

        self.rebnconv1d = REBNCONV(mid_channels*2, out_channels, dilation = 1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = getattr(self, f'rebnconv2')(hx)
        hx = getattr(self, f'pool2')(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = getattr(self, f'rebnconv3d')(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = getattr(self, f'rebnconv2d')(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
    
class RSU4F(nn.Module):
    def __init__(self, in_channels = 3, mid_channels = 12, out_channels = 3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation = 1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation = 1)

        for i in range(2, 5):
            setattr(self, f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilation = pow(2, i-1)))

        # decoder loop
        for i in range(3, 1, -1):
            setattr(self, f'rebnconv{i}d', REBNCONV(mid_channels*2, mid_channels, dilation = pow(2, i-1)))

        self.rebnconv1d = REBNCONV(mid_channels*2, out_channels, dilation = 1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = getattr(self, f'rebnconv2')(hx1)
        hx3 = getattr(self, f'rebnconv3')(hx2)

        hx4 = getattr(self, f'rebnconv4')(hx3)

        hx3d = getattr(self, f'rebnconv3d')(torch.cat((hx4, hx3), 1))
        hx2d = getattr(self, f'rebnconv2d')(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin
    
class U2NET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder stages
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0, d1, d2, d3, d4, d5, d6
