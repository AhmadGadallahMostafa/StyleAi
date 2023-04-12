import torch.nn.functional as F
import torch 
import torch.nn as nn
import torchvision.models as models

# clear the cache
torch.cuda.empty_cache()



# In this file we define various loss functions that will be used the condition generator 
# Also we will define the loss function for the generator and discriminator

# we first define the vgg loss 
# we will use the vgg19 network pre trained from pytorch
# we will use the vgg19 network to extract the features from the generated image and the real image
# we will then use the features to calculate the loss
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]
        ranges = [(0,2), (2,7), (7,12), (12,21), (21,30)]
        
        for i, (start, end) in enumerate(ranges):
            for j in range(start, end):
                slices[i].add_module(str(j), vgg_pretrained_features[j])
                
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        outs = []
        for slice in [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]:
            X = slice(X)
            outs.append(X)
        return outs
    
# we now define the vgg loss which will be used to calculate the loss between the generated image and the real image
class LossVGG(nn.Module):
    def __init__(self):
        super(LossVGG, self).__init__()
        # initialize the vgg network
        self.vgg = Vgg19(requires_grad=False).cuda()
        # we will use the L1 loss to calculate the loss between the generated image and the real image
        self.criterion = nn.L1Loss()
        # we will use the weights to calculate the loss between the generated image and the real image
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # we loop through the vgg layers and calculate the loss between the generated image and the real image
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    

# Next we need the GAN loss which which is the main motive of the generator to generate the image that is similar to the real image
# we will use MCE 
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super().__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_fn = nn.MSELoss() 
        self.register_buffer('real_label', tensor([target_real_label]))
        self.register_buffer('fake_label', tensor([target_fake_label]))

    def get_target_tensor(self, input, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        # we expand the tensor to the size of the input based on the target_is_real or not
        return target_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss_fn(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).cuda()
            return self.loss_fn(input[-1], target_tensor)
        

