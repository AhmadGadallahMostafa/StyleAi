from ConditionGeneratorNetwork import ConditionGenerator, EncapsulatedDiscriminator, Encoder
from Dataset import TryOnDataset, DataLoader
from train_condition_generator import get_options
import torch 
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
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



# Create a ConditionGenerator object
cg = ConditionGenerator(16, 4, 13).cuda()
d = EncapsulatedDiscriminator(4 + 16 + 13).cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create a Dataset object
opt = get_options()
dataset = TryOnDataset(root=opt.dataset, mode='test', data_list=opt.test_list, transform=transform)
# Create a DataLoader object
dt = DataLoader(dataset, shuffle=True, batch_size=1)

# print length of the dataset
print(len(dataset))



# load the weights
cg.load_state_dict(torch.load('condition_generator.pth'))
d.load_state_dict(torch.load('discriminator.pth'))

for i in range(len(dataset)):
    dic = dt.data_loader.__iter__().__next__()
    cloth = dic['cloth'].cuda()
    cloth_mask = dic['cloth_mask']
    cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
    # Pose Input
    dense_pose = dic['dense_pose'].cuda()
    parse_agnostic = dic['parse_agnostic'].cuda()
    # original image
    image = dic['image'].cuda()
    parse = dic['parse'].cuda()




    fake_map, warped_cloth, warped_cloth_mask, flow_list = cg(torch.cat((cloth, cloth_mask), dim = 1), torch.cat((parse_agnostic, dense_pose), dim = 1))

    # occulusion 
    tmp = torch.softmax(fake_map, dim=1)
    cloth_mask_with_body_removed = warped_cloth_mask - ((torch.cat([tmp[:, 1:3, :, :], tmp[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True)) * warped_cloth_mask
    cloth_with_body_removed = warped_cloth * cloth_mask_with_body_removed + torch.ones_like(warped_cloth) * (1 - cloth_mask_with_body_removed)


    cloth_with_body_removed_np = cloth_with_body_removed[0].detach().cpu().numpy().transpose(1, 2, 0)
    tmp = torch.ones_like(fake_map.detach())
    tmp[:, 3:4, :, :] = warped_cloth_mask
    fake_map = fake_map * tmp

    grid = make_grid([image[0].cpu(), cloth[0].cpu(), warped_cloth[0].cpu(), cloth_with_body_removed[0].cpu(), visualize_segmap(fake_map).detach().cpu(), visualize_segmap(parse).detach().cpu()], nrow=3)

    save_image(grid, os.path.join('output_condition_generator/', str(i) + '.png'))
    print(i)
