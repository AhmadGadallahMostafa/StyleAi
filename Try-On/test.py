from ConditionGeneratorNetwork import ConditionGenerator, EncapsulatedDiscriminator, Encoder
from Dataset import TryOnDataset, DataLoader
from train_condition_generator import get_options
import torch 
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

# Create a ConditionGenerator object
cg = ConditionGenerator(16, 4, 13).cuda()
d = EncapsulatedDiscriminator(4 + 16 + 13).cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create a Dataset object
opt = get_options()
dataset = TryOnDataset(root=opt.dataset, mode='train',
                                   data_list=opt.train_list, transform=transform)
# Create a DataLoader object
dt = DataLoader(dataset, shuffle=True, batch_size=1)




# load the weights
cg.load_state_dict(torch.load('condition_generator.pth'))
d.load_state_dict(torch.load('discriminator.pth'))

dic = dt.data_loader.__iter__().__next__()
cloth = dic['cloth'].cuda()
cloth_mask = dic['cloth_mask']
cloth_mask = torch.FloatTensor((cloth_mask.numpy() > 0.5).astype(np.float32)).cuda()
# Pose Input
dense_pose = dic['dense_pose'].cuda()
parse_agnostic = dic['parse_agnostic'].cuda()
# original image
image = dic['image'].cuda()
# plot the original image
image_np = image.detach().cpu().numpy()
plt.imshow(image_np[0].transpose(1, 2, 0))
plt.show()
# plot orginal cloth
cloth_np = cloth.detach().cpu().numpy()
plt.imshow(cloth_np[0].transpose(1, 2, 0))
plt.show()



fake_map, warped_cloth, warped_cloth_mask, flow_list = cg(torch.cat((cloth, cloth_mask), dim = 1), torch.cat((parse_agnostic, dense_pose), dim = 1))

# plot the warped cloth
warped_cloth_np = warped_cloth.detach().cpu().numpy()
plt.imshow(warped_cloth_np[0].transpose(1, 2, 0))
plt.show()





