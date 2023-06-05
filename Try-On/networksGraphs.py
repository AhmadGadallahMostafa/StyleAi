# Import tensor board 
from torch.utils.tensorboard import SummaryWriter
# Import torch
import torch
from ConditionGeneratorNetwork import ConditionGenerator
from ImageGeneratorNetworkOG import ImageGeneratorNetwork

condition_generator = ConditionGenerator(pose_channels= 16, cloth_channels = 4, output_channels = 13)
condition_generator.cuda()
image_generator = ImageGeneratorNetwork(input_channels = 9)
image_generator.cuda()

# # write the graph to tensorboard
# writer = SummaryWriter('runs/condition_generator')
# writer.add_graph(condition_generator, (torch.rand(1, 4, 256, 192).cuda(), torch.rand(1, 16, 256, 192).cuda()))

# write the graph to tensorboard
writer = SummaryWriter('runs/image_generator')
writer.add_graph(image_generator, (torch.rand(1, 9, 1024, 768).cuda(), torch.rand(1, 7, 1024, 768).cuda()))







