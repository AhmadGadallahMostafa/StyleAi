from ConditionGeneratorNetwork import ConditionGenerator, EncapsulatedDiscriminator, Encoder
import torch 

# Create a ConditionGenerator object
cg = ConditionGenerator(16, 4, 13)
d = EncapsulatedDiscriminator(4 + 16 + 13)
# we want to know the number of parameters in the generator and discriminator networks using pytorch
print("Number of parameters in the generator network: ", sum(p.numel() for p in cg.parameters() if p.requires_grad))
print("Number of parameters in the discriminator network: ", sum(p.numel() for p in d.parameters() if p.requires_grad))
