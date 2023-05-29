# Import tensorboard
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# Create a summary writer
writer = SummaryWriter('runs/Spade_Image_Generator')

# read two csv files 
df1 = pd.read_csv('s_image_discriminator_loss.csv')
df2 = pd.read_csv('s_image_generator_loss.csv')

# write the data into tensorboard
for i in range(len(df1)):
    writer.add_scalar('Discriminator Loss', df1.iloc[i,1], df1.iloc[i,0])
for i in range(len(df2)):
    writer.add_scalar('Generator Loss', df2.iloc[i,1], df2.iloc[i,0])


