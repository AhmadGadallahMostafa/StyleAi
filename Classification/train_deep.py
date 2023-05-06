import pandas as pd
import torch
import torch.optim as optim

from utils.dataset import train_valid_split, FashionDataset
from torch.utils.data import DataLoader
from models.resnet_mod import MultiHeadResNet
from tqdm import tqdm
# import cross_entropy_loss
from torch.nn import CrossEntropyLoss
from utils.utils import save_loss_plot, save_model

device = torch.device("cuda")
# New model code
# model = MultiHeadResNet(pre_trained = True, requires_grad = False)
# model.to(device)
# Load model code
model = MultiHeadResNet(pre_trained=False, requires_grad=True).to(device)
checkpoint = torch.load('Classification\outputs\models\model_resnet_after_unfreeze.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
# learning parameters
batch_size = 16
learning_rate = 0.00001
epochs = 10
criteria = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load data
train = pd.read_csv('Classification\DatasetPrep\DeepFashion\\train_cleaned.csv')
valid = pd.read_csv('Classification\DatasetPrep\DeepFashion\\val_cleaned.csv')
train_dataset = FashionDataset(train, is_train = True)
valid_dataset = FashionDataset(valid, is_train = False)

# create data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# train model
def train(model, dataloader, optimizer, loss_fn, dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        category = data['category'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (category)
        # calculate the loss using cross entropy
        loss = CrossEntropyLoss()(outputs, targets)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss

# validation function
def validate(model, dataloader, loss_fn, dataset, device):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        # image = data['image'].to(device)
        # gender = data['gender'].to(device)
        # sub = data['sub'].to(device)
        # article = data['article'].to(device)
        # color = data['color'].to(device)
        # usage = data['usage'].to(device)

        image = data['image'].to(device)
        category = data['category'].to(device)
        
        outputs = model(image)
        targets = (category)
        loss = CrossEntropyLoss()(outputs, targets)
        val_running_loss += loss.item()
        
    val_loss = val_running_loss / counter
    return val_loss

# start the training
train_loss, val_loss = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, CrossEntropyLoss, train_dataset, device
    )
    val_epoch_loss = validate(
        model, valid_loader, CrossEntropyLoss, valid_dataset, device
    )
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    save_model(epochs, model, optimizer, criteria, name = 'model_resnet_after_unfreeze_2.pth')
# save the model to disk
# save_model(epochs, model, optimizer, criteria, name = 'model_1.pth')
# save the training and validation loss plot to disk
save_loss_plot(train_loss, val_loss)