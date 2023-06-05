import pandas as pd
import torch
import torch.optim as optim

from utils.dataset import train_valid_split, FashionDataset_Product
from torch.utils.data import DataLoader
from models.resnet_mod_product_top import MultiHeadResNet_Tops
from tqdm import tqdm
from utils.loss_function import loss_fn
from utils.utils import save_loss_plot, save_model

device = torch.device("cuda")

# New model code
# model = MultiHeadResNet(pre_trained = True, requires_grad = False)
# model.to(device)

# Load model code
model = MultiHeadResNet_Tops(pre_trained=False, requires_grad=True).to(device)
checkpoint = torch.load('Classification\outputs\models\model_resnet_best_top.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# learning parameters
batch_size = 16
learning_rate = 0.0005
epochs = 10
criteria = loss_fn
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load data
data = pd.read_csv('Classification\DatasetPrep\\fashion-dataset\styles_cleaned_top.csv')
train, valid = train_valid_split(data)

train_dataset = FashionDataset_Product(train, is_train = True)
valid_dataset = FashionDataset_Product(valid, is_train = False)

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
        gender = data['gender'].to(device)
        article = data['article'].to(device)
        color = data['color'].to(device)
        usage = data['usage'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (article, color, gender, usage)
        loss = loss_fn(outputs, targets)
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
    correct_1_article = 0
    correct_1_color = 0
    correct_1_gender = 0
    correct_1_usage = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        gender = data['gender'].to(device)
        article = data['article'].to(device)
        color = data['color'].to(device)
        usage = data['usage'].to(device)
        
        outputs = model(image)
        targets = (article, color, gender, usage)
        loss = loss_fn(outputs, targets)
        val_running_loss += loss.item()

        # get accuracy of predicting article type, colour, gender and usage each
        _, pred_article = torch.max(outputs[0].data, 1)
        _, pred_color = torch.max(outputs[1].data, 1)
        _, pred_gender = torch.max(outputs[2].data, 1)
        _, pred_usage = torch.max(outputs[3].data, 1)
        correct_1_article += (pred_article == article).sum().item()
        correct_1_color += (pred_color == color).sum().item()
        correct_1_gender += (pred_gender == gender).sum().item()
        correct_1_usage += (pred_usage == usage).sum().item()

    # calculate loss
    val_loss = val_running_loss / counter
    # calculate accuracy
    val_top_1_article = correct_1_article / len(dataset)
    val_top_1_color = correct_1_color / len(dataset)
    val_top_1_gender = correct_1_gender / len(dataset)
    val_top_1_usage = correct_1_usage / len(dataset)
    print(f"Validation Accuracy Top-1 Article: {val_top_1_article:.4f}")
    print(f"Validation Accuracy Top-1 Color: {val_top_1_color:.4f}")
    print(f"Validation Accuracy Top-1 Gender: {val_top_1_gender:.4f}")
    print(f"Validation Accuracy Top-1 Usage: {val_top_1_usage:.4f}")

    return val_loss

# start the training
train_loss, val_loss = [], []
curr_save = 0
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, loss_fn, train_dataset, device
    )
    val_epoch_loss = validate(
        model, valid_loader, loss_fn, valid_dataset, device
    )
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Validation Loss: {val_epoch_loss:.4f}")

    save_model(epochs, model, optimizer, criteria, name = 'model_resnet_top_' + str(curr_save) + '.pth')
    curr_save += 1

save_loss_plot(train_loss, val_loss)