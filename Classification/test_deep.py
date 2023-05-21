import pandas as pd
import torch
import torch.optim as optim

from utils.dataset import FashionDataset
from torch.utils.data import DataLoader
from models.resnet_mod_deep import MultiHeadResNet
from tqdm import tqdm
# import cross_entropy_loss
from torch.nn import CrossEntropyLoss
from utils.utils import save_loss_plot, save_model

device = torch.device("cuda")

# load data
test = pd.read_csv('Classification\DatasetPrep\DeepFashion\\test_cleaned.csv')
test_dataset = FashionDataset(test, is_train = False)

# create data loader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# test trained model and calculate accuracy top-1 and top-5
def test(model, dataloader, dataset, device):
    model.eval()
    counter = 0
    test_running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            
            # extract the features and labels
            image = data['image'].to(device)
            category = data['category'].to(device)
            
            outputs = model(image)
            targets = (category)
            # calculate the loss using cross entropy
            loss = CrossEntropyLoss()(outputs, targets)
            test_running_loss += loss.item()
            
            # calculate the accuracy
            _, pred = torch.max(outputs.data, 1)
            _, top5_pred = torch.topk(outputs.data, 5, dim=1)
            correct_1 += (pred == targets).sum().item()
            for i in range(len(targets)):
                if targets[i] in top5_pred[i]:
                    correct_5 += 1

    # calculate loss and accuracy
    test_loss = test_running_loss / counter
    test_accuracy_1 = correct_1 / len(dataset)
    test_accuracy_5 = correct_5 / len(dataset)
    return test_loss, test_accuracy_1, test_accuracy_5


# load model
model = MultiHeadResNet(pre_trained=False, requires_grad=True).to(device)

# load model weights
checkpoint = torch.load('Classification\outputs\models\latest_deep_fashion_b4G.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# test model
test_loss, test_accuracy_1, test_accuracy_5 = test(model, test_loader, test_dataset, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc Top-1: {test_accuracy_1:.4f}, Test Acc Top-5: {test_accuracy_5:.4f}')