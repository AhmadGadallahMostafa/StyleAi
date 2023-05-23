import os
import torch
import matplotlib.pyplot as plt


IMAGES_PATH = 'Classification\dataset\seg_images'
OUTPUT_MODELS_PATH = 'Classification\outputs\models'

# save the model, epochs, optimizer, loss using tensorflow
def save_model(epochs, model, optimizer, criterion, name):
    torch.save({'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,}, os.path.join(OUTPUT_MODELS_PATH, name))

# load the model, epochs, optimizer, loss using tensorflow
def load_model(name):
    checkpoint = torch.load(os.path.join(OUTPUT_MODELS_PATH, name))
    return checkpoint

# plot the training and validation loss
def save_loss_plot(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Classification\outputs\plots\loss.jpg')
    plt.show()
