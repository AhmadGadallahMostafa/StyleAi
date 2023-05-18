import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import pandas as pd

CATEGORIES_TO_REMOVE = ['Flip Flops', 'Innerwear', 'Shoe Accessories', 'Fragrance', 'Lips', 
                        'Saree', 'Nails', 'Loungewear and Nightwear', 'Wallets', 'Apparel Set', 
                        'Skin Care', 'Makeup', 'Free Gifts', 'Skin', 'Beauty Accessories', 
                        'Water Bottle', 'Eyes', 'Bath and Body', 'Sports Accessories', 'Cufflinks', 
                        'Sports Equipment', 'Stoles', 'Hair', 'Perfumes', 'Home Furnishing', 
                        'Umbrellas', 'Wristbands', 'Vouchers', 'Socks', 'Belts', 'Sandal', 'Headwear', 'Accessories',
                        'Mufflers'
                        ]

ARTICLES_TO_REMOVE = ['Kurtas', 'Water Bottle', 'Key chain', 'Lehenga Choli', 'Hair Accessory', 'Tablet Sleeve', 'Salwar',
                        'Nehru Jackets', 'Churidar', 'Patiala', 'Salwar and Dupatta', 'Accessory Gift Set', 'Mobile Pouch', 
                        'Kurtis', 'Travel Accessory', 'Bangle', 'Trolley Bag', 'Tunics', 'Mufflers', 'Dupatta', 
                        'Capris', 'Jewellery Set', 'Jewellery Set', 'Belts', 'Sandals', 'Wallets', 'Tracksuits', 'Waistcoat']

IMAGES_PATH = 'Classification\dataset\seg_images'
OUTPUT_MODELS_PATH = 'Classification\outputs\models'

def clean_data(data):
    # remove rows from data which do not have a valid image
    indices_to_remove = []
    print('[INFO]: Checking if all images are present')
    for index, image_id in tqdm(data.iterrows()):
        if not os.path.exists(os.path.join(IMAGES_PATH, str(image_id['id']) + '.jpg')):
            indices_to_remove.append(index)
    print(f"[INFO]: Dropping indices: {indices_to_remove}")
    data.drop(data.index[indices_to_remove], inplace=True)
    # remove columns season, year, productDisplayName
    data = data.drop(['season', 'year', 'productDisplayName'], axis=1)
    # remove rows with the CATEGORIES_TO_REMOVE values in the column 'subCategory'
    data = data[~data['subCategory'].isin(CATEGORIES_TO_REMOVE)]
    # remove rows with usage as nan
    data = data[~data['usage'].isna()]
    # remove rows with baseColour as nan
    data = data[~data['baseColour'].isna()]
    return data

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
