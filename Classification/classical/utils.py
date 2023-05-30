import numpy as np

def load_data():
    # loading fashion mnist datase from dataset folder
    train_images = np.load('dataset/train_images.npy')
