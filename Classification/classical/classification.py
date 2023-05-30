import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST
from hog import get_features_hog

# Fashion MNIST classification using HOG features and SVM
# Load the data
train_dataset = FashionMNIST(root='data/FashionMNIST', train=True, download=True)
test_dataset = FashionMNIST(root='data/FashionMNIST', train=False, download=True)

# Get the images and labels
train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

# Get the HOG features
train_features = []
for image in train_images:
    train_features.append(get_features_hog(image, cells_per_block=(1, 1), pixels_per_cell=(4, 4)))
train_features = np.array(train_features)

test_features = []
for image in test_images:
    test_features.append(get_features_hog(image, cells_per_block=(1, 1), pixels_per_cell=(4, 4)))
test_features = np.array(test_features)

# Reshape the features to 1D
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

# Split the data into train and test
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Train the model
model = svm.SVC()
model.fit(train_features, train_labels)

# Predict the labels
train_predictions = model.predict(train_features)
val_predictions = model.predict(val_features)
test_predictions = model.predict(test_features)
# save the model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Calculate the accuracy
train_accuracy = accuracy_score(train_labels, train_predictions)
val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Print the accuracies
print('Train accuracy: ', train_accuracy)
print('Validation accuracy: ', val_accuracy)
print('Test accuracy: ', test_accuracy)