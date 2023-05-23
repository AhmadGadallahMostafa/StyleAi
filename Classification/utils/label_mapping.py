import pandas as pd
import joblib
import os

LABELS_TO_USE = ['gender', 'articleType', 'baseColour', 'usage']
LABELS_PATH = 'Classification\inputs\labels/shoes'

def get_label_mapping(data):
    # map the labels to integers
    for label in LABELS_TO_USE:
        list_label = data[label].unique()
        label_mapping = {label: index for index, label in enumerate(list_label)}
        print(label_mapping)
        # create a directory to store the label mapping   
        joblib.dump(label_mapping, os.path.join(LABELS_PATH, label + '.pkl'))

# read the data
data = pd.read_csv('Classification\DatasetPrep\\fashion-dataset\styles_cleaned_shoes.csv', error_bad_lines=False)
get_label_mapping(data)