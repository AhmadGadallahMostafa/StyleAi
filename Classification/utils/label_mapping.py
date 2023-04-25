import pandas as pd
import joblib
import os

LABELS_TO_USE = ['gender', 'subCategory', 'articleType', 'baseColour', 'usage']
LABELS_PATH = 'Classification\inputs\labels'

def get_label_mapping(data):
    # map the labels to integers
    for label in LABELS_TO_USE:
        list_label = data[label].unique()
        label_mapping = {label: index for index, label in enumerate(list_label)}
        print(label_mapping)
        joblib.dump(label_mapping, os.path.join(LABELS_PATH, label + '.pkl'))

# read the data
data = pd.read_csv('Classification\dataset\styles_cleaned.csv', error_bad_lines=False)
get_label_mapping(data)