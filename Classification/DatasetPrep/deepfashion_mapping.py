# create a csv which has image name and its corresponding category
# categories are upper body, lower body, full body
# upper body: 1, lower body: 2, full body: 3

import pandas as pd
import numpy as np
import os
# read txt which has category and correesponding typ 1,2,
list_category_cloth = pd.read_csv('Classification\DatasetPrep\DeepFashion\Anno_coarse\list_category_cloth.txt', sep='\s+', skiprows=1, header=None, names=['category_name', 'category_type'])

category_dict = {}
for i in range(1, len(list_category_cloth)):
    # value is tuple of category name and category type
    category_dict[i] = (list_category_cloth['category_name'][i], list_category_cloth['category_type'][i])



# read txt which has image name and its corresponding category label as a number
list_category_img = pd.read_csv('Classification\DatasetPrep\DeepFashion\Anno_coarse\list_category_img.txt', sep='\s+', skiprows=2, header=None, names=['image_name', 'category_label'])
print(list_category_img.head())
# map category label to category name and category type from category_dict
list_category_img['category_name'] = list_category_img['category_label'].map(lambda x: category_dict[x][0])
list_category_img['category_type'] = list_category_img['category_label'].map(lambda x: category_dict[x][1])

# save the dataframe as csv
list_category_img.to_csv('Classification\DatasetPrep\DeepFashion\Anno_coarse\list_category_img.csv', index=False)


