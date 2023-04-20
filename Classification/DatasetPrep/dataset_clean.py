import pandas as pd
import os
from tqdm import tqdm


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

IMAGES_PATH = 'fashion-dataset\images'

def clean_data(data):
    # store all subcategories in a list and write them to a file
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
    subcategories = data['subCategory'].unique()
    with open('subcategories.txt', 'w') as f:
        for subcategory in subcategories:
            f.write(f"{subcategory}\n")
    # remove rows with the ARTICLES_TO_REMOVE values in the column 'articleType'
    data = data[~data['articleType'].isin(ARTICLES_TO_REMOVE)]
    # remove rows with usage as nan
    data = data[~data['usage'].isna()]
    # remove rows with baseColour as nan
    data = data[~data['baseColour'].isna()]
    return data

# read data set from csv file
data = pd.read_csv('fashion-dataset\styles.csv', error_bad_lines=False, warn_bad_lines=False)
# clean the data
data = clean_data(data)
# print all article types
print(data['articleType'].unique())
# print count of each article type
print(data['articleType'].value_counts())
# print count of all rows
print(len(data))
# save the cleaned data to a csv file
data.to_csv('fashion-dataset\styles_cleaned.csv', index=False)