import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib
import argparse
from models.resnet_mod_product_top import MultiHeadResNet_Tops
from models.resnet_mod_product_bottom import MultiHeadResNet_Bottoms
from models.resnet_mod_product_shoes import MultiHeadResNet_Shoes
import json
import glob
import shutil

LABELS_PATH_TOP = 'Classification\inputs\labels\\tops'
LABELS_PATH_BOTTOM = 'Classification\inputs\labels\\bottoms'
LABELS_PATH_SHOES = 'Classification\inputs\labels\\shoes'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_tops = MultiHeadResNet_Tops(pre_trained=False, requires_grad=False)
checkpoint = torch.load('Classification\outputs\models\model_resnet_best_top.pth')
model_tops.load_state_dict(checkpoint['model_state_dict'])
model_tops.to(device)
model_tops.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_bottoms = MultiHeadResNet_Bottoms(pre_trained=False, requires_grad=False)
checkpoint = torch.load('Classification\outputs\models\model_resnet_best_bottom.pth')
model_bottoms.load_state_dict(checkpoint['model_state_dict'])
model_bottoms.to(device)

model_shoes = MultiHeadResNet_Shoes(pre_trained=False, requires_grad=False)
checkpoint = torch.load('Classification\outputs\models\model_resnet_best_shoes.pth')
model_shoes.load_state_dict(checkpoint['model_state_dict'])
model_shoes.to(device)

# read top image
image = cv2.imread('Interface\output_segmentation/top.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model_tops(image)
# extract the five output in order (gender, sub, article, color, usage)
article, color, gender, usage = outputs
# get the index positions of the highest label score
gender_out_label = np.argmax(gender.detach().cpu())
article_out_label = np.argmax(article.detach().cpu())
color_out_label = np.argmax(color.detach().cpu())
usage_out_label = np.argmax(usage.detach().cpu())
# load the label encoder
article_label_list = joblib.load(LABELS_PATH_TOP + '/articleType.pkl')
color_label_list = joblib.load(LABELS_PATH_TOP + '/baseColour.pkl')
gender_label_list = joblib.load(LABELS_PATH_TOP + '/gender.pkl')
usage_label_list = joblib.load(LABELS_PATH_TOP + '/usage.pkl')

gender_keys = list(gender_label_list.keys())
color_keys = list(color_label_list.keys())
article_keys = list(article_label_list.keys())
usage_keys = list(usage_label_list.keys())

gender_values = list(gender_label_list.values())
color_values = list(color_label_list.values())
article_values = list(article_label_list.values())
usage_values = list(usage_label_list.values())

# create json object with the predicted labels
json_object = {
    "Article" : article_keys[article_values.index(article_out_label)],
    "Color" : color_keys[color_values.index(color_out_label)],
        "Gender" : gender_keys[gender_values.index(gender_out_label)],
        "Usage" : usage_keys[usage_values.index(usage_out_label)]
    }

with open(f"Interface\classifieroutput/top.json", "w") as outfile:
    json.dump(json_object, outfile)

# save image to Interface\classifieroutput/ folder
shutil.copy('Interface\output_segmentation/top.jpg', 'Interface\classifieroutput/top.jpg')

# read bottom image
image = cv2.imread('Interface\output_segmentation/bottom.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model_bottoms(image)
# extract labels
article, color, gender, usage = outputs
# get the index positions of the highest label score
gender_out_label = np.argmax(gender.detach().cpu())
article_out_label = np.argmax(article.detach().cpu())
color_out_label = np.argmax(color.detach().cpu())
usage_out_label = np.argmax(usage.detach().cpu())
# load the label encoder
article_label_list = joblib.load(LABELS_PATH_BOTTOM + '/articleType.pkl')
color_label_list = joblib.load(LABELS_PATH_BOTTOM + '/baseColour.pkl')
gender_label_list = joblib.load(LABELS_PATH_BOTTOM + '/gender.pkl')
usage_label_list = joblib.load(LABELS_PATH_BOTTOM + '/usage.pkl')

gender_keys = list(gender_label_list.keys())
color_keys = list(color_label_list.keys())
article_keys = list(article_label_list.keys())
usage_keys = list(usage_label_list.keys())

gender_values = list(gender_label_list.values())
color_values = list(color_label_list.values())
article_values = list(article_label_list.values())
usage_values = list(usage_label_list.values())

# create json object with the predicted labels
json_object = {
    "Article" : article_keys[article_values.index(article_out_label)],
    "Color" : color_keys[color_values.index(color_out_label)],
        "Gender" : gender_keys[gender_values.index(gender_out_label)],
        "Usage" : usage_keys[usage_values.index(usage_out_label)]
    }

with open(f"Interface\classifieroutput/bottom.json", "w") as outfile:
    json.dump(json_object, outfile)

# save image to Interface\classifieroutput/ folder
shutil.copy('Interface\output_segmentation/bottom.jpg', 'Interface\classifieroutput/bottom.jpg')

# read shoes image
image = cv2.imread('Interface\output_segmentation/full_body.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model_shoes(image)
# extract labels
article, color, gender, usage = outputs
# get the index positions of the highest label score
gender_out_label = np.argmax(gender.detach().cpu())
article_out_label = np.argmax(article.detach().cpu())
color_out_label = np.argmax(color.detach().cpu())
usage_out_label = np.argmax(usage.detach().cpu())
# load the label encoder
article_label_list = joblib.load(LABELS_PATH_BOTTOM + '/articleType.pkl')
color_label_list = joblib.load(LABELS_PATH_BOTTOM + '/baseColour.pkl')
gender_label_list = joblib.load(LABELS_PATH_BOTTOM + '/gender.pkl')
usage_label_list = joblib.load(LABELS_PATH_BOTTOM + '/usage.pkl')

gender_keys = list(gender_label_list.keys())
color_keys = list(color_label_list.keys())
article_keys = list(article_label_list.keys())
usage_keys = list(usage_label_list.keys())

gender_values = list(gender_label_list.values())
color_values = list(color_label_list.values())
article_values = list(article_label_list.values())
usage_values = list(usage_label_list.values())

# create json object with the predicted labels
json_object = {
    "Article" : "Dress",
    "Color" : color_keys[color_values.index(color_out_label)],
        "Gender" : "Women",
        "Usage" : usage_keys[usage_values.index(usage_out_label)]
    }

with open(f"Interface\classifieroutput/full_body.json", "w") as outfile:
    json.dump(json_object, outfile)

# save image to Interface\classifieroutput/ folder
shutil.copy('Interface\output_segmentation/full_body.jpg', 'Interface\classifieroutput/full_body.jpg')