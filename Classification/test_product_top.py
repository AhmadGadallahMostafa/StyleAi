import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib
import argparse
from models.resnet_mod_product_top import MultiHeadResNet
import json
import glob

LABELS_PATH = 'Classification\inputs\labels\\tops'

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', help='path to input image', default='test_images/1.jpg')
# args = vars(parser.parse_args())
# # define the computation device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MultiHeadResNet(pre_trained=False, requires_grad=False)
# checkpoint = torch.load('Classification\outputs\models\model_resnet_best_top.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # read an image
# image = cv2.imread(args['input'])
# # keep a copy of the original image for OpenCV functions
# orig_image = image.copy()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # apply image transforms
# image = transform(image)
# # add batch dimension
# image = image.unsqueeze(0).to(device)
# # forward pass the image through the model
# outputs = model(image)
# # extract the five output in order (gender, sub, article, color, usage)
# article, color, gender, usage = outputs
# # get the index positions of the highest label score
# gender_out_label = np.argmax(gender.detach().cpu())
# article_out_label = np.argmax(article.detach().cpu())
# color_out_label = np.argmax(color.detach().cpu())
# usage_out_label = np.argmax(usage.detach().cpu())
# # load the label encoder
# article_label_list = joblib.load(LABELS_PATH + '/articleType.pkl')
# color_label_list = joblib.load(LABELS_PATH + '/baseColour.pkl')
# gender_label_list = joblib.load(LABELS_PATH + '/gender.pkl')
# usage_label_list = joblib.load(LABELS_PATH + '/usage.pkl')

# gender_keys = list(gender_label_list.keys())
# color_keys = list(color_label_list.keys())
# article_keys = list(article_label_list.keys())
# usage_keys = list(usage_label_list.keys())

# gender_values = list(gender_label_list.values())
# color_values = list(color_label_list.values())
# article_values = list(article_label_list.values())
# usage_values = list(usage_label_list.values())

# final_labels = []
# # append by mapping the index position to the label
# final_labels.append(gender_keys[gender_values.index(gender_out_label)])
# final_labels.append(article_keys[article_values.index(article_out_label)])
# final_labels.append(color_keys[color_values.index(color_out_label)])
# final_labels.append(usage_keys[usage_values.index(usage_out_label)])


# # write the label texts on the image
# cv2.putText(
#     orig_image, final_labels[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
#     0.8, (0, 255, 0), 2, cv2.LINE_AA 
# )
# cv2.putText(
#     orig_image, final_labels[1], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
#     0.8, (0, 255, 0), 2, cv2.LINE_AA 
# )
# cv2.putText(
#     orig_image, final_labels[2], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 
#     0.8, (0, 255, 0), 2, cv2.LINE_AA 
# )
# cv2.putText(
#     orig_image, final_labels[3], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
#     0.8, (0, 255, 0), 2, cv2.LINE_AA
# )
# # visualize and save the image
# cv2.imshow('Predicted labels', orig_image)
# cv2.waitKey(0)
# save_name = args['input'].split('/')[-1]
# cv2.imwrite(f"outputs/{save_name}", orig_image)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiHeadResNet(pre_trained=False, requires_grad=False)
checkpoint = torch.load('Classification\outputs\models\model_resnet_best_top.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# read all images to be tested from Classification\inputs\shirts folder
# get the list of all images
images = glob.glob('Classification\inputs\shirts/*.jpg')
# loop through the list of images
for image in images:
    image_name = image.split('/')[-1]
    # read an image
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply image transforms
    image = transform(image)
    # add batch dimension
    image = image.unsqueeze(0).to(device)
    # forward pass the image through the model
    outputs = model(image)
    # extract the five output in order (gender, sub, article, color, usage)
    article, color, gender, usage = outputs
    # get the index positions of the highest label score
    gender_out_label = np.argmax(gender.detach().cpu())
    article_out_label = np.argmax(article.detach().cpu())
    color_out_label = np.argmax(color.detach().cpu())
    usage_out_label = np.argmax(usage.detach().cpu())
    # load the label encoder
    article_label_list = joblib.load(LABELS_PATH + '/articleType.pkl')
    color_label_list = joblib.load(LABELS_PATH + '/baseColour.pkl')
    gender_label_list = joblib.load(LABELS_PATH + '/gender.pkl')
    usage_label_list = joblib.load(LABELS_PATH + '/usage.pkl')

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
    # save the json object to a file
    save_name = image_name.split('.')[0]
    # split image on \\ and get the last element
    save_name = save_name.split('\\')[-1]
    with open(f"Classification/outputs/json/{save_name}.json", "w") as outfile:
        json.dump(json_object, outfile)