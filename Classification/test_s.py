import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib
import argparse
from models.resnet_mod import MultiHeadResNet

LABELS_PATH = 'Classification\inputs\labels'

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to input image')
args = vars(parser.parse_args())
# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiHeadResNet(pre_trained=False, requires_grad=False)
# model = OurEfficientNet(version='b5', num_classes=1000, pretrained=True)
checkpoint = torch.load('Classification\outputs\models\latest_deep_fashion_b4G.pth')
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

# read an image
image = cv2.imread(args['input'])
# keep a copy of the original image for OpenCV functions
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# apply image transforms
image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model(image)

# extract the five output in order (gender, sub, article, color, usage)
category = outputs
# get the index positions of the highest label score
category_out_label = np.argmax(category.detach().cpu())
# get also top 5 predictions descendingly
category_top5 = np.argsort(category.detach().cpu().numpy()[0])[::-1][:5]
# load the label encoder
category_label_list = joblib.load(f'{LABELS_PATH}\category.pkl')

# get the label text by mapping the predicted label index to the actual label text
category_keys = list(category_label_list.keys())

# get the label text by mapping the predicted label index to the actual label text
category_values = list(category_label_list.values())

final_labels = []
# append by mapping the index position to the label
final_labels.append(category_keys[category_out_label])
# print the top 5 predictions using dictionary mapping
print('Top 5 predictions: ')
for i in category_top5:
    print(category_keys[i])

# write the label texts on the image
cv2.putText(
    orig_image, final_labels[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)

# visualize and save the image
cv2.imshow('Predicted labels', orig_image)
cv2.waitKey(0)
save_name = args['input'].split('/')[-1]
cv2.imwrite(f"outputs/{save_name}", orig_image)
