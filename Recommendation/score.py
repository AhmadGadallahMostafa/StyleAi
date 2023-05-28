from models.model import CompatabilityModel
import torch
from PIL import Image
import torchvision
import numpy as np
from data.dataloader import get_dataset
from options import parser
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CompatabilityModel(embedding_dim=1000, need_rep=True, vocabulary_size = 2757)
model.to(device)
model.load_state_dict(torch.load('Recommendation/trained_checkpoint\ckpt_best.pth'))
model.eval()
opt = parser()

IMAGE_DIR = 'Interface\outfit_evaluated'

transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

def load_images():
    # load images
    imgs = []
    # read images from IMAGE_DIR
    top_image = Image.open(IMAGE_DIR + '/top.jpg').convert('RGB')
    bottom_image = Image.open(IMAGE_DIR + '/bottom.jpg').convert('RGB')
    shoes_image = Image.open(IMAGE_DIR + '/shoes.jpg').convert('RGB')
    bag_image = Image.open('Recommendation\data\\bag.png').convert('RGB')
    accessory_image = Image.open('Recommendation\data\\accessory.png').convert('RGB')
    # transform images
    top_image = transform(top_image)
    bottom_image = transform(bottom_image)
    shoes_image = transform(shoes_image)
    bag_image = transform(bag_image)
    accessory_image = transform(accessory_image)
    # add images to list
    imgs.append(top_image)
    imgs.append(bottom_image)
    imgs.append(shoes_image)
    imgs.append(bag_image)
    imgs.append(accessory_image)
    # convert list to tensor
    imgs = torch.stack(imgs)
    # unsqueeze to add batch dimension
    imgs = imgs.unsqueeze(0)
    return imgs

def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None
    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)

    # Forward
    out, *_ = model._compute_score(img)
    one_hot = torch.FloatTensor([[-1]]).to(device)

    # Backward
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)
    
    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def vec2mat(relation, select):
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

def item_diagnosis(relation, select):
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape).byte()
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def retrieve_sub(x, select, order):
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}

    best_score = -1
    best_img_path = dict()
    _, _, test_dataset = get_dataset(opt)
    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in test_dataset.data:
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    score, *_ = model._compute_score(x)
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)


    return best_score, best_img_path

imgs = load_images().to(device)
relation, score = defect_detect(imgs, model)
# save score to txt file
with open('Interface\score.txt', 'w') as f:
    f.write(str(score))

relation = relation.squeeze().cpu().data
result, order = item_diagnosis(relation, [0, 1, 2])
best_score, best_img_path = retrieve_sub(imgs, [0, 1, 2], order)
if best_score < score:
    best_score = score
    best_img_path = {'upper': 'Interface\outfit_evaluated/top.jpg', 'bottom': 'Interface\outfit_evaluated/bottom.jpg', 'shoe': 'Interface\outfit_evaluated\shoes.jpg'}
    
# save best_img_path to json file
import json
with open('Interface/best_img_path.json', 'w') as f:
    json.dump(best_img_path, f)
    
# save best_score to txt file
with open('Interface/best_score.txt', 'w') as f:
    f.write(str(best_score))