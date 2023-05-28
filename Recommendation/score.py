from models.model import CompatabilityModel
import torch
from PIL import Image
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CompatabilityModel(embedding_dim=1000, need_rep=True, vocabulary_size = 2757)
model.to(device)
model.load_state_dict(torch.load('Recommendation/trained_checkpoint\ckpt_best.pth'))
model.eval()

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


imgs = load_images().to(device)
relation, score = defect_detect(imgs, model)
# save score to txt file
with open('Interface\score.txt', 'w') as f:
    f.write(str(score))