from PIL import Image, ImageDraw
import torch
import json
import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v3 as iio

def convert_image(image, new_dict, old_dict, mapping):
    for key in mapping:
        if isinstance(mapping[key], list):
            for value in mapping[key]:
                image[image == old_dict[value]] = new_dict[key]
        else:
            image[image == old_dict[mapping[key]]] = new_dict[key]
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help="output dir")
    parser.add_argument('--parse_path', type=str, help="parse dir")
    args = parser.parse_args()
    
    parse_path = args.parse_path
    output_path = args.output_path

    new_dict = {
        "Background": [0,0,0],
        "Hat": [128,0,0],
        "Hair": [255,0,0],
        "Glove": [0,85,0],
        "Sunglasses": [170,0,51],
        "Upper-clothes": [255,85,0],
        "Dress": [0,0,85],
        "Coat": [0,119,221],
        "Socks": [85,85,0],
        "Pants": [0,85,85],
        "torso-skin": [85,51,0],
        "Scarf": [52,86,128],
        "Skirt": [0,128,0],
        "Face": [0,0,255],
        "Left-arm": [51,170,221],
        "Right-arm": [0,255,255],
        "Left-leg": [85,255,170],
        "Right-leg": [170,255,85],
        "Left-shoe": [255,255,0],
        "Right-shoe": [255,170,0]
        }

    # replace the values in new dict with 0,1,2 etc
    for i, key in enumerate(new_dict):
        new_dict[key] = i


    old_dict = {
        'Background': [0, 0, 0],
        'Hat': [128, 0, 0], 
        'Hair': [0, 128, 0], 
        'Glove': [128, 128, 0], 
        'Sunglasses': [0, 0, 128], 
        'Upper-clothes': [128, 0, 128], 
        'Dress': [0, 128, 128], 
        'Coat': [128, 128, 128], 
        'Socks': [64, 0, 0], 
        'Pants': [192, 0, 0], 
        'Jumpsuits': [64, 128, 0], 
        'Scarf': [192, 128, 0], 
        'Skirt': [64, 0, 128], 
        'Face': [192, 0, 128], 
        'Left-arm': [64, 128, 128], 
        'Right-arm': [192, 128, 128], 
        'Left-leg': [0, 64, 0], 
        'Right-leg': [128, 64, 0], 
        'Left-shoe': [0, 192, 0], 
        'Right-shoe': [128, 192, 0]
        }
    # replace the values in new dict with 0,1,2 etc
    for i, key in enumerate(old_dict):
        old_dict[key] = i

    # reverse the dictionary
    #old_dict = {v: k for k, v in old_dict.items()}

    # From old to new
    mapping = {
        'Background': 'Background',
        'Hat': 'Hat',
        'Hair': 'Hair',
        'Glove': 'Glove',
        'Sunglasses': 'Sunglasses',
        'Upper-clothes': 'Upper-clothes',
        'Coat': 'Coat',
        'Socks': 'Socks',
        'Pants': ['Pants', 'Skirt', 'Jumpsuits', 'Dress'],
        'Scarf': 'Scarf',
        'Face': 'Face',
        'Left-arm': 'Left-arm',
        'Right-arm': 'Right-arm',
        'Left-leg': 'Left-leg',
        'Right-leg': 'Right-leg',
        'Left-shoe': 'Left-shoe',
        'Right-shoe': 'Right-shoe'
        }
    
    
    
    error = 0
    for im_name in tqdm(os.listdir(parse_path)):
        
        # load parse image in png format
        im_parse_old_pil = Image.open(osp.join(parse_path, im_name))
        # convert to numpy array
        im_parse_old = np.array(im_parse_old_pil)
        im_parse_old = convert_image(im_parse_old, new_dict, old_dict, mapping)
        # put new values in image
        im_parse_old_pil = Image.fromarray(im_parse_old)
        # save image
        im_parse_old_pil.save(osp.join("Try-On\PreprocessedImages/ConvertedParse", im_name), mode='L')
        # # show image
        # # plt.imshow(im_parse_old_pil)
        # # plt.show()
        # im_parse_old_pil = Image.open(osp.join("Try-On\PreprocessedImages/testparse", im_name))
        # im_parse_new = Image.open(osp.join("Try-On/data/train/image-parse-v3", im_name))    
        # # plt.imshow(im_parse_new)
        # # plt.show()
        # # convert to torch tensor
        # parse_array_old = np.array(im_parse_old)
        # parse_tensor_old = torch.from_numpy(parse_array_old)
        # parse_array_new = np.array(im_parse_new)
        # parse_tensor_new = torch.from_numpy(parse_array_new)

        # # get unique values and compare them
        # unique_old = np.unique(parse_array_old)
        # unique_new = np.unique(parse_array_new)
        # # remove value 10 from unique_new
        # unique_new = np.delete(unique_new, np.where(unique_new == 10))
        
        # if np.array_equal(unique_old, unique_new):
        #     print("They are equal")
        # else:
        #     print("They are not equal")
        #     print("Old: ", unique_old)
        #     print("New: ", unique_new)
        #     error += 1
        # print("Error: ", error)
if __name__ == '__main__':
    main()