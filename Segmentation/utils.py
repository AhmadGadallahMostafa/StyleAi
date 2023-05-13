import os
import numpy as np
from collections import OrderedDict

import torch

def load_ckpt(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    print ("Loaded checkpoint from {}".format(path))
    return model

def save_ckpt(model, path):
    torch.save(model.state_dict(), path)
    print ("Saved checkpoint to {}".format(path))

def save_ckpt_handler(model, opt, iteration):
    save_ckpt(model, os.path.join(opt.SAVE_PATH, "checkpoints", "ckpt_{}.pth".format(iteration)))

def load_ckpt_handler(model, opt):
    iteration = "best"
    model = load_ckpt(model, os.path.join(opt.SAVE_PATH, "checkpoints", "ckpt_{}.pth".format(iteration)))
    return model