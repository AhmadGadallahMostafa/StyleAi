import os
import sys
import time
import argparse
import numpy as np
import yaml
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import options as opt

from network.u2net import U2NET
from utils import load_ckpt, save_ckpt, save_ckpt_handler, load_ckpt_handler
from options import parser

from data import custom_data_loader

def train_loop(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model = U2NET(in_channels=3, out_channels=4)
    if opt.CONTINUE_TRAIN:
        model = load_ckpt_handler(model, opt)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(),lr=opt.LEARNING_RATE)

    data_loader = custom_data_loader.CustomDatasetDataLoader()
    data_loader.initialize(opt)
    train_loader = data_loader.get_loader()

    dataset_size = len(data_loader)
    print("Total number of images avaliable for training: %d" % dataset_size)

    # loss
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2, 2, 2]).to(device)).to(device)

    # training 
    get_data = custom_data_loader.sample_data(train_loader)

    start_time = time.time()
    print("Entering training loop!")
    for i in range(opt.ITERATIONS):
        image, label = next(get_data)
        image = image.to(device)
        label = label.type(torch.long)
        label = label.to(device)

        # forward
        d0, d1, d2, d3, d4, d5, d6 = model(image)

        # loss
        loss0 = loss_fn(d0, label)
        loss1 = loss_fn(d1, label)
        loss2 = loss_fn(d2, label)
        loss3 = loss_fn(d3, label)
        loss4 = loss_fn(d4, label)
        loss5 = loss_fn(d5, label)
        loss6 = loss_fn(d6, label)
        del d0, d1, d2, d3, d4, d5, d6

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # printing
        if i % opt.PRINT_FREQ == 0:
            pprint.pprint(
                "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(i, time.time() - start_time, loss, loss0))
        
        # saving
        if i % opt.SAVE_MODEL_INTERVAL == 0:
            save_ckpt_handler(model, opt, i)

    print("Training done!")
    itr += 1
    save_ckpt_handler(model, opt, itr)

if __name__ == "__main__":

    opt = parser()

    os.makedirs(opt.SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(opt.SAVE_PATH, "checkpoints"), exist_ok=True)
    train_loop(opt)
