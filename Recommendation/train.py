import torch
from torch import nn
from torchsummary import summary
from torch import optim
import torch.nn.functional as F

from models.model import CompatabilityModel
from utils.utils import AverageMeter
from options import parser

def train_loop(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompatabilityModel(embedding_dim=1000, need_rep=True, vocabulary_size=2757, vse = opt.vse_loss, pe = opt.pe_loss, mlp_layers = opt.mlp_layers, conv_features = opt.conv_features).to(device)

    if opt.continue_train:
        model.load_state_dict(torch.load(opt.MODEL_PATH))
        print("Loaded model from checkpoint")

    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = AverageMeter()
        clf_loss = AverageMeter()
        vse_loss = AverageMeter()



if __name__ == "__main__":
    opt = parser()
    train_loop(opt)