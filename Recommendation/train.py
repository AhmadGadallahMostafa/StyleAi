import torch
from torch import nn
from torchsummary import summary
from torch import optim
import torch.nn.functional as F
import numpy as np
import os

from models.model import CompatabilityModel
from utils.utils import AverageMeter
from sklearn import metrics
from options import parser
from data.dataloader import get_dataset, get_train_loader, get_val_loader, get_test_loader

def train_loop(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompatabilityModel(embedding_dim=1000, need_rep=True, vocabulary_size=2757, vse = opt.vse_loss, pe = opt.pe_loss, mlp_layers = opt.mlp_layers, conv_features = opt.conv_features).to(device)

    if opt.continue_train:
        model.load_state_dict(torch.load(opt.MODEL_PATH))
        print("Loaded model from checkpoint")

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)

    train_dataset, val_dataset, test_dataset = get_dataset(opt)
    train_loader = get_train_loader(opt, train_dataset)
    val_loader = get_val_loader(opt, val_dataset)
    for epoch in range(opt.num_epochs):
        model.train()
        total_losses = AverageMeter()
        vse_losses = AverageMeter()
        clf_losses = AverageMeter()
        for itr, batch in enumerate(train_loader, 1):
            images = batch[1].to(device)
            names = batch[2]
            is_compatible = batch[6].float().to(device)

            out, vse_loss, masks_loss, features_loss = model(images, names)

            target = is_compatible.unsqueeze(1)
            curr_loss = loss_fn(out, target)

            features_loss = 5e-3 * features_loss
            masks_loss = 5e-4 * masks_loss
            total_loss = curr_loss + features_loss + masks_loss + vse_loss

            n = images.shape[0]
            total_losses.update(total_loss.item(), n)
            vse_losses.update(vse_loss.item(), n)
            clf_losses.update(curr_loss.item(), n)

            # Backpropagation
            model.zero_grad()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if itr % 10 == 0:
                print("Epoch: {}/{} | Iter: {}/{} | Total Loss: {:.4f} | VSE Loss: {:.4f} | Clf Loss: {:.4f}"
                      .format(epoch, opt.num_epochs, itr, len(train_loader), total_losses.avg, vse_losses.avg, clf_losses.avg))

            if itr % 100 == 0:
                # Save model
                itr = int(itr)
                torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, "Ckpt_{:.4f}.pth".format(itr)))
                print("Saved model at epoch {} and iteration {}".format(epoch, itr))
        
        print("Epoch: {}/{} | Total Loss: {:.4f} | VSE Loss: {:.4f} | Clf Loss: {:.4f}"
              .format(epoch+1, opt.num_epochs, total_losses.avg, vse_losses.avg, clf_losses.avg))
        
        # Validation
        model.eval()
        clf_losses = AverageMeter()
        out_list = []
        target_list = []
        with torch.no_grad():
            for _, batch in enumerate(val_loader, 1):
                images = batch[1].to(device)
                names = batch[2]
                is_compatible = batch[6].float().to(device)

                out, _, _, _ = model(images, names)

                target = is_compatible.unsqueeze(1)
                curr_loss = loss_fn(out, target)

                n = images.shape[0]
                clf_losses.update(curr_loss.item(), n)

                out_list.append(out)
                target_list.append(target)

        out_list = torch.cat(out_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        out_list = out_list.cpu().data.numpy()
        target_list = target_list.cpu().data.numpy()

        auc = metrics.roc_auc_score(target_list, out_list)
        print("Validation | Clf Loss: {:.4f} | AUC: {:.4f}".format(clf_losses.avg, auc))

        predicts = np.where(out_list > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(predicts, target_list)
        print("Validation | Accuracy: {:.4f}".format(accuracy))

        # Save model
        torch.save(model.state_dict(), os.join(opt.SAVE_PATH, "Ckpt_{:.4f}.pth".format(epoch)))
        

if __name__ == "__main__":
    opt = parser()
    train_loop(opt)