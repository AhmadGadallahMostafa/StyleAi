# Compatability Model
import itertools
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.utils import rnn as rnn_utils
import numpy as np
# import resnet50
from resnet50_rep import ResNet50, ResNetBlock50
from torchsummary import summary

def normalize(x, dim=-1):
    return F.normalize(x, dim=dim)

class CompatabilityModel(nn.Module):
    def __init__(self, need_rep = False, vocabulary_size = None, embedding_dim = 1000, vse = False, pe = False, mlp_layers=2, conv_features="1234"):
        super(CompatabilityModel, self).__init__()
        self.need_rep = need_rep
        self.vse_loss = vse
        self.pe_loss = pe
        self.mlp_layers = mlp_layers
        self.conv_features = conv_features

        self.cnn = ResNet50(ResNetBlock50, image_channels=3, num_classes=1000, include_top=True, weights='imagenet', need_rep=need_rep)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_dim)
        # apply xaiver initialization and constant bias to fc layer
        nn.init.xavier_uniform_(self.cnn.fc.weight)
        nn.init.constant_(self.cnn.fc.bias, 0)
        self.num_rela = 15 * len(conv_features) # 15 relations, 4 conv features
        self.bn = nn.BatchNorm1d(self.num_rela) # batch normalization
        self.input_embedd = 15
        self.output_embedd = 256

        # prediction layers
        if self.mlp_layers > 0:

            pred_layers = []
            for i in range(self.mlp_layers - 1):
                pred_layers.append(nn.Linear(self.num_rela , self.num_rela))
                pred_layers.append(nn.ReLU())
            pred_layers.append(nn.Linear(self.num_rela, 1))

            # apply xaiver initialization and constant bias to all layers in pred_layers
            for layer in pred_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

            self.predictor = nn.Sequential(*pred_layers)
        else:
            self.predictor = nn.Linear(self.num_rela, 1)

        # sigmoid layer for compatibility prediction
        self.sigmoid = nn.Sigmoid()

        # masks for different relations
        self.masks = nn.Embedding(self.input_embedd, embedding_dim)
        setattr(self, "masks", nn.Embedding(self.input_embedd, embedding_dim))
        self.masks_l1 = nn.Embedding(self.input_embedd, self.output_embedd)
        setattr(self, "masks_l1", nn.Embedding(self.input_embedd, self.output_embedd))
        self.masks_l2 = nn.Embedding(self.input_embedd, self.output_embedd * 2)
        setattr(self, "masks_l2", nn.Embedding(self.input_embedd, self.output_embedd * 2))
        self.masks_l3 = nn.Embedding(self.input_embedd, self.output_embedd * 4)
        setattr(self, "masks_l3", nn.Embedding(self.input_embedd, self.output_embedd * 4))

        for mask in [self.masks, self.masks_l1, self.masks_l2, self.masks_l3]:
            mask.weight.data.normal_(0.9, 0.7)

        # embedding layer for word embedding
        self.sem_embedding = nn.Embedding(vocabulary_size, 1000)

        self.image_embedding = nn.Linear(2048, 1000)
        
        # global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, names):

        if self.need_rep:
            out, features, masks, rep = self._compute_score(images)
        else:
            out, features, masks = self._compute_score(images)

        if self.vse_loss:
            # compute vse loss
            vse_loss = self._vse_loss_compute(names, rep)
        else:
            vse_loss = torch.tensor(0.0)

        if self.pe_loss:
            # compute pe loss
            masks_loss, features_loss = self._pe_loss_compute(masks, features)
        else:
            masks_loss = torch.tensor(0.0)
            features_loss = torch.tensor(0.0)

        return out, vse_loss, masks_loss, features_loss

    def _vse_loss_compute(self, names, rep):
        # Normalized Semantic Embedding
        # pad names to the same length and mask the padding
        names_pad = rnn_utils.pad_sequence(names, batch_first=True).to(rep.device)
        mask = torch.gt(names_pad, 0)
        semantic_embedd = self.sem_embedding(names_pad) * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(semantic_embedd.shape[0]).float() * 0.1).to(rep.device),
            word_lengths.float(),
        )
        semantic_embedd = semantic_embedd.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semantic_embedd = normalize(semantic_embedd, dim=1)

        # Normalized Visual Embedding
        visual_embedd = normalize(self.image_embedding(rep), dim=1)

        # VSE Loss
        semantic_embedd = torch.masked_select(semantic_embedd, torch.ge(mask.sum(dim=1), 2))
        visual_embedd = torch.masked_select(visual_embedd, torch.ge(mask.sum(dim=1), 2))
        reshape_dim = [-1, 1000]
        semantic_embedd = semantic_embedd.reshape(reshape_dim)
        visual_embedd = visual_embedd.reshape(reshape_dim)
        scores = torch.matmul(semantic_embedd, visual_embedd.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semantic_embedd.shape[0] ** 2)

        return vse_loss

    def _pe_loss_compute(self, tmasks, features):
        # Type embedding loss
        tmasks_loss = tmasks.norm(1) / len(tmasks)
        features_loss = features.norm(2) / np.sqrt(
            (features.shape[0] * features.shape[1])
        )
        return tmasks_loss, features_loss

    def _compute_score(self, images, activation=True):
        batch_size = images.shape[0]
        item_num = images.shape[1]
        img_size = images.shape[4]
        
        images = torch.reshape(images, (-1, 3, img_size, img_size))
        if self.need_rep:
            features, *rep = self.cnn(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep = rep
            setattr(self, "rep_l1", rep_l1)
            setattr(self, "rep_l2", rep_l2)
            setattr(self, "rep_l3", rep_l3)
            setattr(self, "rep_l4", rep_l4)
            setattr(self, "rep", rep)
        else:
            features = self.cnn(images)

        relations = []
        features = torch.reshape(features, (batch_size, item_num, -1)) 
        masks = F.relu(self.masks.weight)
        # Comparison matrix
        if "4" in self.conv_features:
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0,1,2,3,4], 2)):
                if self.pe_loss:
                    left = normalize(masks[mi] * features[:, i:i+1, :], dim=-1)
                    right = normalize(masks[mi] * features[:, j:j+1, :], dim=-1)
                else:
                    left = normalize(features[:, i:i+1, :], dim=-1) 
                    right = normalize(features[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze()
                relations.append(rela)

        rep_list = []
        masks_list = []
        for i in range(1, len(self.conv_features)):
            rep_list.append(getattr(self, "rep_l{}".format(i)))
            masks_list.append(getattr(self, "masks_l{}".format(i)))

        for rep_itr, masks_itr in zip(rep_list, masks_list):
            rep_itr = self.avgpool(rep_itr).squeeze().reshape(batch_size, item_num, -1)
            masks_itr = F.relu(masks_itr.weight)

            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0,1,2,3,4], 2)):
                left = normalize(masks_itr[mi] * rep_itr[:, i:i+1, :], dim=-1) 
                right = normalize(masks_itr[mi] * rep_itr[:, j:j+1, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze() 
                relations.append(rela)

        if batch_size == 1:
            relations = torch.stack(relations).unsqueeze(0)
        else:
            relations = torch.stack(relations, dim=1)
        relations = self.bn(relations)

        # Predictor
        if self.mlp_layers == 0:
            out = relations.mean(dim=-1, keepdim=True)
        else:
            out = self.predictor(relations)

        if activation:
            out = self.sigmoid(out)
        if self.need_rep:
            return out, features, masks, rep
        else:
            return out, features, masks

# model = CompatabilityModel(embedding_dim=1000, need_rep=True, vocabulary_size=2757,
#                     vse=False, pe=False, mlp_layers=2, conv_features="1234").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# model.load_state_dict(torch.load("Recommendation\models\model_train_relation_vse_type_cond_scales.pth"))
# print(model)