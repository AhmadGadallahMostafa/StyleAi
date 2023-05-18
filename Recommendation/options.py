import os.path as osp
import yaml

class parser(object):
    def __init__(self):
        self.vse_loss = False
        self.pe_loss = False
        self.mlp_layers = 2
        self.con_features = "1234"
        self.MODEL_PATH = "Recommendation/trained_checkpoint/model_train_relation_vse_type_cond_scales.pth"
        self.continue_train = False