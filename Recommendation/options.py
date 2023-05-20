import os.path as osp
import yaml
import torchvision

class parser(object):
    def __init__(self):
        self.vse_loss = False
        self.pe_loss = False
        self.mlp_layers = 2
        self.img_size = 224
        self.conv_features = "1234"
        self.MODEL_PATH = "Recommendation/trained_checkpoint/model_train_relation_vse_type_cond_scales.pth"
        self.SAVE_PATH = "Recommendation/trained_checkpoint/"
        self.continue_train = True
        self.IMAGE_PATH = "Recommendation/data/images/"
        self.DATA_PATH = "Recommendation/data/"
        self.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((self.img_size, self.img_size)),
            torchvision.transforms.ToTensor(),
        ])
        self.batch_size_train = 8
        self.batch_size_valid = 8
        self.batch_size_test = 16
        self.num_workers = 0
        self.num_epochs = 50
        self.mean_img = True
        self.json_train = "train_no_dup_with_category_3more_name.json"
        self.json_valid = "valid_no_dup_with_category_3more_name.json"
        self.json_test = "test_no_dup_with_category_3more_name.json"

