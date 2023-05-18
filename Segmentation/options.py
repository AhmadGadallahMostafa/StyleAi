import os.path as osp
import yaml

class parser(object):
    def __init__(self):
        self.IMAGE_FOLDER = "Segmentation\dataset\\train"
        self.CSV_PATH = "Segmentation\dataset\\train.csv"
        self.IS_TRAIN = True

        self.WIDTH = 768
        self.HEIGHT = 768

        self.BATCH_SIZE = 4
        self.LEARNING_RATE = 0.0001
        self.NUM_THREADS = 0
        self.ITERATIONS = 100000

        self.SAVE_MODEL_INTERVAL = 1000
        self.PRINT_FREQ = 10

        self.SAVE_PATH = osp.join("Segmentation/results/segmentation_train")

        self.CONTINUE_TRAIN = True