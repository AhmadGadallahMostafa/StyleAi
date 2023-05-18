import torch.utils.data as data
from data.aligned_dataset import AlignedDataset

def CreateDataset(opt):
    dataset = None
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
  
class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = data.DataLoader(self.dataset, batch_size=opt.BATCH_SIZE, sampler=data.RandomSampler(self.dataset), num_workers=int(opt.NUM_THREADS), pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), float("inf"))