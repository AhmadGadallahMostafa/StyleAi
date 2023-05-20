from torch.utils.data import DataLoader

from data.polyvore_dataset import CategoryDataset, collate_fn

def get_dataset(opt):
    train_dataset = CategoryDataset(root_dir=opt.IMAGE_PATH, data_dir=opt.DATA_PATH, 
                                    transform=opt.transform, use_mean_img=opt.mean_img, 
                                    data_file=opt.json_train)
    val_dataset = CategoryDataset(root_dir=opt.IMAGE_PATH, data_dir=opt.DATA_PATH,
                                    transform=opt.transform, use_mean_img=opt.mean_img,
                                    data_file=opt.json_valid)
    test_dataset = CategoryDataset(root_dir=opt.IMAGE_PATH, data_dir=opt.DATA_PATH,
                                    transform=opt.transform, use_mean_img=opt.mean_img,
                                    data_file=opt.json_test)
    
    return train_dataset, val_dataset, test_dataset

def get_train_loader(opt, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size_train, shuffle=True, 
                                num_workers=opt.num_workers, collate_fn=collate_fn)
    return train_loader

def get_val_loader(opt, val_dataset):
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_valid, shuffle=True,
                                num_workers=opt.num_workers, collate_fn=collate_fn)
    return val_loader

def get_test_loader(opt, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size_test, shuffle=True,
                                num_workers=opt.num_workers, collate_fn=collate_fn)
    return test_loader