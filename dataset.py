import torch
import numpy as np
from torch.utils.data import Dataset

import warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  

"""
####################################
########## Preparing Data ##########
####################################
"""
#* ================================================================== #
#*                      Random data augmentation                      #
#* ================================================================== #
def DataTransform(sample, config):
    weak_aug = Scaling(sample, config['augmentation']['scale_ratio'])
    strong_aug = Jitter(Permutation(sample, max_segments=config['augmentation']['max_seg']), config['augmentation']['jitter_ratio'])

    return weak_aug, strong_aug

def Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def Permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments) + 1

    ret = np.zeros_like(x)
    if seg_mode == "random":
        split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points)
    else:
        splits = np.array_split(orig_steps, num_segs)
    warp = np.concatenate(np.random.permutation(splits)).ravel()
    ret = x[:, warp] 

    return ret

#* ================================================================== #
#*                           Dataset                                  #
#* ================================================================== #
class MyDataset(Dataset):
    def __init__(self, config, dataset, train):
        super(MyDataset, self).__init__()
        self.train = train
        self.config = config

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        if X_train.shape.index(min(X_train.shape)) != 1:  
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).float()
        else:
            self.x_data = X_train
            self.y_data = y_train

    def __getitem__(self, index):

        y_data = self.y_data[index].float()

        if not self.train:
            return self.x_data[index].float(), y_data
        
        else:
            x_weak, x_strong = DataTransform(self.x_data[index], self.config)

            return self.x_data[index].float(), x_weak.float(), x_strong.float(), y_data, index

    def __len__(self):
        return len(self.x_data)

def get_dataset(config, which="train"):
    if which == "train":
        print("Training on ", config['dataset'])
        train_dataset = torch.load(config["path"]["train"])
        train_dataset = MyDataset(config, train_dataset, train=True)
        print("Training data size: ", len(train_dataset))

        return train_dataset

    elif which == "test":
        print("Evaluating on ", config['dataset'])
        eval_set = torch.load(config["path"]["test"])
        eval_dataset = MyDataset(config, eval_set, train=False)
        print("Evaluation data size: ", len(eval_dataset))

        return eval_dataset

