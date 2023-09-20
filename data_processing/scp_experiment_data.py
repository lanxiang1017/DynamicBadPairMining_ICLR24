''' Adapted from https://github.com/helme/ecg_ptbxl_benchmarking '''

from utils import utils
import os
import pickle
import torch
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from sklearn.model_selection import train_test_split

class SCP_Experiment():
    '''
        Experiment on SCP-ECG statements. All experiments based on SCP are performed and evaluated the same way.
    '''

    def __init__(self, experiment_name, task, datafolder, outputfolder, models, sampling_frequency=100, min_samples=0, train_fold=8, val_fold=9, test_fold=10, folds_type='strat'):
        self.models = models
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.folds_type = folds_type
        self.experiment_name = experiment_name
        self.outputfolder = outputfolder
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

        # create folder structure if needed
        if not os.path.exists(self.outputfolder+self.experiment_name):
            os.makedirs(self.outputfolder+self.experiment_name)
            if not os.path.exists(self.outputfolder+self.experiment_name+'/results/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/results/')
            if not os.path.exists(outputfolder+self.experiment_name+'/models/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/models/')
            if not os.path.exists(outputfolder+self.experiment_name+'/data/'):
                os.makedirs(self.outputfolder+self.experiment_name+'/data/')

    def prepare(self):
        # Load PTB-XL data
        self.data, self.raw_labels = utils.load_dataset(self.datafolder, self.sampling_frequency)

        # Preprocess label data
        self.labels = utils.compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        self.data, self.labels, self.Y, _ = utils.select_data(self.data, self.labels, self.task, self.min_samples, self.outputfolder+self.experiment_name+'/data/')
        self.input_shape = self.data[0].shape

        # Train Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.Y, test_size=0.2, random_state=0)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=0)

        # Preprocess signal data
        self.X_train, self.X_val, self.X_test = utils.preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]

        # Convert to torch tensor
        self.X_train = torch.from_numpy(self.X_train)
        self.X_val = torch.from_numpy(self.X_val)
        self.X_test = torch.from_numpy(self.X_test)

        self.X_train = self.X_train.transpose(1, 2)
        self.X_val = self.X_val.transpose(1, 2)
        self.X_test = self.X_test.transpose(1, 2)

        self.y_train = torch.from_numpy(self.y_train)
        self.y_val = torch.from_numpy(self.y_val)
        self.y_test = torch.from_numpy(self.y_test)        

        self.train_set = {'samples': self.X_train, 'labels': self.y_train} 
        self.val_set = {'samples': self.X_val, 'labels': self.y_val} 
        self.test_set = {'samples': self.X_test, 'labels': self.y_test} 

        # save train and test labels
        base_save_dir = self.outputfolder + self.experiment_name

        torch.save(self.train_set, base_save_dir + '/data/train.pt')
        torch.save(self.val_set, base_save_dir + '/data/val.pt')
        torch.save(self.test_set, base_save_dir + '/data/test.pt')

    