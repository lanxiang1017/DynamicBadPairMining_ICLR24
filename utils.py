import os
import torch
import random
import datetime
import numpy as np
from shutil import copyfile
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     np.random.seed(seed)
     random.seed(seed)

def get_scores(one_hot_labels, probs):

    one_hot_labels = np.array(one_hot_labels)
    probs  = np.array(probs)

    auroc = roc_auc_score(one_hot_labels, probs, average="macro")
    class_auc = roc_auc_score(one_hot_labels, probs, average=None)

    return auroc*100, class_auc*100

def save_model(name, model, PATH):
    torch.save({
        name: model.state_dict()
    }, PATH)

def save_code(log_dir, files_to_same):
    code_folder = os.path.join(log_dir, "code")
    if not os.path.exists(code_folder):
        os.makedirs(code_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(code_folder, os.path.basename(file)))

def _create_model_training_folder(writer):
    model_checkpoints_folder = os.path.join(writer.log_dir, "checkpoints")
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

def get_writer(seed, config, description):
    writer = SummaryWriter(log_dir="logs/"+config["dataset"]+"/"+description+"/"+"seed_"+str(seed)+"/"+\
                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    _create_model_training_folder(writer)
    return writer