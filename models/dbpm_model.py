import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from models.resnet import ResNet34

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils import get_scores, save_model
from models.loss import loss_ntxent
from scipy.stats import norm


class simpleModel(nn.Module):
    def __init__(self, 
                 seed,
                 device, 
                 config,
                 writer):
        super().__init__()    
       
        self.seed  = seed
        self.device = device
        self.config = config
        self.writer = writer
           
        if self.config['cnn']['resnet']:
            self.encoder = ResNet34(input_channels=self.config['cnn']['dim_input'],
                                    projection_output_dim=self.config['cnn']['dim_output'])

        self.head = nn.Sequential(nn.Linear(self.config['cnn']['dim_output'], self.config['head']['head_hidden_dim']),
                                    nn.ReLU(),
                                    nn.Linear(self.config['head']['head_hidden_dim'], self.config['head']['head_output_dim']))
        
        self.optimizer = torch.optim.Adam([{'params':self.encoder.parameters()},
                                            {'params':self.head.parameters()}], 
                                            lr=self.config['optimizer']['lr'])
        
    def fit(self,
            n_epoch,
            batch_size,
            train_dataset,
            test_dataset=None):
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=self.config['trainer']['num_workers'])
        self.batch_size=batch_size

        #* indicies of instance in dataset
        indicies = np.arange(len(train_loader.dataset))
        #* dictionary to record instance training loss (the memory module)
        memory_module = dict([(k,[]) for k in indicies])

        for epoch in tqdm(range(n_epoch)):
            epoch_loss = 0

            if epoch < self.config['trainer']['warm_epochs']:
                print('\n warmup training epoch ', epoch)
            else:
                print('\n dbpm training epoch ', epoch)

            if test_dataset != None:
                if epoch+1==self.config['trainer']['max_epochs']:
                    
                    train_dataset.train = False
                    train_repr, train_labels = self.getrepr(train_dataset)
                    test_repr, test_labels = self.getrepr(test_dataset)

                    evaluator = LinearClassifier(self.config['cnn']['dim_output'],
                                                self.config["classifier"]["prediction_size"],
                                                self.config,
                                                self.device).to(self.device)

                    eval_result = evaluator.eval_classification(train_repr, train_labels, test_repr, test_labels)
                    train_dataset.train = True

                    print('per class auroc:', np.round(eval_result[1], decimals=3))
                    with open(self.writer.log_dir+"/"+"seed{}_results.txt".format(self.seed),"w") as f:
                        f.write(f"Average auroc: {eval_result[0]: .3f} \n")
                        f.write(f"Per-class auroc: {np.round(eval_result[1], decimals=3)} \n")

                        model_dir = os.path.join(self.writer.log_dir, "checkpoints/seed{}_model.pth".format(self.seed))
                        save_model('simple_model', self, model_dir)

            if epoch >= self.config['trainer']['warm_epochs']:
                #* calculate weighted average training loss for each instance 
                epoch_weight = [(1+i)/self.config['trainer']['max_epochs'] for i in range(epoch)]
                instance_mean = {k: np.mean(np.array(v)*epoch_weight) for k, v in sorted(memory_module.items(), key=lambda item: item[1])}

                mu = np.mean(list(instance_mean.values()))
                sd = np.std(list(instance_mean.values()))

                #* global statistic
                gaussian_norm = norm(mu, sd)

                #* thresholds for noisy positive pairs and faulty positive pairs
                np_bound = mu-self.config['correction']['cut_np']*sd
                fp_bound = mu+self.config['correction']['cut_fp']*sd

                #* identify potential bad positive pairs
                np_index = [k for k in instance_mean.keys() if instance_mean[k]<=np_bound]
                fp_index = [k for k in instance_mean.keys() if instance_mean[k]>=fp_bound]

            for _, (_, x1, x2, _, index) in tqdm(enumerate(train_loader)):   
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                self.head.train()
                self.encoder.train()

                self.optimizer.zero_grad()

                z1 = self.head(self.encoder(x1))
                z2 = self.head(self.encoder(x2))

                pos_loss = loss_ntxent([z1, z2], self.device)

                #* update the memory module
                for i in range(len(index)):
                    memory_module[index[i].cpu().item()].append(pos_loss[i].cpu().item())

                #* DBPM re-weighting process
                if epoch >= self.config['trainer']['warm_epochs']:
                    l = pos_loss.detach().cpu()
                    w = gaussian_norm.pdf(l)
                    for i in range(len(index)):
                        _id = index[i].cpu().item()
                        if _id in np_index:
                            #* penalize potential np
                            pos_loss[i] *= w[i]
                        elif _id in fp_index:
                            #* penalize potential fp  
                            pos_loss[i] *= w[i]

                loss = torch.mean(pos_loss)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.detach().cpu()
                
            self.writer.add_scalar("seed {} loss".format(self.seed), epoch_loss, global_step=epoch) 
        
        return eval_result
    
    def getrepr(self, test_dataset, batch_size=1024):
        self.encoder.eval()
        test_loader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=self.config['trainer']['num_workers'])

        repr_list=[]
        label_list=[]
        with torch.no_grad():
            for x,labels in test_loader:
                x=x.to(self.device)
                z=self.encoder(x)
                repr_list.append(z.cpu())
                label_list.append(labels)
            return torch.cat(repr_list,dim=0),torch.cat(label_list,dim=0)
        

class LinearClassifier(nn.Module):
    def __init__(self, 
                 input_size, 
                 num_classes, 
                 config,
                 device):
        
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

        self.optimizer=torch.optim.Adam(self.fc.parameters(), lr=0.001)
        self.device=device
        self.config=config
        self.scaler=StandardScaler()

        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes

        if self.config['classifier']['multi_label']:
            self.criterion=nn.BCEWithLogitsLoss()

        else:
            self.criterion=nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
    def eval_classification(self,
                            train_repr,
                            train_labels,
                            test_repr,
                            test_labels):
        
        train_repr=torch.Tensor(self.scaler.fit_transform(train_repr)).to(self.device)
        test_repr=torch.Tensor(self.scaler.transform(test_repr))
        
        if not self.config['classifier']['multi_label']:
            train_labels=torch.LongTensor(train_labels).to(self.device)
        train_labels = train_labels.to(self.device)

        train_set=TensorDataset(train_repr, train_labels)
        train_loader=DataLoader(train_set,
                                batch_size=self.config['trainer']['batch_size'],
                                shuffle=True,
                                num_workers=0)
        
        self.train()
        for _ in tqdm(range(self.config['trainer']['tune_epochs'])):
            for repre, label in train_loader:
                
                self.optimizer.zero_grad()
                pred = self.forward(repre)
                loss = self.criterion(pred, label)
                loss.backward() 
                self.optimizer.step() 
            
        self.to('cpu')
        self.eval()
        with torch.no_grad():
            logits = self.forward(test_repr)

            probs = torch.sigmoid(logits).cpu().detach().numpy()

            auroc, class_auc = get_scores(test_labels, probs)

            return auroc, class_auc