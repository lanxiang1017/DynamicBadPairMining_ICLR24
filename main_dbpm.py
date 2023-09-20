import os
import yaml
import torch
import argparse
import numpy as np

from dataset import get_dataset
from models.dbpm_model import simpleModel
from utils import setup_seed, save_code, get_writer

if __name__ == "__main__":

    home_dir = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='ECG', type=str,
                        help='Dataset of choice')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cpu or cuda')
    parser.add_argument('--description', default='', type=str,
                        help='description of experiments')    
    parser.add_argument('--cutnp', default=2, type=int,
                        help='threshold to identify np')   
    parser.add_argument('--cutfp', default=2, type=int,
                        help='threshold to identify fp') 
    parser.add_argument('--nclass', default=44, type=int,
                        help='number of class for task') 
    
    args = parser.parse_args()

    if args.dataset == 'ECG':
        config_dir = "config/config_ecg.yaml"
        config = yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader)

    config["correction"]["cut_np"] = args.cutnp
    config["correction"]["cut_fp"] = args.cutfp
    config["classifier"]["prediction_size"] = args.nclass

    print("set cut np to ", config["correction"]["cut_np"])
    print("set cut fp to ", config["correction"]["cut_fp"])
    print("number of categories: ", config["classifier"]["prediction_size"])

    base_dir = os.path.join("logs",config["dataset"]+'/'+args.description)
    save_code(base_dir, files_to_same=[config_dir, "models/dbpm_model.py", "models/loss.py",
                                                   "models/resnet.py", "utils.py", "main_dbpm.py", "dataset.py", "run.sh"])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using gpu: {device}{args.device}")

    train_dataset = get_dataset(config, which='train')
    test_dataset = get_dataset(config, which='test')

    auc_list = []
    for i in config['seeds']:
        setup_seed(i)

        print('==================================')
        print('Seed: ', i)
        print('==================================')

        writer = get_writer(i, config, args.description)

        model = simpleModel(i, device, config, writer).to(device)
        
        print('\n Phase: minimizing contrastive loss' )
        linear_results = model.fit(
                                    n_epoch=config['trainer']['max_epochs'],
                                    batch_size=config['trainer']['batch_size'],
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset
                                    )
        
        auc_list.append(linear_results[0])

        del(model)
        torch.cuda.empty_cache()

    with open(base_dir+"/"+"test_result.txt","w") as f:
        f.writelines("AUROC Mean: %.3f, AUROC STD: %.3f \n" % (np.mean(auc_list), np.std(auc_list)))


