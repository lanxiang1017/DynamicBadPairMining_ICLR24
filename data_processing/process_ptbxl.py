''' Adapted from https://github.com/helme/ecg_ptbxl_benchmarking '''

from scp_experiment_data import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *

def main():
    
    # specify raw data path and output path
    datafolder = ''
    outputfolder = ''

    models = [
        conf_fastai_xresnet1d101,
        conf_fastai_resnet1d_wang,
        conf_fastai_lstm,
        conf_fastai_lstm_bidir,
        conf_fastai_fcn_wang,
        conf_fastai_inception1d,
        conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('Diagnostic', 'diagnostic')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()

if __name__ == "__main__":
    main()
