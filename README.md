# DynamicBadPairMining_ICLR24
This repository contains codes for demonstration of integrating DBPM into SimCLR. Codes will be made publicly available.

## Dataset and preprocessing
1. Here we use the PTB-XL dataset as an example. To acquire the data, kindly follow the guidelines at [https://physionet.org/content/ptb-xl/1.0.1/].
2. Specify both the raw data path and output path for processed data in 'data_processing/process_ptbxl.py'.
3. `python process_ptbxl.py`

## Run task
1. Specify the processed data path in config/config_ecg.yaml
2. `. run.sh`

## Main dependencies
```
python==3.7.10
pytorch==1.11.0
numpy==1.20.3
scikit-learn==0.24.2
scipy==1.6.3
```
