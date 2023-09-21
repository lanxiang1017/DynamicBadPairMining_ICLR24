# DynamicBadPairMining_ICLR24
This repository includes codes to demonstrate the integration of DBPM into SimCLR. Codes will be made publicly available.

## Dataset and preprocessing
1. Here we use the PTB-XL dataset for demonstration. Data can be downloaded at https://physionet.org/content/ptb-xl/1.0.1/.
2. Specify the paths for both raw data and processed data in 'data_processing/process_ptbxl.py'.
3. `python process_ptbxl.py`

## Run task
1. Specify the path for processed data within 'config/config_ecg.yaml'.
2. `. run.sh`

## Main dependencies
```
python==3.7.10
pytorch==1.11.0
numpy==1.20.3
scikit-learn==0.24.2
scipy==1.6.3
```
