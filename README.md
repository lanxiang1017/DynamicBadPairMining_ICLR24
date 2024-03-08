# Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach (ICLR 2024)
This repository includes codes to demonstrate the integration of DBPM into SimCLR. 📃[Read the paper](https://openreview.net/pdf?id=K2c04ulKXn).

>**Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach** \
>*The Twelfth International Conference on Learning Representations (ICLR 2024)* \
>Xiang Lan, Hanshu Yan, Shenda Hong, Mengling Feng

*Last update on 08 Mar 2024*

## Dataset and preprocessing
1. We use the PTB-XL dataset for demonstration. Data can be downloaded at [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/).
2. Specify the paths for both raw data and processed data in *'data_processing/process_ptbxl.py'*.
3. `python process_ptbxl.py`

## Run task
1. Specify the path for processed data within *'config/config_ecg.yaml'*.
2. `. run.sh`

## Main dependencies
```
python==3.7.10
pytorch==1.11.0
numpy==1.20.3
scikit-learn==0.24.2
scipy==1.6.3
```

## Reference
We appreciate your citations if you find our paper related and useful to your research!

```
@inproceedings{
lan2024towards,
title={Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach},
author={Xiang Lan and Hanshu Yan and Shenda Hong and Mengling Feng},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=K2c04ulKXn}
}
```
