# Attacks-Zhiyuan

PyTorch implementations of `Badnets: Identifying vulnerabilities in the machine learning model supply chain` on MNIST and CIFAR10.

## Setup

```
cd BadNet
conda activate attacks
```

## To Run

### Download Dataset

```
python data_download.py
```


### Evaluation

```
python main.py --dataset mnist --attack_type single --only_eval true
python main.py --dataset mnist --attack_type all --only_eval true
python main.py --dataset cifar --attack_type single --only_eval true
python main.py --dataset cifar --attack_type all --only_eval true
```

### Training

```
python main.py --dataset mnist --epoch 15 --attack_type single
python main.py --dataset mnist --epoch 15 --attack_type all
python main.py --dataset cifar --epoch 15 --attack_type single
python main.py --dataset cifar --attack_type all
```

### Help
```
usage: main.py [-h] [--dataset DATASET] [--proportion PROPORTION]
               [--trigger_label TRIGGER_LABEL] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--attack_type ATTACK_TYPE]
               [--only_eval ONLY_EVAL]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     "cifar" or "mnist".
  --proportion PROPORTION
                        proportion of poisoned training data.
  --trigger_label TRIGGER_LABEL
                        target label of poisoned training, for single attack.
  --batch_size BATCH_SIZE
                        batch size for training.
  --epochs EPOCHS       number of epochs.
  --attack_type ATTACK_TYPE
                        "single" or "all".
  --only_eval ONLY_EVAL
                        set to true to evaluate trained loaded models
```

## Structure

```
.
├── models/   # saved models.
├── dataset/       # saved datasets, mnist and cifar10.
├── README.md
├── main.py   # main entry file
├── util.py   # utilitiy functions for train/eval/data loading
├── poisoned_dataset.py   # dataset class to add trigger pattern
├── data_download.py   # download mnist and cifar10 datasets
├── badnet_model.py   # BadNet model class
└── requirements.txt
```


## Reference
- [badnets-pytorch from verazuo](https://github.com/verazuo/badnets-pytorch)
- [badnets from GeorgeTzannetos](https://github.com/GeorgeTzannetos/badnets)