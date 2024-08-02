# ML-Leaks

PyTorch implementations of `ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models` on MNIST and CIFAR10.

## To Run
```
cd ML-Leaks
conda activate attacks
python download_data.py
```
MNIST
```
python main.py --dataset mnist --epoch 30
```
CIFAR10
```
python main.py --dataset cifar --epoch 40
```
### Example

```
Testing Accuracy: 0.6453
              precision    recall  f1-score   support

           0       0.80      0.39      0.52     12500
           1       0.60      0.90      0.72     12500

    accuracy                           0.65     25000
   macro avg       0.70      0.65      0.62     25000
weighted avg       0.70      0.65      0.62     25000

Attacker: Acc: 64.5320 Prec: 59.6269 Recall: 90.0080
```

## Structure

```
.
├── README.md
├── main.py   # main entry file
├── target_utils.py   # train/eval for target and shadow CNN
├── attacker_utils.py   # train/eval for attacker MLP
├── download_data.py   # download mnist and cifar10 datasets
├── models.py   # CNN and Attacker model
├── custom_dataloader.py # split dataset to 4 subsets
```

## Reference
- [from GeorgeTzannetos](https://github.com/GeorgeTzannetos/ml-leaks-pytorch)
- [from JiePKU](https://github.com/JiePKU/ML-Leaks)