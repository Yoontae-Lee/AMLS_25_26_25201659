# AMLS BreastMNIST Benchmark (SVM & ResNet)

This repository benchmarks two models on BreastMNIST (28x28 grayscale breast ultrasound images, binary classification):
- Model A: classical SVM 
- Model B: deep model ResNet

## Requirements

Python 3.10+ recommended.
Install dependencies:
```bash
pip install -U numpy scikit-learn torch torchvision pandas matplotlib openpyxl
```

## Dataset setup
Place the dataset at:
```
Datasets/BreastMNIST/breastmnist.npz
```
The loader supports common BreastMNIST/MedMNIST-style keys inside the .npz.

## Repository structure (expected)

```
AMLS_25_26_25201659
├── main.py
├── outputs/                    
├── Datasets/
│   └── BreastMNIST/
│       └── breastmnist.npz
└── Code/
    ├── model_a/
    │   ├── a_data.py
    │   ├── a_metrics.py
    │   └── svm_model.py
    └── model_b/
        ├── b_data.py
        ├── b_metrics.py
        └── resnet_model.py
```

## Basic Command
Options:
```bash
python main.py -h
```
Common flags:
- --model {A,B}
- --seed <int>
- --eval-test

### Model A (SVM)
Run:
```bash
python main.py --model A
```
Parameter options:
```bash
# Kernel: rbf or linear
python main.py --model A --svm-kernel rbf
python main.py --model A --svm-kernel linear

# Preprocessing: none or scale (StandardScaler)
python main.py --model A --svm-preproc none
python main.py --model A --svm-preproc scale

# C value
python main.py --model A --c 1

# Evaluate on test split 
python main.py --model A --eval-test

# Offline augmentation on the training split only
python main.py --model A --svm-augmentation --aug-repeats 1
```

### Model B (ResNet)
Run:
```bash
python main.py --model B
```
Parameter options:

```bash
# Depth (capacity): 18/34/50/101/152
python main.py --model B --b-depth 18
python main.py --model B --b-depth 101

# Epochs (budget)
python main.py --model B --b-epochs 50

# Evaluate on test split (optional)
python main.py --model B --eval-test

# On-the-fly augmentation in the training loader only (optional)
python main.py --model B --resnet-augmentation

```


## Outputs

- JSON summaries are written to:
  - outputs/result_A.json
  - outputs/result_B.json

- Model B additionally saves an Excel log and plots.


## Reproducibility
Default seed is 42. You can override it via:
```bash
python main.py --model A --seed 42
python main.py --model B --seed 42
```
