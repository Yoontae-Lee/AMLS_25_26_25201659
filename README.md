# AMLS BreastMNIST Benchmark (SVM vs ResNet)

This repository benchmarks two models on BreastMNIST (28x28 grayscale breast ultrasound images, binary classification):
- Model A: classical SVM (scikit-learn)
- Model B: deep model (ResNet, PyTorch)

Entry point: main.py

Metrics reported: Accuracy, Precision, Recall, F1, PR-AUC


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

The loader supports common BreastMNIST/MedMNIST-style keys inside the .npz (e.g., train/val/test splits).
Images are normalized to [0, 1] by the data loader.


## Repository structure (expected)

```
.
├── main.py
├── outputs/                      # result_A.json / result_B.json (auto-created)
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

Note: main.py imports Model B from Code/model_b/model_b.py if it exists; otherwise it falls back to Code/model_b/resnet_model.py.


## Usage

Show all options:

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

Useful options:

```bash
# Kernel: rbf or linear
python main.py --model A --svm-kernel rbf
python main.py --model A --svm-kernel linear

# Preprocessing: none or scale (StandardScaler)
python main.py --model A --svm-preproc none
python main.py --model A --svm-preproc scale

# C value
python main.py --model A --c 1

# Evaluate on test split (optional)
python main.py --model A --eval-test

# Offline augmentation on the training split only (optional)
python main.py --model A --svm-augmentation --aug-repeats 3
```


### Model B (ResNet)

Run:

```bash
python main.py --model B
```

Useful options:

```bash
# Depth (capacity): 18/34/50/101/152
python main.py --model B --b-depth 18
python main.py --model B --b-depth 101

# Epochs (budget)
python main.py --model B --b-epochs 50

# Training hyperparameters
python main.py --model B --b-batch-size 64 --b-lr 1e-3 --b-weight-decay 1e-4

# On-the-fly augmentation in the training loader only (optional)
python main.py --model B --resnet-augmentation

# Evaluate on test split (optional)
python main.py --model B --eval-test
```


## Outputs

- JSON summaries are written to:
  - outputs/result_A.json
  - outputs/result_B.json

- Model B additionally saves an Excel log and plots (by default under outputs_model_b/ unless configured otherwise).


## Reproducibility

Default seed is 42. You can override it via:

```bash
python main.py --model A --seed 123
python main.py --model B --seed 123
```


## Notes

- Model B uses CUDA automatically if available; otherwise it runs on CPU.
