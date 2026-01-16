from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import pandas as pd
from Code.model_a.a_metrics import Metrics, compute_metrics

# Global parameters
DEFAULT_SEED: int = 42
DEFAULT_DATASET_RATIO: float = 1.0

# Data augmentation
DEFAULT_USE_AUGMENTATION: bool = False
DEFAULT_AUG_FLIP: bool = True
DEFAULT_AUG_NOISE_STD: float = 0.02
DEFAULT_AUG_BRIGHTNESS_DELTA: float = 0.15

# Model depth
Depth = Literal[18, 34, 50, 101, 152]
DEFAULT_DEPTH: Depth = 18

# Learning parameter
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4

DEFAULT_OUTDIR: str = "outputs"
DEFAULT_EXCEL_NAME: str = "resnet_results.xlsx"
DEFAULT_EVAL_TEST: bool = True

def _seed_worker(worker_id: int) -> None:
    s = (DEFAULT_SEED + worker_id) % (2**32)
    np.random.seed(s)
    torch.manual_seed(s)

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _sample_by_ratio(x: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0,1], got {ratio}")
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    n = x.shape[0]
    k = max(1, int(np.floor(n * ratio)))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return x[idx], y[idx]

class _BreastDatasetTorch:
    def __init__(self,x: np.ndarray,y: np.ndarray,train: bool,
                 use_augmentation: bool,do_flip: bool,noise_std: float,brightness_delta: float):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (N,28,28), got {x.shape}")
        self.x = torch.from_numpy(x)  # (N,28,28)
        self.y = torch.from_numpy(y)  # (N,)
        self.train = bool(train)
        self.use_augmentation = bool(use_augmentation)
        self.do_flip = bool(do_flip)
        self.noise_std = float(noise_std)
        self.brightness_delta = float(brightness_delta)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        img = self.x[idx].unsqueeze(0)  # (1,28,28)
        lab = self.y[idx]

        if self.train and self.use_augmentation:
            if self.do_flip and torch.rand(()) < 0.5:
                img = torch.flip(img, dims=[2])
            if self.noise_std > 0:
                img = img + torch.randn_like(img) * self.noise_std
            if self.brightness_delta > 0:
                scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * self.brightness_delta
                img = img * scale
            img = torch.clamp(img, 0.0, 1.0)
        return img, lab

def _build_resnet(depth: int, num_classes: int = 2) -> nn.Module:
    ctor = {18: models.resnet18,34: models.resnet34,50: models.resnet50,101: models.resnet101,152: models.resnet152}.get(int(depth))
    if ctor is None:
        raise ValueError("depth should be 18/34/50/101/152.")
    m = ctor(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _evaluate_arrays(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    yt, yp, ys = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            yt.append(yb.detach().cpu().numpy())
            yp.append(pred.detach().cpu().numpy())
            ys.append(probs_pos.detach().cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(ys)

def _evaluate_loss_and_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, Metrics]:
    model.eval()
    total_loss, total_n = 0.0, 0
    yt, yp, ys = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = int(yb.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs

            probs_pos = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            yt.append(yb.detach().cpu().numpy())
            yp.append(pred.detach().cpu().numpy())
            ys.append(probs_pos.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total_n)
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    ys = np.concatenate(ys)
    return avg_loss, compute_metrics(yt, yp, ys)

def _train_one_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = int(yb.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)

def _metrics_to_str(m: Metrics) -> str:
    return f"acc={m.acc:.4f}  prec={m.prec:.4f}  rec={m.rec:.4f}  f1={m.f1:.4f}  pr_auc={m.pr_auc:.4f}"

def _save_metric_plots(rows: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    ep = df["epoch"].values
    plt.figure()
    plt.plot(ep, df["train_loss"].values, label="train")
    plt.plot(ep, df["val_loss"].values, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=200)
    plt.close()

    for mn in ["acc", "prec", "rec", "f1", "pr_auc"]:
        plt.figure()
        plt.plot(ep, df[f"train_{mn}"].values, label="train")
        plt.plot(ep, df[f"val_{mn}"].values, label="val")
        plt.xlabel("epoch"); plt.ylabel(mn); plt.title(f"{mn} vs Epoch")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{mn}.png", dpi=200)
        plt.close()

def _save_results_excel(rows: List[Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(out_path, sheet_name="results", index=False)

def run_resnet(x_train: np.ndarray,y_train: np.ndarray,x_val: np.ndarray,y_val: np.ndarray,x_test: np.ndarray,y_test: np.ndarray):
    _set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr, y_tr = _sample_by_ratio(x_train, y_train, DEFAULT_DATASET_RATIO, DEFAULT_SEED)
    ds_tr = _BreastDatasetTorch(x_tr, y_tr,train=True,use_augmentation=DEFAULT_USE_AUGMENTATION,do_flip=DEFAULT_AUG_FLIP,
                                noise_std=DEFAULT_AUG_NOISE_STD,brightness_delta=DEFAULT_AUG_BRIGHTNESS_DELTA)
    
    ds_va = _BreastDatasetTorch(x_val, y_val, train=False, use_augmentation=False, do_flip=False, noise_std=0.0, brightness_delta=0.0)
    ds_te = _BreastDatasetTorch(x_test, y_test, train=False, use_augmentation=False, do_flip=False, noise_std=0.0, brightness_delta=0.0)
    g = torch.Generator().manual_seed(DEFAULT_SEED)
    dl_tr = DataLoader(ds_tr, batch_size=DEFAULT_BATCH_SIZE, shuffle=True,  num_workers=0, generator=g, worker_init_fn=_seed_worker, pin_memory=torch.cuda.is_available())
    dl_va = DataLoader(ds_va, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=0, worker_init_fn=_seed_worker, pin_memory=torch.cuda.is_available())
    dl_te = DataLoader(ds_te, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=0, worker_init_fn=_seed_worker, pin_memory=torch.cuda.is_available())

    model = _build_resnet(int(DEFAULT_DEPTH), 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)

    rows: List[Dict[str, float]] = []

    for epoch in range(1, int(DEFAULT_EPOCHS) + 1):
        tr_loss = _train_one_epoch(model, dl_tr, device, criterion, optimizer)
        va_loss, va_m = _evaluate_loss_and_metrics(model, dl_va, device, criterion)
        _, tr_m = _evaluate_loss_and_metrics(model, dl_tr, device, criterion)

        rows.append({
            "epoch": float(epoch),
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "train_acc": float(tr_m.acc),
            "train_prec": float(tr_m.prec),
            "train_rec": float(tr_m.rec),
            "train_f1": float(tr_m.f1),
            "train_pr_auc": float(tr_m.pr_auc),
            "val_acc": float(va_m.acc),
            "val_prec": float(va_m.prec),
            "val_rec": float(va_m.rec),
            "val_f1": float(va_m.f1),
            "val_pr_auc": float(va_m.pr_auc),
        })

        print(
            f"epoch {epoch:03d} | train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}\n"
            f"  train: {_metrics_to_str(tr_m)}\n"
            f"  val  : {_metrics_to_str(va_m)}"
        )

    # final metrics
    y_true_tr, y_pred_tr, y_score_tr = _evaluate_arrays(model, dl_tr, device)
    train_m = compute_metrics(y_true_tr, y_pred_tr, y_score_tr)

    y_true_va, y_pred_va, y_score_va = _evaluate_arrays(model, dl_va, device)
    val_m = compute_metrics(y_true_va, y_pred_va, y_score_va)

    test_m: Optional[Metrics] = None
    if DEFAULT_EVAL_TEST:
        y_true_te, y_pred_te, y_score_te = _evaluate_arrays(model, dl_te, device)
        test_m = compute_metrics(y_true_te, y_pred_te, y_score_te)

    print("final metrics")
    print(f"  train: {_metrics_to_str(train_m)}")
    print(f"  val  : {_metrics_to_str(val_m)}")
    if test_m is not None:
        print(f"  test : {_metrics_to_str(test_m)}")

    out_dir = Path(DEFAULT_OUTDIR)
    out_xlsx = out_dir / DEFAULT_EXCEL_NAME

    _save_metric_plots(rows, out_dir)
    _save_results_excel(rows, out_xlsx)
    print(f"saved excel: {out_xlsx}")

    return model, train_m, val_m, test_m
