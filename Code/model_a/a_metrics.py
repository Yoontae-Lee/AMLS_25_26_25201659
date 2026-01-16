from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

@dataclass(frozen=True)
class Metrics:
    acc: float
    prec: float
    rec: float
    f1: float
    pr_auc: float 

    def metrics_dict(self) -> Dict[str, float]:
        return {"acc": self.acc, "prec": self.prec, "rec": self.rec, "f1": self.f1, "pr_auc": self.pr_auc}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Metrics:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # binary task: benign vs malignant (0/1)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # PR-AUC (Average Precision)
    if y_score is None:
        pr_auc = float("nan")
    else:
        y_score = np.asarray(y_score).reshape(-1)
        pr_auc = float(average_precision_score(y_true, y_score))

    return Metrics(acc=acc, prec=prec, rec=rec, f1=f1, pr_auc=pr_auc)
