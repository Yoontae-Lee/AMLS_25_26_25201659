from __future__ import annotations
from typing import Literal, Optional, Tuple
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from Code.model_a.a_metrics import Metrics, compute_metrics
from Code.model_a.a_data import offline_augmentation

# Global Parameters
DEFAULT_SEED: int = 42
DEFAULT_DATASET_RATIO: float = 1.0

# Data augmentation
DEFAULT_USE_AUGMENTATION: bool = False
DEFAULT_AUG_REPEATS: int = 1

# Preprocessing
Preproc = Literal["none", "scale"] 
DEFAULT_PREPROC: Preproc = "scale"

# SVM kernel
Kernel = Literal["rbf", "linear"]
DEFAULT_KERNEL: Kernel = "rbf"

# Dataset class
DEFAULT_CLASS_WEIGHT: Optional[str] = "balanced"

# SVM param setting
DEFAULT_C: float = 1
DEFAULT_GAMMA: str | float = "scale"
DEFAULT_PROB: bool = False

# Test on/off
DEFAULT_EVAL_TEST: bool = True

def _flatten_images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Unexpected x shape: {x.shape}")

def _sample_by_ratio(x: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0,1], got {ratio}")
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    n = x.shape[0]
    k = max(1, int(np.floor(n * ratio)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return x[idx], y[idx]

def _get_y_score(estimator, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        except Exception:
            pass

    if hasattr(estimator, "decision_function"):
        try:
            scores = estimator.decision_function(X)
            return np.asarray(scores).reshape(-1)
        except Exception:
            pass

    return None

def run_svm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[Pipeline, Metrics, Metrics, Optional[Metrics]]:
    # 1) train sampling
    x_tr, y_tr = _sample_by_ratio(x_train, y_train, DEFAULT_DATASET_RATIO, DEFAULT_SEED)

    # 2) offline augmentation
    if DEFAULT_USE_AUGMENTATION and DEFAULT_AUG_REPEATS > 0:
        x_tr, y_tr = offline_augmentation(x_tr,y_tr,seed=DEFAULT_SEED,repeats=DEFAULT_AUG_REPEATS,augmentation=True)

    # 3) build pipeline
    steps = [("feat", FunctionTransformer(_flatten_images, validate=False))]

    if DEFAULT_PREPROC == "scale":
        steps.append(("scaler", StandardScaler()))
    elif DEFAULT_PREPROC == "none":
        pass
    else:
        raise ValueError(f"Unknown preproc: {DEFAULT_PREPROC}")

    steps.append(
        ("svm", SVC(kernel=DEFAULT_KERNEL,C=DEFAULT_C,gamma=(DEFAULT_GAMMA if DEFAULT_KERNEL == "rbf" else "scale"),
                    class_weight=DEFAULT_CLASS_WEIGHT,random_state=DEFAULT_SEED,probability=DEFAULT_PROB))
    )

    pipe = Pipeline(steps)

    # 4) train
    pipe.fit(x_tr, y_tr)

    # 5) metrics
    yhat_tr = pipe.predict(x_tr)
    train_m = compute_metrics(y_tr, yhat_tr, _get_y_score(pipe, x_tr))

    yhat_va = pipe.predict(x_val)
    val_m = compute_metrics(y_val, yhat_va, _get_y_score(pipe, x_val))

    test_m: Optional[Metrics] = None
    if DEFAULT_EVAL_TEST:
        yhat_te = pipe.predict(x_test)
        test_m = compute_metrics(y_test, yhat_te, _get_y_score(pipe, x_test))

    return pipe, train_m, val_m, test_m
