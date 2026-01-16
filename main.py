from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from Code.model_a.a_data import load_breastmnist
import Code.model_a.a_model as a_mod
try:
    import Code.model_b.model_b as b_mod  
except Exception:
    import Code.model_b.resnet_model as b_mod  

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True, parents=True)

def configure_svm_defaults(args: argparse.Namespace) -> None:
    if args.seed is not None:
        a_mod.DEFAULT_SEED = int(args.seed)
    if args.dataset_ratio is not None:
        a_mod.DEFAULT_DATASET_RATIO = float(args.dataset_ratio)
    if args.svm_kernel is not None:
        a_mod.DEFAULT_KERNEL = args.svm_kernel
    if args.svm_preproc is not None:
        a_mod.DEFAULT_PREPROC = args.svm_preproc
    if args.pca_dim is not None:
        a_mod.DEFAULT_PCA_DIM = int(args.pca_dim)
    if args.class_weight is not None:
        a_mod.DEFAULT_CLASS_WEIGHT = None if args.class_weight.lower() == "none" else args.class_weight
    if args.c is not None:
        a_mod.DEFAULT_C = float(args.c)
    if args.gamma is not None:
        a_mod.DEFAULT_GAMMA = args.gamma
    if args.aug_repeats is not None:
        a_mod.DEFAULT_AUG_REPEATS = int(args.aug_repeats)
    if args.svm_augmentation:
        a_mod.DEFAULT_USE_AUGMENTATION = True
    if args.prob:
        a_mod.DEFAULT_PROB = True
    if args.eval_test:
        a_mod.DEFAULT_EVAL_TEST = True

def configure_resnet_defaults(args: argparse.Namespace) -> None:
    if hasattr(b_mod, "DEFAULT_SEED") and args.seed is not None:
        b_mod.DEFAULT_SEED = int(args.seed)
    if hasattr(b_mod, "DEFAULT_DEPTH") and args.b_depth is not None:
        b_mod.DEFAULT_DEPTH = int(args.b_depth)
    if hasattr(b_mod, "DEFAULT_EPOCHS") and args.b_epochs is not None:
        b_mod.DEFAULT_EPOCHS = int(args.b_epochs)
    if hasattr(b_mod, "DEFAULT_BATCH_SIZE") and args.b_batch_size is not None:
        b_mod.DEFAULT_BATCH_SIZE = int(args.b_batch_size)
    if hasattr(b_mod, "DEFAULT_LR") and args.b_lr is not None:
        b_mod.DEFAULT_LR = float(args.b_lr)
    if hasattr(b_mod, "DEFAULT_WEIGHT_DECAY") and args.b_weight_decay is not None:
        b_mod.DEFAULT_WEIGHT_DECAY = float(args.b_weight_decay)
    if hasattr(b_mod, "DEFAULT_EVAL_TEST") and args.eval_test:
        b_mod.DEFAULT_EVAL_TEST = True
    if hasattr(b_mod, "DEFAULT_USE_AUGMENTATION") and args.resnet_augmentation:
        b_mod.DEFAULT_USE_AUGMENTATION = True

def result_print(result: Dict[str, Any]) -> None:
    def fmt(split: str) -> str:
        m = result.get(split)
        if m is None:
            return f"{split}: None"
        return (
            f"{split}  "
            f"acc={m['acc']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}  "
            f"f1={m['f1']:.4f}  pr_auc={m['pr_auc']:.4f}"
        )
    print(fmt("train"))
    print(fmt("val"))
    print(fmt("test"))

def run_model_a(args: argparse.Namespace) -> Dict[str, Any]:
    configure_svm_defaults(args)
    dataset_root = ROOT / "Datasets" / "BreastMNIST"
    ds = load_breastmnist(dataset_root)
    _, train_m, val_m, test_m = a_mod.run_svm(ds.x_train, ds.y_train,ds.x_val, ds.y_val,ds.x_test, ds.y_test)
    return {
        "train": train_m.metrics_dict(),
        "val": val_m.metrics_dict(),
        "test": None if test_m is None else test_m.metrics_dict(),
    }

def run_model_b(args: argparse.Namespace) -> Dict[str, Any]:
    configure_resnet_defaults(args)
    dataset_root = ROOT / "Datasets" / "BreastMNIST"
    if hasattr(b_mod, "run_resnet_single"):
        train_m, val_m, test_m = b_mod.run_resnet_single( 
            dataset_root=dataset_root,
            seed=int(args.seed) if args.seed is not None else 42,
            augmentation=bool(args.resnet_augmentation),
            eval_test=bool(args.eval_test),
            out_dir=OUTDIR,
            depth=int(args.b_depth) if args.b_depth is not None else None,
            epochs=int(args.b_epochs) if args.b_epochs is not None else None,
            batch_size=int(args.b_batch_size) if args.b_batch_size is not None else None,
            lr=float(args.b_lr) if args.b_lr is not None else None,
            weight_decay=float(args.b_weight_decay) if args.b_weight_decay is not None else None,
        )
        return {
            "train": train_m.metrics_dict(),
            "val": val_m.metrics_dict(),
            "test": None if test_m is None else test_m.metrics_dict(),
        }

    if hasattr(b_mod, "run_resnet"):
        ds = load_breastmnist(dataset_root)
        _, train_m, val_m, test_m = b_mod.run_resnet( 
            ds.x_train, ds.y_train,
            ds.x_val, ds.y_val,
            ds.x_test, ds.y_test,
        )
        return {
            "train": train_m.metrics_dict(),
            "val": val_m.metrics_dict(),
            "test": None if test_m is None else test_m.metrics_dict(),
        }

    raise RuntimeError("Model B module does not provide run_resnet_single or run_resnet.")

def main() -> None:
    p = argparse.ArgumentParser(description="AMLS BreastMNIST - Model A(SVM) / Model B(ResNet)")
    p.add_argument("--model", required=True, choices=["A", "B"])

    # Common
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-test", action="store_true")

    # ---------------- Model A: SVM ----------------
    p.add_argument("--svm-augmentation", action="store_true")
    p.add_argument("--svm-kernel", choices=["rbf", "linear"], default=None)
    p.add_argument("--svm-preproc", choices=["none", "scale"], default=None)
    p.add_argument("--c", type=float, default=None)

    p.add_argument("--aug-repeats", type=int, default=None)

    # ---------------- Model B: ResNet ----------------
    p.add_argument("--resnet-augmentation", action="store_true")
    p.add_argument("--b-depth", type=int, choices=[18, 34, 50, 101, 152], default=None)
    p.add_argument("--b-epochs", type=int, default=None)

    args = p.parse_args()

    if args.model == "A":
        result = run_model_a(args)
        out_path = OUTDIR / "result_A.json"
    else:
        result = run_model_b(args)
        out_path = OUTDIR / "result_B.json"

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result_print(result)

if __name__ == "__main__":
    main()
