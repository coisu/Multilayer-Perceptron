# load saved npz(weights + meta), excute forward and valify accurancy
import os, csv
import argparse
import numpy as np
from split import N_COLS
from mlp_manual import load_npz, cross_entropy
from train import one_hot


RESET  = "\033[0m"
YELLOW = "\033[33m"

# defaults (used when no CLI argument is given)
# MODEL_PATH = "models/saved_model.npz"
MODELS_DIR = "models/"
LATEST_PATH = "models/latest"
INPUT_CSV  = "datasets/data_valid.csv"


def parse_args():
    # current constants are the defaults; pass args to override
    p = argparse.ArgumentParser(description="Load a saved model and evaluate it (accuracy + BCE)")
    p.add_argument("--input",      default=INPUT_CSV,   help=f"csv to predict on (default: {INPUT_CSV})")
    p.add_argument("--model",      default=None,        help="model .npz filename inside models_dir (default: read from latest)")
    p.add_argument("--models_dir", default=MODELS_DIR,  help=f"model dir (default: {MODELS_DIR})")
    p.add_argument("--latest",     default=LATEST_PATH, help=f"latest pointer file (default: {LATEST_PATH})")
    return p.parse_args()


def read_latest_model_path(latest_file="models/latest") -> str:
    if not os.path.isfile(latest_file):
        raise FileNotFoundError(f"missing latest.txt: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        name = f.readline().strip()
    if not name:
        raise ValueError(f"cannot get npz file name: {latest_file}")
    return name


def read_dataset(path):
    X_list, y_list = [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: 
                continue
            assert len(row) == N_COLS, f"Expected {N_COLS}, got {len(row)}"
            y = 1 if row[1].strip() == "M" else 0
            feats = [float(v) for v in row[2:]]
            y_list.append(y)
            X_list.append(feats)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def standardize_transform(X, mean, std):
    return (X - mean) / std

def binary_cross_entropy(y_true: np.ndarray, p_true: np.ndarray) -> float:
    # E = -(1/N) * sum( y*log(p) + (1-y)*log(1-p) )
    # Σ no needs np.sum(..., axis=1)
    eps = 1e-12
    y = y_true.astype(np.float64)
    p = np.clip(p_true.astype(np.float64), eps, 1.0 - eps)  # 1.0 - eps: for log(1.0 - p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def main(args=None):
    if args is None:
        args = parse_args()

    MODEL_PATH = args.model if args.model else read_latest_model_path(args.latest)
    print(f"{YELLOW}[predict] {RESET}load model file: {os.path.join(args.models_dir, MODEL_PATH)}{RESET}")
    mlp, meta = load_npz(os.path.join(args.models_dir, MODEL_PATH))
    mean = np.array(meta["mean"], dtype=np.float64)
    std  = np.array(meta["std"],  dtype=np.float64)
    activations = meta["activations"]

    # 2) data load + preprocessing in the samw way
    X, y = read_dataset(args.input)
    X = standardize_transform(X, mean, std)

    # 3) predict
    proba = mlp.predict_proba(X)
    yhat = proba.argmax(axis=1)
    acc = (yhat == y).mean()

    # strict BCE evaluation (use p for class 1)
    p_pos = proba[:, 1]                  # shape (N,)
    bce = binary_cross_entropy(y, p_pos)

    print(f"{YELLOW}[predict] accuracy={acc:.4f}, binary_cross_entropy={bce:.4f}{RESET}")

if __name__ == "__main__":
    main()
