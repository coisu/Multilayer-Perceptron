# load saved npz(weights + meta), excute forward and valify accurancy
import os, csv
import numpy as np
from split import N_COLS
from mlp_manual import load_npz, cross_entropy
from train import one_hot


RESET  = "\033[0m"
YELLOW = "\033[33m"

# MODEL_PATH = "models/saved_model.npz"
MODELS_DIR = "models/"
LATEST_PATH = "models/latest"
INPUT_CSV  = "datasets/data_valid.csv"


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
            y_list.append(y); X_list.append(feats)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def standardize_transform(X, mean, std):
    return (X - mean) / std


def main():
    MODEL_PATH = read_latest_model_path(LATEST_PATH)
    print(f"{YELLOW}[predict] load latest model file: {MODELS_DIR}{MODEL_PATH}{RESET}")
    mlp, meta = load_npz(os.path.join(MODELS_DIR, MODEL_PATH))
    mean = np.array(meta["mean"], dtype=np.float64)
    std  = np.array(meta["std"],  dtype=np.float64)
    activations = meta["activations"]

    # 2) data load + preprocessing in the samw way
    X, y = read_dataset(INPUT_CSV)
    X = standardize_transform(X, mean, std)

    # 3) predict
    proba = mlp.predict_proba(X)
    yhat = proba.argmax(axis=1)
    acc = (yhat == y).mean()

    Y = one_hot(y, n_classes=2)
    binary_cross_entropy = cross_entropy(proba, Y)

    print(f"{YELLOW}[predict]{RESET} accuracy={acc:.4f}, binary_cross_entropy={binary_cross_entropy:.4f}")

if __name__ == "__main__":
    main()
