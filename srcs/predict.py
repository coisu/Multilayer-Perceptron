# load saved npz(weights + meta), excute forward and valify accurancy
import csv
import numpy as np
from split import N_COLS
from mlp_manual import load_npz

MODEL_PATH = "models/saved_model.npz"
INPUT_CSV  = "datasets/data_valid.csv"


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
    mlp, meta = load_npz(MODEL_PATH)
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
    print(f"[predict] accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
