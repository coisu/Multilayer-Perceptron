import argparse
import csv
import numpy as np
from mlp_numpy import MLP, one_hot, cross_entropy, accuracy

def read_dataset(path):
    X_list = []
    y_list = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            assert len(row) == 32, f"Expected 32 columns, got {len(row)}"
            diag = row[1].strip()
            y = 1 if diag == "M" else 0
            feats = [float(v) for v in row[2:]]
            y_list.append(y)
            X_list.append(feats)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    mlp, meta = MLP.load(args.model)
    X, y = read_dataset(args.dataset)

    mu = np.array(meta["mu"], dtype=np.float32)
    sigma = np.array(meta["sigma"], dtype=np.float32)
    Xs = (X - mu) / sigma

    y_oh = one_hot(y, 2)
    proba = mlp.predict_proba(Xs)
    loss = cross_entropy(proba, y_oh)
    acc = accuracy(proba, y_oh)
    print(f"loss: {loss:.4f}, acc: {acc:.4f}")

    # save predictions.csv
    with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["proba_B", "proba_M", "pred", "true"])
        pred_label = np.argmax(proba, axis=1)
        for pr, pl, yt in zip(proba, pred_label, y):
            pred = "M" if pl == 1 else "B"
            tru = "M" if yt == 1 else "B"
            w.writerow([f"{pr[0]:.6f}", f"{pr[1]:.6f}", pred, tru])

if __name__ == "__main__":
    main()
