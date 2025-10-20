import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mlp_numpy import MLP, one_hot, train

def read_dataset(path):
    # no header, 32 cols: id, diagnosis(M/B), f1..f30
    X_list = []
    y_list = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            assert len(row) == 32, f"Expected 32 columns, got {len(row)}"
            diag = row[1].strip()
            y = 1 if diag == "M" else 0  # M=1, B=0
            feats = [float(v) for v in row[2:]]  # 30 features
            y_list.append(y)
            X_list.append(feats)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set", required=True)
    ap.add_argument("--valid_set", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[24, 24])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    ap.add_argument("--out", default="saved_model.npz")
    ap.add_argument("--output_dir", default=".")
    args = ap.parse_args()

    X_tr, y_tr = read_dataset(args.train_set)
    X_va, y_va = read_dataset(args.valid_set)

    mu, sigma = standardize_fit(X_tr)
    X_tr = standardize_apply(X_tr, mu, sigma)
    X_va = standardize_apply(X_va, mu, sigma)

    y_tr_oh = one_hot(y_tr, 2)
    y_va_oh = one_hot(y_va, 2)

    input_size = X_tr.shape[1]
    layer_sizes = [input_size] + args.layers + [2]
    activations = [args.activation] * len(args.layers) + ["softmax"]

    mlp = MLP(layer_sizes, activations, seed=42)
    hist = train(
        mlp, X_tr, y_tr_oh, X_va, y_va_oh,
        epochs=args.epochs, lr=args.learning_rate,
        batch_size=args.batch_size, verbose=True
    )

    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "seed": 42,
    }
    mlp.save(args.out, meta)
    print(f"> saved model to '{args.out}'")

    os.makedirs(args.output_dir, exist_ok=True)
    # Loss plot
    plt.figure()
    plt.plot(hist["loss"], label="loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("cross-entropy")
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, "loss.png"))
    # Acc plot
    plt.figure()
    plt.plot(hist["acc"], label="acc")
    plt.plot(hist["val_acc"], label="val_acc")
    plt.legend(); plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir, "accuracy.png"))
    print("> saved plots: loss.png, accuracy.png")

if __name__ == "__main__":
    main()
