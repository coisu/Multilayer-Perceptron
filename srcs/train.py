import os, csv
import numpy as np
import matplotlib.pyplot as plt
from mlp_numpy import MLP, one_hot, train
from split import SEED, N_COLS

TRAIN_SET   = "datasets/data_train.csv"
VALID_SET   = "datasets/data_valid.csv"
MODEL_OUT   = "models/saved_model.npz"
OUTPUT_DIR  = "outputs"

LAYERS      = [32, 32]
EPOCHS      = 30
LR          = 0.05  # learning rate
BATCH       = 64
ACT         = "relu"

def read_dataset(path):
    X_list, y_list = [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: 
                continue
            assert len(row) == N_COLS, f"Expected {N_COLS}, got {len(row)}"
            y = 1 if row[1].strip() == "M" else 0
            feats = [float(v) for v in row[2:]]  # 30 features
            y_list.append(y); X_list.append(feats)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sigma

def zscore_apply(X, mu, sigma):
    return (X - mu) / sigma

def main():
    # 1) load
    X_tr, y_tr = read_dataset(TRAIN_SET)
    X_va, y_va = read_dataset(VALID_SET)

    # 2) standardize by train stats
    mu, sigma = zscore_fit(X_tr)
    X_tr = zscore_apply(X_tr, mu, sigma)
    X_va = zscore_apply(X_va, mu, sigma)

    y_tr_oh = one_hot(y_tr, 2)
    y_va_oh = one_hot(y_va, 2)

    # 3) model
    input_size = X_tr.shape[1]
    layer_sizes = [input_size] + LAYERS + [2]
    activations = [ACT] * len(LAYERS) + ["softmax"]
    mlp = MLP(layer_sizes, activations, seed=42)

    # 4) train
    hist = train(
        mlp, X_tr, y_tr_oh, X_va, y_va_oh,
        epochs=EPOCHS, lr=LR, batch_size=BATCH, verbose=True
    )

    # 5) save model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "seed": SEED,
    }
    mlp.save(MODEL_OUT, meta)
    print(f"[train] saved model -> {MODEL_OUT}")

    # 6) plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(); plt.plot(hist["loss"], label="loss"); plt.plot(hist["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))

    plt.figure(); plt.plot(hist["acc"], label="acc"); plt.plot(hist["val_acc"], label="val_acc")
    plt.legend(); plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
    print(f"[train] plots -> {os.path.join(OUTPUT_DIR,'loss.png')}, {os.path.join(OUTPUT_DIR,'accuracy.png')}")

if __name__ == "__main__":
    main()
