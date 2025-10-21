import os, csv
import numpy as np
import matplotlib.pyplot as plt
from mlp_numpy import MLP, train
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

def one_hot(y, n_classes):
    y = np.asarray(y, dtype=int)
    Y = np.zeros((len(y), n_classes), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y

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

def standardize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8 # avoid sigma = 0
    return mean, std

def standardize_features(X, mean, std):
    return (X - mean) / std

def main():
    # 1) load
    X_tr, y_tr = read_dataset(TRAIN_SET)
    X_va, y_va = read_dataset(VALID_SET)
    print(f"\nX_tr shape: {X_tr.shape}")
    print(f"X_va shape: {X_va.shape}\n")

    # 2) standardize by train stats
    mean_train, std_train = standardize_fit(X_tr)
    X_tr = standardize_features(X_tr, mean_train, std_train)
    X_va = standardize_features(X_va, mean_train, std_train)

    #   vectorized
    y_tr_vec = one_hot(y_tr, 2)
    y_va_vec = one_hot(y_va, 2)

    # 3) model
    input_size = X_tr.shape[1]
    layer_sizes = [input_size] + LAYERS + [2]   # [30(input), 32, 32, 2(output: [M, B])]
    activations = [ACT] * len(LAYERS) + ["softmax"]
    mlp = MLP(layer_sizes, activations, SEED)   # LR * sum + bias

    # 4) train
    hist = train(
        mlp, X_tr, y_tr_vec, X_va, y_va_vec,
        epochs=EPOCHS, lr=LR, batch_size=BATCH, verbose=True
    )

    # 5) save model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mean": mean_train.tolist(),
        "standard_division": std_train.tolist(),
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
