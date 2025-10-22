import os, csv
import numpy as np
import matplotlib.pyplot as plt
from split import SEED, N_COLS
from mlp_manual import MLP, train_loop, save_npz

TRAIN_SET   = "datasets/data_train.csv"
VALID_SET   = "datasets/data_valid.csv"
MODEL_OUT   = "models/saved_model.npz"
OUTPUT_DIR  = "outputs"

LAYERS      = [32, 32]
EPOCHS      = 30
LR          = 0.05
BATCH       = 64
ACT         = "relu"

PATIENCE        = 5         # for early stopping
LR_DECAY_AT     = 10        # at EPOCHS 10
LR_DECAY_FACTOR = 0.5
WEIGHT_DECAY    = 0.0       # L2
MAX_GRAD_NORM   = 5.0

def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
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
            # col0=id, col1=diagnosis(M/B), col2..31=30 features
            y = 1 if row[1].strip() == "M" else 0
            feats = [float(v) for v in row[2:]]
            y_list.append(y); X_list.append(feats)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def standardize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def standardize_transform(X, mean, std):
    return (X - mean) / std


def main():
    # load data
    X_tr, y_tr = read_dataset(TRAIN_SET)
    X_va, y_va = read_dataset(VALID_SET)
    print(f"X_tr: {X_tr.shape}, X_va: {X_va.shape}")

    # standardize
    mean_tr, std_tr = standardize_fit(X_tr)
    X_tr = standardize_transform(X_tr, mean_tr, std_tr)
    X_va = standardize_transform(X_va, mean_tr, std_tr)

    # one-hot
    Y_tr = one_hot(y_tr, 2)
    Y_va = one_hot(y_va, 2)

    # define model
    input_size = X_tr.shape[1]
    layer_sizes = [input_size] + LAYERS + [2]     # [30, 32, 32, 2]
    activations = [ACT] * len(LAYERS) + ["softmax"]

    # create model (init SEED = shuffle SEED)
    mlp = MLP(layer_sizes, activations, seed=SEED)

    # train
    hist, best_epoch = train_loop(
        mlp,
        X_tr, Y_tr, X_va, Y_va,
        epochs=EPOCHS, lr=LR, batch_size=BATCH,
        seed=SEED, patience=PATIENCE,
        lr_decay_at=LR_DECAY_AT, lr_decay_factor=LR_DECAY_FACTOR,
        weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM,
        verbose=True
    )

    # save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mean": mean_tr.tolist(),
        "std": std_tr.tolist(),
        "seed": int(SEED),
        "best_epoch": int(best_epoch),
    }
    save_npz(MODEL_OUT, mlp, meta)
    print(f"[train] saved model -> {MODEL_OUT}")

    # graph
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(hist["loss"], label="loss")
    plt.plot(hist["val_loss"], label="val_loss")

    plt.legend()

    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))

    plt.figure()
    plt.plot(hist["acc"], label="acc")
    plt.plot(hist["val_acc"], label="val_acc")

    plt.legend()
    
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))

    print(f"[train] plots -> {os.path.join(OUTPUT_DIR,'loss.png')}, {os.path.join(OUTPUT_DIR,'accuracy.png')}")

if __name__ == "__main__":
    main()
