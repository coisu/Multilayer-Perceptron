import os, csv
import numpy as np
import matplotlib.pyplot as plt
from split import SEED, N_COLS
from mlp_manual import MLP, train_loop, save_npz

RESET  = "\033[0m"
GRAY   = "\033[90m"
BLUE   = "\033[94m"

TRAIN_SET   = "datasets/data_train.csv"
VALID_SET   = "datasets/data_valid.csv"
MODELS_DIR  = "models/"
# MODEL_OUT   = "models/saved_model.npz"
OUTPUT_DIR  = "outputs"

LAYERS      = [64, 32]
EPOCHS      = 60
LR          = 0.05
BATCH       = 32
ACT         = "relu"

PATIENCE        = 5         # for early stopping
LR_DECAY_AT     = 10        # at EPOCHS 10
LR_DECAY_FACTOR = 0.5
WEIGHT_DECAY    = 0.0       # L2
MAX_GRAD_NORM   = 5.0

def write_latest(models_dir: str, model_filename: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    latest = os.path.join(MODELS_DIR, "latest")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(os.path.basename(model_filename).strip() + "\n")  # 파일명만 저장
        f.flush()
        os.fsync(f.fileno())
    print(f"{BLUE}[train]{GRAY} saved latest model -> {MODELS_DIR}latest{RESET}")

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
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mean": mean_tr.tolist(),
        "std": std_tr.tolist(),
        "seed": int(SEED),
        "best_epoch": int(best_epoch),
    }

    arch_all = "-".join(str(s) for s in layer_sizes)

    # output_size = 2
    # arch_model = "-".join(str(s) for s in (LAYERS + [output_size]))  # e.g. "64-32-2"

    best_val_loss = min(hist["val_loss"]) if isinstance(hist, dict) else min(h["val_loss"] for h in hist)
    MODEL_OUT = f"model_e{best_epoch}-of-{EPOCHS}_vl{best_val_loss:.4f}_b{BATCH}_seed{SEED}_arch{arch_all}.npz"

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_npz(os.path.join(MODELS_DIR, MODEL_OUT), mlp, meta)
    print(f"{BLUE}[train] saved model -> {MODELS_DIR}{MODEL_OUT}{RESET}")
    write_latest(MODELS_DIR, MODEL_OUT)

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

    print(f"{BLUE}[train]{RESET} plots -> {os.path.join(OUTPUT_DIR,'loss.png')}, {os.path.join(OUTPUT_DIR,'accuracy.png')}")

if __name__ == "__main__":
    main()
