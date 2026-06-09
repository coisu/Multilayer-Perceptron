import os, csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from split import SEED, N_COLS
from mlp_manual import MLP, train_loop, save_npz

RESET  = "\033[0m"
GRAY   = "\033[90m"
BLUE   = "\033[94m"

# defaults (used when no CLI argument is given)
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


def parse_args():
    # current constants are the defaults; pass args to override
    p = argparse.ArgumentParser(description="Train the MLP and save model + learning curves")
    p.add_argument("--layer", nargs="+", type=int, default=LAYERS,
                   help=f"hidden layer sizes, space separated (default: {' '.join(map(str, LAYERS))})")
    p.add_argument("--epochs", type=int, default=EPOCHS, help=f"epochs (default: {EPOCHS})")
    p.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=LR,
                   help=f"learning rate (default: {LR})")
    p.add_argument("--batch_size", type=int, default=BATCH, help=f"batch size (default: {BATCH})")
    p.add_argument("--activation", default=ACT, choices=["relu"],
                   help=f"hidden activation (default: {ACT})")
    p.add_argument("--loss", default="categoricalCrossentropy", choices=["categoricalCrossentropy"],
                   help="training loss (default: categoricalCrossentropy)")
    p.add_argument("--seed", type=int, default=SEED, help=f"seed (default: {SEED})")
    # regularization / schedule / early stopping
    p.add_argument("--patience", type=int, default=PATIENCE, help=f"early-stopping patience (default: {PATIENCE})")
    p.add_argument("--lr_decay_at", type=int, default=LR_DECAY_AT, help=f"epoch to decay lr, 0=off (default: {LR_DECAY_AT})")
    p.add_argument("--lr_decay_factor", type=float, default=LR_DECAY_FACTOR, help=f"lr decay factor (default: {LR_DECAY_FACTOR})")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help=f"L2 weight decay (default: {WEIGHT_DECAY})")
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM, help=f"grad clip norm, 0=off (default: {MAX_GRAD_NORM})")
    # paths
    p.add_argument("--train_set", default=TRAIN_SET, help=f"train csv (default: {TRAIN_SET})")
    p.add_argument("--valid_set", default=VALID_SET, help=f"valid csv (default: {VALID_SET})")
    p.add_argument("--models_dir", default=MODELS_DIR, help=f"model output dir (default: {MODELS_DIR})")
    p.add_argument("--output_dir", default=OUTPUT_DIR, help=f"plot output dir (default: {OUTPUT_DIR})")
    return p.parse_args()

def write_latest(models_dir: str, model_filename: str):
    os.makedirs(models_dir, exist_ok=True)
    latest = os.path.join(models_dir, "latest")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(os.path.basename(model_filename).strip() + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"{BLUE}[train]{GRAY} saved latest model -> {os.path.join(models_dir, 'latest')}{RESET}")


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes), dtype=np.float64) # align witj softmax
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
            y_list.append(y)
            X_list.append(feats)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def standardize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def standardize_transform(X, mean, std): #np
    return (X - mean) / std


def main(args=None):
    if args is None:
        args = parse_args()

    # load data
    X_tr, y_tr = read_dataset(args.train_set)
    X_va, y_va = read_dataset(args.valid_set)
    print(f"X_tr: {X_tr.shape}, X_va: {X_va.shape}")
    print(f"y_tr: {y_tr.shape}")

    # one-hot
    Y_tr = one_hot(y_tr, 2)
    Y_va = one_hot(y_va, 2)
    print(f"Y_tr: {Y_tr.shape}")
    # standardize
    mean_tr, std_tr = standardize_fit(X_tr)             # avoiding data leakage, to not use validation data
    X_tr = standardize_transform(X_tr, mean_tr, std_tr)
    X_va = standardize_transform(X_va, mean_tr, std_tr)

    # define model
    input_size = X_tr.shape[1]
    layer_sizes = [input_size] + args.layer + [2]               # [30, 64, 32, 2]
    activations = [args.activation] * len(args.layer) + ["softmax"]  # relu, relu, softmax

    # create model (init seed = shuffle seed)
    mlp = MLP(layer_sizes, activations, seed=args.seed)

    # train
    history, best_epoch = train_loop(
        mlp,
        X_tr, Y_tr, X_va, Y_va,
        epochs=args.epochs, lr=args.learning_rate, batch_size=args.batch_size,
        seed=args.seed, patience=args.patience,
        lr_decay_at=args.lr_decay_at, lr_decay_factor=args.lr_decay_factor,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        verbose=True
    )

    # save
    meta = {
        "layer_sizes": layer_sizes,
        "activations": activations,
        "mean": mean_tr.tolist(),
        "std": std_tr.tolist(),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
    }

    arch_all = "-".join(str(s) for s in layer_sizes)

    # output_size = 2
    # arch_model = "-".join(str(s) for s in (LAYERS + [output_size]))

    best_val_loss = min(history["val_loss"]) if isinstance(history, dict) else min(h["val_loss"] for h in history)
    MODEL_OUT = f"model_e{best_epoch}-of-{args.epochs}_vl{best_val_loss:.4f}_b{args.batch_size}_seed{args.seed}_arch{arch_all}.npz"

    os.makedirs(args.models_dir, exist_ok=True)
    save_npz(os.path.join(args.models_dir, MODEL_OUT), mlp, meta)
    print(f"{BLUE}[train] saved model -> {os.path.join(args.models_dir, MODEL_OUT)}{RESET}")
    write_latest(args.models_dir, MODEL_OUT)

    # graph
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")

    plt.legend()

    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))

    plt.figure()
    plt.plot(history["acc"], label="acc")
    plt.plot(history["val_acc"], label="val_acc")

    plt.legend()
    
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))

    print(f"{BLUE}[train]{RESET} plots -> {os.path.join(OUTPUT_DIR,'loss.png')}, {os.path.join(OUTPUT_DIR,'accuracy.png')}")

if __name__ == "__main__":
    main()
