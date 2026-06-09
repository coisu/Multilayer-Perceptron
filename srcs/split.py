import csv
import argparse
import numpy as np

RESET  = "\033[0m"
PINK    = "\033[95m"

# defaults (used when no CLI argument is given)
INPUT_PATH  = "datasets/data.csv"
TRAIN_OUT   = "datasets/data_train.csv"
VALID_OUT   = "datasets/data_valid.csv"
VALID_RATIO = 0.2
SEED        = 42
N_COLS      = 32

def read_csv_no_header(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            rows.append(row)
    return rows  # list[list[str]]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

def parse_args():
    # current constants are the defaults; pass args to override
    p = argparse.ArgumentParser(description="Split data.csv into train/valid sets")
    p.add_argument("--input",       default=INPUT_PATH, help=f"source csv (default: {INPUT_PATH})")
    p.add_argument("--train-out",   default=TRAIN_OUT,  help=f"train output (default: {TRAIN_OUT})")
    p.add_argument("--valid-out",   default=VALID_OUT,  help=f"valid output (default: {VALID_OUT})")
    p.add_argument("--valid-ratio", type=float, default=VALID_RATIO, help=f"valid fraction (default: {VALID_RATIO})")
    p.add_argument("--seed",        type=int,   default=SEED, help=f"shuffle seed (default: {SEED})")
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    rows = read_csv_no_header(args.input)
    assert rows, "input is empty or not found"
    assert len(rows[0]) == N_COLS, f"Expected {N_COLS} columns, got {len(rows[0])}"

    n_rows = len(rows)
    idx = np.arange(n_rows)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    split = int(n_rows * (1 - args.valid_ratio))
    train_idx, valid_idx = idx[:split], idx[split:]
    write_csv(args.train_out, [rows[i] for i in train_idx])
    write_csv(args.valid_out, [rows[i] for i in valid_idx])

    print(f"{PINK}[split]{RESET} train={len(train_idx)}, valid={len(valid_idx)}")
    print(f"{PINK}[split] saved -> {args.train_out}, {args.valid_out}{RESET}")

if __name__ == "__main__":
    main()