import csv
import numpy as np

RESET  = "\033[0m"
PINK    = "\033[95m"

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

def main():
    rows = read_csv_no_header(INPUT_PATH)
    assert rows, "INPUT_PATH is empty or not found"
    assert len(rows[0]) == N_COLS, f"Expected {N_COLS} columns, got {len(rows[0])}"

    n_rows = len(rows)
    idx = np.arange(n_rows)
    rng = np.random.default_rng(SEED)
    print("rng: ", rng)
    rng.shuffle(idx)

    split = int(n_rows * (1 - VALID_RATIO))
    train_idx, valid_idx = idx[:split], idx[split:]
    write_csv(TRAIN_OUT, [rows[i] for i in train_idx])
    write_csv(VALID_OUT, [rows[i] for i in valid_idx])

    print(f"{PINK}[split]{RESET} train={len(train_idx)}, valid={len(valid_idx)}")
    print(f"{PINK}[split] saved -> {TRAIN_OUT}, {VALID_OUT}{RESET}")

if __name__ == "__main__":
    main()