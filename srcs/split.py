import argparse
import csv
import numpy as np

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data.csv (no header, 32 cols)")
    ap.add_argument("--train_out", default="data_train.csv")
    ap.add_argument("--valid_out", default="data_valid.csv")
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_csv_no_header(args.input)
    assert len(rows) > 0, "empty input"
    ncols = len(rows[0])
    assert ncols == 32, f"Expected 32 columns, got {ncols}"

    n = len(rows)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    split = int(n * (1 - args.valid_ratio))
    train_idx, valid_idx = idx[:split], idx[split:]

    train_rows = [rows[i] for i in train_idx]
    valid_rows = [rows[i] for i in valid_idx]

    write_csv(args.train_out, train_rows)
    write_csv(args.valid_out, valid_rows)
    print(f"Saved {len(train_rows)} rows to {args.train_out} and {len(valid_rows)} rows to {args.valid_out}")

if __name__ == "__main__":
    main()
