# export_isolet_only.py
import os, sys
import numpy as np
import pandas as pd

def write_csv(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(arr).to_csv(path, header=False, index=False)

def main(root):
    # Accept both root/isolet*.data and root/isolet/isolet*.data
    candidates = [
        os.path.join(root, "isolet1+2+3+4.data"),
        os.path.join(root, "isolet", "isolet1+2+3+4.data"),
    ]
    train_path = next((p for p in candidates if os.path.isfile(p)), None)

    candidates = [
        os.path.join(root, "isolet5.data"),
        os.path.join(root, "isolet", "isolet5.data"),
    ]
    test_path = next((p for p in candidates if os.path.isfile(p)), None)

    if not train_path or not test_path:
        print("[ISOLET] ERROR: could not find isolet1+2+3+4.data and/or isolet5.data")
        print("  looked for:", train_path, "and", test_path)
        sys.exit(1)

    print(f"[ISOLET] reading:\n  {train_path}\n  {test_path}")
    A = pd.read_csv(train_path, header=None).values
    B = pd.read_csv(test_path, header=None).values
    if A.shape[1] != B.shape[1]:
        print("[ISOLET] ERROR: feature mismatch between train/test")
        sys.exit(1)

    feat = A.shape[1] - 1
    X = np.vstack([A[:, :feat], B[:, :feat]]).astype(np.float32)
    y = np.concatenate([A[:, -1], B[:, -1]]).astype(np.int64)
    # shift 1..26 -> 0..25 (guard if already zero-based)
    if y.min() == 1:
        y = y - 1

    out_dir = os.path.join(root, "isolet")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(X, os.path.join(out_dir, "X.csv"))
    write_csv(y.reshape(-1,1), os.path.join(out_dir, "y.csv"))
    print(f"[ISOLET] wrote:\n  {out_dir}/X.csv (shape {X.shape})\n  {out_dir}/y.csv (shape {y.shape})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_isolet_only.py /path/to/Datasets")
        sys.exit(1)
    # normalize trailing slash
    root = sys.argv[1].rstrip("/")

    main(root)
