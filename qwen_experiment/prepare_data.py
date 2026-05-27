from __future__ import annotations

import argparse
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
from src.data import (
    load_xsum_from_hf,
    load_coedit_from_hf,
    safe_jsonl_write,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, val = load_xsum_from_hf(args.max_train, args.max_val, seed=args.seed)

    sum_train = os.path.join(args.out_dir, "sum_train.jsonl")
    sum_val = os.path.join(args.out_dir, "sum_val.jsonl")
    safe_jsonl_write(sum_train, train)
    safe_jsonl_write(sum_val, val)

    para_train_rows = load_coedit_from_hf("paraphrase", args.max_train, args.max_val, seed=args.seed + 1)[0]
    para_val_rows = load_coedit_from_hf("paraphrase", args.max_train, args.max_val, seed=args.seed + 2)[1]

    para_train = os.path.join(args.out_dir, "para_train.jsonl")
    para_val = os.path.join(args.out_dir, "para_val.jsonl")
    safe_jsonl_write(para_train, para_train_rows)
    safe_jsonl_write(para_val, para_val_rows)

    print("Wrote:")
    print(" ", sum_train)
    print(" ", sum_val)
    print(" ", para_train)
    print(" ", para_val)
    print(f"Train sizes: sum={len(train)} paraphrase={len(para_train_rows)}")
    print(f"Val sizes:   sum={len(val)} paraphrase={len(para_val_rows)}")

if __name__ == "__main__":
    main()
