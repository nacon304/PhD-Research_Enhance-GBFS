#!/usr/bin/env python3
from __future__ import annotations
import os, csv, argparse, glob
from typing import List, Dict, Any, Set, Iterable, Tuple

def _iter_candidate_files(out_root: str) -> Tuple[List[str], List[str]]:
    # Only match exactly 3-level deep: dataset_*/algo/run_*/{front_train_cv.csv,test_points.csv}
    pat_front = os.path.join(out_root, "dataset_*", "*", "run_*", "front_train_cv.csv")
    pat_test  = os.path.join(out_root, "dataset_*", "*", "run_*", "test_points.csv")
    front_files = sorted(glob.glob(pat_front))
    test_files  = sorted(glob.glob(pat_test))
    return front_files, test_files

def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
    return [h.strip() for h in hdr if h is not None]

def _union_fieldnames(files: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for p in files:
        hdr = _read_header(p)
        for k in hdr:
            if k not in seen:
                seen.add(k); out.append(k)
    return out

def _stream_rows(files: List[str]) -> Iterable[Dict[str, Any]]:
    for p in files:
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    yield row
        except Exception as e:
            print(f"[WARN] skip {p} ({e})")

def _write_union_csv(path: str, files: List[str], fieldnames: List[str]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in _stream_rows(files):
            w.writerow(row)
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="Folder containing dataset_* subfolders")
    ap.add_argument("--out_front", type=str, default=None, help="Optional override output front csv path")
    ap.add_argument("--out_test", type=str, default=None, help="Optional override output test csv path")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    if not os.path.isdir(out_root):
        raise FileNotFoundError(out_root)

    front_files, test_files = _iter_candidate_files(out_root)

    if not front_files:
        print("[WARN] no front_train_cv.csv files found under:", out_root)
    if not test_files:
        print("[WARN] no test_points.csv files found under:", out_root)

    out_front = args.out_front or os.path.join(out_root, "compare_front_train.csv")
    out_test  = args.out_test  or os.path.join(out_root, "compare_test_points.csv")

    # Union headers (stream-friendly)
    fn_front = _union_fieldnames(front_files) if front_files else []
    fn_test  = _union_fieldnames(test_files)  if test_files  else []

    # If some required cols missing, we still write union; analysis will validate.
    n_front = _write_union_csv(out_front, front_files, fn_front) if front_files else 0
    n_test  = _write_union_csv(out_test,  test_files,  fn_test)  if test_files  else 0

    print("MERGE DONE.")
    print("Front:", out_front, "files=", len(front_files), "rows=", n_front)
    print("Test :", out_test,  "files=", len(test_files),  "rows=", n_test)

if __name__ == "__main__":
    main()

# python .\Compare\merge_results.py --out_root "D:\PhD\The First Paper\Code Implement\GBFS-SND\_local_out"