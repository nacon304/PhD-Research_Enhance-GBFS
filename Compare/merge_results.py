import os
import csv
import argparse
from typing import Dict, List, Any, Set


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def write_csv_union(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        print("No rows to write:", path)
        return

    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    args = ap.parse_args()

    out_root = args.out_root

    front_all: List[Dict[str, Any]] = []
    test_all: List[Dict[str, Any]] = []

    for ds_name in sorted(os.listdir(out_root)):
        ds_path = os.path.join(out_root, ds_name)
        if not (os.path.isdir(ds_path) and ds_name.startswith("dataset_")):
            continue

        for run_name in sorted(os.listdir(ds_path)):
            run_path = os.path.join(ds_path, run_name)
            if not (os.path.isdir(run_path) and run_name.startswith("run_")):
                continue

            f_front = os.path.join(run_path, "front_train_all.csv")
            f_test  = os.path.join(run_path, "test_points_all.csv")

            if os.path.exists(f_front):
                front_all.extend(read_csv_rows(f_front))
            if os.path.exists(f_test):
                test_all.extend(read_csv_rows(f_test))

    out_front = os.path.join(out_root, "compare_front_train.csv")
    out_test  = os.path.join(out_root, "compare_test_points.csv")

    write_csv_union(out_front, front_all)
    write_csv_union(out_test, test_all)

    print("MERGE DONE.")
    print("Front:", out_front, "rows=", len(front_all))
    print("Test :", out_test, "rows=", len(test_all))


if __name__ == "__main__":
    main()
