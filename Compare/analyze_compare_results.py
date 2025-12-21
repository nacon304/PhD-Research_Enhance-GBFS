from __future__ import annotations

import os
import re
import glob
import hashlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# CONFIG
# =========================
OUT_ROOT = r"D:\PhD\The First Paper\Code Implement\GBFS-SND\Compare\Results"

ANALYSIS_SUBFOLDER = "analysis_results"
OUT_DIR = os.path.join(OUT_ROOT, ANALYSIS_SUBFOLDER)

TRAIN_CSV = os.path.join(OUT_ROOT, "compare_front_train.csv")
TEST_CSV  = os.path.join(OUT_ROOT, "compare_test_points.csv")

INJECT_GBFS_FROM_FOLDERS = True

HV_REF = (1.0, 1.0)
IGD_P = 2

PLOT_FIGSIZE = (10, 6)
LEGEND_FONTSIZE = 8
LEGEND_OUTSIDE = True


# =========================
# Helpers: Pareto, HV, IGD
# =========================

def _as_float_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = np.nan_to_num(a, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return a


def nondominated_2d_min(fr: np.ndarray, er: np.ndarray, eps: float = 1e-12):
    fr = _as_float_array(fr)
    er = _as_float_array(er)
    if fr.size == 0:
        return np.array([]), np.array([])

    pts = np.column_stack([fr, er])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return np.array([]), np.array([])

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    kept = []
    best_er = np.inf
    last_fr = None

    for f, e in pts:
        if last_fr is None or abs(f - last_fr) > eps:
            if e + eps < best_er:
                kept.append((f, e))
                best_er = e
            last_fr = f
        else:
            if e + eps < best_er:
                kept.append((f, e))
                best_er = e

    kept = np.asarray(kept, dtype=float)
    if kept.size == 0:
        return np.array([]), np.array([])
    return kept[:, 0], kept[:, 1]


def _hv_max2d(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = _as_float_array(xs)
    ys = _as_float_array(ys)
    pts = np.column_stack([xs, ys])
    pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
    if pts.size == 0:
        return 0.0

    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in range(len(pts)):
            if i == j or not keep[j]:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
                (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])):
                keep[i] = False
                break

    pts = pts[keep]
    if pts.size == 0:
        return 0.0

    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    max_y = 0.0
    for x, y in pts:
        if y > max_y:
            hv += x * (y - max_y)
            max_y = y
    return float(hv)


def hypervolume_min2d(fr: np.ndarray, er: np.ndarray, ref=(1.0, 1.0)) -> float:
    fr = _as_float_array(fr)
    er = _as_float_array(er)
    if fr.size == 0:
        return 0.0
    xs = np.clip(ref[0] - fr, 0.0, None)
    ys = np.clip(ref[1] - er, 0.0, None)
    return _hv_max2d(xs, ys)


def igd_min2d(front_fr: np.ndarray, front_er: np.ndarray,
              ref_fr: np.ndarray, ref_er: np.ndarray, p: int = 2) -> float:
    front_fr = _as_float_array(front_fr)
    front_er = _as_float_array(front_er)
    ref_fr   = _as_float_array(ref_fr)
    ref_er   = _as_float_array(ref_er)

    if front_fr.size == 0 or ref_fr.size == 0:
        return float("nan")

    A = np.column_stack([front_fr, front_er])
    R = np.column_stack([ref_fr, ref_er])

    diff = R[:, None, :] - A[None, :, :]
    dist = np.linalg.norm(diff, ord=p, axis=2)
    dmin = np.min(dist, axis=1)
    return float(np.mean(dmin))

def _stable_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _method_category(method: str) -> str:
    m = (method or "").lower()
    if "gbfs_enhanced" in m:
        return "enhanced"
    if "gbfs_baseline" in m:
        return "baseline"
    return "other"


def _muted_color_cycle():
    return [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]


def _enhanced_color_cycle():
    return [
        "tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple",
        "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray",
    ]


def _build_other_method_style_map(methods: list[str]) -> dict[str, dict]:
    others = sorted([m for m in methods if _method_category(m) == "other"])
    colors = _muted_color_cycle()
    markers = ["o", "s", "^", "v", "<", ">", "P", "X", "d", "h"]

    mp: dict[str, dict] = {}
    for i, m in enumerate(others):
        mp[m] = dict(
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=1.7,
            markersize=4,
            alpha=0.7,
            zorder=2,
        )
    return mp


def _build_enhanced_color_map(methods: list[str]) -> dict[str, str]:
    enh = sorted([m for m in methods if _method_category(m) == "enhanced"])
    palette = _enhanced_color_cycle()
    mp: dict[str, str] = {}
    for i, m in enumerate(enh):
        mp[m] = palette[i % len(palette)]
    return mp


def _style_for_train(method_train: str,
                     other_map: dict[str, dict],
                     enhanced_color_map: dict[str, str]) -> dict:
    cat = _method_category(method_train)

    if cat == "enhanced":
        col = enhanced_color_map.get(method_train, "tab:red")
        return dict(color=col, linewidth=2.6, alpha=1.0, zorder=20)

    if cat == "baseline":
        return dict(color="deepskyblue", linewidth=2.4, alpha=0.7, zorder=12)

    st = other_map.get(method_train, dict(color="0.45", linewidth=1.7, alpha=0.7, zorder=2))
    return dict(
        color=st["color"],
        linewidth=st.get("linewidth", 1.7),
        alpha=0.7,
        zorder=st.get("zorder", 2)
    )


def _method_base(method: str) -> str:
    mlow = (method or "").lower()
    k = mlow.rfind("__post")
    if k >= 0:
        return method[:k]
    return method


def _post_tag(method_test: str) -> str:
    m = (method_test or "")
    mm = m.lower()
    k = mm.rfind("__post")
    if k < 0:
        return "nopost"
    return mm[k+2:]


def _marker_for_post(tag: str) -> str:
    t = (tag or "").lower()
    if t == "postbuddy":
        return "s"
    if t == "postnormal":
        return "o"
    if t.startswith("post"):
        return "^"
    return "D"


def _fmt(mean: float, std: float, nd: int = 3) -> str:
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f}±{std:.{nd}f}"


# =========================
# Folder ingest
# =========================

def _dataset_folder(ds: int) -> str:
    return os.path.join(OUT_ROOT, f"dataset_{ds:02d}")


def _ablation_path(ds: int, kind: str) -> str:
    return os.path.join(_dataset_folder(ds), f"gbfs_{kind}", f"{ds:02d}", "ablation_summary.csv")


def _ks_key_from_dir(ks_dir_name: str) -> str:
    name = (ks_dir_name or "").strip()
    name = re.sub(r"^ks[_\-]?", "", name)
    name = name.replace("_", "")
    return name.lower()


def _read_ablation_test_rows(ds: int, dataset_name: str, kind: str) -> pd.DataFrame:
    p = _ablation_path(ds, kind)
    if not os.path.exists(p):
        return pd.DataFrame()

    df = pd.read_csv(p)
    if "run" not in df.columns:
        return pd.DataFrame()

    run_num = pd.to_numeric(df["run"], errors="coerce")
    df = df[run_num.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["run"] = pd.to_numeric(df["run"], errors="coerce").astype(int)

    combo_keys = sorted({c[len("acc_"):] for c in df.columns if c.startswith("acc_")})
    rows = []

    prefix = f"GBFS_{kind}"

    for _, row in df.iterrows():
        r = int(row["run"])
        for ck in combo_keys:
            acc = row.get(f"acc_{ck}", np.nan)
            fnum = row.get(f"fnum_{ck}", np.nan)
            red  = row.get(f"red_{ck}", np.nan)
            ttot = row.get(f"time_{ck}", np.nan)

            if not np.isfinite(acc) or not np.isfinite(fnum):
                continue

            method = f"{prefix}__{ck}"
            rows.append({
                "dataset_idx": int(ds),
                "dataset": str(dataset_name),
                "method": method,
                "run": int(r),
                "test_mode": "selected",
                "acc_test": float(acc),
                "eRate_test": float(1.0 - float(acc)),
                "fnum": float(fnum),
                "red": float(red) if np.isfinite(red) else np.nan,
                "time_total": float(ttot) if np.isfinite(ttot) else np.nan,
            })

    return pd.DataFrame(rows)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == c.lower():
                return c
    return None


def _read_latest_gen_train_points(gen_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(gen_dir, "gen_*.csv"))
    if not files:
        return pd.DataFrame()

    def _gen_num(path: str) -> int:
        m = re.search(r"gen_(\d+)\.csv$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    files = sorted(files, key=_gen_num)
    last = files[-1]

    try:
        df = pd.read_csv(last)
    except Exception:
        return pd.DataFrame()

    cols_l = {c.lower().strip(): c for c in df.columns}

    if "obj1" in cols_l and "obj2" in cols_l:
        fr = pd.to_numeric(df[cols_l["obj1"]], errors="coerce")
        er = pd.to_numeric(df[cols_l["obj2"]], errors="coerce")
        out = pd.DataFrame({
            "fRatio": fr,
            "eRate_train_cv": er,
            "acc_train_cv": 1.0 - er,
            "fnum": np.nan,
        }).dropna(subset=["fRatio", "eRate_train_cv"])
        return out

    fr_col = _find_col(df, ["fRatio", "fratio", "f_ratio", "ratio"])
    er_col = _find_col(df, ["eRate_train_cv", "erate_train_cv", "eRate", "erate", "e_rate"])
    acc_col = _find_col(df, ["acc_train_cv", "acc_cv", "acc"])
    fn_col = _find_col(df, ["fnum", "nfeat", "num_features", "features"])

    if fr_col is None and df.shape[1] == 2:
        cA, cB = df.columns[0], df.columns[1]
        a = pd.to_numeric(df[cA], errors="coerce")
        b = pd.to_numeric(df[cB], errors="coerce")
        if a.notna().any() and b.notna().any():
            out = pd.DataFrame({
                "fRatio": a,
                "eRate_train_cv": b,
                "acc_train_cv": 1.0 - b,
                "fnum": np.nan,
            }).dropna(subset=["fRatio", "eRate_train_cv"])
            return out

    if fr_col is None:
        return pd.DataFrame()

    fr = pd.to_numeric(df[fr_col], errors="coerce")

    if er_col is not None:
        er = pd.to_numeric(df[er_col], errors="coerce")
        acc = 1.0 - er
    elif acc_col is not None:
        acc = pd.to_numeric(df[acc_col], errors="coerce")
        er = 1.0 - acc
    else:
        return pd.DataFrame()

    out = pd.DataFrame({
        "fRatio": fr,
        "eRate_train_cv": er,
        "acc_train_cv": acc,
    })

    out["fnum"] = pd.to_numeric(df[fn_col], errors="coerce") if fn_col is not None else np.nan
    out = out.dropna(subset=["fRatio", "eRate_train_cv"])
    return out


def _scan_gbfs_train_rows(ds: int, dataset_name: str, kind: str) -> pd.DataFrame:
    root = os.path.join(_dataset_folder(ds), f"gbfs_{kind}", f"{ds:02d}")
    if not os.path.isdir(root):
        return pd.DataFrame()

    prefix = f"GBFS_{kind}"
    rows = []

    run_dirs = sorted([d for d in os.listdir(root) if d.lower().startswith("run_")])
    for rd in run_dirs:
        m = re.search(r"run_(\d+)", rd, flags=re.IGNORECASE)
        if not m:
            continue
        run_id = int(m.group(1))
        run_path = os.path.join(root, rd)

        init_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
        for init_mode in init_dirs:
            init_path = os.path.join(run_path, init_mode)

            ks_dirs = [d for d in os.listdir(init_path) if os.path.isdir(os.path.join(init_path, d))]
            for ks_dir in ks_dirs:
                ks_path = os.path.join(init_path, ks_dir)
                if not os.path.basename(ks_path).lower().startswith("ks"):
                    continue

                ks_key = _ks_key_from_dir(ks_dir)
                method = f"{prefix}__{init_mode.lower()}__ks{ks_key}"

                dfp = _read_latest_gen_train_points(ks_path)
                if dfp.empty:
                    print(f"[WARN] Skip TRAIN (no usable gen_*.csv columns) : {method} | {ks_path}")
                    continue

                for _, r in dfp.iterrows():
                    rows.append({
                        "dataset_idx": int(ds),
                        "dataset": str(dataset_name),
                        "method": str(method),
                        "run": int(run_id),
                        "fRatio": float(r["fRatio"]),
                        "eRate_train_cv": float(r["eRate_train_cv"]),
                        "acc_train_cv": float(r["acc_train_cv"]),
                        "fnum": float(r.get("fnum", np.nan)),
                    })

    return pd.DataFrame(rows)


def _inject_gbfs(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_tr is None or df_te is None:
        return df_tr, df_te

    ds_list = sorted(set(
        pd.to_numeric(df_tr.get("dataset_idx"), errors="coerce").dropna().astype(int).tolist()
        + pd.to_numeric(df_te.get("dataset_idx"), errors="coerce").dropna().astype(int).tolist()
    ))

    def _is_gbfs(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.contains("gbfs_", na=False)

    df_tr_keep = df_tr[~_is_gbfs(df_tr["method"])].copy() if "method" in df_tr.columns else df_tr.copy()
    df_te_keep = df_te[~_is_gbfs(df_te["method"])].copy() if "method" in df_te.columns else df_te.copy()

    gbfs_tr_all = []
    gbfs_te_all = []

    for ds in ds_list:
        ds_name = None
        t = df_tr[df_tr["dataset_idx"] == ds] if "dataset_idx" in df_tr.columns else pd.DataFrame()
        if (not t.empty) and ("dataset" in t.columns):
            ds_name = str(t["dataset"].iloc[0])
        u = df_te[df_te["dataset_idx"] == ds] if "dataset_idx" in df_te.columns else pd.DataFrame()
        if ds_name is None and (not u.empty) and ("dataset" in u.columns):
            ds_name = str(u["dataset"].iloc[0])
        if ds_name is None:
            ds_name = f"dataset_{ds:02d}"

        gbfs_te_all.append(_read_ablation_test_rows(ds, ds_name, "enhanced"))
        gbfs_te_all.append(_read_ablation_test_rows(ds, ds_name, "baseline"))

        gbfs_tr_all.append(_scan_gbfs_train_rows(ds, ds_name, "enhanced"))
        gbfs_tr_all.append(_scan_gbfs_train_rows(ds, ds_name, "baseline"))

    parts_tr = [d for d in gbfs_tr_all if isinstance(d, pd.DataFrame) and (not d.empty)]
    parts_te = [d for d in gbfs_te_all if isinstance(d, pd.DataFrame) and (not d.empty)]
    df_tr_gbfs = pd.concat(parts_tr, ignore_index=True) if parts_tr else pd.DataFrame()
    df_te_gbfs = pd.concat(parts_te, ignore_index=True) if parts_te else pd.DataFrame()

    df_tr_out = pd.concat([df_tr_keep, df_tr_gbfs], ignore_index=True) if (not df_tr_gbfs.empty) else df_tr_keep
    df_te_out = pd.concat([df_te_keep, df_te_gbfs], ignore_index=True) if (not df_te_gbfs.empty) else df_te_keep
    return df_tr_out, df_te_out


# =========================
# Main
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(TRAIN_CSV)
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(TEST_CSV)

    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    if INJECT_GBFS_FROM_FOLDERS:
        df_tr, df_te = _inject_gbfs(df_tr, df_te)

    need_cols_tr = {"dataset_idx", "dataset", "method", "run", "fRatio", "eRate_train_cv", "acc_train_cv", "fnum"}
    missing_tr = need_cols_tr - set(df_tr.columns)
    if missing_tr:
        raise ValueError(f"TRAIN csv missing columns (after inject): {missing_tr}")

    need_cols_te = {"dataset_idx", "dataset", "method", "run", "test_mode", "acc_test", "fnum"}
    missing_te = need_cols_te - set(df_te.columns)
    if missing_te:
        raise ValueError(f"TEST csv missing columns (after inject): {missing_te}")

    for c in ["dataset_idx", "run", "fRatio", "eRate_train_cv", "acc_train_cv", "fnum"]:
        df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce")
    for c in ["dataset_idx", "run", "acc_test", "fnum"]:
        df_te[c] = pd.to_numeric(df_te[c], errors="coerce")

    all_summary_train = []
    all_summary_test = []
    all_summary_join = []
    all_summary_join_base = []

    datasets = sorted(df_tr["dataset_idx"].dropna().unique().astype(int).tolist())

    for ds in datasets:
        df_tr_ds = df_tr[df_tr["dataset_idx"] == ds].copy()
        if df_tr_ds.empty:
            continue

        dataset_name = str(df_tr_ds["dataset"].iloc[0])
        print(f"\n=== DATASET {ds:02d} | {dataset_name} ===")

        fr_all = df_tr_ds["fRatio"].to_numpy(dtype=float)
        er_all = df_tr_ds["eRate_train_cv"].to_numpy(dtype=float)
        ref_fr, ref_er = nondominated_2d_min(fr_all, er_all)
        print(f"Reference front size (TRAIN, global ND): {len(ref_fr)}")

        records = []
        grouped = df_tr_ds.groupby(["method", "run"], dropna=False)

        for (method, run_id), g in grouped:
            g = g.dropna(subset=["fRatio", "eRate_train_cv"])
            fr = g["fRatio"].to_numpy(dtype=float)
            er = g["eRate_train_cv"].to_numpy(dtype=float)

            fr_nd, er_nd = nondominated_2d_min(fr, er)
            hv = hypervolume_min2d(fr_nd, er_nd, ref=HV_REF)
            igd = igd_min2d(fr_nd, er_nd, ref_fr, ref_er, p=IGD_P)

            if len(er) > 0:
                idx_best = int(np.nanargmin(er))
                best_er = float(er[idx_best])
                best_acc = float(1.0 - best_er)
                best_fr = float(fr[idx_best])
                best_fnum = float(g["fnum"].iloc[idx_best]) if "fnum" in g.columns else float("nan")
            else:
                best_er = 1.0
                best_acc = 0.0
                best_fr = float("nan")
                best_fnum = float("nan")

            records.append({
                "dataset_idx": ds,
                "dataset": dataset_name,
                "method": str(method),
                "run": int(run_id) if not pd.isna(run_id) else -1,
                "hv_train": float(hv),
                "igd_train": float(igd),
                "best_acc_train": float(best_acc),
                "best_eRate_train": float(best_er),
                "best_fRatio_train": float(best_fr),
                "best_fnum_train": float(best_fnum),
                "front_size_nd": int(len(fr_nd)),
            })

        df_metrics = pd.DataFrame(records)
        if df_metrics.empty:
            print("No train metrics computed.")
            continue

        agg = df_metrics.groupby("method").agg(
            hv_mean=("hv_train", "mean"),
            hv_std=("hv_train", "std"),
            igd_mean=("igd_train", "mean"),
            igd_std=("igd_train", "std"),
            best_acc_mean=("best_acc_train", "mean"),
            best_acc_std=("best_acc_train", "std"),
            best_fnum_mean=("best_fnum_train", "mean"),
            best_fnum_std=("best_fnum_train", "std"),
            runs=("run", "nunique"),
        ).reset_index()

        agg["dataset_idx"] = ds
        agg["dataset"] = dataset_name
        agg = agg.sort_values(["hv_mean", "igd_mean"], ascending=[False, True]).reset_index(drop=True)

        out_train_summary = os.path.join(OUT_DIR, f"summary_train_hv_igd_dataset_{ds:02d}.csv")
        agg.to_csv(out_train_summary, index=False)
        all_summary_train.append(agg)
        print(f"Saved: {out_train_summary}")

        df_te_ds = df_te[df_te["dataset_idx"] == ds].copy()

        df_test_sum = pd.DataFrame()
        if not df_te_ds.empty:
            df_te_ds["test_mode"] = df_te_ds["test_mode"].astype(str)

            test_records = []
            for method, gm in df_te_ds.groupby("method"):
                method = str(method)
                gm = gm.dropna(subset=["acc_test"])

                per_run = []
                for run_id, gr in gm.groupby("run"):
                    gr = gr.copy()

                    if method.lower().startswith("gbfs"):
                        cand = gr[gr["test_mode"] == "selected"]
                        if cand.empty:
                            cand = gr
                        pick = cand.iloc[int(np.nanargmax(cand["acc_test"].to_numpy(dtype=float)))]
                    else:
                        cand = gr[gr["test_mode"] == "k_ref"]
                        if cand.empty:
                            cand = gr[gr["test_mode"] == "ratio"]
                        if cand.empty:
                            cand = gr
                        pick = cand.iloc[int(np.nanargmax(cand["acc_test"].to_numpy(dtype=float)))]

                    per_run.append({
                        "dataset_idx": ds,
                        "dataset": dataset_name,
                        "method": method,
                        "run": int(run_id) if not pd.isna(run_id) else -1,
                        "picked_test_mode": str(pick.get("test_mode", "")),
                        "acc_test": float(pick.get("acc_test", 0.0)),
                        "fnum_test": float(pick.get("fnum", float("nan"))),
                    })

                pr = pd.DataFrame(per_run)
                if pr.empty:
                    continue

                test_records.append({
                    "dataset_idx": ds,
                    "dataset": dataset_name,
                    "method": method,
                    "method_base": _method_base(method),
                    "post_tag": _post_tag(method),
                    "runs": int(pr["run"].nunique()),
                    "acc_test_mean": float(pr["acc_test"].mean()),
                    "acc_test_std": float(pr["acc_test"].std()),
                    "fnum_test_mean": float(pr["fnum_test"].mean()),
                    "fnum_test_std": float(pr["fnum_test"].std()),
                    "mode_note": "GBFS:selected | others:k_ref->ratio->best",
                })

            df_test_sum = pd.DataFrame(test_records).sort_values(
                ["acc_test_mean"], ascending=[False]
            ).reset_index(drop=True)

            out_test_summary = os.path.join(OUT_DIR, f"summary_test_dataset_{ds:02d}.csv")
            df_test_sum.to_csv(out_test_summary, index=False)
            all_summary_test.append(df_test_sum)
            print(f"Saved: {out_test_summary}")

            joined = agg.merge(
                df_test_sum.drop(columns=["method_base", "post_tag"]),
                on=["dataset_idx", "dataset", "method"],
                how="left",
                suffixes=("", "_test"),
            )
            out_join = os.path.join(OUT_DIR, f"summary_joined_dataset_{ds:02d}.csv")
            joined.to_csv(out_join, index=False)
            all_summary_join.append(joined)
            print(f"Saved: {out_join}")

            train_base = agg.copy().rename(columns={"method": "method_train"})
            train_base["method_base"] = train_base["method_train"].map(_method_base)

            test_keep = df_test_sum.copy().rename(columns={"method": "method_test", "runs": "runs_test"})

            joined_base = train_base.merge(
                test_keep,
                on=["dataset_idx", "dataset", "method_base"],
                how="left",
                suffixes=("_train", "_test"),
            )

            out_join_base = os.path.join(OUT_DIR, f"summary_joined_by_base_dataset_{ds:02d}.csv")
            joined_base.to_csv(out_join_base, index=False)
            all_summary_join_base.append(joined_base)
            print(f"Saved: {out_join_base}")

        chosen = []
        for m, gm in df_metrics.groupby("method"):
            hv_arr = gm["hv_train"].to_numpy(dtype=float)
            if hv_arr.size == 0 or np.all(~np.isfinite(hv_arr)):
                continue
            med = float(np.nanmedian(hv_arr))
            idx = int(np.nanargmin(np.abs(hv_arr - med)))
            chosen_run = int(gm.iloc[idx]["run"])
            chosen.append((str(m), chosen_run))

        all_methods = sorted(df_tr_ds["method"].dropna().astype(str).unique().tolist())
        other_map = _build_other_method_style_map(all_methods)
        enhanced_color_map = _build_enhanced_color_map(all_methods)   # <<< new

        test_by_base: dict[str, list[dict]] = {}
        if df_test_sum is not None and (not df_test_sum.empty):
            for _, r in df_test_sum.iterrows():
                b = str(r["method_base"])
                test_by_base.setdefault(b, []).append({
                    "method_test": str(r["method"]),
                    "post_tag": str(r.get("post_tag", "nopost")),
                    "acc_test_mean": float(r.get("acc_test_mean", np.nan)),
                    "acc_test_std": float(r.get("acc_test_std", np.nan)),
                })

            def _rank_tag(tag: str) -> int:
                t = (tag or "").lower()
                if t == "postnormal":
                    return 0
                if t == "postbuddy":
                    return 1
                if t == "nopost":
                    return 2
                return 3

            for b in list(test_by_base.keys()):
                test_by_base[b] = sorted(
                    test_by_base[b],
                    key=lambda d: (_rank_tag(d.get("post_tag", "")), d.get("method_test", ""))
                )

        def _draw_order_key(item):
            method = (item[0] or "").lower()
            cat = _method_category(method)
            if cat == "enhanced":
                return 2
            if cat == "baseline":
                return 1
            return 0

        chosen_sorted = sorted(chosen, key=_draw_order_key)

        plt.figure(figsize=PLOT_FIGSIZE)
        legend_handles = []
        legend_labels = []

        for method_train, run_id in chosen_sorted:
            g = df_tr_ds[(df_tr_ds["method"] == method_train) & (df_tr_ds["run"] == run_id)]
            g = g.dropna(subset=["fRatio", "eRate_train_cv"])
            fr = g["fRatio"].to_numpy(dtype=float)
            er = g["eRate_train_cv"].to_numpy(dtype=float)
            fr_nd, er_nd = nondominated_2d_min(fr, er)
            if len(fr_nd) == 0:
                continue

            st = _style_for_train(method_train, other_map, enhanced_color_map)

            plt.plot(fr_nd, er_nd, color=st["color"], linewidth=st["linewidth"], alpha=st["alpha"], zorder=st["zorder"])
            plt.scatter(
                fr_nd, er_nd,
                s=55 if _method_category(method_train) in ["enhanced", "baseline"] else 20,
                c=st["color"],
                alpha=st["alpha"],
                edgecolors="black" if _method_category(method_train) in ["enhanced", "baseline"] else st["color"],
                linewidths=0.8 if _method_category(method_train) in ["enhanced", "baseline"] else 0.0,
                zorder=st["zorder"] + 1,
            )

            base = _method_base(method_train)
            variants = test_by_base.get(base, [])

            if (_method_category(method_train) in ["enhanced", "baseline"]) and variants:
                for v in variants:
                    tag = v.get("post_tag", "nopost")
                    mk = _marker_for_post(tag)
                    acc_txt = _fmt(v.get("acc_test_mean", np.nan), v.get("acc_test_std", np.nan), nd=3)
                    lab = f"{v.get('method_test', base)} | acc_test={acc_txt}"

                    legend_handles.append(
                        Line2D([0], [0],
                               color=st["color"], lw=st["linewidth"], alpha=st["alpha"],
                               marker=mk, markersize=7,
                               markerfacecolor=st["color"],
                               markeredgecolor="black")
                    )
                    legend_labels.append(lab)
            else:
                legend_handles.append(
                    Line2D([0], [0],
                           color=st["color"], lw=st["linewidth"], alpha=st["alpha"],
                           marker="o", markersize=5,
                           markerfacecolor=st["color"],
                           markeredgecolor="black" if _method_category(method_train) in ["enhanced", "baseline"] else st["color"])
                )
                legend_labels.append(f"{method_train} (HV~median)")

        plt.xlabel("fRatio (#selected / #total)")
        plt.ylabel("eRate_train_cv (1 - CV accuracy)")
        plt.title(f"Median-HV fronts (TRAIN) | Dataset {ds:02d} - {dataset_name}")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if legend_handles:
            if LEGEND_OUTSIDE:
                plt.legend(
                    legend_handles, legend_labels,
                    fontsize=LEGEND_FONTSIZE,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                )
                plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
            else:
                plt.legend(legend_handles, legend_labels, fontsize=LEGEND_FONTSIZE)
                plt.tight_layout()
        else:
            plt.tight_layout()

        fig_path = os.path.join(OUT_DIR, f"front_medianHV_dataset_{ds:02d}.png")
        plt.savefig(fig_path, dpi=220)
        plt.close()
        print(f"Saved: {fig_path}")

    if all_summary_train:
        df_all_train = pd.concat(all_summary_train, ignore_index=True)
        p = os.path.join(OUT_DIR, "summary_train_hv_igd.csv")
        df_all_train.to_csv(p, index=False)
        print("\nSaved:", p)

    if all_summary_test:
        df_all_test = pd.concat(all_summary_test, ignore_index=True)
        p = os.path.join(OUT_DIR, "summary_test.csv")
        df_all_test.to_csv(p, index=False)
        print("Saved:", p)

    if all_summary_join:
        df_all_join = pd.concat(all_summary_join, ignore_index=True)
        p = os.path.join(OUT_DIR, "summary_joined.csv")
        df_all_join.to_csv(p, index=False)
        print("Saved:", p)

    if all_summary_join_base:
        df_all_join_base = pd.concat(all_summary_join_base, ignore_index=True)
        p = os.path.join(OUT_DIR, "summary_joined_by_base.csv")
        df_all_join_base.to_csv(p, index=False)
        print("Saved:", p)

    print("\nDONE.")
    print("Outputs folder:", OUT_DIR)


if __name__ == "__main__":
    main()
