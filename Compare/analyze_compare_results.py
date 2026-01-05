#!/usr/bin/env python3
from __future__ import annotations
import os, re, hashlib, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Pareto / HV / IGD helpers
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
                kept.append((f, e)); best_er = e
            last_fr = f
        else:
            if e + eps < best_er:
                kept.append((f, e)); best_er = e
    kept = np.asarray(kept, dtype=float)
    if kept.size == 0:
        return np.array([]), np.array([])
    return kept[:, 0], kept[:, 1]

def _hv_max2d(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = _as_float_array(xs); ys = _as_float_array(ys)
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
    fr = _as_float_array(fr); er = _as_float_array(er)
    if fr.size == 0:
        return 0.0
    xs = np.clip(ref[0] - fr, 0.0, None)
    ys = np.clip(ref[1] - er, 0.0, None)
    return _hv_max2d(xs, ys)

def igd_min2d(front_fr: np.ndarray, front_er: np.ndarray,
              ref_fr: np.ndarray, ref_er: np.ndarray, p: int = 2) -> float:
    front_fr = _as_float_array(front_fr); front_er = _as_float_array(front_er)
    ref_fr   = _as_float_array(ref_fr);   ref_er   = _as_float_array(ref_er)
    if front_fr.size == 0 or ref_fr.size == 0:
        return float("nan")
    A = np.column_stack([front_fr, front_er])
    R = np.column_stack([ref_fr, ref_er])
    diff = R[:, None, :] - A[None, :, :]
    dist = np.linalg.norm(diff, ord=p, axis=2)
    dmin = np.min(dist, axis=1)
    return float(np.mean(dmin))

# =========================
# Styling / method helpers
# =========================

def _method_category(method: str) -> str:
    m = (method or "").lower()
    if "gbfs_enhanced" in m:
        return "enhanced"
    if "gbfs_baseline" in m:
        return "baseline"
    return "other"

def _muted_color_cycle():
    return [
        "tab:blue", "tab:orange", "tab:green", "tab:purple",
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
        mp[m] = dict(color=colors[i % len(colors)],
                     marker=markers[i % len(markers)],
                     linewidth=1.6, markersize=4, alpha=0.65, zorder=2)
    return mp

def _build_enhanced_color_map(methods: list[str]) -> dict[str, str]:
    enh = sorted([m for m in methods if _method_category(m) == "enhanced"])
    palette = _enhanced_color_cycle()
    return {m: palette[i % len(palette)] for i, m in enumerate(enh)}

def _style_for_train(method_train: str,
                     other_map: dict[str, dict],
                     enhanced_color_map: dict[str, str]) -> dict:
    cat = _method_category(method_train)
    if cat == "enhanced":
        col = enhanced_color_map.get(method_train, "tab:red")
        return dict(color=col, linewidth=2.6, alpha=1.0, zorder=20)
    if cat == "baseline":
        return dict(color="deepskyblue", linewidth=2.3, alpha=0.75, zorder=12)
    st = other_map.get(method_train, dict(color="0.45", linewidth=1.6, alpha=0.65, zorder=2))
    return dict(color=st["color"], linewidth=st.get("linewidth", 1.6), alpha=st.get("alpha", 0.65), zorder=st.get("zorder", 2))

def _method_base(method: str) -> str:
    # remove __post... suffix if present
    mlow = (method or "").lower()
    k = mlow.rfind("__post")
    if k >= 0:
        return method[:k]
    return method

def _post_tag(method_test: str, post_mode: str | None = None) -> str:
    if post_mode is not None and str(post_mode).strip() != "" and str(post_mode).lower() != "nan":
        return f"post{str(post_mode).lower()}"
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

def _ensure_cols(df: pd.DataFrame, required: set[str], name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

# =========================
# Ablation (GBFS enhanced)
# =========================

def _ablation_enhanced(df_te_ds: pd.DataFrame) -> pd.DataFrame:
    if df_te_ds is None or df_te_ds.empty:
        return pd.DataFrame()
    m = df_te_ds["method"].astype(str).str.lower()
    df = df_te_ds[m.str.contains("gbfs_enhanced", na=False)].copy()
    if df.empty:
        return pd.DataFrame()

    # normalize
    for c in ["init_mode", "ks_mode", "post_mode", "test_mode"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)

    df["acc_test"] = pd.to_numeric(df["acc_test"], errors="coerce")
    df["fnum"] = pd.to_numeric(df.get("fnum", np.nan), errors="coerce")
    df["red"] = pd.to_numeric(df.get("red", np.nan), errors="coerce")
    df["time_total"] = pd.to_numeric(df.get("time_total", np.nan), errors="coerce")

    # keep selected only if exists
    if (df["test_mode"].str.lower() == "selected").any():
        df = df[df["test_mode"].str.lower() == "selected"].copy()

    # per-run pick by (init, ks, post): there is typically 1 row already
    grp = df.groupby(["init_mode", "ks_mode", "post_mode"], dropna=False)
    out = grp.agg(
        runs=("run", "nunique") if "run" in df.columns else ("acc_test", "size"),
        acc_mean=("acc_test", "mean"),
        acc_std=("acc_test", "std"),
        fnum_mean=("fnum", "mean"),
        fnum_std=("fnum", "std"),
        red_mean=("red", "mean"),
        red_std=("red", "std"),
        time_total_mean=("time_total", "mean"),
        time_total_std=("time_total", "std"),
    ).reset_index()

    # rank by acc
    out = out.sort_values(["acc_mean", "fnum_mean"], ascending=[False, True]).reset_index(drop=True)
    return out

# =========================
# Main analysis
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="Folder containing compare_front_train.csv and compare_test_points.csv")
    ap.add_argument("--analysis_subfolder", type=str, default="analysis_results_runone")
    ap.add_argument("--hv_ref_x", type=float, default=1.0)
    ap.add_argument("--hv_ref_y", type=float, default=1.0)
    ap.add_argument("--igd_p", type=int, default=2)
    ap.add_argument("--plot_fig_w", type=float, default=10.0)
    ap.add_argument("--plot_fig_h", type=float, default=6.0)
    ap.add_argument("--legend_fontsize", type=int, default=8)
    ap.add_argument("--legend_outside", action="store_true")
    args = ap.parse_args()

    OUT_ROOT = os.path.abspath(args.out_root)
    OUT_DIR = os.path.join(OUT_ROOT, args.analysis_subfolder)
    os.makedirs(OUT_DIR, exist_ok=True)

    TRAIN_CSV = os.path.join(OUT_ROOT, "compare_front_train.csv")
    TEST_CSV  = os.path.join(OUT_ROOT, "compare_test_points.csv")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(TRAIN_CSV)
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(TEST_CSV)

    df_tr = pd.read_csv(TRAIN_CSV, low_memory=True)
    df_te = pd.read_csv(TEST_CSV, low_memory=True)

    # Ensure essentials exist (allow extra columns)
    _ensure_cols(df_tr, {"dataset_idx", "dataset", "method", "run", "fRatio"}, "TRAIN")
    if "eRate_train_cv" not in df_tr.columns:
        if "acc_train_cv" in df_tr.columns:
            df_tr["acc_train_cv"] = pd.to_numeric(df_tr["acc_train_cv"], errors="coerce")
            df_tr["eRate_train_cv"] = 1.0 - df_tr["acc_train_cv"]
        else:
            raise ValueError("TRAIN needs eRate_train_cv or acc_train_cv")
    if "acc_train_cv" not in df_tr.columns:
        df_tr["eRate_train_cv"] = pd.to_numeric(df_tr["eRate_train_cv"], errors="coerce")
        df_tr["acc_train_cv"] = 1.0 - df_tr["eRate_train_cv"]

    _ensure_cols(df_te, {"dataset_idx", "dataset", "method", "run", "acc_test", "test_mode"}, "TEST")
    if "fnum" not in df_te.columns:
        df_te["fnum"] = np.nan

    # numeric cast
    for c in ["dataset_idx", "run"]:
        df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce")
        df_te[c] = pd.to_numeric(df_te[c], errors="coerce")
    for c in ["fRatio", "eRate_train_cv", "acc_train_cv", "fnum"]:
        if c in df_tr.columns:
            df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce")
    for c in ["acc_test", "fnum", "red", "time_total"]:
        if c in df_te.columns:
            df_te[c] = pd.to_numeric(df_te[c], errors="coerce")

    # drop nonsense
    df_tr = df_tr.dropna(subset=["dataset_idx", "method", "run", "fRatio", "eRate_train_cv"]).copy()
    df_te = df_te.dropna(subset=["dataset_idx", "method", "run", "acc_test"]).copy()

    HV_REF = (args.hv_ref_x, args.hv_ref_y)
    IGD_P = int(args.igd_p)

    all_summary_train = []
    all_summary_test = []
    all_summary_join = []
    all_summary_join_base = []
    all_ablation = []

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

        # per (method, run) metrics
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
                best_er = 1.0; best_acc = 0.0; best_fr = float("nan"); best_fnum = float("nan")

            records.append({
                "dataset_idx": int(ds),
                "dataset": dataset_name,
                "method": str(method),
                "run": int(run_id),
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

        agg["dataset_idx"] = int(ds)
        agg["dataset"] = dataset_name
        agg = agg.sort_values(["hv_mean", "igd_mean"], ascending=[False, True]).reset_index(drop=True)

        out_train_summary = os.path.join(OUT_DIR, f"summary_train_hv_igd_dataset_{ds:02d}.csv")
        agg.to_csv(out_train_summary, index=False)
        all_summary_train.append(agg)
        print(f"Saved: {out_train_summary}")

        # =========================
        # TEST summary
        # =========================
        df_te_ds = df_te[df_te["dataset_idx"] == ds].copy()
        df_test_sum = pd.DataFrame()

        if not df_te_ds.empty:
            df_te_ds["test_mode"] = df_te_ds["test_mode"].astype(str)

            # build method_test with post tag if exists
            if "post_mode" in df_te_ds.columns:
                pm = df_te_ds["post_mode"].astype(str)
                use = pm.notna() & (pm.str.strip() != "") & (pm.str.lower() != "nan")
                df_te_ds["method_test"] = df_te_ds["method"].astype(str)
                df_te_ds.loc[use, "method_test"] = df_te_ds.loc[use, "method"].astype(str) + "__post" + df_te_ds.loc[use, "post_mode"].astype(str).str.lower()
            else:
                df_te_ds["method_test"] = df_te_ds["method"].astype(str)

            df_te_ds["method_base"] = df_te_ds["method_test"].map(_method_base)
            df_te_ds["post_tag"] = [
                _post_tag(mt, pm) if "post_mode" in df_te_ds.columns else _post_tag(mt, None)
                for mt, pm in zip(df_te_ds["method_test"].astype(str).tolist(),
                                  df_te_ds.get("post_mode", pd.Series([None]*len(df_te_ds))).tolist())
            ]

            # pick per run per method_test
            test_records = []
            for method_test, gm in df_te_ds.groupby("method_test"):
                method_test = str(method_test)
                gm = gm.dropna(subset=["acc_test"])
                per_run = []
                for run_id, gr in gm.groupby("run"):
                    gr = gr.copy()
                    cat = _method_category(str(gr["method"].iloc[0]))
                    if cat in ["enhanced", "baseline"]:
                        cand = gr[gr["test_mode"].astype(str).str.lower() == "selected"]
                        if cand.empty:
                            cand = gr
                        pick = cand.iloc[int(np.nanargmax(cand["acc_test"].to_numpy(dtype=float)))]
                    else:
                        cand = gr[gr["test_mode"].astype(str).str.lower() == "k_ref"]
                        if cand.empty:
                            cand = gr[gr["test_mode"].astype(str).str.lower() == "ratio"]
                        if cand.empty:
                            cand = gr
                        pick = cand.iloc[int(np.nanargmax(cand["acc_test"].to_numpy(dtype=float)))]

                    per_run.append({
                        "dataset_idx": int(ds),
                        "dataset": dataset_name,
                        "method": method_test,
                        "method_base": _method_base(method_test),
                        "post_tag": _post_tag(method_test, pick.get("post_mode", None)),
                        "run": int(run_id),
                        "picked_test_mode": str(pick.get("test_mode", "")),
                        "acc_test": float(pick.get("acc_test", 0.0)),
                        "fnum_test": float(pick.get("fnum", float("nan"))),
                        "red_test": float(pick.get("red", float("nan"))) if "red" in pick else float("nan"),
                        "time_total": float(pick.get("time_total", float("nan"))) if "time_total" in pick else float("nan"),
                    })

                pr = pd.DataFrame(per_run)
                if pr.empty:
                    continue
                test_records.append({
                    "dataset_idx": int(ds),
                    "dataset": dataset_name,
                    "method": method_test,
                    "method_base": _method_base(method_test),
                    "post_tag": pr["post_tag"].iloc[0],
                    "runs": int(pr["run"].nunique()),
                    "acc_test_mean": float(pr["acc_test"].mean()),
                    "acc_test_std": float(pr["acc_test"].std()),
                    "fnum_test_mean": float(pr["fnum_test"].mean()),
                    "fnum_test_std": float(pr["fnum_test"].std()),
                    "red_test_mean": float(pr["red_test"].mean()),
                    "red_test_std": float(pr["red_test"].std()),
                    "time_total_mean": float(pr["time_total"].mean()),
                    "time_total_std": float(pr["time_total"].std()),
                    "mode_note": "GBFS:selected | others:k_ref->ratio->best",
                })

            df_test_sum = pd.DataFrame(test_records).sort_values(
                ["acc_test_mean"], ascending=[False]
            ).reset_index(drop=True)

            out_test_summary = os.path.join(OUT_DIR, f"summary_test_dataset_{ds:02d}.csv")
            df_test_sum.to_csv(out_test_summary, index=False)
            all_summary_test.append(df_test_sum)
            print(f"Saved: {out_test_summary}")

            # Join (exact)
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

            # Join by base
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

            # Ablation (GBFS enhanced only)
            abl = _ablation_enhanced(df_te_ds)
            if not abl.empty:
                abl.insert(0, "dataset_idx", int(ds))
                abl.insert(1, "dataset", dataset_name)
                out_abl = os.path.join(OUT_DIR, f"ablation_gbfs_enhanced_dataset_{ds:02d}.csv")
                abl.to_csv(out_abl, index=False)
                all_ablation.append(abl)
                print(f"Saved: {out_abl}")

        # =========================
        # Plot median-HV fronts (TRAIN)
        # =========================
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
        enhanced_color_map = _build_enhanced_color_map(all_methods)

        # For legend: show test variants per base (method_base)
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
                if t == "postnormal": return 0
                if t == "postbuddy":  return 1
                if t == "nopost":     return 2
                return 3
            for b in list(test_by_base.keys()):
                test_by_base[b] = sorted(test_by_base[b], key=lambda d: (_rank_tag(d.get("post_tag","")), d.get("method_test","")))

        def _draw_order_key(item):
            method = (item[0] or "").lower()
            cat = _method_category(method)
            if cat == "enhanced": return 2
            if cat == "baseline": return 1
            return 0

        chosen_sorted = sorted(chosen, key=_draw_order_key)

        plt.figure(figsize=(args.plot_fig_w, args.plot_fig_h))
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
            plt.scatter(fr_nd, er_nd,
                        s=55 if _method_category(method_train) in ["enhanced", "baseline"] else 18,
                        c=st["color"], alpha=st["alpha"],
                        edgecolors="black" if _method_category(method_train) in ["enhanced", "baseline"] else st["color"],
                        linewidths=0.8 if _method_category(method_train) in ["enhanced", "baseline"] else 0.0,
                        zorder=st["zorder"] + 1)

            base = _method_base(method_train)
            variants = test_by_base.get(base, [])

            if (_method_category(method_train) in ["enhanced", "baseline"]) and variants:
                for v in variants:
                    tag = v.get("post_tag", "nopost")
                    mk = _marker_for_post(tag)
                    acc_txt = _fmt(v.get("acc_test_mean", np.nan), v.get("acc_test_std", np.nan), nd=3)
                    lab = f"{v.get('method_test', base)} | acc_test={acc_txt}"

                    legend_handles.append(Line2D([0], [0],
                        color=st["color"], lw=st["linewidth"], alpha=st["alpha"],
                        marker=mk, markersize=7,
                        markerfacecolor=st["color"], markeredgecolor="black"))
                    legend_labels.append(lab)
            else:
                legend_handles.append(Line2D([0], [0],
                    color=st["color"], lw=st["linewidth"], alpha=st["alpha"],
                    marker="o", markersize=5,
                    markerfacecolor=st["color"],
                    markeredgecolor="black" if _method_category(method_train) in ["enhanced", "baseline"] else st["color"]))
                legend_labels.append(f"{method_train} (HV~median)")

        plt.xlabel("fRatio (#selected / #total)")
        plt.ylabel("eRate_train_cv (1 - CV accuracy)")
        plt.title(f"Median-HV fronts (TRAIN) | Dataset {ds:02d} - {dataset_name}")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        if legend_handles:
            if args.legend_outside:
                plt.legend(legend_handles, legend_labels, fontsize=args.legend_fontsize,
                           loc="upper left", bbox_to_anchor=(1.02, 1.0),
                           borderaxespad=0.0, frameon=True)
                plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
            else:
                plt.legend(legend_handles, legend_labels, fontsize=args.legend_fontsize)
                plt.tight_layout()
        else:
            plt.tight_layout()

        fig_path = os.path.join(OUT_DIR, f"front_medianHV_dataset_{ds:02d}.png")
        plt.savefig(fig_path, dpi=220)
        plt.close()
        print(f"Saved: {fig_path}")

    # aggregate outputs
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

    if all_ablation:
        df_all_abl = pd.concat(all_ablation, ignore_index=True)
        p = os.path.join(OUT_DIR, "ablation_gbfs_enhanced_all.csv")
        df_all_abl.to_csv(p, index=False)
        print("Saved:", p)

    print("\nDONE.")
    print("Outputs folder:", OUT_DIR)

if __name__ == "__main__":
    main()

# python .\Compare\analyze_compare_results.py --out_root "D:\PhD\The First Paper\Code Implement\GBFS-SND\_local_out" --legend_outside
