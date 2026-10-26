#!/usr/bin/env python3
"""
Use the Moran's I / Geary's C metrics as features for classification.

Reads the .txt files produced by spatial_metrics.py:

    metrics/<queen|rook>/<dataset>/<dimensions>/<variant>/<image>.txt

and builds ONE feature row per sample by putting together the files of that
sample from ALL dimensions and BOTH contiguity methods. With 5 dimensions this
gives 5 x 2 x 2 = 20 features (queen_64x64_I, queen_64x64_C, rook_64x64_I, ...).

A "sample" is one source image, identified by (dataset, variant, file name).
Its files in the different dimension folders are matched by file name
(--match-by name, default) or, if the names differ between dimension folders,
by their position in the sorted file list (--match-by order).

Two kinds of classification task are run, each with repeated stratified
k-fold cross-validation and several classifiers:

  inter-dataset : label = dataset, all samples of all datasets together
  intra-dataset : label = variant, one task per dataset

Imputation and scaling are fitted inside each training fold only (no leakage).
With --group-by-name, samples with the same file name inside a dataset (e.g.
the same source image in different variants) are always kept in the same fold.

Output (in --out-dir):
    features.csv                    the feature table
    summary.csv                     scores for every task x model
    confusion_<task>_<model>.csv    confusion matrices (summed over repeats)
    importance_<task>.csv           random forest importance per feature
    importance_grouped_<task>.csv   the same, summed per method / dimension / metric

Usage:
    python classify_metrics.py
    python classify_metrics.py --quadrants          # add quadrant summaries as features
    python classify_metrics.py --group-by-name --match-by order
"""

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

METHODS = ["queen", "rook"]
DIM_RE = re.compile(r"\d+x\d+")
ID_COLS = ["dataset", "variant", "sample"]

RE_I = re.compile(r"^Moran's I:\s*(\S+)")
RE_C = re.compile(r"^Geary's C:\s*(\S+)")
RE_LEVEL = re.compile(r"^# Level (\d+):")


# --------------------------------------------------------------------------
# Reading the metrics files
# --------------------------------------------------------------------------
def parse_metrics_file(path, quadrants):
    """Return {feature: value} for one .txt file: global I and C, and (if
    requested) mean/std/min/max of I and C over the quadrants of every level."""
    feats = {}
    levels = defaultdict(lambda: {"I": [], "C": []})
    depth = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = RE_I.match(line)
        if m:
            feats["I"] = float(m.group(1))
            continue
        m = RE_C.match(line)
        if m:
            feats["C"] = float(m.group(1))
            continue
        m = RE_LEVEL.match(line)
        if m:
            depth = int(m.group(1))
            continue
        if depth is not None and "|" in line and not line.startswith("#"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 5:
                levels[depth]["I"].append(float(parts[3]))
                levels[depth]["C"].append(float(parts[4]))

    if quadrants:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for d, vals in levels.items():
                for k in ("I", "C"):
                    a = np.array(vals[k], dtype=float)
                    feats[f"L{d}_{k}_mean"] = np.nanmean(a)
                    feats[f"L{d}_{k}_std"] = np.nanstd(a)
                    feats[f"L{d}_{k}_min"] = np.nanmin(a)
                    feats[f"L{d}_{k}_max"] = np.nanmax(a)
    return feats


def build_feature_table(metrics_dir, methods, match_by, quadrants):
    rows = {}
    for method in methods:
        mdir = metrics_dir / method
        if not mdir.is_dir():
            sys.exit(f"Metrics folder not found: {mdir}")
        groups = defaultdict(list)
        for p in sorted(mdir.rglob("*.txt")):
            parts = p.relative_to(mdir).parts
            if len(parts) != 4 or not DIM_RE.fullmatch(parts[1]):
                continue  # not <dataset>/<dimensions>/<variant>/<file>
            groups[(parts[0], parts[1], parts[2])].append(p)

        for (ds, dim, var), paths in groups.items():
            for i, p in enumerate(paths):
                key = p.stem if match_by == "name" else f"{i:05d}"
                row = rows.setdefault((ds, var, key),
                                      {"dataset": ds, "variant": var, "sample": key})
                for name, value in parse_metrics_file(p, quadrants).items():
                    row[f"{method}_{dim}_{name}"] = value

    if not rows:
        sys.exit(f"No metrics .txt files found under {metrics_dir}")
    df = pd.DataFrame(list(rows.values()))

    def sort_key(col):
        method, dim, rest = col.split("_", 2)
        w, h = map(int, dim.split("x"))
        return (methods.index(method), w * h, rest)

    feat_cols = sorted((c for c in df.columns if c not in ID_COLS), key=sort_key)
    return df[ID_COLS + feat_cols].sort_values(ID_COLS).reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS]


def report_completeness(df):
    feats = feature_columns(df)
    # only count columns that exist for the dataset at all
    print("\nSamples per dataset / variant (complete = has every feature "
          "that exists for that dataset):")
    lines = []
    for (ds, var), g in df.groupby(["dataset", "variant"]):
        ds_cols = [c for c in feats if df.loc[df["dataset"] == ds, c].notna().any()]
        complete = g[ds_cols].notna().all(axis=1).sum() if ds_cols else 0
        lines.append((ds, var, len(g), complete))
    for ds, var, n, c in lines:
        flag = "" if n == c else "   <-- some samples are missing features"
        print(f"  {ds:20s} {var:20s} n={n:4d} complete={c:4d}{flag}")
    if any(n != c for _, _, n, c in lines):
        print("  Missing values are imputed (median of the training fold). If "
              "this is unexpected, check that file names match across dimension "
              "folders, or try --match-by order.")


# --------------------------------------------------------------------------
# Models and evaluation
# --------------------------------------------------------------------------
def get_models(seed):
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
    return {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "LogReg": make_pipeline(*pre(), LogisticRegression(max_iter=5000)),
        "SVM-RBF": make_pipeline(*pre(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "kNN": make_pipeline(*pre(), KNeighborsClassifier(n_neighbors=5)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    }


def usable_columns(df, y_col):
    """Columns that carry information for this task. A column that is entirely
    missing for ANY class's group is dropped, otherwise 'is the value missing'
    would itself reveal the class (e.g. a dimension that only one dataset has)."""
    cols = []
    for c in feature_columns(df):
        present = df.groupby(y_col)[c].apply(lambda s: s.notna().any())
        if present.all():
            cols.append(c)
    return cols


def evaluate_task(task, df, y_col, args, models, out_dir):
    """Cross-validate all models for one task. Returns summary rows."""
    cols = usable_columns(df, y_col)
    if not cols:
        print(f"[{task}] no usable features, skipped", file=sys.stderr)
        return []
    data = df
    if args.drop_incomplete:
        data = df[df[cols].notna().all(axis=1)]
    X = data[cols].to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    y = data[y_col].to_numpy()
    classes = sorted(set(y))
    counts = pd.Series(y).value_counts()

    n_splits = min(args.folds, int(counts.min()))
    if len(classes) < 2 or n_splits < 2:
        print(f"[{task}] needs >= 2 classes with >= 2 samples each, skipped",
              file=sys.stderr)
        return []

    groups = None
    if args.group_by_name:
        groups = (data["dataset"] + "/" + data["sample"]).to_numpy()
        # a fold split needs each class to still have enough distinct groups
        n_groups = pd.DataFrame({"y": y, "g": groups}).groupby("y")["g"].nunique().min()
        n_splits = min(n_splits, int(n_groups))
        if n_splits < 2:
            print(f"[{task}] too few distinct groups for grouped CV, skipped",
                  file=sys.stderr)
            return []

    print(f"\n=== {task}: {len(y)} samples, {len(classes)} classes, "
          f"{len(cols)} features, {n_splits}-fold x {args.repeats} ===")
    print("    classes: " + ", ".join(f"{c} ({counts[c]})" for c in classes))

    rows = []
    for mname, model in models.items():
        accs, bals, f1s = [], [], []
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for rep in range(args.repeats):
            seed = args.seed + rep
            if groups is not None:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                          random_state=seed)
                pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed)
                pred = cross_val_predict(model, X, y, cv=cv)
            accs.append(float(np.mean(pred == y)))
            bals.append(balanced_accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro", zero_division=0))
            cm += confusion_matrix(y, pred, labels=classes)

        rows.append({
            "task": task, "model": mname, "n_samples": len(y),
            "n_classes": len(classes), "n_features": len(cols),
            "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
            "balanced_accuracy": np.mean(bals), "balanced_accuracy_std": np.std(bals),
            "macro_f1": np.mean(f1s), "macro_f1_std": np.std(f1s),
            "chance_balanced_accuracy": 1.0 / len(classes),
        })
        safe = re.sub(r"[^\w.-]+", "_", f"{task}_{mname}")
        pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                     columns=[f"pred_{c}" for c in classes]
                     ).to_csv(out_dir / f"confusion_{safe}.csv")
        print(f"    {mname:13s} bal.acc {np.mean(bals):.3f} ± {np.std(bals):.3f}   "
              f"acc {np.mean(accs):.3f}   macro-F1 {np.mean(f1s):.3f}")

    save_importance(task, X, y, cols, args, out_dir)
    return rows


def save_importance(task, X, y, cols, args, out_dir):
    """Random forest feature importance on the full data (descriptive only)."""
    rf = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1))
    rf.fit(X, y)
    imp = pd.Series(rf[-1].feature_importances_, index=cols)
    safe = re.sub(r"[^\w.-]+", "_", task)
    imp.sort_values(ascending=False).rename("importance").to_csv(
        out_dir / f"importance_{safe}.csv", index_label="feature")

    parts = pd.DataFrame({
        "feature": cols, "importance": imp.values,
        "method": [c.split("_", 2)[0] for c in cols],
        "dimension": [c.split("_", 2)[1] for c in cols],
        "metric": ["Moran's I" if re.search(r"(^|_)I(_|$)", c.split("_", 2)[2])
                   else "Geary's C" for c in cols],
    })
    grouped = []
    for key in ("method", "dimension", "metric"):
        g = parts.groupby(key)["importance"].sum().sort_values(ascending=False)
        for k, v in g.items():
            grouped.append({"group_by": key, "value": k, "importance": v})
    pd.DataFrame(grouped).to_csv(out_dir / f"importance_grouped_{safe}.csv",
                                 index=False)
    top = imp.sort_values(ascending=False).head(5)
    print("    top features (RF): " +
          ", ".join(f"{k} {v:.2f}" for k, v in top.items()))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", default="metrics")
    ap.add_argument("--out-dir", default="classification")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--match-by", choices=["name", "order"], default="name",
                    help="how to match one sample's files across dimension "
                         "folders (default: same file name)")
    ap.add_argument("--quadrants", action="store_true",
                    help="also use quadrant statistics (mean/std/min/max of I and "
                         "C per level) found in the .txt files as features")
    ap.add_argument("--group-by-name", action="store_true",
                    help="keep samples with the same file name within a dataset "
                         "(e.g. the same source image in different variants) in "
                         "the same CV fold")
    ap.add_argument("--drop-incomplete", action="store_true",
                    help="drop samples missing any feature instead of imputing")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["inter", "intra"],
                    help="run only the inter-dataset or only the intra-dataset tasks")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    metrics_dir = root / args.metrics_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_table(metrics_dir, args.methods, args.match_by, args.quadrants)
    df.to_csv(out_dir / "features.csv", index=False)
    print(f"Feature table: {len(df)} samples x {len(feature_columns(df))} features "
          f"-> {out_dir / 'features.csv'}")
    print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
    report_completeness(df)

    models = get_models(args.seed)
    summary = []

    if args.only != "intra":
        if df["dataset"].nunique() >= 2:
            summary += evaluate_task("inter-dataset (label=dataset)", df,
                                     "dataset", args, models, out_dir)
        else:
            print("\nOnly one dataset found, skipping inter-dataset task.",
                  file=sys.stderr)

    if args.only != "inter":
        for ds, sub in df.groupby("dataset"):
            if sub["variant"].nunique() < 2:
                print(f"\n[{ds}] only one variant, skipping intra-dataset task.",
                      file=sys.stderr)
                continue
            summary += evaluate_task(f"intra-dataset {ds} (label=variant)",
                                     sub.reset_index(drop=True), "variant",
                                     args, models, out_dir)

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(out_dir / "summary.csv", index=False)
        show = s.assign(**{
            "bal.acc": s["balanced_accuracy"].map("{:.3f}".format) + " ± "
                       + s["balanced_accuracy_std"].map("{:.3f}".format),
            "chance": s["chance_balanced_accuracy"].map("{:.3f}".format),
        })[["task", "model", "n_samples", "bal.acc", "chance"]]
        print("\n" + "=" * 70 + "\nSUMMARY (balanced accuracy, mean ± std over repeats)\n"
              + "=" * 70)
        print(show.to_string(index=False))
        print(f"\nAll results in {out_dir}")


if __name__ == "__main__":
    main()