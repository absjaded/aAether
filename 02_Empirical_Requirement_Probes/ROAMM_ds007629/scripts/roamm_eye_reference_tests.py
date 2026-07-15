from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
EPS = 1e-3
DEFAULT_FEATURES = [
    "fix_dur", "fix_sd", "pupil", "pupil_sd", "sacc_amp", "sacc_v",
    "blink", "n_fix", "n_sacc", "x", "y", "x_sd", "y_sd", "fp", "tot",
]


def feature_columns(include_time: bool) -> tuple[list[str], list[str]]:
    cat = ["eye"]
    num = DEFAULT_FEATURES.copy()
    if include_time:
        num = ["run", "t"] + num
    return cat, num


def pipeline(cat: list[str], num: list[str]):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), ("num", "passthrough", num)],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=0.01,
        random_state=SEED,
        class_weight="balanced",
    )
    return make_pipeline(pre, clf)


def balance_by_subject(df: pd.DataFrame, min_pos: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("sub"):
        pos = g[g["mw"] == 1]
        neg = g[g["mw"] == 0]
        if len(pos) < min_pos or len(neg) < min_pos:
            continue
        neg_sample = neg.sample(n=len(pos), random_state=seed)
        parts.append(pd.concat([pos, neg_sample], ignore_index=False))
    if not parts:
        raise ValueError("No subjects survived balancing")
    out = pd.concat(parts, ignore_index=False)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def clean_frame(path: Path, include_time: bool, min_pos: int, seed: int) -> pd.DataFrame:
    cat, num = feature_columns(include_time)
    cols = ["sub", "run", "mw"] + cat + num
    df = pd.read_csv(path)
    df = df.dropna(subset=sorted(set(cols))).copy()
    df["mw"] = df["mw"].astype(int)
    df["sub"] = df["sub"].astype(str)
    return balance_by_subject(df, min_pos=min_pos, seed=seed)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    on_task_acc = tn / max(1, tn + fp)
    mw_acc = tp / max(1, tp + fn)
    return {
        "n": float(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float((on_task_acc + mw_acc) / 2),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "on_task_acc": float(on_task_acc),
        "mw_acc": float(mw_acc),
    }


def loso_predictions(df: pd.DataFrame, include_time: bool, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cat, num = feature_columns(include_time)
    xcols = cat + num
    rows = []
    pred_parts = []
    for i, sub in enumerate(sorted(df["sub"].unique()), 1):
        train = df[df["sub"] != sub]
        test = df[df["sub"] == sub]
        model = pipeline(cat, num)
        model.fit(train[xcols], train["mw"])
        p = model.predict_proba(test[xcols])[:, 1]
        m = metrics(test["mw"].to_numpy(), p)
        rows.append({"subject": sub, **m})
        part = test[["sub", "run", "mw"]].copy()
        part["p_group_loso"] = p
        pred_parts.append(part)
        print(f"LOSO {i:02d}/{df['sub'].nunique()} sub={sub} acc={m['acc']:.3f} auc={m['auc']:.3f}", flush=True)
    per_subject = pd.DataFrame(rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    overall = metrics(preds["mw"].to_numpy(), preds["p_group_loso"].to_numpy())
    per_subject.to_csv(out_dir / "shared_reference_loso_by_subject.csv", index=False)
    preds.to_csv(out_dir / "shared_reference_loso_predictions.csv", index=False)
    return per_subject, preds, overall


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    x = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, random_state=SEED)
    cal.fit(x, y)
    return cal


def apply_calibrator(cal: LogisticRegression | None, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    if cal is None:
        return p
    return cal.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def calibration_cost(preds: pd.DataFrame, sizes: list[int], repeats: int, seed: int, out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sub, g0 in preds.groupby("sub"):
        g = g0.reset_index(drop=True)
        pos_idx = np.flatnonzero(g["mw"].to_numpy() == 1)
        neg_idx = np.flatnonzero(g["mw"].to_numpy() == 0)
        for n in sizes:
            if n == 0:
                m = metrics(g["mw"].to_numpy(), g["p_group_loso"].to_numpy())
                rows.append({"subject": sub, "calibration_trials": 0, "repeat": 0, "usable": True, **m})
                continue
            n_pos = n // 2
            n_neg = n - n_pos
            if len(pos_idx) <= n_pos or len(neg_idx) <= n_neg:
                rows.append({"subject": sub, "calibration_trials": n, "repeat": -1, "usable": False})
                continue
            for r in range(repeats):
                cal_pos = rng.choice(pos_idx, size=n_pos, replace=False)
                cal_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                cal_idx = np.concatenate([cal_pos, cal_neg])
                test_mask = np.ones(len(g), dtype=bool)
                test_mask[cal_idx] = False
                test = g[test_mask]
                if test["mw"].nunique() < 2:
                    rows.append({"subject": sub, "calibration_trials": n, "repeat": r, "usable": False})
                    continue
                cal = fit_calibrator(g.loc[cal_idx, "p_group_loso"].to_numpy(), g.loc[cal_idx, "mw"].to_numpy())
                p_cal = apply_calibrator(cal, test["p_group_loso"].to_numpy())
                m = metrics(test["mw"].to_numpy(), p_cal)
                rows.append({"subject": sub, "calibration_trials": n, "repeat": r, "usable": True, **m})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "calibration_cost_trials.csv", index=False)
    summary = out[out["usable"] == True].groupby("calibration_trials").agg(
        reps=("repeat", "count"),
        subjects=("subject", "nunique"),
        acc_mean=("acc", "mean"),
        acc_sd=("acc", "std"),
        auc_mean=("auc", "mean"),
        log_loss_mean=("log_loss", "mean"),
        balanced_acc_mean=("balanced_acc", "mean"),
        on_task_acc_mean=("on_task_acc", "mean"),
        mw_acc_mean=("mw_acc", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "calibration_cost_summary.csv", index=False)
    return summary


def cross_run_loso(df_all: pd.DataFrame, include_time: bool, min_pos: int, seed: int, out_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    cat, num = feature_columns(include_time)
    xcols = cat + num
    rows = []
    pred_parts = []
    subjects = sorted(df_all["sub"].unique())
    for i, sub in enumerate(subjects, 1):
        train_raw = df_all[(df_all["sub"] != sub) & (df_all["run"].isin([1, 2]))]
        test_raw = df_all[(df_all["sub"] == sub) & (df_all["run"].isin([4, 5]))]
        try:
            train = balance_by_subject(train_raw, min_pos=min_pos, seed=seed)
            test = balance_by_subject(test_raw, min_pos=max(4, min_pos // 2), seed=seed)
        except ValueError:
            continue
        if test["mw"].nunique() < 2:
            continue
        model = pipeline(cat, num)
        model.fit(train[xcols], train["mw"])
        p = model.predict_proba(test[xcols])[:, 1]
        m = metrics(test["mw"].to_numpy(), p)
        rows.append({"subject": sub, **m})
        part = test[["sub", "run", "mw"]].copy()
        part["p_cross_run"] = p
        pred_parts.append(part)
        print(f"RUN  {i:02d}/{len(subjects)} sub={sub} acc={m['acc']:.3f} auc={m['auc']:.3f}", flush=True)
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(out_dir / "cross_run_loso_by_subject.csv", index=False)
    if pred_parts:
        preds = pd.concat(pred_parts, ignore_index=True)
        preds.to_csv(out_dir / "cross_run_loso_predictions.csv", index=False)
        overall = metrics(preds["mw"].to_numpy(), preds["p_cross_run"].to_numpy())
    else:
        overall = {}
    return per_subject, overall


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ROAMM shared-reference and calibration-cost tests")
    ap.add_argument("--csv", default="data/roamm_epochs_44subj.csv.gz")
    ap.add_argument("--out", default="artifacts/roamm_reference_tests")
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--calibration-repeats", type=int, default=200)
    ap.add_argument("--include-time", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean_frame(Path(args.csv), include_time=args.include_time, min_pos=args.min_pos, seed=SEED)
    print(f"balanced n={len(df)} subjects={df['sub'].nunique()} include_time={args.include_time}", flush=True)
    per_subject, preds, shared = loso_predictions(df, args.include_time, out_dir)
    cost = calibration_cost(preds, sizes=[0, 5, 10, 20, 50], repeats=args.calibration_repeats, seed=SEED, out_dir=out_dir)

    cat, num = feature_columns(args.include_time)
    all_cols = ["sub", "run", "mw"] + cat + num
    df_all = pd.read_csv(args.csv).dropna(subset=sorted(set(all_cols))).copy()
    df_all["mw"] = df_all["mw"].astype(int)
    df_all["sub"] = df_all["sub"].astype(str)
    cross_subject, cross = cross_run_loso(df_all, args.include_time, args.min_pos, SEED, out_dir)

    summary = pd.DataFrame([
        {"test": "shared_reference_loso", **shared},
        {"test": "cross_run_train12_test45_loso", **cross},
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("CALIBRATION")
    print(cost.to_string(index=False))


if __name__ == "__main__":
    main()

