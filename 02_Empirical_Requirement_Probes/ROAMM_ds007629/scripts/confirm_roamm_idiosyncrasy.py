from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

import residual_screen
from residual_screen import _pipeline
from replicate_roamm import DEFAULT_FEATURES, balance_by_subject

EPS = 1e-3


def binary_ll(y, p):
    return log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])


def split_subject(g: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = g[g["_y"] == 1].sample(frac=1, random_state=seed)
    neg = g[g["_y"] == 0].sample(frac=1, random_state=seed)
    h_pos = len(pos) // 2
    h_neg = len(neg) // 2
    cal = pd.concat([pos.iloc[:h_pos], neg.iloc[:h_neg]], ignore_index=True)
    test = pd.concat([pos.iloc[h_pos:], neg.iloc[h_neg:]], ignore_index=True)
    return (
        cal.sample(frac=1, random_state=seed).reset_index(drop=True),
        test.sample(frac=1, random_state=seed).reset_index(drop=True),
    )


def fit_calibrator(frame: pd.DataFrame) -> LogisticRegression | None:
    y = frame["_y"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return None
    x = logit(np.clip(frame["_p"].to_numpy(dtype=float), EPS, 1 - EPS)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, random_state=residual_screen.SEED)
    cal.fit(x, y)
    return cal


def apply_calibrator(cal: LogisticRegression | None, p: np.ndarray) -> np.ndarray:
    if cal is None:
        return np.clip(p, EPS, 1 - EPS)
    return cal.predict_proba(logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1))[:, 1]


def heldout_group_predictions(df: pd.DataFrame, cat: list[str], num: list[str], model: str) -> pd.DataFrame:
    R = df.dropna(subset=["mw", "sub"] + cat + num).copy().reset_index(drop=True)
    R["_y"] = R["mw"].astype(int)
    R["_p"] = np.nan
    X = R[cat + num]
    pipe = _pipeline(cat, num, model)
    for tr, te in GroupKFold(n_splits=min(5, R["sub"].nunique())).split(X, R["_y"], groups=R["sub"]):
        pipe.fit(X.iloc[tr], R["_y"].iloc[tr])
        R.loc[R.index[te], "_p"] = pipe.predict_proba(X.iloc[te])[:, 1]
    R["_p"] = R["_p"].clip(EPS, 1 - EPS)
    return R


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    residual_screen.SEED = args.seed
    df = pd.read_csv(args.csv)
    if args.epoch_stride > 1:
        df = df[df["t"].astype(int) % args.epoch_stride == 0].copy()
    df = balance_by_subject(df, args.min_pos, args.seed)

    cat = ["eye"] if args.include_eye else []
    num = DEFAULT_FEATURES.copy()
    if args.include_time:
        num = ["run", "t"] + num

    R = heldout_group_predictions(df, cat, num, args.model)
    splits = {}
    for subject, g in R.groupby("sub"):
        if int(g["_y"].sum()) < args.min_pos or int((1 - g["_y"]).sum()) < args.min_pos:
            continue
        splits[subject] = split_subject(g, args.seed)

    all_cal = pd.concat([v[0] for v in splits.values()], ignore_index=True)
    pooled_cal = fit_calibrator(all_cal)

    rows = []
    for subject, (sub_cal_frame, sub_test_frame) in splits.items():
        y = sub_test_frame["_y"].to_numpy(dtype=int)
        p_group = sub_test_frame["_p"].to_numpy(dtype=float)
        group_ll = binary_ll(y, p_group)

        # Global leave-one-subject-out calibration: generic calibration learned from other people.
        global_frame = pd.concat([v[0] for s, v in splits.items() if s != subject], ignore_index=True)
        global_cal = fit_calibrator(global_frame)
        p_global = apply_calibrator(global_cal, p_group)
        global_ll = binary_ll(y, p_global)

        p_pooled = apply_calibrator(pooled_cal, p_group)
        pooled_ll = binary_ll(y, p_pooled)

        subject_cal = fit_calibrator(sub_cal_frame)
        p_subject = apply_calibrator(subject_cal, p_group)
        subject_ll = binary_ll(y, p_subject)

        rows.append({
            "subject": subject,
            "n_test": len(sub_test_frame),
            "group_ll": group_ll,
            "pooled_cal_ll": pooled_ll,
            "global_loso_cal_ll": global_ll,
            "subject_cal_ll": subject_ll,
            "pooled_gain": group_ll - pooled_ll,
            "global_loso_gain": group_ll - global_ll,
            "subject_gain": group_ll - subject_ll,
            "subject_over_global": global_ll - subject_ll,
            "subject_over_pooled": pooled_ll - subject_ll,
            "subject_slope": np.nan if subject_cal is None else float(subject_cal.coef_[0, 0]),
            "subject_intercept": np.nan if subject_cal is None else float(subject_cal.intercept_[0]),
            "global_slope": np.nan if global_cal is None else float(global_cal.coef_[0, 0]),
            "global_intercept": np.nan if global_cal is None else float(global_cal.intercept_[0]),
        })

    out = pd.DataFrame(rows)

    def wtest(col: str) -> float:
        v = out[col].to_numpy(dtype=float)
        return float(wilcoxon(v).pvalue) if len(v) >= 6 and np.any(v != 0) else np.nan

    rng = np.random.default_rng(args.seed)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "epoch_stride": args.epoch_stride,
        "n_subjects": len(out),
        "mean_group_ll": out["group_ll"].mean(),
        "mean_global_loso_gain": out["global_loso_gain"].mean(),
        "global_loso_gain_p": wtest("global_loso_gain"),
        "mean_subject_gain": out["subject_gain"].mean(),
        "subject_gain_p": wtest("subject_gain"),
        "mean_subject_over_global": out["subject_over_global"].mean(),
        "subject_over_global_p": wtest("subject_over_global"),
        "subject_over_global_improved": int((out["subject_over_global"] > 0).sum()),
        "mean_subject_over_pooled": out["subject_over_pooled"].mean(),
        "subject_over_pooled_p": wtest("subject_over_pooled"),
        "slope_mean": out["subject_slope"].mean(),
        "slope_sd": out["subject_slope"].std(ddof=1),
        "intercept_mean": out["subject_intercept"].mean(),
        "intercept_sd": out["subject_intercept"].std(ddof=1),
    }

    v = out["subject_over_global"].to_numpy(dtype=float)
    obs = v.mean()
    signs = rng.choice([-1, 1], size=(args.sign_perm, len(v)))
    null = (signs * v).mean(axis=1)
    summary["subject_over_global_sign_perm_p"] = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Test whether ROAMM calibration gain is subject-idiosyncratic")
    ap.add_argument("--csv", default="data/roamm_epochs_44subj.csv", help="Local ROAMM epoch table; data is not included")
    ap.add_argument("--model", choices=["gbm", "logistic"], default="gbm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--epoch-stride", type=int, default=1)
    ap.add_argument("--sign-perm", type=int, default=100000)
    ap.add_argument("--include-eye", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-time", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-prefix", default="work/results/roamm_idiosyncrasy_gbm")
    args = ap.parse_args()

    out, summary = run(args)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(prefix.with_suffix(".subjects.csv"), index=False)
    pd.DataFrame([summary]).to_csv(prefix.with_suffix(".summary.csv"), index=False)

    print("=" * 72)
    print(f"ROAMM IDIOSYNCRASY TEST ({args.model}, seed={args.seed}, stride={args.epoch_stride})")
    print("=" * 72)
    print(f"subjects: {summary['n_subjects']}")
    print(f"global LOSO calibration gain: {summary['mean_global_loso_gain']:+0.4f} (p={summary['global_loso_gain_p']:0.4g})")
    print(f"subject calibration gain     : {summary['mean_subject_gain']:+0.4f} (p={summary['subject_gain_p']:0.4g})")
    print(f"subject over global LOSO     : {summary['mean_subject_over_global']:+0.4f} ({summary['subject_over_global_improved']}/{summary['n_subjects']}, p={summary['subject_over_global_p']:0.4g}, sign-perm={summary['subject_over_global_sign_perm_p']:0.4g})")
    print(f"subject slope mean/sd        : {summary['slope_mean']:+0.3f} / {summary['slope_sd']:+0.3f}")
    print(f"subject intercept mean/sd    : {summary['intercept_mean']:+0.3f} / {summary['intercept_sd']:+0.3f}")
    print(f"wrote {prefix.with_suffix('.summary.csv')}")
    print(f"wrote {prefix.with_suffix('.subjects.csv')}")


if __name__ == "__main__":
    main()

