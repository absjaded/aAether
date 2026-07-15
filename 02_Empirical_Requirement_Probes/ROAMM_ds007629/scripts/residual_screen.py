"""
Subject-level residual tests for ROAMM epoch tables.

The code first fits a group-level checker with held-out-subject predictions, then
tests whether per-subject calibration improves held-out prediction for that same
subject.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EPS = 1e-3
ModelName = Literal["logistic", "gbm"]


@dataclass(frozen=True)
class ScreenResult:
    dataset: str
    n_trials: int
    n_subjects: int
    baseline: float
    checker_acc: float
    checker_auc: float
    checker_log_loss: float
    checker_edge: float
    resid_split_half_r: float
    resid_sb: float
    resid_perm_p: float
    intercept_acc_gain: float
    intercept_ll_gain: float
    intercept_p: float
    intercept_improved: int
    recal_acc_gain: float
    recal_ll_gain: float
    recal_p: float
    recal_improved: int
    full_acc_gain: float
    full_ll_gain: float
    full_p: float
    full_improved: int
    full_perm_p: float
    verdict: str


def _model(name: ModelName):
    if name == "logistic":
        return LogisticRegression(max_iter=4000, random_state=SEED)
    if name == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=SEED,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model: {name}")


def _pipeline(cat_features: list[str], num_features: list[str], model_name: ModelName):
    if model_name == "logistic":
        num_step = StandardScaler()
    else:
        num_step = "passthrough"
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", num_step, num_features),
        ],
        remainder="drop",
    )
    return make_pipeline(pre, _model(model_name))


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    acc = ((p >= 0.5).astype(int) == y).mean()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ll = log_loss(y, p, labels=[0, 1])
    return float(acc), float(auc), float(ll)


def _fit_log_odds_intercept(y: np.ndarray, p: np.ndarray) -> float:
    y_rate = np.clip(np.mean(y), EPS, 1 - EPS)
    p_rate = np.clip(np.mean(p), EPS, 1 - EPS)
    return float(logit(y_rate) - logit(p_rate))


def _fit_recalibrator(y: np.ndarray, p: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    x = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, random_state=SEED)
    cal.fit(x, y)
    return cal


def _split_half_reliability(frame: pd.DataFrame, subject_col: str, rng: np.random.Generator) -> float:
    a, b = [], []
    for _, g in frame.groupby(subject_col):
        if len(g) < 8:
            continue
        idx = rng.permutation(len(g))
        h = len(idx) // 2
        a.append(g["_res"].to_numpy()[idx[:h]].mean())
        b.append(g["_res"].to_numpy()[idx[h:]].mean())
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else np.nan


def _personal_tests(
    R: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
    subject_col: str,
    model_name: ModelName,
    min_pos: int,
    n_personal_perm: int,
    rng: np.random.Generator,
):
    rows = []
    null_means = []
    X_cols = cat_features + num_features
    base_model = _pipeline(cat_features, num_features, model_name)

    for subject, g0 in R.groupby(subject_col):
        if int(g0["_y"].sum()) < min_pos or int((1 - g0["_y"]).sum()) < min_pos:
            continue
        pos = g0[g0["_y"] == 1].sample(frac=1, random_state=SEED)
        neg = g0[g0["_y"] == 0].sample(frac=1, random_state=SEED)
        h_pos = len(pos) // 2
        h_neg = len(neg) // 2
        tr = pd.concat([pos.iloc[:h_pos], neg.iloc[:h_neg]], ignore_index=True)
        te = pd.concat([pos.iloc[h_pos:], neg.iloc[h_neg:]], ignore_index=True)
        tr = tr.sample(frac=1, random_state=SEED).reset_index(drop=True)
        te = te.sample(frac=1, random_state=SEED).reset_index(drop=True)
        y_tr = tr["_y"].to_numpy(dtype=int)
        y_te = te["_y"].to_numpy(dtype=int)
        p_group = np.clip(te["_p"].to_numpy(dtype=float), EPS, 1 - EPS)
        group_acc, _, group_ll = _binary_metrics(y_te, p_group)

        delta = _fit_log_odds_intercept(y_tr, tr["_p"].to_numpy(dtype=float))
        p_intercept = expit(logit(p_group) + delta)
        int_acc, _, int_ll = _binary_metrics(y_te, p_intercept)

        cal = _fit_recalibrator(y_tr, tr["_p"].to_numpy(dtype=float))
        if cal is None:
            p_recal = p_intercept
        else:
            p_recal = cal.predict_proba(logit(p_group).reshape(-1, 1))[:, 1]
        recal_acc, _, recal_ll = _binary_metrics(y_te, p_recal)

        personal = clone(base_model)
        personal.fit(tr[X_cols], y_tr)
        p_full = personal.predict_proba(te[X_cols])[:, 1]
        full_acc, _, full_ll = _binary_metrics(y_te, p_full)

        row = {
            "subject": subject,
            "n": len(tr) + len(te),
            "group_acc": group_acc,
            "group_ll": group_ll,
            "intercept_acc_gain": int_acc - group_acc,
            "intercept_ll_gain": group_ll - int_ll,
            "recal_acc_gain": recal_acc - group_acc,
            "recal_ll_gain": group_ll - recal_ll,
            "full_acc_gain": full_acc - group_acc,
            "full_ll_gain": group_ll - full_ll,
        }
        rows.append(row)

        if n_personal_perm:
            subject_null = []
            for _ in range(n_personal_perm):
                y_perm = rng.permutation(y_tr)
                if len(np.unique(y_perm)) < 2:
                    continue
                null_model = clone(base_model)
                null_model.fit(tr[X_cols], y_perm)
                p_null = null_model.predict_proba(te[X_cols])[:, 1]
                _, _, null_ll = _binary_metrics(y_te, p_null)
                subject_null.append(group_ll - null_ll)
            if subject_null:
                null_means.append(subject_null)

    P = pd.DataFrame(rows)
    if not len(P):
        raise ValueError("No subjects passed min_pos/min_neg threshold for personal tests")

    full_perm_p = np.nan
    if null_means:
        null_arr = np.array(null_means, dtype=float)
        obs = P["full_ll_gain"].mean()
        null_subjects = null_arr.shape[0]
        null_iters = null_arr.shape[1]
        null_distribution = null_arr.mean(axis=0) if null_subjects == len(P) else np.nanmean(null_arr, axis=0)
        full_perm_p = float((np.sum(null_distribution >= obs) + 1) / (null_iters + 1))

    return P, full_perm_p


def screen(
    df: pd.DataFrame,
    label_col: str,
    cat_features: Iterable[str] = (),
    num_features: Iterable[str] = (),
    subject_col: str = "sub",
    name: str = "dataset",
    model_name: ModelName = "gbm",
    n_perm: int = 500,
    n_personal_perm: int = 0,
    min_pos: int = 16,
    verbose: bool = True,
) -> tuple[ScreenResult, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    cat = list(cat_features)
    num = list(num_features)
    R = df.dropna(subset=[label_col, subject_col] + cat + num).copy().reset_index(drop=True)
    R["_y"] = R[label_col].astype(int)
    X = R[cat + num]

    pipe = _pipeline(cat, num, model_name)
    n_sub = R[subject_col].nunique()
    R["_p"] = np.nan
    for train_idx, test_idx in GroupKFold(n_splits=min(5, n_sub)).split(X, R["_y"], groups=R[subject_col]):
        pipe.fit(X.iloc[train_idx], R["_y"].iloc[train_idx])
        R.loc[R.index[test_idx], "_p"] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    R["_p"] = R["_p"].clip(EPS, 1 - EPS)

    y = R["_y"].to_numpy(dtype=int)
    p = R["_p"].to_numpy(dtype=float)
    acc, auc, ll = _binary_metrics(y, p)
    base = float(max(y.mean(), 1 - y.mean()))
    R["_res"] = y - p

    observed_rs = np.array([_split_half_reliability(R, subject_col, rng) for _ in range(n_perm)])
    resid_r = float(np.nanmean(observed_rs))
    resid_sb = float(2 * resid_r / (1 + resid_r)) if resid_r > -1 else np.nan
    null = []
    for _ in range(n_perm):
        Rp = R.copy()
        Rp["_res"] = rng.permutation(Rp["_res"].to_numpy())
        null.append(_split_half_reliability(Rp, subject_col, rng))
    null = np.array(null)
    resid_perm_p = float((np.nansum(null >= resid_r) + 1) / (np.sum(~np.isnan(null)) + 1))

    personal, full_perm_p = _personal_tests(
        R, cat, num, subject_col, model_name, min_pos, n_personal_perm, rng
    )

    def wp(col: str) -> float:
        values = personal[col].to_numpy(dtype=float)
        return float(wilcoxon(values).pvalue) if len(values) >= 6 and np.any(values != 0) else np.nan

    full_p = wp("full_ll_gain")
    recal_p = wp("recal_ll_gain")
    recal_helps = bool(personal["recal_ll_gain"].mean() > 0 and recal_p < 0.05)
    full_helps = bool(personal["full_ll_gain"].mean() > 0 and full_p < 0.05)
    checker_strong = bool((acc - base) > 0.10 or auc >= 0.70)
    verdict = "RESIDUAL PRESENT" if (recal_helps or full_helps) else "NO RESIDUAL"

    result = ScreenResult(
        dataset=name,
        n_trials=len(R),
        n_subjects=int(n_sub),
        baseline=base,
        checker_acc=acc,
        checker_auc=auc,
        checker_log_loss=ll,
        checker_edge=acc - base,
        resid_split_half_r=resid_r,
        resid_sb=resid_sb,
        resid_perm_p=resid_perm_p,
        intercept_acc_gain=float(personal["intercept_acc_gain"].mean()),
        intercept_ll_gain=float(personal["intercept_ll_gain"].mean()),
        intercept_p=wp("intercept_ll_gain"),
        intercept_improved=int((personal["intercept_ll_gain"] > 0).sum()),
        recal_acc_gain=float(personal["recal_acc_gain"].mean()),
        recal_ll_gain=float(personal["recal_ll_gain"].mean()),
        recal_p=recal_p,
        recal_improved=int((personal["recal_ll_gain"] > 0).sum()),
        full_acc_gain=float(personal["full_acc_gain"].mean()),
        full_ll_gain=float(personal["full_ll_gain"].mean()),
        full_p=full_p,
        full_improved=int((personal["full_ll_gain"] > 0).sum()),
        full_perm_p=full_perm_p,
        verdict=verdict,
    )

    if verbose:
        print_result(result, model_name, checker_strong, len(personal))
    return result, personal, R


def print_result(result: ScreenResult, model_name: str, checker_strong: bool, subjects_tested: int) -> None:
    print("=" * 72)
    print(f"RESIDUAL SCREEN - {result.dataset} ({model_name})")
    print("=" * 72)
    print(f"{result.n_trials} trials | {result.n_subjects} subjects")
    print("[1] Checker, held-out subjects")
    print(f"    baseline accuracy : {100 * result.baseline:5.1f}%")
    print(f"    checker accuracy  : {100 * result.checker_acc:5.1f}%")
    print(f"    checker AUC       : {result.checker_auc:0.3f}")
    print(f"    checker log-loss  : {result.checker_log_loss:0.4f}")
    print(f"    checker is {'STRONG' if checker_strong else 'WEAK'}")
    print("[2] Residual reliability")
    print(f"    split-half r      : {result.resid_split_half_r:+0.3f}")
    print(f"    Spearman-Brown    : {result.resid_sb:+0.3f}")
    print(f"    permutation p     : {result.resid_perm_p:0.4f}")
    print("[3] Personal-model gain over group checker")
    print(f"    subjects tested   : {subjects_tested}")
    print(f"    intercept acc     : {100 * result.intercept_acc_gain:+0.1f} pts")
    print(f"    intercept LL      : {result.intercept_ll_gain:+0.4f} ({result.intercept_improved}/{subjects_tested}, p={result.intercept_p:0.4g})")
    print(f"    recal acc         : {100 * result.recal_acc_gain:+0.1f} pts")
    print(f"    recal LL          : {result.recal_ll_gain:+0.4f} ({result.recal_improved}/{subjects_tested}, p={result.recal_p:0.4g})")
    print(f"    full acc          : {100 * result.full_acc_gain:+0.1f} pts")
    print(f"    full LL           : {result.full_ll_gain:+0.4f} ({result.full_improved}/{subjects_tested}, p={result.full_p:0.4g})")
    if not np.isnan(result.full_perm_p):
        print(f"    full perm p       : {result.full_perm_p:0.4f}")
    print(f"VERDICT: {result.verdict}")
    print("=" * 72)





