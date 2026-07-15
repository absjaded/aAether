from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import residual_screen
from residual_screen import screen

DEFAULT_FEATURES = [
    "fix_dur", "fix_sd", "pupil", "pupil_sd", "sacc_amp", "sacc_v",
    "blink", "n_fix", "n_sacc", "x", "y", "x_sd", "y_sd", "fp", "tot",
]


def load_roamm(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def balance_by_subject(df: pd.DataFrame, min_pos: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("sub"):
        pos = g[g["mw"] == 1]
        neg = g[g["mw"] == 0]
        if len(pos) < min_pos or len(neg) < min_pos:
            continue
        neg_sample = neg.sample(n=len(pos), random_state=seed)
        parts.append(pd.concat([pos, neg_sample], ignore_index=True))
    if not parts:
        raise ValueError("No subjects survived balancing")
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_once(args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    residual_screen.SEED = args.seed
    df = load_roamm(Path(args.csv))
    if args.epoch_stride and args.epoch_stride > 1:
        df = df[df["t"].astype(int) % args.epoch_stride == 0].copy()
    if args.balanced:
        df = balance_by_subject(df, args.min_pos, args.seed)

    cat = ["eye"] if args.include_eye else []
    num = DEFAULT_FEATURES.copy()
    if args.include_time:
        num = ["run", "t"] + num

    result, personal, _ = screen(
        df,
        label_col="mw",
        cat_features=cat,
        num_features=num,
        subject_col="sub",
        name=args.name,
        model_name=args.model,
        n_perm=args.n_perm,
        n_personal_perm=args.n_personal_perm,
        min_pos=args.min_pos,
        verbose=True,
    )
    return asdict(result), personal


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ROAMM residual-screen replication")
    ap.add_argument("--csv", default="data/roamm_epochs_44subj.csv", help="Local ROAMM epoch table; data is not included")
    ap.add_argument("--model", choices=["gbm", "logistic"], default="gbm")
    ap.add_argument("--name", default="ROAMM balanced 2s")
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-personal-perm", type=int, default=0)
    ap.add_argument("--epoch-stride", type=int, default=1, help="Keep only integer t divisible by this value")
    ap.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-eye", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-time", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-prefix", default="artifacts/results/roamm_replication")
    args = ap.parse_args()

    result, personal = run_once(args)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(prefix.with_suffix(".summary.csv"), index=False)
    personal.to_csv(prefix.with_suffix(".subjects.csv"), index=False)
    print(f"wrote {prefix.with_suffix('.summary.csv')}")
    print(f"wrote {prefix.with_suffix('.subjects.csv')}")


if __name__ == "__main__":
    main()




