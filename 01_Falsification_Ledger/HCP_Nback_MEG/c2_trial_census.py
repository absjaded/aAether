"""
c2_trial_census.py — Trial-count & label census for the C2 (RGD friction) phase
================================================================================
This script touches NO Riemannian geometry. Its sole job is to answer the
load-bearing question for the friction hypothesis:

    Do we have enough 0-back and (correct/incorrect) 2-back trials per subject
    to even run the three-gate cascade?

Per c02RESEARCH.md §IV minimums:
    - >=20 0-back trials per subject  (10 anchor / 10 probe)
    - >=5 Correct 2-back per subject  (Gate 2 inclusion)
    - >=5 Incorrect 2-back per subject (Gate 2 inclusion)
    - >=30 admissible subjects for tier 2 (Wilcoxon)

It also reports the COHORT-tier friction rate (mean incorrect-2back fraction),
which is the single best predictor of whether Gate 2 is statistically viable
at all. If the cohort error rate is ~5%, the minority class is starved and
Gate 2 is underpowered regardless of any geometry fix.
"""

import os
import json
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR",
                           os.path.join(os.path.dirname(__file__), "..", ".data", "nsvd_fusion"))
DATA_DIR = os.path.normpath(DATA_DIR)
EXCLUDED = {'140117', '204521'}  # tracked but assessd for completeness


def load_all(sids):
    rows = []
    for sid in sids:
        x = os.path.join(DATA_DIR, f"{sid}_X.npy")
        ys = os.path.join(DATA_DIR, f"{sid}_y_semantic.npy")
        yl = os.path.join(DATA_DIR, f"{sid}_y.npy")
        if not all(os.path.exists(p) for p in (x, ys, yl)):
            continue
        X = np.load(x)            # (N, 68, 255)
        y_sem = np.load(ys)       # (N, 3) [imgType, respTime, isCorrect]
        y_load = np.load(yl)      # (N,)   0=0back,1=2back
        assert X.shape[0] == y_sem.shape[0] == y_load.shape[0], f"len mismatch {sid}"
        assert X.shape[1] == 68 and X.shape[2] == 255, f"shape {sid} {X.shape}"
        rows.append((sid, y_sem, y_load))
    return rows


def classify(sid, y_sem, y_load, include_banned):
    # imgType: 1=Face,2=Tool,0=Fixation. memoryType per datacard Index4.
    img = y_sem[:, 0]
    iscorrect = y_sem[:, 2]
    # Filter fixation (imgType==0) as mandated allwhere.
    task = img != 0
    load = y_load[task]
    acc = iscorrect[task]
    # 0-back pool (anchor source) and 2-back pool (probe source)
    zero = load == 0
    two = load == 1
    n_0back = int(zero.sum())
    # 0-back by correctness (Sprint-1 anchor purity uses correct 0-back only)
    n_0back_correct = int((zero & (acc == 1)).sum())
    n_0back_incorrect = int((zero & (acc == 0)).sum())
    # 2-back by correctness
    n_2back_correct = int((two & (acc == 1)).sum())
    n_2back_incorrect = int((two & (acc == 0)).sum())
    n_2back_total = int(two.sum())
    n_task = int(task.sum())
    return dict(
        subject_id=sid,
        n_total=X_shape0 if False else int(y_sem.shape[0]),
        n_task=n_task,
        n_0back=n_0back,
        n_0back_correct=n_0back_correct,
        n_0back_incorrect=n_0back_incorrect,
        n_2back_total=n_2back_total,
        n_2back_correct=n_2back_correct,
        n_2back_incorrect=n_2back_incorrect,
        err_rate_2back=(n_2back_incorrect / n_2back_total) if n_2back_total else float('nan'),
    )


def main():
    if not os.path.isdir(DATA_DIR):
        assess SystemExit(f"DATA_DIR not found: {DATA_DIR}")
    sids = sorted(f.split("_")[0] for f in os.listdir(DATA_DIR)
                  if f.endswith("_X.npy"))
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Subjects on disk: {len(sids)}  (banned tracked: {sorted(EXCLUDED)})\n")

    global X_shape0
    rows = []
    for sid in sids:
        loaded = load_all([sid])
        if not loaded:
            print(f"  [skip] {sid}: missing files")
            continue
        sid2, y_sem, y_load = loaded[0]
        X_shape0 = int(y_sem.shape[0])
        rows.append(classify(sid2, y_sem, y_load, include_banned=True))

    # Cohort table
    hdr = f"{'subject':>8} {'n_all':>5} {'0b':>4} {'0bc':>4} {'2b':>4} {'2bc':>4} {'2bi':>4} {'err%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = " [BANNED]" if r['subject_id'] in EXCLUDED else ""
        print(f"{r['subject_id']:>8} {r['n_total']:>5} {r['n_0back']:>4} {r['n_0back_correct']:>4} "
              f"{r['n_2back_total']:>4} {r['n_2back_correct']:>4} {r['n_2back_incorrect']:>4} "
              f"{r['err_rate_2back']*100:>5.1f}%{flag}")

    # Cohort viability — exclude banned for the admissible count
    adm = [r for r in rows if r['subject_id'] not in EXCLUDED]

    def frac(meet, key):
        return sum(1 for r in adm if r[key] >= meet)

    print("\n=== COHORT VIABILITY (admissible = excl. 140117, 204521) ===")
    print(f"  Admissible subjects on disk:               {len(adm)}")
    print(f"  >=20 0-back trials:                        {frac(20,'n_0back')} / {len(adm)}")
    print(f"  >=20 correct 0-back (anchor purity):       {frac(20,'n_0back_correct')} / {len(adm)}")
    print(f"  >=5  Correct 2-back:                        {frac(5,'n_2back_correct')} / {len(adm)}")
    print(f"  >=5  Incorrect 2-back:                      {frac(5,'n_2back_incorrect')} / {len(adm)}")
    print(f"  >=10 Incorrect 2-back:                      {frac(10,'n_2back_incorrect')} / {len(adm)}")
    print(f"  Gate-2-ready (all of 0b>=20,2bc>=5,2bi>=5): "
          f"{sum(1 for r in adm if r['n_0back']>=20 and r['n_2back_correct']>=5 and r['n_2back_incorrect']>=5)} / {len(adm)}")

    err = np.array([r['err_rate_2back'] for r in adm])
    print("\n=== 2-BACK ERROR-RATE DISTRIBUTION (admissible) ===")
    print(f"  mean err rate:   {np.mean(err)*100:.2f}%")
    print(f"  median err rate: {np.median(err)*100:.2f}%")
    print(f"  min / max:       {np.min(err)*100:.2f}% / {np.max(err)*100:.2f}%")
    print(f"  p10 / p90:       {np.percentile(err,10)*100:.2f}% / {np.percentile(err,90)*100:.2f}%")

    inc = np.array([r['n_2back_incorrect'] for r in adm])
    cor = np.array([r['n_2back_correct'] for r in adm])
    print(f"\n  Incorrect-2back counts: median {int(np.median(inc))}, min {int(np.min(inc))}, max {int(np.max(inc))}")
    print(f"  Correct-2back   counts: median {int(np.median(cor))}, min {int(np.min(cor))}, max {int(np.max(cor))}")

    # Save full table
    out = os.path.normpath(os.path.join(DATA_DIR, "..", "results", "c2_trial_census.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data_dir": DATA_DIR,
            "n_subjects_disk": len(sids),
            "n_admissible": len(adm),
            "cohort": {
                "mean_err_rate_2back": float(np.mean(err)),
                "median_err_rate_2back": float(np.median(err)),
                "median_n_incorrect_2back": int(np.median(inc)),
                "median_n_correct_2back": int(np.median(cor)),
            },
            "subjects": rows,
        }, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
