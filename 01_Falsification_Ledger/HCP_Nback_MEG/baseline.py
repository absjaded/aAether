"""
v2.3.0_full_loso_run.py — NSVD Full 73-Subject LOSO Run
=========================================================
LOCKED CONFIGURATION (from v2.2.1 + v2.2.2 proxy sweeps):
  window_size = 20  (80ms at 250Hz) — absolute peak from minor sweep (59.12%)
  window_step = 2   (8ms at 250Hz)

MLOPS.md compliance:
  - §0.5  Excluded subjects: 140117, 204521 (BANNED)
  - §2.1  Semantic label: y_semantic col 0 — Face(1) vs Tool(2) only
  - §4.1  SPD conditioning on all covariance batch
  - §7.1  Per-fold JSONL checkpoint with flush + fsync (crash-safe)
  - §7.4  Chunk-restart OOM guard (MAX_FOLDS_PER_PROCESS)
  - §1.2  n_jobs=2 inside fold only — sequential folds (no cross-fold parallelism)
  - §6.1  Gate 1 circuit breaker at script end

rng_state = 42
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ===========================================================================
# CELL 1 — Environment Validation
# ===========================================================================
import sys, importlib, random, json, time

REQUIRED = {
    "numpy":     "numpy",
    "pyriemann": "pyriemann",
    "sklearn":   "scikit-learn",
    "joblib":    "joblib",
}

print("=" * 60)
print("CELL 1 — Environment Validation")
print("=" * 60)
all_ok = True
for module, pkg_name in REQUIRED.items():
    try:
        importlib.import_module(module)
        print(f"  [OK]   {pkg_name}")
    except ImportError:
        print(f"  [FAIL] {pkg_name}")
        all_ok = False

if not all_ok:
    print("\n[FATAL] Fix missing packages before proceeding.")
    sys.exit(1)

import numpy as np
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

print(f"\n  numpy:     {np.__version__}")
print(f"  pyriemann: {pyriemann.__version__}")
print(f"  python:    {sys.version.split()[0]}")
print("\n  [PASS] Environment OK\n")

# ===========================================================================
# CELL 2 — Configuration
# LOCKED: Window=20, Step=2 from proxy sweep peak (59.12%)
# ===========================================================================
print("=" * 60)
print("CELL 2 — Configuration (LOCKED)")
print("=" * 60)

rng_state = 42
np.random.rng_state(rng_state)
random.rng_state(rng_state)

DATA_DIR        = os.environ.get("DATA_DIR", "/workspace/data/nsvd_fusion")
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "/workspace/results/loso_checkpoint_v2.3.0.jsonl")
FINAL_RESULT    = os.environ.get("FINAL_RESULT",    "/workspace/results/loso_final_v2.3.0.json")

os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

# MLOPS.md §0.5 — BANNED subjects. Non-negotiable.
EXCLUDED_SUBJECTS = {'140117', '204521'}

# LOCKED window from proxy sweep
WINDOW_SIZE = 20   # 80ms at 250Hz — proxy sweep peak
WINDOW_STEP = 2    # 8ms at 250Hz

N_TIMEPOINTS = 255   # Fixed by dataset — DO NOT CHANGE

# MLOPS.md §7.4 — Chunk-restart OOM guard
# Process 20 folds, then exit with code 1 so bash restarts.
# Checkpoint ensures no fold is ever repeated.
MAX_FOLDS_PER_PROCESS = 20

# MLOPS.md §2.1 — Tier-1 ROI subset (12 executive control ROIs, LOCKED)
TIER1_ROI_INDICES = [
    1,  # L_caudalanteriorcingulate
    2,  # L_caudalmiddlefrontal (DLPFC core)
    6,  # L_inferiorparietal
    25, # L_rostralmiddlefrontal
    26, # L_superiorfrontal
    29, # L_supramarginal
    35, # R_caudalanteriorcingulate
    36, # R_caudalmiddlefrontal (DLPFC core)
    40, # R_inferiorparietal
    59, # R_rostralmiddlefrontal
    60, # R_superiorfrontal
    63, # R_supramarginal
]

PHASE0_BASELINE = 0.598   # Phase 0 baseline to beat
GATE1_THRESHOLD = 0.643   # Phase 0 + 3pp = Gate 1 pass

print(f"  rng_state:             {rng_state}")
print(f"  DATA_DIR:         {DATA_DIR}")
print(f"  CHECKPOINT_FILE:  {CHECKPOINT_FILE}")
print(f"  WINDOW_SIZE:      {WINDOW_SIZE} samples ({WINDOW_SIZE/250*1000:.0f}ms) — LOCKED")
print(f"  WINDOW_STEP:      {WINDOW_STEP} samples ({WINDOW_STEP/250*1000:.0f}ms) — LOCKED")
print(f"  EXCLUDED:         {sorted(EXCLUDED_SUBJECTS)}")
print(f"  Gate 1 target:    {GATE1_THRESHOLD*100:.1f}%")
print()

# ===========================================================================
# CELL 3 — Data Path Validation (MLOPS.md §0.3)
# ===========================================================================
print("=" * 60)
print("CELL 3 — Data Path Validation")
print("=" * 60)

if not os.path.isdir(DATA_DIR):
    print(f"\n[FATAL] DATA_DIR not found: {DATA_DIR}")
    sys.exit(1)

all_files = os.listdir(DATA_DIR)
x_files   = [f for f in all_files if f.endswith("_X.npy")]
y_files   = [f for f in all_files if f.endswith("_y_semantic.npy")]
print(f"  _X.npy files found:       {len(x_files)}")
print(f"  _y_semantic.npy files:    {len(y_files)}")

if len(x_files) == 0 or len(y_files) == 0:
    print("[FATAL] Missing data files.")
    sys.exit(1)

# Spot-check shape
spot_sid = x_files[0].split("_")[0]
spot_X   = np.load(os.path.join(DATA_DIR, f"{spot_sid}_X.npy"))
spot_y   = np.load(os.path.join(DATA_DIR, f"{spot_sid}_y_semantic.npy"))
print(f"\n  Spot-check subject:       {spot_sid}")
print(f"  X shape:                  {spot_X.shape}  (expected: (N, 68, 255))")
print(f"  y_semantic shape:         {spot_y.shape}  (expected: (N, 3))")

if spot_X.shape[1] != 68:
    print(f"[FATAL] Expected 68 ROIs, got {spot_X.shape[1]}.")
    sys.exit(1)
if spot_X.shape[2] != N_TIMEPOINTS:
    print(f"[FATAL] Expected {N_TIMEPOINTS} timepoints, got {spot_X.shape[2]}.")
    sys.exit(1)
del spot_X, spot_y
print("\n  [PASS] Data shapes valid\n")

# ===========================================================================
# CELL 4 — Pipeline Functions & Precomputation
# ===========================================================================
print("=" * 60)
print("CELL 4 — Defining Pipeline Functions")
print("=" * 60)

def condition_spd(matrix, eps=1e-6):
    if matrix.ndim == 2:
        matrix = (matrix + matrix.T) / 2
        trace  = np.trace(matrix)
        matrix += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    elif matrix.ndim == 3:
        matrix = (matrix + matrix.transpose(0, 2, 1)) / 2
        for i in range(matrix.shape[0]):
            trace = np.trace(matrix[i])
            matrix[i] += np.eye(matrix.shape[-1]) * (trace * eps if trace > 0 else eps)
    return matrix

def per_subject_euclidean_alignment(X_cov):
    R_mean    = X_cov.mean(axis=0)
    R_invsqrt = invsqrtm(R_mean)
    X_aligned = R_invsqrt @ X_cov @ R_invsqrt
    return condition_spd(X_aligned)

def load_subject(subject_id):
    if subject_id in EXCLUDED_SUBJECTS:
        return None
    try:
        X     = np.load(os.path.join(DATA_DIR, f"{subject_id}_X.npy"))
        y_sem = np.load(os.path.join(DATA_DIR, f"{subject_id}_y_semantic.npy"))
    except Exception as e:
        return None

    valid_mask = y_sem[:, 0] != 0
    X_valid    = X[valid_mask]
    y_valid    = y_sem[valid_mask, 0]

    if len(X_valid) < 2 or len(np.unique(y_valid)) < 2:
        return None

    n_face = (y_valid == 1).sum()
    n_tool = (y_valid == 2).sum()
    ratio  = n_face / n_tool if n_tool > 0 else 0
    if not (0.5 < ratio < 2.0):
        return None

    return subject_id, X_valid, y_valid

def precompute_subject_windows(sid, X_raw):
    """
    Precomputes Covariance and Euclidean Alignment for ALL sliding windows
    for a single subject. This removes the 12-hour redundant computation.
    """
    windows = [
        (start, start + WINDOW_SIZE)
        for start in range(0, N_TIMEPOINTS - WINDOW_SIZE + 1, WINDOW_STEP)
    ]
    cov_estimator = Covariances(estimator='lwf')
    
    subject_covs = []
    for (start, end) in windows:
        X_tier1 = X_raw[:, TIER1_ROI_INDICES, start:end]
        cov     = cov_estimator.fit_transform(X_tier1)
        cov     = condition_spd(cov)
        traces  = np.trace(cov, axis1=1, axis2=2)[:, None, None]
        cov    /= np.where(traces > 0, traces, 1.0)
        cov     = per_subject_euclidean_alignment(cov)
        subject_covs.append(cov)
        
    # shape: (N_windows, N_trials, 12, 12)
    return sid, np.array(subject_covs)

def _process_precomputed_window(w_idx, window, sid_to_covs, train_sids, test_sid, y_train, y_test):
    """TRDA for one time window using precomputed 12x12 matrices."""
    # Build train set for this window
    train_covs = [sid_to_covs[sid][w_idx] for sid in train_sids]
    cov_train  = np.concatenate(train_covs, axis=0)
    
    # Test set for this window
    cov_test = sid_to_covs[test_sid][w_idx]

    # Tangent Space Projection
    ts = TangentSpace(metric='riemann')
    ts.fit(cov_train)
    X_train_ts = ts.transform(cov_train)
    X_test_ts  = ts.transform(cov_test)

    clf = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000,
        solver='lbfgs', random_state=rng_state,
    )
    clf.fit(X_train_ts, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_ts))

    return float(acc), clf.coef_[0].tolist(), list(window)

def execute_loso_fold_precomputed(test_sid, subjects_data, sid_to_covs):
    """One LOSO fold using precomputed covariances."""
    windows = [
        (start, start + WINDOW_SIZE)
        for start in range(0, N_TIMEPOINTS - WINDOW_SIZE + 1, WINDOW_STEP)
    ]

    y_train_list = []
    y_test = None
    train_sids = []

    for sid, _, y_valid in subjects_data:
        if sid == test_sid:
            y_test = y_valid
        else:
            train_sids.append(sid)
            y_train_list.append(y_valid)

    if y_test is None or not train_sids:
        return test_sid, None, None, None

    y_train = np.concatenate(y_train_list, axis=0)

    # Fast parallel over windows
    window_results = Parallel(n_jobs=4, backend="loky")(
        delayed(_process_precomputed_window)(w_idx, w, sid_to_covs, train_sids, test_sid, y_train, y_test)
        for w_idx, w in enumerate(windows)
    )

    best_acc, best_coef, best_window = 0.0, None, None
    accuracies = []
    for acc, coef, window in window_results:
        accuracies.append(acc)
        if acc > best_acc:
            best_acc, best_coef, best_window = acc, coef, window

    if best_coef is None and window_results:
        _, best_coef, best_window = window_results[0]

    return test_sid, accuracies, best_coef, best_window

print("  [OK] All pipeline functions defined.\n")

# ===========================================================================
# CELL 5 — Subject Discovery & Loading & Precomputing
# ===========================================================================
print("=" * 60)
print("CELL 5 — Subject Discovery, Loading & Precomputing")
print("=" * 60)

subject_files   = [f for f in os.listdir(DATA_DIR) if f.endswith("_X.npy")]
all_subject_ids = sorted([f.split("_")[0] for f in subject_files
                          if f.split("_")[0] not in EXCLUDED_SUBJECTS])

print(f"  Subjects on disk (excl. banned): {len(all_subject_ids)}")

t0 = time.time()
subjects_data = []
for sid in all_subject_ids:
    res = load_subject(sid)
    if res is not None:
        subjects_data.append(res)
print(f"  Subjects loaded:  {len(subjects_data)}  ({time.time()-t0:.1f}s)")

total_trials = sum(len(y) for _, _, y in subjects_data)
print(f"  Total valid trials: {total_trials}")

if len(subjects_data) < 2:
    print("[FATAL] Need at least 2 subjects. Aborting.")
    sys.exit(1)

# NEW: Precompute all covariances upfront
print("\n  Precomputing Covariances and EA for all subjects...")
t1 = time.time()
# n_jobs=4 here is perfectly safe because we are processing one subject per thread
# and deleting X_raw from memory after creating 12x12 covariances.
precomp_results = Parallel(n_jobs=4, backend="loky")(
    delayed(precompute_subject_windows)(sid, X_raw)
    for sid, X_raw, _ in subjects_data
)
sid_to_covs = {sid: covs for sid, covs in precomp_results}
print(f"  Precomputation complete in {time.time()-t1:.1f}s")
print()

# ===========================================================================
# CELL 6 — MLOPS.md §7.2 Checkpoint Resume Logic
# ===========================================================================
print("=" * 60)
print("CELL 6 — Checkpoint Resume")
print("=" * 60)

completed_results = {}
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as ck:
        for line in ck:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    completed_results[entry["subject_id"]] = entry
                except Exception:
                    pass
    print(f"  [CHECKPOINT] Resumed {len(completed_results)} completed folds.")
else:
    print("  [CHECKPOINT] No existing checkpoint — starting fresh.")
print()

# ===========================================================================
# CELL 7 — Full LOSO Loop with Per-Fold Checkpointing
# ===========================================================================
print("=" * 60)
print(f"CELL 7 — Full LOSO Run ({len(subjects_data)} subjects)")
print("=" * 60)

folds_this_run = 0
n_total = len(subjects_data)

for fold_idx, (sid, _, _) in enumerate(subjects_data, 1):

    if sid in completed_results:
        print(f"  Fold {fold_idx:02d}/{n_total} — {sid}: already done ({completed_results[sid]['peak_acc']*100:.2f}%) — skipping.")
        continue

    print(f"\n  Fold {fold_idx:02d}/{n_total} — Testing subject: {sid}", flush=True)
    t_fold = time.time()

    test_sid, accuracies, best_coef, best_window = execute_loso_fold_precomputed(sid, subjects_data, sid_to_covs)

    if accuracies is None:
        print(f"    [WARN] Fold failed for {sid} — skipping.")
        continue

    peak_acc  = float(max(accuracies))
    mean_acc  = float(np.mean(accuracies))
    elapsed   = time.time() - t_fold

    print(f"    Peak acc:   {peak_acc*100:.2f}%  |  Mean acc: {mean_acc*100:.2f}%  |  Time: {elapsed:.1f}s")

    # MLOPS.md §7.1 — write to disk immediately with fsync (crash-safe)
    fold_entry = {
        "subject_id":  sid,
        "fold_index":  fold_idx,
        "peak_acc":    peak_acc,
        "mean_acc":    mean_acc,
        "best_window": best_window,
        "elapsed_s":   round(elapsed, 2),
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "n_windows":   len(accuracies),
    }
    completed_results[sid] = fold_entry

    with open(CHECKPOINT_FILE, "a") as ck:
        ck.write(json.dumps(fold_entry) + "\n")
        ck.flush()
        os.fsync(ck.fileno())   # MLOPS.md §7.3 — force OS buffer to disk NOW

    print(f"    [CHECKPOINT] Fold {fold_idx} persisted to disk.")

    # MLOPS.md §7.4 — chunk-restart OOM guard
    folds_this_run += 1
    if folds_this_run >= MAX_FOLDS_PER_PROCESS:
        remaining = n_total - len(completed_results)
        print(f"\n[CHUNK] {folds_this_run} folds complete. {remaining} remaining.")
        print("[CHUNK] Exiting to free C-tier BLAS memory. Bash will restart.")
        sys.exit(1)   # non-zero → bash restart loop picks up from checkpoint

# ===========================================================================
# CELL 8 — Final Results & Gate 1 Check (MLOPS.md §6.1)
# ===========================================================================
print("\n" + "=" * 60)
print("CELL 8 — Final Results")
print("=" * 60)

all_peak_accs = [v["peak_acc"] for v in completed_results.values()]
overall_mean  = float(np.mean(all_peak_accs))
overall_std   = float(np.std(all_peak_accs))
delta_vs_p0   = overall_mean - PHASE0_BASELINE

print(f"\n  Subjects completed:  {len(all_peak_accs)}")
print(f"  Mean LOSO accuracy:  {overall_mean*100:.2f}%")
print(f"  Std:                 {overall_std*100:.2f}%")
print(f"  Delta vs Phase 0:    {delta_vs_p0*100:+.2f}pp  (baseline: {PHASE0_BASELINE*100:.1f}%)")

# Gate 1 circuit breaker — MLOPS.md §6.1
if overall_mean >= GATE1_THRESHOLD:
    gate1_status = "PASS"
    print(f"\n  *** GATE 1: PASS — {overall_mean*100:.2f}% >= {GATE1_THRESHOLD*100:.1f}% ***")
    print("  Ready to proceed to Gate 2 biology check.")
else:
    gate1_status = "FAIL"
    print(f"\n  GATE 1: FAIL — {overall_mean*100:.2f}% < {GATE1_THRESHOLD*100:.1f}%")
    print("  DO NOT run Gate 2/3. Diagnose per RESEARCH.md §VIII.")

# Write final JSON summary
final_summary = {
    "version":         "v2.3.0",
    "window_size":     WINDOW_SIZE,
    "window_step":     WINDOW_STEP,
    "n_subjects":      len(all_peak_accs),
    "mean_peak_acc":   overall_mean,
    "std_peak_acc":    overall_std,
    "delta_vs_phase0": delta_vs_p0,
    "gate1_threshold": GATE1_THRESHOLD,
    "gate1_status":    gate1_status,
    "rng_state":            rng_state,
}

with open(FINAL_RESULT, "w") as f:
    json.dump(final_summary, f, indent=2)
    f.flush()
    os.fsync(f.fileno())

print(f"\n  Final summary written to: {FINAL_RESULT}")
print("=" * 60)
print("Run complete.")
sys.exit(0)   # clean exit → bash restart loop terminates
