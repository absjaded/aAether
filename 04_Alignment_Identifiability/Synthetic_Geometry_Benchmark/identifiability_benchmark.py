#!/usr/bin/env python3
"""Synthetic failure map for cross-space alignment identifiability.

The generator creates two observation spaces from one hidden task state. Labels
and true point pairs are retained only for evaluation. Correspondence-free
methods receive intra-space distances; anchor methods receive exactly the
declared paired points.
"""

import argparse
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_CUPY", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

SEED = 20260715
RICHNESS = ("binary", "ordinal", "compositional")
SYMMETRY = ("symmetric", "asymmetric")


@dataclass(frozen=True)
class Cell:
    richness: str
    symmetry: str
    distortion: float
    noise: float
    rep: int
    n_points: int
    obs_dim: int
    anchors: tuple
    epsilon: float
    fused_alpha: float
    null_permutations: int
    seed: int


def parse_csv(text, cast=float):
    return tuple(cast(x.strip()) for x in text.split(",") if x.strip())


def orthogonal_matrix(dim, rng):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    q *= np.sign(np.diag(r) + 1e-12)
    return q


def standardized(x):
    x = np.asarray(x, dtype=float)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x - x.mean(axis=0)) / scale

def normalized_space(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
    return x / max(float(scale), 1e-12)

def latent_features(richness, symmetry, n_points, rng):
    """Return latent features and labels; no observation-space basis is shared."""
    asymmetric = symmetry == "asymmetric"
    if richness == "binary":
        n1 = int(round(n_points * (0.65 if asymmetric else 0.50)))
        y = np.r_[np.zeros(n_points - n1, dtype=int), np.ones(n1, dtype=int)]
        rng.shuffle(y)
        s = 2.0 * y - 1.0
        phi = s[:, None]
        point_noise_scale = np.where(y == 0, 0.65 if asymmetric else 1.0, 1.35 if asymmetric else 1.0)
        return phi, y, point_noise_scale

    if richness == "ordinal":
        t = np.linspace(-1.0, 1.0, n_points)
        rng.shuffle(t)
        bend = 0.55 * t**2
        if asymmetric:
            bend += 0.35 * t**3 + 0.12 * t
        phi = np.c_[t, bend]
        y = np.digitize(t, [-0.5, 0.0, 0.5]).astype(int)
        return phi, y, np.ones(n_points)

    if richness == "compositional":
        side = max(3, int(round(math.sqrt(n_points))))
        axis = np.linspace(-1.0, 1.0, side)
        x, z = np.meshgrid(axis, axis, indexing="ij")
        x, z = x.ravel(), z.ravel()
        order = rng.permutation(len(x))
        x, z = x[order], z[order]
        if asymmetric:
            xa = x + 0.22 * (z + 1.0) ** 2 + 0.08 * x * z
            za = z + 0.13 * x**2 - 0.05 * x
            phi = np.c_[xa, za, 0.55 * x * z, 0.22 * (x**2 + z**2)]
        else:
            phi = np.c_[x, z]
        y = ((x >= 0).astype(int) + 2 * (z >= 0).astype(int)).astype(int)
        return phi, y, np.ones(len(x))

    raise ValueError(f"unknown richness: {richness}")


def make_spaces(cell):
    rng = np.random.default_rng(cell.seed)
    phi, labels, point_noise_scale = latent_features(
        cell.richness, cell.symmetry, cell.n_points, rng
    )
    n = len(phi)
    latent = standardized(phi)
    if latent.shape[1] > cell.obs_dim:
        raise ValueError("obs_dim must be at least the latent feature dimension")
    canonical = np.zeros((n, cell.obs_dim), dtype=float)
    canonical[:, : latent.shape[1]] = latent
    canonical = normalized_space(canonical)

    qa = orthogonal_matrix(cell.obs_dim, rng)
    qb = orthogonal_matrix(cell.obs_dim, rng)
    a_clean = canonical @ qa

    warp_basis = orthogonal_matrix(cell.obs_dim, rng)
    warped = np.tanh(1.3 * canonical @ warp_basis) @ warp_basis.T
    b_clean = (canonical + cell.distortion * warped) @ qb

    base_sd = float(np.sqrt(np.mean(canonical**2)))
    na = rng.normal(size=(n, cell.obs_dim)) * point_noise_scale[:, None]
    nb = rng.normal(size=(n, cell.obs_dim)) * point_noise_scale[:, None]
    a = normalized_space(a_clean + cell.noise * base_sd * na)
    b = normalized_space(b_clean + cell.noise * base_sd * nb)

    b_order = rng.permutation(n)
    b = b[b_order]
    labels_b = labels[b_order]
    true_pair = np.empty(n, dtype=int)
    true_pair[b_order] = np.arange(n)
    return a, b, labels, labels_b, true_pair


def normalized_distances(x):
    d = squareform(pdist(x, metric="sqeuclidean"))
    scale = float(d.max())
    return d / max(scale, 1e-12)


def distance_signatures(d, width=24):
    q = np.linspace(0.02, 0.98, min(width, max(4, d.shape[0] - 1)))
    sig = np.quantile(d, q, axis=1).T
    return standardized(sig)


def coupling_entropy(t):
    p = t / max(float(t.sum()), 1e-30)
    h = -float(np.sum(p * np.log(p + 1e-30)))
    return h / max(math.log(p.size), 1e-12)


def coupling_to_map(t):
    rows, cols = linear_sum_assignment(-np.asarray(t))
    mapping = np.empty(t.shape[0], dtype=int)
    mapping[rows] = cols
    return mapping


def cost_to_map(cost):
    rows, cols = linear_sum_assignment(np.asarray(cost))
    mapping = np.empty(cost.shape[0], dtype=int)
    mapping[rows] = cols
    return mapping


def score_map(
    mapping,
    labels_a,
    labels_b,
    true_pair,
    da,
    db,
    method,
    rng,
    null_permutations,
    entropy=None,
    anchors=None,
):
    n = len(mapping)
    label_acc = float(np.mean(labels_a == labels_b[mapping]))
    label_chance = 0.0
    for cls in np.union1d(labels_a, labels_b):
        label_chance += float(np.mean(labels_a == cls) * np.mean(labels_b == cls))
    pair_acc = float(np.mean(mapping == true_pair))
    null_accuracies = np.empty(null_permutations, dtype=float)
    for i in range(null_permutations):
        shuffled = labels_b[rng.permutation(n)]
        null_accuracies[i] = np.mean(labels_a == shuffled[mapping])
    label_perm_p = float(
        (1 + np.sum(null_accuracies >= label_acc)) / (null_permutations + 1)
    )
    tri = np.triu_indices(n, 1)
    mapped_db = db[np.ix_(mapping, mapping)]
    corr = spearmanr(da[tri], mapped_db[tri]).statistic
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "method": method,
        "anchors": anchors,
        "label_accuracy": label_acc,
        "label_chance": label_chance,
        "label_edge": label_acc - label_chance,
        "label_perm_p": label_perm_p,
        "label_perm_significant": bool(label_perm_p <= 0.05),
        "pair_accuracy": pair_acc,
        "pair_chance": 1.0 / n,
        "structure_r": float(corr),
        "degeneracy": None if entropy is None else float(entropy),
    }


def gw_coupling(da, db, epsilon):
    import ot

    p = ot.unif(len(da))
    q = ot.unif(len(db))
    return ot.gromov.entropic_gromov_wasserstein(
        da,
        db,
        p,
        q,
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=300,
        tol=1e-7,
        verbose=False,
    )


def fused_gw_coupling(da, db, sa, sb, alpha):
    import ot

    m = cdist(sa, sb, metric="sqeuclidean")
    positive = m[m > 0]
    if positive.size:
        m /= np.median(positive)
    p = ot.unif(len(da))
    q = ot.unif(len(db))
    return ot.gromov.fused_gromov_wasserstein(
        m,
        da,
        db,
        p,
        q,
        loss_fun="square_loss",
        alpha=alpha,
        armijo=False,
        max_iter=300,
        tol_rel=1e-7,
        tol_abs=1e-7,
    )


def procrustes_map(a, b, a_idx, b_idx):
    r, _ = orthogonal_procrustes(a[a_idx], b[b_idx])
    return cost_to_map(cdist(a @ r, b, metric="sqeuclidean"))


def ridge_map(a, b, a_idx, b_idx, alpha=1e-2):
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha, fit_intercept=True).fit(a[a_idx], b[b_idx])
    return cost_to_map(cdist(model.predict(a), b, metric="sqeuclidean"))


def cell_run(cell_dict):
    cell = Cell(**cell_dict)
    rng = np.random.default_rng(cell.seed + 17)
    a, b, labels_a, labels_b, true_pair = make_spaces(cell)
    da, db = normalized_distances(a), normalized_distances(b)
    sa, sb = distance_signatures(da), distance_signatures(db)
    results = []

    signature_map = cost_to_map(cdist(sa, sb, metric="sqeuclidean"))
    results.append(
        score_map(
            signature_map,
            labels_a,
            labels_b,
            true_pair,
            da,
            db,
            "distance_signature",
            rng,
            cell.null_permutations,
        )
    )

    t_gw = gw_coupling(da, db, cell.epsilon)
    results.append(
        score_map(
            coupling_to_map(t_gw),
            labels_a,
            labels_b,
            true_pair,
            da,
            db,
            "gw",
            rng,
            cell.null_permutations,
            coupling_entropy(t_gw),
        )
    )

    t_fgw = fused_gw_coupling(da, db, sa, sb, cell.fused_alpha)
    results.append(
        score_map(
            coupling_to_map(t_fgw),
            labels_a,
            labels_b,
            true_pair,
            da,
            db,
            "signature_fgw",
            rng,
            cell.null_permutations,
            coupling_entropy(t_fgw),
        )
    )

    order = rng.permutation(len(a))
    for k in cell.anchors:
        k_eff = min(int(k), len(a) - 2)
        anchor_idx = order[:k_eff]
        mapping = procrustes_map(a, b, anchor_idx, true_pair[anchor_idx])
        results.append(
            score_map(
                mapping,
                labels_a,
                labels_b,
                true_pair,
                da,
                db,
                "anchor_procrustes",
                rng,
                cell.null_permutations,
                anchors=k_eff,
            )
        )

    train_n = max(cell.obs_dim + 2, len(a) // 2)
    train_idx = order[:train_n]
    results.append(
        score_map(
            procrustes_map(a, b, train_idx, true_pair[train_idx]),
            labels_a,
            labels_b,
            true_pair,
            da,
            db,
            "paired_procrustes",
            rng,
            cell.null_permutations,
            anchors=train_n,
        )
    )
    results.append(
        score_map(
            ridge_map(a, b, train_idx, true_pair[train_idx]),
            labels_a,
            labels_b,
            true_pair,
            da,
            db,
            "paired_ridge",
            rng,
            cell.null_permutations,
            anchors=train_n,
        )
    )

    base = {
        "richness": cell.richness,
        "symmetry": cell.symmetry,
        "distortion": cell.distortion,
        "noise": cell.noise,
        "rep": cell.rep,
        "n_points": len(a),
        "obs_dim": cell.obs_dim,
        "seed": cell.seed,
    }
    return [{**base, **row} for row in results]


def mean_or_none(series):
    values = [float(x) for x in series if x is not None and np.isfinite(float(x))]
    return float(np.mean(values)) if values else None


def summarize(rows):
    keys = ("richness", "symmetry", "distortion", "noise", "method", "anchors")
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        out = dict(zip(keys, key))
        out["reps"] = len(group)
        for metric in (
            "label_accuracy",
            "label_chance",
            "label_edge",
            "label_perm_p",
            "label_perm_significant",
            "pair_accuracy",
            "pair_chance",
            "structure_r",
            "degeneracy",
        ):
            out[metric] = mean_or_none([r[metric] for r in group])
        output.append(out)
    return sorted(
        output,
        key=lambda r: (
            RICHNESS.index(r["richness"]),
            SYMMETRY.index(r["symmetry"]),
            r["distortion"],
            r["noise"],
            r["method"],
            -1 if r["anchors"] is None else r["anchors"],
        ),
    )


def method_pass(row):
    nondegenerate = row["degeneracy"] is None or row["degeneracy"] <= 0.97
    return (
        row["label_edge"] >= 0.15
        and row["label_perm_significant"] >= 0.70
        and row["structure_r"] >= 0.50
        and nondegenerate
    )


def classify_limits(summary):
    design_keys = ("richness", "symmetry", "distortion", "noise")
    grouped = {}
    for row in summary:
        key = tuple(row[k] for k in design_keys)
        grouped.setdefault(key, []).append(row)
    limits = []
    cf_methods = {"distance_signature", "gw", "signature_fgw"}
    for key, group in grouped.items():
        by_method = {}
        for row in group:
            by_method.setdefault(row["method"], []).append(row)
        cf = [r for r in group if r["method"] in cf_methods and method_pass(r)]
        anchors = sorted(
            [r for r in by_method.get("anchor_procrustes", []) if method_pass(r)],
            key=lambda r: r["anchors"],
        )
        paired_proc = by_method.get("paired_procrustes", [None])[0]
        paired_ridge = by_method.get("paired_ridge", [None])[0]
        if cf:
            best = max(cf, key=lambda r: (r["label_edge"], r["structure_r"]))
            verdict = "world_b"
            winning_method = best["method"]
            min_anchors = 0
        elif anchors:
            verdict = f"world_a_{int(anchors[0]['anchors'])}"
            winning_method = "anchor_procrustes"
            min_anchors = int(anchors[0]["anchors"])
        elif paired_ridge is not None and method_pass(paired_ridge):
            verdict = "paired_only"
            winning_method = "paired_ridge"
            min_anchors = None
        else:
            verdict = "representation_limit"
            winning_method = None
            min_anchors = None
        limits.append(
            {
                **dict(zip(design_keys, key)),
                "verdict": verdict,
                "winning_method": winning_method,
                "min_anchors": min_anchors,
                "paired_procrustes_pass": bool(
                    paired_proc is not None and method_pass(paired_proc)
                ),
                "paired_ridge_pass": bool(
                    paired_ridge is not None and method_pass(paired_ridge)
                ),
            }
        )
    return sorted(
        limits,
        key=lambda r: (
            RICHNESS.index(r["richness"]),
            SYMMETRY.index(r["symmetry"]),
            r["distortion"],
            r["noise"],
        ),
    )


def write_csv(path, rows):
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--richness", default=",".join(RICHNESS))
    parser.add_argument("--symmetry", default=",".join(SYMMETRY))
    parser.add_argument("--distortion-values", default="0,0.15,0.30")
    parser.add_argument("--noise-values", default="0.03,0.10,0.20")
    parser.add_argument("--anchor-values", default="1,2,4,8,16,32")
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--fused-alpha", type=float, default=0.70)
    parser.add_argument("--null-permutations", type=int, default=99)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default="04_Alignment_Identifiability/Synthetic_Geometry_Benchmark/outputs/identifiability")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    richness = parse_csv(args.richness, str)
    symmetry = parse_csv(args.symmetry, str)
    bad_richness = sorted(set(richness) - set(RICHNESS))
    bad_symmetry = sorted(set(symmetry) - set(SYMMETRY))
    if bad_richness or bad_symmetry:
        raise SystemExit(f"invalid regimes: richness={bad_richness}, symmetry={bad_symmetry}")

    distortions = parse_csv(args.distortion_values, float)
    noises = parse_csv(args.noise_values, float)
    anchors = parse_csv(args.anchor_values, int)
    reps = args.reps
    if args.quick:
        richness = ("ordinal",)
        symmetry = ("symmetric", "asymmetric")
        distortions = (0.0, 0.30)
        noises = (0.03, 0.20)
        anchors = (1, 4, 16)
        reps = min(reps, 2)

    cells = []
    cell_index = 0
    for r in richness:
        for s in symmetry:
            for d in distortions:
                for n in noises:
                    for rep in range(reps):
                        cells.append(
                            Cell(
                                richness=r,
                                symmetry=s,
                                distortion=d,
                                noise=n,
                                rep=rep,
                                n_points=args.n_points,
                                obs_dim=args.obs_dim,
                                anchors=anchors,
                                epsilon=args.epsilon,
                                fused_alpha=args.fused_alpha,
                                null_permutations=args.null_permutations,
                                seed=args.seed + 10007 * cell_index + rep,
                            )
                        )
                    cell_index += 1

    print(f"cells: {len(cells)} | workers: {min(args.workers, len(cells))}", flush=True)
    rows = []
    worker_count = min(max(1, args.workers), len(cells))
    if worker_count == 1:
        for i, cell in enumerate(cells, 1):
            rows.extend(cell_run(asdict(cell)))
            print(f"completed {i}/{len(cells)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(cell_run, asdict(cell)): cell for cell in cells}
            done = 0
            for future in as_completed(futures):
                rows.extend(future.result())
                done += 1
                if done == 1 or done % max(1, len(cells) // 20) == 0 or done == len(cells):
                    print(f"completed {done}/{len(cells)}", flush=True)

    rows = sorted(
        rows,
        key=lambda r: (
            RICHNESS.index(r["richness"]),
            SYMMETRY.index(r["symmetry"]),
            r["distortion"],
            r["noise"],
            r["rep"],
            r["method"],
            -1 if r["anchors"] is None else r["anchors"],
        ),
    )
    summary = summarize(rows)
    limits = classify_limits(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "identifiability_trials.csv", rows)
    write_csv(out_dir / "identifiability_summary.csv", summary)
    write_csv(out_dir / "identifiability_limits.csv", limits)
    payload = {
        "config": vars(args),
        "thresholds": {
            "label_edge": 0.15,
            "structure_r": 0.50,
            "max_degeneracy": 0.97,
            "min_label_permutation_pass_rate": 0.70,
        },
        "limits": limits,
        "summary": summary,
    }
    with (out_dir / "identifiability.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    counts = {}
    for row in limits:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()

